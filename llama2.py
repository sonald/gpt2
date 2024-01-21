import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum
from transformers.modeling_utils import no_init_weights, ContextManagers
from transformers.activations import ACT2FN
from transformers import AutoTokenizer
import torch
import math
import json
import os


model_path = '/data/sonald/ai_models/model_weights/Llama-2-7b-hf'


class LlamaRMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rms_norm_eps = config['rms_norm_eps']
        self.weight = nn.Parameter(torch.ones(config['hidden_size']))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True) + self.rms_norm_eps
        x = x * torch.rsqrt(var)
        a = self.weight * x.to(dtype=dtype)
        return a


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config['hidden_size'], config['intermediate_size'], bias=False)
        self.up_proj = nn.Linear(
            config['hidden_size'], config['intermediate_size'], bias=False)
        self.down_proj = nn.Linear(
            config['intermediate_size'], config['hidden_size'], bias=False)
        self.act = ACT2FN[config['hidden_act']]

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.act(self.gate_proj(
            hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, base=10000, device=None):
        super().__init__()
        self.seq_len = 0
        self.d = config['hidden_size'] // config['num_attention_heads']
        inv_freq = 1.0 / \
            (base ** (torch.arange(0, self.d, 2, dtype=torch.float32).to(device) / self.d))
        self.register_buffer('inv_freq', inv_freq)
        self._update_sin_cos_cache(config['max_position_embeddings'])

    def _update_sin_cos_cache(self, seq_len):
        pos = torch.arange(0, seq_len, dtype=torch.float32).to(
            self.inv_freq.device)
        rot = einsum(pos, self.inv_freq, 'i,j -> i j')  # S, D
        theta = torch.cat([rot, rot], dim=-1)
        self.register_buffer('sin_cached', theta.sin()[
                             None, None, :, :], persistent=False)
        self.register_buffer('cos_cached', theta.cos()[
                             None, None, :, :], persistent=False)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[-2]
        if self.seq_len < seq_len:
            self._update_sin_cos_cache(seq_len)

        return (
            self.cos_cached[..., :seq_len, :].to(dtype=x.dtype),
            self.sin_cached[..., :seq_len, :].to(dtype=x.dtype)
        )


class LlamaMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(
            config['hidden_size'], config['hidden_size'], bias=False)
        self.k_proj = nn.Linear(
            config['hidden_size'], config['hidden_size'], bias=False)
        self.v_proj = nn.Linear(
            config['hidden_size'], config['hidden_size'], bias=False)
        self.o_proj = nn.Linear(
            config['hidden_size'], config['hidden_size'], bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(config)
        self.num_heads = config['num_attention_heads']

        D = config['max_position_embeddings']
        self.register_buffer('bias', torch.tril(torch.ones(D, D))[
                             None, None, :, :], persistent=False)

    def rotate_half(self, x):
        D = x.shape[-1] // 2
        x1 = x[..., :D]
        x2 = x[..., D:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, hidden_states: torch.Tensor):
        q = self.q_proj(hidden_states)  # B, S, D
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)

        cos, sin = self.rotary_emb(v)
        cos = cos.to(q.device)
        sin = sin.to(q.device)

        # apply RoPE
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin

        q_seq_len, kv_seq_len = q.shape[-2], v.shape[-2]
        # SDPA is the same
        # attn_mask = self.bias[:, :, :q_seq_len,
        #                       :kv_seq_len] == 1  # sdpa needs bool mask
        # attn = F.scaled_dot_product_attention(
        #     q, k, v, is_causal=True, dropout_p=0.0, attn_mask=attn_mask)
        scores = einsum(q, k, 'b h q d, b h k d -> b h q k') / \
            math.sqrt(q.shape[-1])
        scores = scores.masked_fill(
            self.bias[:, :, :q_seq_len, :kv_seq_len] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        attn = attn @ v

        attn = rearrange(attn, 'b h s d -> b s (h d)')
        hidden_states = self.o_proj(attn)
        return hidden_states


class LlamaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaMultiHeadAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)

    def forward(self, hidden_states):
        pre_normed = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(pre_normed) + hidden_states

        post_normed = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(post_normed) + hidden_states
        return hidden_states


class Llama2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config['vocab_size'], config['hidden_size'])
        self.layers = nn.ModuleList([LlamaLayer(config)
                                    for _ in range(config['num_hidden_layers'])])
        self.norm = LlamaRMSNorm(config)

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Llama2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Llama2Model(config)
        self.lm_head = nn.Linear(
            config['hidden_size'], config['vocab_size'], bias=False)

        if 'tie_word_embeddings' in config and config['tie_word_embeddings']:
            self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits.float(), hidden_states

    def from_pretrained(model_path: str):
        with open(os.path.join(model_path, 'config.json'), mode='r') as f:
            config = json.load(f)

        match config['torch_dtype']:
            case 'float16':
                dtype = torch.float16
            case 'bfloat16':
                dtype = torch.bfloat16
            case _:
                dtype = torch.float32
        torch.set_default_dtype(dtype)

        init_contexts = [no_init_weights(_enable=True)]
        with ContextManagers(init_contexts):
            # Let's make sure we don't run the init function of buffer modules
            model = Llama2ForCausalLM(config)
            model.eval()

        sd = model.state_dict()
        sd_keys = sd.keys()

        from tqdm import tqdm
        inited = set()
        print('loading state dict...')
        files = ['/pytorch_model-00001-of-00002.bin',
                 '/pytorch_model-00002-of-00002.bin']
        for shard in files:
            bin_sd = torch.load(model_path + shard)
            for key in tqdm(bin_sd.keys(), desc=shard):
                if key in sd_keys:
                    with torch.no_grad():
                        sd[key].copy_(bin_sd[key])
                        inited.add(key)

        # print(set(sd_keys) ^ (inited))
        return model

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, do_sample=False, max_new_tokens: int = 100):
        input_ids = input_ids[:, -
                              self.config['max_position_embeddings']:]  # (B, S)
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids)  # B, S, V
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)  # B, V
            if do_sample:
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)  # B, 1
            input_ids = torch.concat((input_ids, next_id), dim=1)

        return input_ids


def predict(model, tokenizer, prompt, do_sample=False, max_new_tokens=100):
    inputs = tokenizer(prompt, padding=True, return_tensors="pt")
    output = model.generate(inputs['input_ids'].to(
        'cuda'), do_sample=do_sample, max_new_tokens=max_new_tokens)
    response = tokenizer.batch_decode(output, skip_special_tokens=True)
    return response


def predict2(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer([prompt], padding=True, return_tensors="pt").to('cuda')
    output = model.generate(**inputs, do_sample=False,
                            temperature=1.0, max_new_tokens=max_new_tokens)
    response = tokenizer.batch_decode(output, skip_special_tokens=True)
    return response


if __name__ == "__main__":
    model = Llama2ForCausalLM.from_pretrained(model_path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0

    debug = False
    if debug:
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='auto')
        predict2(base, tokenizer, 'I enjoy walking with my cute dog',
                 max_new_tokens=50)
        exit

    text = 'I enjoy walking with my cute dog'
    resp = predict(model, tokenizer, [text], max_new_tokens=50)
    print(resp)

    truth = 'I enjoy walking with my cute dog, reading, and watching movies.\nI am a very friendly person and I love to help people. I am a very hard worker and I am very dedicated to my work. I am very patient and I am very good at listening to people'

    if resp[0] != truth:
        print("predict wrong")
