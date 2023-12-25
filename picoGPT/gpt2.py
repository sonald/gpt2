import numpy as np
import torch

from utils import load_encoder_hparams_and_params


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, epsilon=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + epsilon) * g + b


def linear(x, w, b):
    return x @ w + b


def attention(q, k, v, mask=None):
    a = q @ k.T / np.sqrt(k.shape[-1])
    if mask is not None:
        a = a + mask
    return softmax(a) @ v


def casual_self_attention(x, c_attn, c_proj):
    x = linear(x, **c_attn)
    q, k, v = np.split(x, 3, axis=-1)
    casual_mask = (1.0 - np.tri(x.shape[0])) * 1e-10
    a = attention(q, k, v, casual_mask)
    return linear(x, **c_proj)


def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    # print(f"{x.shape}")

    # S, 3*D -> S, 3, n_head, D // n_head -> 3, n_head, S, D // n_head
    qkv_heads = x.reshape(x.shape[0], 3, n_head,
                          x.shape[-1] // n_head // 3).transpose([1, 2, 0, 3])
    # print(f"{qkv_heads.shape=}")

    casual_mask = (1.0 - np.tri(x.shape[0])) * -1e10
    # n_head, S, D // n_head
    scores = [attention(q, k, v, casual_mask) for q, k, v in zip(*qkv_heads)]
    x = np.hstack(scores)

    x = linear(x, **c_proj)
    return x


def ffn(x, c_fc, c_proj):
    a = gelu(linear(x, **c_fc))
    return linear(a, **c_proj)


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def gpt2(input_ids, wte, wpe, blocks, ln_f, n_head):
    x = wte[input_ids] + wpe[range(len(input_ids))]  # (B, n_seq, n_embd)

    # print("-----------> ", x.shape)

    for b in blocks:
        x = transformer_block(x, **b, n_head=n_head)

    x = layer_norm(x, **ln_f)
    return x @ wte.T


def generate(inputs, params, n_head, max_new_tokens):
    from tqdm import tqdm

    for _ in tqdm(range(max_new_tokens), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_token_id = np.argmax(logits[-1])
        inputs.append(next_token_id)

    return inputs[len(inputs) - max_new_tokens:]


def load_model(model_size: str = "124M", models_dir: str = "models"):
    pass


def main(prompt: str, max_new_tokens: int = 10, model_size: str = "124M", models_dir: str = "models"):

    tokenizer, hparams, params = load_encoder_hparams_and_params(
        model_size, models_dir)

    input_ids = tokenizer.encode(prompt)

    output_ids = generate(
        input_ids, params, n_head=hparams["n_head"], max_new_tokens=max_new_tokens)
    response = tokenizer.decode(output_ids)
    print(response)
    return response


if __name__ == "__main__":
    import fire

    fire.Fire(main)
