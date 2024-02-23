from dataclasses import dataclass
from trl import DPOTrainer
from typing import Any, Dict, Optional
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, HfArgumentParser, Seq2SeqTrainingArguments
from datasets import load_dataset
import torch
import argparse
import logging
import sys
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
def get_current_device() -> str:
    import accelerate
    from accelerate import Accelerator
    dummy_accelerator = Accelerator()
    if accelerate.utils.is_xpu_available():
        return "xpu:{}".format(dummy_accelerator.local_process_index)
    else:
        return dummy_accelerator.local_process_index if torch.cuda.is_available() else "cpu"


print(f"{get_current_device()=}")


default_model_path = '/data/sonald/ai_models/model_weights/Llama-2-7b-hf'
default_output_dir = 'dpo-output'

def_args = TrainingArguments(
    output_dir=default_output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    optim='paged_adamw_32bit',
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=1000,
    eval_steps=100,
    evaluation_strategy='steps',
    save_total_limit=2,
    lr_scheduler_type='cosine',
    # report_to='wandb',
    load_best_model_at_end=True,
    remove_unused_columns=False,
    bf16=True,
)


@dataclass
class DPOArguments:
    """ arguments for dpo training """
    model_name_or_path: Optional[str] = default_model_path
    dataset: Optional[str] = 'Anthropic/hh-rlhf'
    finetuning_type: Optional[str] = 'lora'


parser = HfArgumentParser([DPOArguments, Seq2SeqTrainingArguments])
dpo_args, train_args = parser.parse_args_into_dataclasses()
# print(dpo_args)
# print(train_args)


def build_datasets(ds_name):
    def common_prefix(s, t):
        from itertools import takewhile
        return ''.join(c[0] for c in takewhile(lambda x: x[0] == x[1], zip(s, t)))

    def build_dpo_prompt_and_reponses(samples) -> Dict[str, Any]:
        s, t = samples['chosen'], samples['rejected']
        prefix = common_prefix(s, t)

        return {
            'prompt': prefix,
            'chosen': s[len(prefix):],
            'rejected': t[len(prefix):],
        }

    ds = load_dataset(ds_name, split='train')
    ds = ds.map(build_dpo_prompt_and_reponses, batched=False,
                num_proc=32, remove_columns=ds.column_names)
    dss = ds.train_test_split(test_size=0.1)
    return dss


dss = build_datasets(dpo_args.dataset)
train_ds = dss['train'].select(range(2000))
eval_ds = dss['test'].select(range(200))


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)
model = AutoModelForCausalLM.from_pretrained(
    dpo_args.model_name_or_path,
    quantization_config=bnb_config,
    # this is used to place the model on the right device, if not provided, the model will be placed on the first available device. which conflicts with the accelerator
    device_map={"": get_current_device()},
    # this is used to reduce the memory usage on the cpu, if not provided, the memory usage will be high
    low_cpu_mem_usage=True,
)
model.config.use_cache = False  # during the training, reduces the memory usage
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(dpo_args.model_name_or_path)
# tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)
model.train()

if dpo_args.finetuning_type == 'lora':
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['o_proj', 'q_proj', 'k_proj', 'v_proj'],
    )
# peft_model = get_peft_model(model, lora_config)
else:
    raise ValueError(f"Unknown finetuning type {dpo_args.finetuning_type}")


dpo = DPOTrainer(model=model,
                 ref_model=None,
                 tokenizer=tokenizer,
                 args=train_args,
                 beta=0.1,
                 train_dataset=train_ds,
                 eval_dataset=eval_ds,
                 dataset_num_proc=32,
                 max_prompt_length=2048,
                 max_length=4096,
                 peft_config=peft_config)
dpo.train()

dpo.save_model(train_args.output_dir + '/dpo-model')
