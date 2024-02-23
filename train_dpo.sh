model='/data/sonald/ai_models/model_weights/Llama-2-7b-hf'
model='/data/sonald/ai_models/model_weights/deepseek-coder-6.7b-base'

accelerate launch stackllama.py  \
    --model_name_or_path $model \
    --output_dir dpo-output \
    --do_train \
    --do_eval \
    --dataset Anthropic/hh-rlhf \
    --finetuning_type lora \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 50 \
    --eval_steps 10 \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --learning_rate 1e-4 \
    --optim paged_adamw_32bit \
    --bf16
