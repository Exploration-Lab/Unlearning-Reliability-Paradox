#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

export HF_HOME="/DATA3/divyaksh/.cache/huggingface"

models=(
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)

splits=(
    "forget10 holdout10 retain90"
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
)

base_lora_path="../saves/unlearn/tofu"

methods=(
    "GradAscent"
    "GradDiff"
    "NPO"
    "DPO"
)


for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    
    for model in "${models[@]}"; do
        for method in "${methods[@]}"; do

            echo "Attribution for model: ${model}, method: ${method}, forget_split: ${forget_split}"
            CUDA_VISIBLE_DEVICES=0 python llama_inseq_token_attribution_unlearned.py \
                --model_name="meta-llama/${model}" \
                --model_path="meta-llama/${model}" \
                --lora_path="${base_lora_path}_${model}_${forget_split}_${method}" \
                --method_name="${method}" \
                --prompt_template="mcq" \
                --dataset_path="data/mcqa/${forget_split}.json" \
                --shuffle_choices \
                # --use_chat_template

            echo "Attribution for model: ${model}, method: ${method}, retain_split: ${retain_split}"
            CUDA_VISIBLE_DEVICES=0 python llama_inseq_token_attribution_unlearned.py \
                --model_name="meta-llama/${model}" \
                --model_path="meta-llama/${model}" \
                --lora_path="${base_lora_path}_${model}_${forget_split}_${method}" \
                --method_name="${method}" \
                --prompt_template="mcq" \
                --dataset_path="data/mcqa/${retain_split}.json" \
                --shuffle_choices \
                # --use_chat_template
        done
    done
done