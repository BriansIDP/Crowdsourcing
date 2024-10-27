ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}

# trainfile="data/artificial.json"
# trainfile="data/wikibio_crosscheck_gpt3.json"
expdir=exp/finetune_llama3_8B_ultrachat
mkdir -p $expdir

# torchrun --nproc_per_node=1 --master_port=19995 \
#     finetune_llm.py \
# accelerate launch --config_file config/default_config.yaml --main_process_port=19995 finetune_llm.py \
torchrun --nproc_per_node=8 --master_port=19995 \
    finetune_llm.py \
        --train_data_path ../data/train.json \
        --val_data_path ../data/validation.json \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --batch_size 1 \
        --eval_batch_size 1 \
        --learning_rate 5e-5 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 30 \
        --num_warmup_steps 0.03 \
        --weight_decay 0.0 \
        --lr_scheduler_type cosine \
        --outputdir $expdir \
        --logfile $expdir/log.txt \
        --log_interval 1 \
        --lora_config config/lora_config.json \