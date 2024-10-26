. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination



# trainfile="data/artificial.json"
# trainfile="data/wikibio_crosscheck_gpt3.json"
expdir=exp/finetune_llama3_8B_ultrachat
mkdir -p $expdir

python finetune_llm.py \
    --train_data_path /data/milsrg1/corpora/ultrachat/train.json \
    --val_data_path /data/milsrg1/corpora/ultrachat/validation.json \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --batch_size 1 \
    --eval_batch_size 1 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 30 \
    --num_warmup_steps 0.03 \
    --weight_decay 0.0 \
    --lr_scheduler_type cosine \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 100 \
    --lora_config config/lora_config.json \