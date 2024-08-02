. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

trainfile="data/halueval_dialogue.json"
# trainfile="data/wikibio_crosscheck_gpt3.json"
# expdir="exp/crosscheck_pew_gpt2_logistic_relu"
expdir=exp/tfm_gpt2_mse_relu
mkdir -p $expdir

python train_nn.py \
    --model_path gpt2 \
    --batch_size 8 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --num_warmup_steps 0.03 \
    --weight_decay 0.0 \
    --lr_scheduler_type cosine \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 200 \
    --train_data_path $trainfile \
    --evidence_llm "llama3,beluga,mistral,zephyr,starling" \
    --regression mse \
    --mode tfm \
    # --split -0.5 \
    # --evidence_llm "mistral,llama2,vicuna,beluga,starling,openorca,gpt3" \
