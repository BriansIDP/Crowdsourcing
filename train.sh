# . /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

trainfile="data/halueval_dialogue.json"
# trainfile="data/artificial.json"
# trainfile="data/wikibio_crosscheck_gpt3.json"
# expdir="exp/crosscheck_pew_gpt2_logistic_relu"
expdir=exp/brian_pewcrowd_K_10
# expdir=exp/pewcrowd_llama3_mse_direct_crowdlayer_constrained_freeze
# expdir=exp/transformer_gpt2_mse_artificial
mkdir -p $expdir
regression=hardlabel
mode=pewcrowd

python train_nn.py \
    --model_path gpt2 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --num_warmup_steps 0.03 \
    --weight_decay 0.0 \
    --lr_scheduler_type cosine \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 100 \
    --train_data_path $trainfile \
    --evidence_llm "llama3,beluga,mistral,zephyr,starling,openorca,dolphin,mistral1,hermes2,hermes25" \
    --regression $regression \
    --mode $mode \
    --split 0.1 \
    --freeze_epoch 200 \
    # --evidence_llm "llama3,beluga,mistral,zephyr,starling" \
    # --evidence_llm "mistral,llama2,vicuna,beluga,starling,openorca,gpt3" \
    # "system_0,system_1,system_2,system_3,system_4" \
    #  "llama3,beluga,mistral,zephyr,starling,openorca,dolphin,mistral1,hermes2,hermes25" \
# meta-llama/Meta-Llama-3-8B-Instruct \


# expdir=${expdir}_2
# 
# mkdir -p $expdir
# python train_nn.py \
#     --model_path gpt2 \
#     --batch_size 32 \
#     --learning_rate 1e-4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 10 \
#     --num_warmup_steps 0.03 \
#     --weight_decay 0.0 \
#     --lr_scheduler_type cosine \
#     --outputdir $expdir \
#     --logfile $expdir/log.txt \
#     --log_interval 200 \
#     --train_data_path $trainfile \
#     --evidence_llm "llama3,beluga,mistral,zephyr,starling,openorca,dolphin,mistral1,hermes2,hermes25" \
#     --regression $regression \
#     --mode $mode \
#     --split 0.2 \
# 
# expdir=${expdir}_3
# 
# mkdir -p $expdir
# python train_nn.py \
#     --model_path gpt2 \
#     --batch_size 32 \
#     --learning_rate 1e-4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 10 \
#     --num_warmup_steps 0.03 \
#     --weight_decay 0.0 \
#     --lr_scheduler_type cosine \
#     --outputdir $expdir \
#     --logfile $expdir/log.txt \
#     --log_interval 200 \
#     --train_data_path $trainfile \
#     --evidence_llm "llama3,beluga,mistral,zephyr,starling,openorca,dolphin,mistral1,hermes2,hermes25" \
#     --regression $regression \
#     --mode $mode \
#     --split 0.3 \

# expdir=${expdir}_4
# 
# mkdir -p $expdir
# python train_nn.py \
#     --model_path gpt2 \
#     --batch_size 8 \
#     --learning_rate 1e-4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 10 \
#     --num_warmup_steps 0.03 \
#     --weight_decay 0.0 \
#     --lr_scheduler_type cosine \
#     --outputdir $expdir \
#     --logfile $expdir/log.txt \
#     --log_interval 200 \
#     --train_data_path $trainfile \
#     --evidence_llm "llama3,beluga,mistral,zephyr,starling,openorca,dolphin,mistral1,hermes2,hermes25" \
#     --regression $regression \
#     --mode pewcrowd \
#     --split 0.4 \
# 
# expdir=${expdir}_5
# 
# mkdir -p $expdir
# python train_nn.py \
#     --model_path gpt2 \
#     --batch_size 8 \
#     --learning_rate 1e-4 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 10 \
#     --num_warmup_steps 0.03 \
#     --weight_decay 0.0 \
#     --lr_scheduler_type cosine \
#     --outputdir $expdir \
#     --logfile $expdir/log.txt \
#     --log_interval 200 \
#     --train_data_path $trainfile \
#     --evidence_llm "llama3,beluga,mistral,zephyr,starling,openorca,dolphin,mistral1,hermes2,hermes25" \
#     --regression $regression \
#     --mode pewcrowd \
#     --split 0.5 \
