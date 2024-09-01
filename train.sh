# . /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

# trainfile="data/halueval_dialogue.json"
trainfile=data/truthfulQA/truthful_qa.json
regression=hardlabel
# mode=pewcrowdae
# mode=compression
# task=halueval
mode=gt
task=truthfulqa

# trainfile="data/artificial.json"
# trainfile="data/wikibio_crosscheck_gpt3.json"
# expdir=exp/pewcrowd_gpt2_mse_direct_crowdlayer_constrained_Xtcondition
# expdir=exp/pewcrowd_gpt2_mse_direct_crowdlayer_${mode}_${regression}_${task}_5beluga
# expdir=exp/worker_compression_encoder_decoder_5beluga
expdir=exp/truthfulqa/gt_brian_sept_1
mkdir -p $expdir


# python train_nn.py \
#     --model_path gpt2 \
#     --batch_size 64 \
#     --learning_rate 2.5 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 50 \
#     --num_warmup_steps 0.1 \
#     --weight_decay 0.0 \
#     --lr_scheduler_type linear \
#     --outputdir $expdir \
#     --logfile $expdir/log.txt \
#     --log_interval 100 \
#     --train_data_path $trainfile \
#     --evidence_llm "llama3,beluga,mistral,zephyr,starling,openorca,beluga,beluga,beluga,beluga" \
#     --regression $regression \
#     --mode $mode \
#     --split 0.9 \
#     --target_nllms 7 \

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
#     --log_interval 100 \
#     --train_data_path $trainfile \
#     --evidence_llm "llama3,beluga,mistral,zephyr,starling,openorca,beluga,beluga,beluga,beluga" \
#     --regression $regression \
#     --mode $mode \
#     --split 0.1 \
#     --freeze_epoch 200 \
#     --reg_factor 0.5 \
#     --target_nllms 7 \
#     --encdecpath exp/worker_compression_encoder_decoder_CE_5beluga/checkpoint.49/pytorch_model.pt \
#     # --evidence_llm "mistral,llama2,vicuna,beluga,starling,openorca,gpt3" \
#     # "system_0,system_1,system_2,system_3,system_4" \
#     #  "llama3,beluga,mistral,zephyr,starling,openorca,dolphin,mistral1,hermes2,hermes25" \


python train_nn.py \
    --model_path gpt2 \
    --task truthfulqa \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --num_warmup_steps 0.03 \
    --weight_decay 0.0 \
    --lr_scheduler_type cosine \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 100 \
    --train_data_path $trainfile \
    --evidence_llm "llama3,mistral,zephyr,starling,openorca,hermes2,hermes25" \
    --regression $regression \
    --mode $mode \
    --split 0.1 \
    --freeze_epoch 200 \
    --reg_factor 0.05 \


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
