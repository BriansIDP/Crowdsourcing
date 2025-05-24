trainfile="data/halueval_dialogue.json"
# trainfile=data/truthfulQA/truthful_qa.json
regression=skill
# regression=hardlabel
mode=pewcrowdimp
# mode=gt
# mode=compression
task=halueval
# task=truthfulqa

# trainfile="data/artificial.json"
# trainfile="data/wikibio_crosscheck_gpt3.json"
# expdir=exp/pewcrowd_roberta_mse_direct_crowdlayer_${mode}_${regression}_${task}_70B
expdir=exp/pewcrowd_gpt2_mse_direct_crowdlayer_${mode}_${regression}_${task}_70B
# expdir=exp/worker_compression_encoder_decoder_CE_01bias_1sworkers
mkdir -p $expdir


python train_nn.py \
    --model_path "gpt2" \
    --batch_size 8 \
    --learning_rate 10e-5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --num_warmup_steps 0.03 \
    --weight_decay 0.0 \
    --lr_scheduler_type cosine \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 100 \
    --train_data_path $trainfile \
    --evidence_llm "hermes70B,llama370B,mixtral,athene,qwen272B" \
    --regression $regression \
    --mode $mode \
    --split 0.9 \
    --freeze_epoch 200 \
    --reg_factor 0.1 \
    # --lora_rank 8 \
    # --target_nllms 9 \
    # --encdecpath exp/worker_compression_encoder_decoder_CE_01bias_9workers_adv/checkpoint.49/pytorch_model.pt \
#     --mode $mode \
#     --split 0.1 \
#     --freeze_epoch 200 \
#     --reg_factor 0.5 \
