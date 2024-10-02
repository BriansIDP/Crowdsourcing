. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

regression=skill
# regression=hardlabel
mode=pewcrowdimp
# mode=gt
# mode=compression
# task=halueval
# task=arenabinary
# task=truthfulqa
task=mmlujudge

# trainfile="data/halueval_dialogue.json"
# trainfile=data/truthfulQA/truthful_qa.json
# trainfile=data/Arena/arena_hard_binary_reverse_short.json
trainfile=data/MMLU_judge/mmlu_binary_short.json
# expdir=exp/pewcrowd_gpt2_mse_direct_crowdlayer_${mode}_${regression}_${task}_reverse_7B_seed1
expdir=exp/pewcrowd_gpt2_mse_direct_crowdlayer_${mode}_${regression}_${task}_reg
# expdir=exp/pewcrowd_roberta_mse_direct_crowdlayer_${mode}_${regression}_${task}_reverse_7B
# expdir=exp/worker_compression_encoder_decoder_CE_${task}_1workers
# expdir=exp/worker_compression_encoder_decoder_CE_01bias_1sworkers

# python neuralEM.py \
python predict.py \
    --model_path $expdir \
    --model_ckpt checkpoint.29 \
    --bsize 8 \
    --testfile $trainfile \
    --aggregation mean \
    # --sigma_path exp/pewcrowd_gpt2_mse_direct_crowdlayer_constrained_largebatch/sigma.npy \
