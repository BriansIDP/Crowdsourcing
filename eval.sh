# . /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

regression=skill
# regression=hardlabel
mode=pewcrowdaext
# mode=gt
# mode=compression
# task=halueval
# task=arenabinary
task=truthfulqa

# trainfile="data/halueval_dialogue.json"
trainfile=data/truthfulQA/truthful_qa.json
# trainfile=data/Arena/arena_hard_binary_short.json
expdir=exp/pewcrowd_gpt2_mse_direct_crowdlayer_${mode}_${regression}_${task}
# expdir=exp/pewcrowd_llama3_mse_direct_crowdlayer_${mode}_${regression}_${task}
# expdir=exp/worker_compression_encoder_decoder_CE_${task}_6workers

# python neuralEM.py \
python predict.py \
    --model_path $expdir \
    --model_ckpt checkpoint.9 \
    --bsize 8 \
    --testfile $trainfile \
    --aggregation hardEM \
    # --sigma_path exp/pewcrowd_gpt2_mse_direct_crowdlayer_constrained_largebatch/sigma.npy \
