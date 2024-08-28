. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

trainfile="data/halueval_dialogue.json"
# trainfile="data/artificial.json"
# trainfile="data/wikibio_crosscheck_gpt3.json"
expdir=exp/pewcrowd_gpt2_mse_direct_crowdlayer_pewcrowdimp_skill_allbeluga
# expdir=exp/pewcrowd_llama3_mse_direct_crowdlayer_constrained
# expdir=exp/worker_compression_encoder_decoder

# python neuralEM.py \
python predict.py \
    --model_path $expdir \
    --model_ckpt checkpoint.1 \
    --bsize 8 \
    --testfile $trainfile \
    --aggregation hardEM \
    # --sigma_path exp/pewcrowd_gpt2_mse_direct_crowdlayer_constrained_largebatch/sigma.npy \
