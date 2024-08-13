. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

trainfile="data/halueval_dialogue.json"
# trainfile="data/artificial.json"
# trainfile="data/wikibio_crosscheck_gpt3.json"
expdir="exp/pewcrowd_gpt2_mse_direct_crowdlayer_constrained"
# expdir="exp/transformer_gpt2_mse_artificial"

python predict.py \
    --model_path $expdir \
    --model_ckpt checkpoint.2 \
    --bsize 8 \
    --testfile $trainfile \
    --aggregation mean \
