# . /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

trainfile="data/halueval_dialogue.json"
# trainfile="data/artificial.json"
# trainfile="data/wikibio_crosscheck_gpt3.json"
# expdir="exp/pewcrowd_gpt2_mse_direct_crowdlayer_constrained_largebatch"
# expdir="exp/transformer_gpt2_mse_artificial"
expdir=exp/brian_pewcrowd_K_10

python predict.py \
    --model_path $expdir \
    --model_ckpt checkpoint.3 \
    --bsize 16 \
    --testfile $trainfile \
    --aggregation mean \
