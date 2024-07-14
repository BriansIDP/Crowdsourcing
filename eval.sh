. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

trainfile="data/halueval_dialogue.json"
expdir="exp/pew_gpt2_logistic"

python predict.py \
    --model_path $expdir \
    --model_ckpt checkpoint.9 \
    --bsize 8 \
    --testfile $trainfile \
    --aggregation grad \
