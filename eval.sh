. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

trainfile="data/halueval_dialogue.json"
# trainfile="data/wikibio_crosscheck_gpt3.json"
expdir="exp/pew_gpt2_mse_relu_noinput_secondhalf"
# expdir="exp/crosscheck_pew_gpt2_logistic_relu"

python predict.py \
    --model_path $expdir \
    --model_ckpt checkpoint.9 \
    --bsize 8 \
    --testfile $trainfile \
    --aggregation mean \
