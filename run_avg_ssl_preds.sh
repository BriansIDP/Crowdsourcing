. /scratch/OpenSource/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate hallucination

data_loader_params=(
    "data_loader=full_context_data"
    "data_loader.params.data_path='data/Arena/arena_hard_binary_short.json'"
    "data_loader.params.task=arena"
    "data_loader.params.evidence_llm=['hermes70B','llama370B','mixtral','athene','qwen272B']"
    "data_loader.params.cross_val=False"
    "data_loader.params.nfolds=5"
    "data_loader.params.probs=False"
)

    # "data_loader.params.data_path='data/truthfulQA/truthful_qa.json'"
    # "data_loader.params.task=truthfulqa"

policy_params=(
    "policy=avg_ssl_preds_lm"
    "policy.params.model_dir='exp/arena/pew_lm/2024-09-12_20-40-12/fold_0'"
    "policy.params.lr=5e-5"
    "policy.params.weight_decay=5e-6"
    "policy.params.batch_size=8"
    "policy.params.log_interval=100"
    "policy.params.probs=False"
    "policy.params.max_grad_steps=6000"
    "policy.params.gradient_accumulation_steps=1"
)

    # "policy.params.model_dir='exp/truthfulqa/pew_lm'"

neural_net_params=(
    "neural_net=multihead_net"
    "neural_net.params.dropout_prob=0.1"
    "neural_net.params.hidden_size_ow=128"
)

# Combine all parameters into one array
all_params=("${data_loader_params[@]}" "${policy_params[@]}" "${neural_net_params[@]}")
# Run the Python script with the combined parameters
python eval_checkpoint.py "${all_params[@]}"
