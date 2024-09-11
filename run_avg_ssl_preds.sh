data_loader_params=(
    "data_loader=full_context_data"
    "data_loader.params.data_path='data/Arena/arena_binary.json'"
    "data_loader.params.task=arena"
    "data_loader.params.evidence_llm=['hermes70B','llama370B','mixtral','athene','qwen272B']"
    "data_loader.params.cross_val=True"
    "data_loader.params.nfolds=5"
    "data_loader.params.probs=False"
)

    # "data_loader.params.data_path='data/truthfulQA/truthful_qa.json'"
    # "data_loader.params.task=truthfulqa"

policy_params=(
    "policy=avg_ssl_preds_lm"
    "policy.params.model_dir='exp/arena/pew_lm'"
    "policy.params.lr=1e-4"
    "policy.params.weight_decay=1e-5"
    "policy.params.batch_size=16"
    "policy.params.log_interval=100"
    "policy.params.probs=False"
    "policy.params.max_grad_steps=3000"
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
python main_nn.py "${all_params[@]}"
