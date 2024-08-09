# Crowdsourcing
Basic usage with standard Gaussian Prior:
```
python EM.py --algorithm em_orig
```

Basic usage with bi-modal Gaussian Prior:
```
python EM.py --algorithm em_bimodal
```

Run with artificial data:
```
python EM.py --algorithm em_orig --datapath artificial
```

Run on binary data:
```
pip install -r requirements.txt
pip install -e .
python main.py policy=em_sym_bin policy.params.seed=0 data_loader=halu_dialogue_bin
python main.py policy=em_asym_bin policy.params.seed=0 data_loader=halu_dialogue_bin
python main.py policy=majority_vote data_loader=halu_dialogue_bin
```

Basic usage of EM with Gaussian mixture model (GMM):
```
python main.py data_loader=halu_dialogue_logit policy=em_gmm policy.params.max_iter=200
```

Basic usage of EM original
```
python main.py data_loader=halu_dialogue_logit policy=em_orig policy.params.prior_var_of_cov=10
```

## Train PEW using GPT-2:
```
./train.sh
```
You need to specify `expdir` where the experimental data and model checkpoints will be stored. The first line in this file is to activate the conda environment. Please replace it with your own conda env.

### To run with ground truth labels and GPT2
Set the following parameters in `train.sh`:
```
--mode gt
--split 0.5
```
Note that the `--split` determines the split between train and validation - 0.5 means 50% of data is used for training and 50% for validation. 

### Inference with trained model
```
./eval.sh
```
There are some local directories - please change to your local ones.

AvgSSLPreds (with no contexts)
Basic usage
```
python train.py data_loader=halu_dialogue_logit policy=avg_ssl_preds neural_net.params.hidden_size=100 neural_net.params.seed=0
```
Passing learning rate and weight decay
```
python train.py data_loader=halu_dialogue_logit policy=avg_ssl_preds policy.params.lr=0.001 policy.params.weight_decay=0.01 neural_net.params.hidden_size=100 neural_net.params.seed=0
```