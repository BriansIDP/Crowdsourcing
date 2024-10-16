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
python train.py policy=em_sym_bin policy.params.seed=0 data_loader=halu_dialogue_bin
python train.py policy=em_asym_bin policy.params.seed=0 data_loader=halu_dialogue_bin
python train.py policy=majority_vote data_loader=halu_dialogue_bin
```

Basic usage of EM with Gaussian mixture model (GMM):
```
python train.py data_loader=halu_dialogue_logit policy=em_gmm
```

Basic usage of EM original
```
python train.py data_loader=halu_dialogue_logit policy=em_orig
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
