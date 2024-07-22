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