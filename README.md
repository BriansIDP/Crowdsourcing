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

Train PEW using GPT-2:
```
./train.sh
```
Predict with PEW
```
./eval.sh
```
There are loads of local directories - please change to your local ones.
