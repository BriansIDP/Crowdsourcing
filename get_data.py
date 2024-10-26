from datasets import load_dataset
import json
from tqdm import tqdm

ds = load_dataset("HuggingFaceH4/ultrachat_200k")

alltrain = []
for datapiece in tqdm(ds['train_sft']['messages']):
    alltrain.append(datapiece)
with open("train.json", "w") as fout:
    json.dump(alltrain, fout, indent=4)

allval = []
for datapiece in tqdm(ds['test_sft']['messages']):
    allval.append(datapiece)
with open("validation.json", "w") as fout:
    json.dump(allval, fout, indent=4)
