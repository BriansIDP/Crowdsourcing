import json
import numpy as np


data = []
with open("truthful_qa.unrolled.json") as fin:
    origdata = json.load(fin)
    for datapiece in origdata:
        label = "yes" if datapiece["label"] == "correct" else "no"
        data.append({"query": datapiece["question"], "response": datapiece["answer"], "question": "Is the answer truthful to the question?", "ref": label})

with open("Meta-Llama-3-8B-Instruct.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[i]["llama3"] = list(probs)

with open("Mistral-7B-Instruct-v0.2.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[i]["mistral"] = list(probs)

with open("OpenHermes-2.5-Mistral-7B.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[i]["hermes25"] = list(probs)

with open("OpenHermes-2-Mistral-7B.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[i]["hermes2"] = list(probs)

with open("Starling-LM-7B-alpha.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[i]["starling"] = list(probs)

with open("zephyr-7b-beta.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[i]["zephyr"] = list(probs)

with open("zephyr-7b-beta.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[i]["zephyr"] = list(probs)

with open("Mistral-7B-OpenOrca.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[i]["openorca"] = list(probs)

with open("Mistral-7B-Instruct-v0.1.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[i]["mistral1"] = list(probs)

with open("truthful_qa.json", "w") as fout:
    json.dump(data, fout, indent=4)
