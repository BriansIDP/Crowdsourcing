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
        data[info["i"]]["llama3"] = list(probs)

with open("Mistral-7B-Instruct-v0.2.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[info["i"]]["mistral"] = list(probs)

with open("OpenHermes-2.5-Mistral-7B.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[info["i"]]["hermes25"] = list(probs)

with open("OpenHermes-2-Mistral-7B.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[info["i"]]["hermes2"] = list(probs)

with open("Starling-LM-7B-alpha.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[info["i"]]["starling"] = list(probs)

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
        data[info["i"]]["openorca"] = list(probs)

with open("Mistral-7B-Instruct-v0.1.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[info["i"]]["mistral1"] = list(probs)

with open("StableBeluga-7B.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[info["i"]]["beluga"] = list(probs)

with open("dolphin-2.1-mistral-7b.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[info["i"]]["dolphin"] = list(probs)

with open("Hermes-3-Llama-3.1-70B.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[info["i"]]["hermes70B"] = list(probs)

with open("Meta-Llama-3-70B-Instruct.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[info["i"]]["llama370B"] = list(probs)

with open("Mixtral-8x7B-Instruct-v0.1.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[info["i"]]["mixtral"] = list(probs)

with open("Athene-70B.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[info["i"]]["athene"] = list(probs)

with open("Qwen2-72B-Instruct.jsonl") as fin:
    for i, line in enumerate(fin):
        info = json.loads(line)
        probs = np.array([info["logit_yes"], info["logit_no"]])
        probs = np.exp(probs) / np.exp(probs).sum()
        data[info["i"]]["qwen272B"] = list(probs)

with open("truthful_qa.json", "w") as fout:
    json.dump(data, fout, indent=4)
