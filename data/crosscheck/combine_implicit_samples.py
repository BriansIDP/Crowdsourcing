import json
import os


target_model = "gpt3"
evidence_models = ["starling", "mistral", "llama2", "vicuna", "openorca", "beluga"]
all_data = [{} for i in range(238)]
for evidence_model in evidence_models:
    with open("crosscheck_implicit_sample/crosscheck_prompt_{}_implicit_cot_checked_by_{}.json".format(target_model, evidence_model)) as fin:
        data = json.load(fin)
        labels = data["labels"]
        data = data["results"]
    for i, document in enumerate(data):
        newdoc = [sum(utt)/len(utt) for utt in document[evidence_model]]
        all_data[i][evidence_model] = newdoc
with open("crosscheck_prompt_{}_implicit_cot_sample.json".format(target_model), "w") as fout:
    json.dump({"labels": labels, "results": all_data}, fout, indent=4)
