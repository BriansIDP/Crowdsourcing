import json
from pathlib import Path

import numpy as np

class HaluDialogueBinary:
    def __init__(self, datapath, model_list):
        self.datapath = datapath
        self.model_list = model_list

    def get_data(self):
        est_dict = {}
        for model in self.model_list:
            hits = 0
            est_dict[model] = []
            outcomes = []
            filepath = Path(self.datapath) / "halueval_dialogue_{}.json".format(model)
            with open(filepath) as fin:
                modeldata = json.load(fin)[model]
            for datapiece in modeldata:
                outcome = 1 if datapiece["ref"] == "yes" else 0
                outcomes.append(outcome)
                # est_dict[model].append(datapiece["prob"])
                est = np.argmax(datapiece["prob"])
                est_dict[model].append(est)
                if datapiece["prob"][0] > datapiece["prob"][1] and outcome == 0:
                    hits += 1
                elif datapiece["prob"][1] > datapiece["prob"][0] and outcome == 1:
                    hits += 1
            # print("{} Acc: {:.3f}".format(model, hits/len(est_dict[model])))
        return est_dict, outcomes

class HaluQABinary:
    def __init__(self, datapath, model_list):
        self.datapath = datapath
        self.model_list = model_list

    def get_data(self):
        est_dict = {}
        for model in self.model_list:
            hits = 0
            est_dict[model] = []
            outcomes = []
            filepath = Path(self.datapath) / "halueval_qa_{}.json".format(model)
            with open(filepath) as fin:
                modeldata = json.load(fin)[model]
            for datapiece in modeldata:
                outcome = 1 if datapiece["ref"] == "yes" else 0
                outcomes.append(outcome)
                # est_dict[model].append(datapiece["prob"])
                est = np.argmax(datapiece["prob"])
                est_dict[model].append(est)
                if datapiece["prob"][0] > datapiece["prob"][1] and outcome == 0:
                    hits += 1
                elif datapiece["prob"][1] > datapiece["prob"][0] and outcome == 1:
                    hits += 1
            # print("{} Acc: {:.3f}".format(model, hits/len(est_dict[model])))
        return est_dict, outcomes
