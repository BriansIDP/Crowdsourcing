import os
import random
import argparse
import math
import pickle
import time
import json
from collections import OrderedDict

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from transformers import SchedulerType, AdamW, get_scheduler

from model import WorkerPredictor, WorkerCompressor
from dataloader import WorkerDataset, collate_fn
from torch.utils.data import DataLoader


def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    with open(os.path.join(args.model_path, "model_config.json")) as fin:
        train_args = json.load(fin)
    tokenizer = AutoTokenizer.from_pretrained(train_args["model_path"])
    llm_list = train_args["evidence_llm"].split(',')
    task = train_args["task"] if "task" in train_args else "halueval"
    task = "artificial" if "artificial" in args.testfile else task

    testdata = WorkerDataset(
        args.testfile,
        tokenizer,
        evidence_llm=llm_list,
        evalmode=True,
        task=task,
        split=1.0, # train_args['split'] if 'split' in train_args else 1.0,
        mode="gt", # train_args["mode"] if "pewcrowd" not in train_args["mode"] else "gt",
    )
    test_dataloader = DataLoader(
        testdata,
        batch_size=args.bsize,
        shuffle=False,
        collate_fn=collate_fn,
    )
    ## Initialise model
    if train_args["mode"] == "compression":
        model = WorkerCompressor(len(llm_list), train_args["target_nllms"])
        model.to(device)
    else:
        lora_config = {}
        lora_config["lora_rank"] = train_args["lora_rank"]
        lora_config["lora_alpha"] = train_args["lora_alpha"]
        lora_config["lora_dropout"] = train_args["lora_dropout"]
        lora_config["lora_module"] = train_args["lora_module"]

        model = WorkerPredictor(
            train_args["model_path"],
            train_args["target_nllms"] if train_args["mode"] == "pewcrowdae" else len(llm_list),
            tokenizer,
            train_args["regression"],
            mode=train_args["mode"],
            lora_config=lora_config,
        ).to(device)
    modelpath = os.path.join(args.model_path, args.model_ckpt, "pytorch_model.pt")
    trained_params = torch.load(modelpath)
    msg = model.load_state_dict(trained_params, strict=False)

    if train_args["mode"] == "pewcrowdae" and train_args["encdecpath"] != "":
        ae_model = WorkerCompressor(len(llm_list), train_args["target_nllms"]).to(device)
        state_dict = torch.load(train_args["encdecpath"])
        ae_model.load_state_dict(state_dict, strict=False)
    else:
        ae_model = None

    model.eval()

    total_hits = 0
    total_samples = 0
    predictions = []
    all_errors = []
    all_labels = []
    all_sigmas = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader)):
            inputs, workers, labels = batch
            if task == "artificial":
                for worker in workers:
                    prediction = model.density_estimtion(
                        inputs,
                        worker.unsqueeze(0),
                    )
                    all_sigmas.append(prediction)
            else:
                if train_args["mode"] == "compression":
                    loss, prediction = model(workers)
                else:
                    if train_args["mode"] == "pewcrowdae":
                        aeloss, workers = ae_model(workers)
                        # workers = 1 - workers
                    prediction, probs = model.predict(
                        inputs,
                        workers,
                        aggregation=args.aggregation,
                        expected_error=all_errors,
                        labels=(workers < 0.5).float() if "hard" in args.aggregation else 1-workers,
                        withEM=True if "EM" in args.aggregation else False,
                    )
                if train_args["mode"] in ["gt", "pewcrowd", "pewcrowdimp", "pewcrowdimpxt", "pewcrowdae"]:
                    predictions.extend(prediction[:, 0].tolist())
                    prediction = prediction[:, 0] < 0.5
                    total_hits += (prediction == labels[:, 0]).sum()
                elif train_args["mode"] == "compression":
                    predictions.extend((prediction < 0.5).tolist())
                else:
                    total_hits += (prediction == labels[:, 0]).sum()
                total_samples += prediction.size(0)
                all_labels.extend(labels[:, 0].tolist())
    # all_sigmas = np.array(all_sigmas)
    if task in ["halueval", "truthfulqa", "arenabinary"]:
        predictions = np.array(predictions)
        all_labels = np.array(all_labels)
        if train_args["mode"] == "compression":
            total_samples = predictions.shape[0]
            for k in range(predictions.shape[1]):
                hits = (predictions[:, k] == all_labels).sum(axis=0)
                print("Accuracy worker {}: {:.5f}".format(k, hits/total_samples))
            total_hits = ((predictions.mean(axis=-1) > 0.5) == all_labels).sum()
        # np.save("outputs/sigma_reg.npy", np.array(predictions))
        print("Accuracy: {:.5f}".format(total_hits / total_samples))
    elif task == "crosscheck":
        predictions = np.array(predictions)
        all_labels = np.array(all_labels)
        print("PCC: {:.2f}".format(pearsonr(predictions, all_labels)[0]*100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM finetuning")
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to the model file",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="",
        help="Checkpoint of the model file",
    )
    parser.add_argument(
        "--bsize",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--testfile",
        type=str,
        default="dataset/gt_nbest_sel.json",
        help="Path to the model file",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="eval_log.output",
        help="Path to the model file",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        help="Way to combine",
    )
    args = parser.parse_args()
    main(args)
