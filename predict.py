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

from model import WorkerPredictor
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
    task = "halueval" if "halueval" in args.testfile else "crosscheck"

    testdata = WorkerDataset(
        args.testfile,
        tokenizer,
        evidence_llm=llm_list,
        evalmode=True,
        task=task,
        split=1.0, # train_args['split'] if 'split' in train_args else 1.0,
        mode=train_args["mode"],
    )
    test_dataloader = DataLoader(
        testdata,
        batch_size=args.bsize,
        shuffle=False,
        collate_fn=collate_fn,
    )
    ## Initialise model
    model = WorkerPredictor(
        train_args["model_path"],
        len(llm_list),
        tokenizer,
        train_args["regression"],
        mode=train_args["mode"],
    ).to(device)
    modelpath = os.path.join(args.model_path, args.model_ckpt, "pytorch_model.pt")
    trained_params = torch.load(modelpath)
    msg = model.load_state_dict(trained_params, strict=False)

    model.eval()

    total_hits = 0
    total_samples = 0
    predictions = []
    all_errors = []
    all_labels = []
    if args.aggregation == "grad":
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_dataloader)):
                inputs, workers, labels = batch
                prediction, probs = model.predict(
                    inputs,
                    workers,
                    aggregation="ex_error",
                )
                all_errors.append(prediction)
            all_errors = torch.cat(all_errors, dim=0)
            all_errors = all_errors.mean(dim=0)
    for i, batch in enumerate(tqdm(test_dataloader)):
        inputs, workers, labels = batch
        prediction, probs = model.predict(
            inputs,
            workers,
            aggregation=args.aggregation,
            expected_error=all_errors,
        )
        total_hits += (prediction == labels[:, 0]).sum()
        total_samples += prediction.size(0)
        if task == "halueval":
            predictions.extend(prediction.tolist())
        elif task == "crosscheck":
            predictions.extend(probs.tolist())
        all_labels.extend(labels[:, 0].tolist())
    if task == "halueval":
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
