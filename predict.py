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

from model import WorkerPredictor, WorkerCompressor, NDM
from dataloader import WorkerDataset, collate_fn
from torch.utils.data import DataLoader
from scipy.stats import beta
from sklearn.calibration import calibration_curve


def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')


def train_desc(args, predictions, all_labels):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Hyper-parameters
    lr = 0.2
    epochs = 100
    batch_size = 32
    predictions = torch.tensor(predictions, dtype=torch.float32).to(device)
    labels = np.array(all_labels)
    one_indices = np.where(labels==1)
    zero_indices = np.where(labels==0)
    predictions = torch.cat([predictions[one_indices], predictions[zero_indices]], dim=0)
    ndm_model = NDM(predictions.size(-1)).to(device)
    optimizer = torch.optim.SGD(ndm_model.parameters(), lr=lr)
    nbatches = predictions.size(0) // batch_size
    predictions = predictions[:batch_size*nbatches].view(nbatches, batch_size, -1)
    orders = [i for i in range(nbatches)]
    bestloss = 1e9
    for epoch in range(epochs):
        random.shuffle(orders)
        total_loss = 0
        total_count = 0
        for i in orders:
            optimizer.zero_grad()
            sample = predictions[i]
            shuffled_indices = torch.argsort(torch.rand_like(sample), dim=0)
            shuffled_sample = torch.gather(sample, dim=0, index=shuffled_indices)
            labels = torch.cat([torch.ones(sample.size(0)), torch.zeros(sample.size(0))], dim=0).long().to(device)
            loss = ndm_model(torch.cat((sample, shuffled_sample), dim=0), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_count += 1
        avg_loss = total_loss / total_count
        print("Epoch {}\tAverage NDM Loss: {:.5f}".format(epoch, avg_loss))
        if avg_loss < bestloss:
            bestloss = avg_loss
        else:
            lr = lr // 2
            for g in optimizer.param_groups:
                g['lr'] = lr
    return ndm_model


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE) of a classifier.

    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like, shape (n_samples,)
        Predicted probabilities or confidence scores.
    n_bins : int, optional (default=10)
        Number of bins to discretize the predicted probabilities.

    Returns:
    --------
    ece : float
        The Expected Calibration Error.
    """
    # Compute the calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Bin edges used by calibration_curve
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Digitize the predicted probabilities into bins
    bin_indices = np.digitize(y_prob, bins=bin_edges, right=True) - 1
    # Handle edge case where y_prob == 1
    bin_indices[bin_indices == n_bins] = n_bins - 1

    n = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        # Select samples in the current bin
        bin_mask = bin_indices == i
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            # Compute the average accuracy and confidence in the bin
            bin_accuracy = np.mean(y_true[bin_mask])
            bin_confidence = np.mean(y_prob[bin_mask])
            # Update ECE
            ece += (bin_size / n) * abs(bin_accuracy - bin_confidence)

    return ece


def get_indep_samples(all_workers, all_labels):
    all_workers = np.array(all_workers)
    all_labels = np.array(all_labels)
    one_indices = np.where(all_labels==1)
    zero_indices = np.where(all_labels==0)
    one_probs = all_workers[one_indices]
    zero_probs = all_workers[zero_indices]
    indep_samples = []
    for i in range(all_workers.shape[1]-1):
        one_alpha_1, one_alpha_2, loc, scale = beta.fit(one_probs[:, i], floc=0, fscale=1)
        one_samples = beta.rvs(one_alpha_1, one_alpha_2, size=all_workers.shape[0]//2)
        zero_alpha_1, zero_alpha_2, _, _ = beta.fit(zero_probs[:, i], floc=0, fscale=1)
        zero_samples = beta.rvs(zero_alpha_1, zero_alpha_2, size=all_workers.shape[0]//2)
        samples = list(one_samples) + list(zero_samples)
        indep_samples.append(samples)
    return np.array(indep_samples).T

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
        evalmode=True if train_args['split'] >= 0.9 else False,
        task=task,
        split=1.0 if train_args['split'] >= 0.9 else train_args['split'],
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
            train_args["target_nllms"] if train_args["mode"] in ["pewcrowdae", "pewcrowdaext"] else len(llm_list),
            tokenizer,
            train_args["regression"],
            mode=train_args["mode"],
            lora_config=lora_config,
        ).to(device)
    modelpath = os.path.join(args.model_path, args.model_ckpt, "pytorch_model.pt")
    trained_params = torch.load(modelpath)
    msg = model.load_state_dict(trained_params, strict=False)

    if train_args["mode"] in ["pewcrowdae", "pewcrowdaext"] and train_args["encdecpath"] != "":
        ae_model = WorkerCompressor(len(llm_list), train_args["target_nllms"]).to(device)
        state_dict = torch.load(train_args["encdecpath"])
        ae_model.load_state_dict(state_dict, strict=False)
    else:
        ae_model = None

    model.eval()

    total_hits = 0
    total_samples = 0
    predictions = []
    all_workers = []
    all_labels = []
    all_sigmas = []
    all_pred_workers = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader)):
            inputs, workers, labels = batch
            all_workers.extend(workers.tolist())
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
                    if train_args["mode"] in ["pewcrowdae", "pewcrowdaext"]:
                        aeloss, workers = ae_model(workers)
                        # workers = 1 - workers
                    prediction, probs = model.predict(
                        inputs,
                        workers,
                        aggregation=args.aggregation,
                        labels=(workers < 0.5).float() if "hard" in args.aggregation else 1-workers,
                        withEM=True if "EM" in args.aggregation else False,
                    )
                if train_args["mode"] in ["gt", "pewcrowd", "pewcrowdimp", "pewcrowdimpxt", "pewcrowdae", "pewcrowdaext", "pewcrowdaepost"]:
                    predictions.extend(prediction[:, 0].tolist())
                    prediction = prediction[:, 0] < 0.5
                    total_hits += (prediction == labels[:, 0]).sum()
                    all_pred_workers.extend(probs.view(workers.size(0), -1, 2).tolist())
                elif train_args["mode"] == "compression":
                    # predictions.extend((prediction < 0.5).tolist())
                    predictions.extend(prediction.tolist())
                else:
                    total_hits += (prediction == labels[:, 0]).sum()
                total_samples += prediction.size(0)
                all_labels.extend(labels[:, 0].tolist())

    if train_args["mode"] == "compression" and args.aggregation == "ndm":
        print("Training NDM for measuring independence")
        ndm_model = train_desc(args, predictions, all_labels)
        # independent_samples = get_indep_samples(all_workers, all_labels)
        # ndm_model = train_desc(args, all_workers, all_labels)

    # all_sigmas = np.array(all_sigmas)
    if task in ["halueval", "truthfulqa", "arenabinary", "mmlujudge"]:
        predictions = np.array(predictions)
        all_labels = np.array(all_labels)
        all_workers = np.array(all_workers)
        all_pred_workers = np.array(all_pred_workers)
        np.save(os.path.join(args.model_path, "predictions_mean.npy"), predictions)
        np.save(os.path.join(args.model_path, "workers.npy"), all_workers)
        np.save(os.path.join(args.model_path, "labels.npy"), all_labels)
        np.save(os.path.join(args.model_path, "pred_workers.npy"), all_pred_workers)

        majority_voting = ((all_workers<0.5).sum(axis=-1) > (all_workers.shape[-1] // 2))
        hits = (majority_voting == all_labels).sum()
        print("Acc via majority voting: {:.5f}".format(hits / all_labels.size))

        if train_args["mode"] == "compression":
            # Compute ECE for latent workers
            for k in range(all_workers.shape[1]):
                ece = compute_ece(all_labels, all_workers[:, k], n_bins=10)
                print("ECE original worker {}: {:.5f}".format(k, ece))
            for k in range(predictions.shape[1]):
                ece = compute_ece(all_labels, predictions[:, k], n_bins=10)
                print("ECE latent worker {}: {:.5f}".format(k, ece))

            total_samples = predictions.shape[0]
            total_hits = ((predictions.mean(axis=-1) < 0.5) == all_labels).sum()
            predictions = predictions < 0.5
            for k in range(predictions.shape[1]):
                hits = (predictions[:, k] == all_labels).sum(axis=0)
                print("Accuracy worker {}: {:.5f}".format(k, hits/total_samples))
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
