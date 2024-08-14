import os
import random
import argparse
import math
import pickle
import time
import json
from collections import OrderedDict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from transformers import SchedulerType, AdamW, get_scheduler

from model import WorkerPredictor
from dataloader import WorkerDataset, collate_fn
from torch.utils.data import DataLoader


torch.manual_seed(3407)
random.seed(3407)


def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    with open(os.path.join(args.outputdir, 'model_config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    ## Initialise data
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm_list = args.evidence_llm.split(',')
    task = "halueval" if "halueval" in args.train_data_path else "crosscheck"
    traindata = WorkerDataset(
        args.train_data_path,
        tokenizer,
        evidence_llm=llm_list,
        task=task,
        split=args.split,
        mode=args.mode,
    )
    train_dataloader = DataLoader(
        traindata,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    validdata = WorkerDataset(
        args.train_data_path,
        tokenizer,
        evidence_llm=llm_list,
        task=task,
        split=args.split,
        mode=args.mode,
        evalmode=True,
    )
    valid_dataloader = DataLoader(
        validdata,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    ## Initialise model
    lora_config = {}
    lora_config["lora_rank"] = args.lora_rank
    lora_config["lora_alpha"] = args.lora_alpha,
    lora_config["lora_dropout"] = args.lora_dropout,
    lora_config["lora_module"] = args.lora_module,

    model = WorkerPredictor(
        args.model_path,
        len(llm_list),
        tokenizer,
        args.regression,
        mode=args.mode,
        lora_config=lora_config,
    ).to(device)

    ## Optimiser
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = args.num_warmup_steps * max_train_steps

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Train loop
    for epoch in range(args.num_train_epochs):
        model.train()
        if epoch > args.freeze_epoch and "pewcrowd" in args.mode:
            model.freeze_model()
        model = train_one_epoch(
            args,
            epoch,
            model,
            train_dataloader,
            optimizer,
            lr_scheduler,
            tokenizer,
        )
        if args.split < 1.0:
            model.eval()
            eval_one_epoch(args, epoch, model, valid_dataloader, tokenizer)

        current_lr = optimizer.param_groups[0]["lr"]
        if epoch > args.freeze_epoch and "pewcrowd" in args.mode:
            model.unfreeze_model()
        save_checkpoint(model, tokenizer, args.outputdir, epoch)


def train_one_epoch(
    args,
    epoch,
    model,
    train_dataloader,
    optimizer,
    lr_scheduler,
    tokenizer,
):
    optimizer.zero_grad()
    trainsize = len(train_dataloader)
    start = time.time()
    total_loss = 0
    total_count = 0
    for i, batch in enumerate(train_dataloader):
        inputs, workers, labels = batch
        loss = model(
            inputs,
            workers,
            labels,
        )
        total_loss += loss.item()
        total_count += 1
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        if (i + 1) % args.gradient_accumulation_steps == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if (i + 1) % args.log_interval == 0:
            elasped_time = time.time() - start
            # loss = loss.item() * args.gradient_accumulation_steps
            loss_print = total_loss / total_count
            logging(f"Epoch {epoch} | Batch {i+1}/{trainsize} | loss: {loss_print} | time {elasped_time}", args.logfile)

    return model

def eval_one_epoch(
    args,
    epoch,
    model,
    valid_dataloader,
    tokenizer,
):
    hits = 0
    total = 0
    group_hits = 0
    group_total = 0
    for batch in tqdm(valid_dataloader):
        inputs, workers, labels = batch
        pred, hidden = model.predict(
            inputs,
            workers,
            labels=labels < 0.5,
        )
        if "pewcrowd" in args.mode:
            group_labels = labels > 0.5
            group_hits += (group_labels.view(-1) == hidden.max(dim=-1)[1]).sum()
            group_total += group_labels.view(-1).size(0)
            labels = ((labels < 0.5).sum(dim=-1) > labels.size(-1) // 2).long()
        if len(pred.size()) > 1:
            hits += sum(labels.view(-1) == pred.max(dim=-1)[1])
        else:
            hits += (labels[:, 0] == pred).sum()
        total += pred.size(0)
    if "pewcrowd" in args.mode:
        logging("Group Accuracy: {:.5f}".format(group_hits/group_total), args.logfile)
    logging("Accuracy: {:.5f}".format(hits/total), args.logfile)


def save_checkpoint(model, tokenizer, outputdir, epoch):
    fulloutput = os.path.join(outputdir, "checkpoint.{}".format(epoch))
    os.system(f"mkdir -p {fulloutput}")
    checkpoint = OrderedDict()
    for k, v in model.named_parameters():
        if v.requires_grad:
            checkpoint[k] = v
    torch.save(checkpoint, f'{fulloutput}/pytorch_model.pt')
    # save tokenizer
    tokenizer.save_pretrained(fulloutput)
    return checkpoint


if __name__ == "__main__":
    ## Parameter groups
    parser = argparse.ArgumentParser(description="Worker prediction")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./hf_models",
        help="Path to the model file",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="./hf_models",
        help="Path to the train data file",
    )
    parser.add_argument(
        "--evidence_llm",
        type=str,
        default="llama3",
        help="Path to the train data file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="Weight decay."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=float, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default='./log.txt',
        help="Path to the log file",
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        default='./exp/clip_vlm',
        help="Path to the output dir",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="log interval",
    )
    parser.add_argument(
        "--regression",
        type=str,
        default='mse',
        help="Regression method",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=1.0,
        help="split train set size",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='pew',
        help="Aggregation method",
    )
    parser.add_argument(
        "--freeze_epoch",
        type=int,
        default=100,
        help="Number of epochs after which the llm is frozen",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--lora_module",
        type=list,
        default=["q_proj", "v_proj", "o_proj", "fc1", "fc2"],
        help="LoRA module",
    )
    args = parser.parse_args()
    main(args)
