import os
import math
import time
from collections import OrderedDict

import hydra
import torch
from transformers import get_scheduler
from torch.optim import AdamW
from omegaconf import OmegaConf

import worker_aggregation
from torch.utils.data import DataLoader

def get_dataloader(cfg, split_type='train'):
    data_constructor = worker_aggregation.__dict__[cfg.data_loader.name]
    data_dict = OmegaConf.to_container(cfg.data_gen.params, resolve=True, throw_on_missing=True)
    if split_type == 'val':
        data_dict['evalmode'] = True
    data = data_constructor(**data_dict)
    dataloader = DataLoader(
        data,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        collate_fn=data.collate_fn,
    )
    return dataloader

def get_model(cfg, 
               device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    model_constructor = worker_aggregation.__dict__[cfg.policy.name]
    model = model_constructor(**cfg.policy.params).to(device)
    return model

def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):
    # with open(os.path.join(cfg.main.model_dir, 'model_config.json'), 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)
    train_dataloader = get_dataloader(cfg, split_type='train')
    valid_dataloader = get_dataloader(cfg, split_type='val')
    model = get_model(cfg) 

    ## Optimiser
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.trainer.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.trainer.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.trainer.gradient_accumulation_steps)
    max_train_steps = cfg.trainer.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = cfg.trainer.num_warmup_steps * max_train_steps

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Train loop
    for epoch in range(cfg.trainer.num_train_epochs):
        model.train()
        model = train_one_epoch(
            cfg,
            epoch,
            model,
            train_dataloader,
            optimizer,
            lr_scheduler,
        )
        # if args.split < 1.0:
        model.eval()
        eval_one_epoch(model, valid_dataloader)

        # current_lr = optimizer.param_groups[0]["lr"]
        tokenizer = model.tokenizer
        save_checkpoint(model, tokenizer, cfg.trainer.model_dir, epoch)


def train_one_epoch(
    cfg,
    epoch,
    model,
    train_dataloader,
    optimizer,
    lr_scheduler,
):
    optimizer.zero_grad()
    trainsize = len(train_dataloader)
    start = time.time()
    for i, batch in enumerate(train_dataloader):
        inputs, workers, labels = batch
        loss = model(
            inputs,
            workers,
            labels,
        )
        loss = loss / cfg.trainer.gradient_accumulation_steps
        loss.backward()

        if (i + 1) % cfg.trainer.gradient_accumulation_steps == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if (i + 1) % cfg.trainer.log_interval == 0:
            elasped_time = time.time() - start
            loss = loss.item() * cfg.trainer.gradient_accumulation_steps
            logfile = cfg.trainer.model_dir + '/train.log'
            logging(f"Epoch {epoch} | Batch {i+1}/{trainsize} | loss: {loss} | time {elasped_time}", 
                    logfile)

    return model

def eval_one_epoch(
    model,
    valid_dataloader,
):
    hits = 0
    total = 0
    for i, batch in enumerate(valid_dataloader):
        inputs, workers, labels = batch
        pred, hidden = model.predict(
            inputs,
            workers,
        )
        hits += sum(labels.view(-1) == pred.max(dim=-1)[1])
        total += pred.size(0)
    print("Accuracy: {:.2f}".format(hits/total))


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
    main()


