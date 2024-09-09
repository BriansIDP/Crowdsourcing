import json
from pathlib import Path

# from transformers import BertTokenizer, BertModel
# from transformers import GPT2Tokenizer, GPT2Model
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import worker_agg

# if __name__ == '__main__':
#     path = Path('data/halueval_dialogue.json')
#     with open(path) as fin:
#         data = json.load(fin)
#     # model_name='bert-base-uncased'
#     # tokenizer = BertTokenizer.from_pretrained(model_name)
#     # model = BertModel.from_pretrained(model_name)
#     model_name = 'gpt2'
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#     model = GPT2Model.from_pretrained(model_name)
#     all_embeddings = []
#     for datum in tqdm(data):
#         input_str = "Query: {}\nResponse: {}\nIs there any non-factual or \
#             hallucinated information in the response?".format(datum["query"], datum["response"])
#         # inputs = tokenizer(input_str, return_tensors='pt', padding=True, truncation=True)
#         inputs = tokenizer(input_str, return_tensors='pt')
#         with torch.no_grad():
#             outputs = model(**inputs)
#             embeddings = outputs.last_hidden_state.mean(dim=1)
#         all_embeddings.append(embeddings.numpy())
    
#     all_embeddings = np.concatenate(all_embeddings, axis=0)
#     np.save('data/gpt2_embeddings.npy', all_embeddings)

def get_data(cfg, split_type='train', with_gt=False):
    data_constructor = worker_agg.__dict__[cfg.data_loader.name]
    data_dict = OmegaConf.to_container(cfg.data_loader.params, resolve=True, throw_on_missing=True)
    if split_type == 'val':
        data_dict['evalmode'] = True
    data_dict['with_gt'] = with_gt
    data = data_constructor(**data_dict)
    return data


@hydra.main(version_base=None, config_path="./../conf", config_name="config")
def main(cfg):
    # Determine device: use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = cfg.neural_net.params.model_path

    # Load the pre-trained language model and move to the correct device
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cfg.neural_net.params.cache_dir
    ).to(device)

    # Load the tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_path)

    all_embeddings = []
    all_ests = []
    all_outcomes = []
    for split_type in ['train', 'val']:
        print(f"Processing {split_type} data")
        data = get_data(cfg, split_type=split_type, with_gt=True)
        dataloader = DataLoader(
                    data,
                    batch_size=cfg.policy.params.batch_size,
                    shuffle=False,
                    collate_fn=data.collate_fn,
                )
        for i, batch in enumerate(tqdm(dataloader)):
            inputs, ests, outcomes = batch
            # Ensure inputs are on the correct device
            attention_mask = inputs["attention_mask"].to(device)
            outputs = llm(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            insizes = attention_mask.sum(dim=-1) - 1
            pred_hidden = outputs.hidden_states[-1][torch.arange(insizes.size(0)), insizes]
            # Convert the embeddings to numpy and append to list
            all_embeddings.append(pred_hidden.cpu().detach().numpy())
            all_ests.append(ests.cpu().detach().numpy())
            all_outcomes.append(outcomes.cpu().detach().numpy())

    # Concatenate all embeddings along the first dimension (batch size)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print("shape of all_embeddings:", all_embeddings.shape)
    all_ests = np.concatenate(all_ests, axis=0)
    print("shape of all_ests:", all_ests.shape)
    all_outcomes = np.concatenate(all_outcomes, axis=0)
    all_outcomes = all_outcomes.flatten()
    print("shape of all_outcomes:", all_outcomes.shape)
    # npz file
    np.savez('data/gpt2_embeddings_arena.npz', 
             embeddings=all_embeddings, ests=all_ests, outcomes=all_outcomes)
    # np.save('data/gpt2_embeddings.npy', all_embeddings)

if __name__ == '__main__':
    main()