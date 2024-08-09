import json
from pathlib import Path

# from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    path = Path('data/halueval_dialogue.json')
    with open(path) as fin:
        data = json.load(fin)
    # model_name='bert-base-uncased'
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertModel.from_pretrained(model_name)
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)
    all_embeddings = []
    for datum in tqdm(data):
        input_str = "Query: {}\nResponse: {}\nIs there any non-factual or \
            hallucinated information in the response?".format(datum["query"], datum["response"])
        # inputs = tokenizer(input_str, return_tensors='pt', padding=True, truncation=True)
        inputs = tokenizer(input_str, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save('data/gpt2_embeddings.npy', all_embeddings)