#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from transformers import pipeline
import math
import numpy as np
import torch
import os
from tqdm import tqdm


from transformers import AutoTokenizer, AutoModel


def columns_to_list(df, columns):
    return list(zip(*[df[c].to_list() for c in columns]))


def get_files_info(path):
    files = os.listdir(path)
    result = {}
    for f in files:
        if f[-4:] == ".npy":
            parts = f[:-4].split("-")
        else:
            parts = f.split("-")
        i = int(parts[-1])
        start_line = int(parts[-3])
        if i in result:
            print("Conflict!")
        result[i] = (f, start_line)
    return result

tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1", cache_dir='cache/')
model = AutoModel.from_pretrained("huggingface/CodeBERTa-small-v1", cache_dir='cache/')

device = torch.device("cuda:0")
model.to(device)


def tokenize_code(code):
    tokenized = tokenizer(code, add_special_tokens=False)
    if len(tokenized["input_ids"]) <= 510:
        input_ids = [[0] + tokenized["input_ids"] + [2]]
        attention_mask = [[1] + tokenized["attention_mask"] + [1]]
        return {
            "input_ids": torch.tensor(input_ids).long().cuda(),
            "attention_mask": torch.tensor(attention_mask).long().cuda()
        }, len(tokenized["input_ids"])
    input_ids = []
    attention_mask = []
    for i in range(0, len(tokenized["input_ids"]), 510):
        if len(tokenized["input_ids"]) >= i + 510:
            input_ids.append([0] + tokenized["input_ids"][i:i+510] + [2])
            attention_mask.append([1] + tokenized["attention_mask"][i:i+510] + [1])
        else:
            delta = (i + 510) - len(tokenized["input_ids"])
            input_ids.append([0] + tokenized["input_ids"][i:len(tokenized["input_ids"])] + [2] + [1] * delta)
            attention_mask.append([1] + tokenized["attention_mask"][i:len(tokenized["input_ids"])] + [1] + [0] * delta)
    return {
        "input_ids": torch.tensor(input_ids).long().cuda(),
        "attention_mask": torch.tensor(attention_mask).long().cuda()
    }, len(tokenized["input_ids"])



def preprocess_part(name):
    df = pd.read_csv(f"data/{name}.csv")
    code_parts = columns_to_list(df, ["file_id", "methodLoc", "startLine"])
    id2file = get_files_info(f"data/{name}")
    os.makedirs(f"data/embeddings/{name}", exist_ok=True)
    
    for file_id, loc, start_ in tqdm(code_parts, smoothing=0.01):
        if math.isnan(loc):
            continue
        if not file_id in id2file:
            continue
        filename, start = id2file[file_id]
        with open(f"data/{name}/{filename}", "r") as f:
            for _ in range(int(start) - 1):
                f.readline()
            lines = []
            for _ in range(int(loc)):
                lines.append(f.readline())
        code = "\n".join(lines)
        tokenized, l = tokenize_code(code)
        with torch.no_grad():
            embs = model(**tokenized)[0].cpu().numpy()
        embs = embs[:, 0, :]

        np.save(f'data/embeddings/{name}/{filename}', embs)


#preprocess_part("val")
#preprocess_part("test")
#preprocess_part("train")
#exit(0)

from torch import nn
from torch.nn import functional as F

class EmbeddingCombiner(nn.Module):
    def __init__(self, embedding_size=768, hidden_size=512, head_size=128, n_heads=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, head_size * n_heads)
        )
        self.attention = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, n_heads)
        )
        self.n_heads = n_heads
        self.head_size = head_size
        
    def forward(self, x):
        if len(x) == 1:
            return self.model(x)[0]
        x_enc = self.model(x)
        x_attn = F.softmax(self.attention(x), dim=0).unsqueeze(-1).expand(-1, self.n_heads, self.head_size).reshape(-1, self.n_heads*self.head_size)
        x = (x_enc * x_attn.expand_as(x_enc)).sum(0)
        return x
    
class RegressionNN(nn.Module):
    def __init__(self, input_size=512, hidden_size=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.model(x)

from torch.utils.data import Dataset, DataLoader
import time
import random

class EmbeddingDataset(Dataset):
    def __init__(self, name, target_metric):
        target_df = pd.read_csv(f"data/{name}_metrics.csv")
        #target_df = pd.read_csv(TARGETS_PATH / f'{data_type}_metrics.csv')
        orig_df = pd.read_csv(f'data/{name}.csv')
        merged = pd.concat([orig_df, target_df], axis=1)[['filename', target_metric]]
        #file_metric_df = target_df[['id', target_metric]]
        self.targets = merged.set_index('filename').to_dict()[target_metric]
        
        prev_keys = list(self.targets.keys())
        for key in prev_keys:
            if key in self.targets and self.targets[key] != self.targets[key]: # nan check
                del self.targets[key]
        
        #self.code_parts = columns_to_list(target_df, ["id", target_name])
        #self.id2file = get_files_info(f"data/embeddings/{name}")
        #print('id2file', len(self.id2file))
        #print(list(sorted(self.id2file.keys()))[:10])
        self.name = name
        self.avg_y = np.mean([y for y in self.targets.values()])
        self.std_y = np.std([y for y in self.targets.values()])
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        filename = list(self.targets.keys())[idx]
        y = self.targets[filename]
    
        x = np.load(f"data/embeddings/{self.name}/{filename}.npy")
        return x, y
    
class MyDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def get_batch(self):
        #keys = list(self.dataset.id2file.keys())
        idx = np.random.randint(0, len(self.dataset), self.batch_size)
        xs = []
        ys = []
        for i in idx:
            x, y = self.dataset[i]
            xs.append(x)
            ys.append(y)
        return xs, ys


def model_inference(encoder, regression, dataset, batches, batch_size, norm):
    dataloader = MyDataLoader(dataset, batch_size=batch_size)
    mae, mse, r2 = 0, 0, 0
    with torch.no_grad():
        for _ in range(batches):
            xs, ys = dataloader.get_batch()
            tx = []
            for x in xs:
                x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
                tx.append(encoder(x))
            xs = torch.stack(tx)
            ys_pred = norm[1] * regression(xs).view(-1) + norm[0]
            ys = torch.tensor(np.array(ys), dtype=torch.float32, device=device)
            mse += ((ys_pred - ys)**2).mean().item()
            mae += ((ys_pred - ys).abs()).mean().item()
            r2 += (1 - ((ys_pred - ys)**2).mean() / ((dataset.avg_y - ys)**2).mean()).item()
        mse /= batches
        mae /= batches
        r2 /= batches
    return mse, mae, r2


def train_model(target_name, middle_size=512, head_size=128, n_heads=4, hidden_size=256, 
                lr=1e-4, epoch=90, batch_size=64, batches_per_epoch=1000, batches_per_eval=100):
    print(f"Training model for {target_name} with prarms {(middle_size, head_size, n_heads, hidden_size, lr)}")
    encoder_model = EmbeddingCombiner(768, middle_size, head_size, n_heads)
    regression_model = RegressionNN(middle_size, hidden_size)
    encoder_model.to(device)
    regression_model.to(device)
    optim = torch.optim.Adam(list(encoder_model.parameters()) + list(regression_model.parameters()), lr=lr)
    
    train_dataset = EmbeddingDataset("train", target_name)
    test_dataset = EmbeddingDataset("test", target_name)
    val_dataset = EmbeddingDataset("val", target_name)

    norm = train_dataset.avg_y, train_dataset.std_y
    
    train_errors = []
    test_errors = []
    val_errors = []
    for i in range(epoch):
        start = time.time()
        dataloader = MyDataLoader(train_dataset, batch_size=batch_size)
        
        print(f"Epoch {i+1}")
        print("Train")
        # Training step
        for _ in tqdm(range(batches_per_epoch)):
            xs, ys = dataloader.get_batch()
            tx = []
            for x in xs:
                x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
                tx.append(encoder_model(x))
            xs = torch.stack(tx)
            ys_pred = regression_model(xs).view(-1)
            ys = torch.tensor(np.array(ys), dtype=torch.float32, device=device)
            ys = (ys - norm[0]) / norm[1]
            loss = F.mse_loss(ys_pred, ys)
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        print("Test")
        mse, mae, r2 = model_inference(encoder_model, regression_model, train_dataset, batches_per_eval, batch_size, norm)
        train_errors.append((mse, mae, r2))
        
        mse, mae, r2 = model_inference(encoder_model, regression_model, val_dataset, batches_per_eval, batch_size, norm)
        val_errors.append((mse, mae, r2))
        
        mse, mae, r2  = model_inference(encoder_model, regression_model, test_dataset, batches_per_eval, batch_size, norm)
        test_errors.append((mse, mae, r2))
        print(f"ERRORS INFO")
        print(f"Train    | mse: {train_errors[-1][0]}, mae: {train_errors[-1][1]}, r2: {train_errors[-1][2]}")
        print(f"Validate | mse: {val_errors[-1][0]}, mae: {val_errors[-1][1]}, r2: {val_errors[-1][2]}")
        print(f"Epoch time: {(time.time() - start) / 60}")
    return train_errors, test_errors, val_errors



def train_for_all_metrics(metrics, epoch=3, is_complex=False):
    for m in tqdm(metrics):
        os.makedirs(f"logs/{m}", exist_ok=True)
        train_errors, test_errors, val_errors = train_model(m, epoch=epoch, middle_size=1024, hidden_size=1024, n_heads=8, lr=1e-3, batches_per_epoch=500, batch_size=128)
        
        with open(f'logs/{m}/train.txt', 'w') as train_output:
            train_output.write(train_errors)
            
        with open(f'logs/{m}/test.txt', 'w') as test_output:
            train_output.write(test_errors)
            
        with open(f'logs/{m}/val.txt', 'w') as val_output:
            train_output.write(val_errors)
        
        np.save(f'logs/{m}/train', np.array(train_errors))
        np.save(f'logs/{m}/test', np.array(test_errors))
        np.save(f'logs/{m}/val', np.array(val_errors))


complex_metrics1 = ['CyclomaticComplexity', 'HalsteadDifficultyMethod', 'DesignComplexity', 'HalsteadEffortMethod']
complex_metrics2 = ['HalsteadVolumeMethod', 'HalsteadBugsMethod', 'HalsteadLengthMethod']
complex_metrics3 = [ 'HalsteadVocabularyMethod', 'EssentialCyclomaticComplexity', 'ControlDensity']
complex_metrics4 = ['QCPCorrectness', 'QCPMaintainability', 'QCPReliability']


train_for_all_metrics(complex_metrics1)
