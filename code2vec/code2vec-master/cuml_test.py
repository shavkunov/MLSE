#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb

import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool
from pathlib import Path
from sklearn.metrics import r2_score as r2, mean_squared_error as mse
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torch.optim import Adam
from torch.nn import functional as F
import random
import os

TARGETS_PATH = Path('../target_metrics/snippets')

DATASET_PATH = Path('../samples')
EMBEDDINGS_PATH =  DATASET_PATH / 'embeddings'
    

class EmbeddingDataset(IterableDataset):
    def __init__(self, data_type, target_metric):
        emb_path = EMBEDDINGS_PATH / data_type
        self.files = os.listdir(emb_path)
        self.total_files = len(self.files)
        self.emb_path = emb_path
        
        path = Path('../samples/') / data_type
        file2anonfile = {}
        for idx, file in enumerate(os.listdir(path)):
            file2anonfile[file] = f'{data_type}_{idx}'
            
        self.file2anonfile = file2anonfile    
        target_df = pd.read_csv(TARGETS_PATH / f'{data_type}_metrics.csv')
        file_metric_df = target_df[['id', target_metric]]
        self.targets = file_metric_df.set_index('id').to_dict()[target_metric]
        self.index = 0

    def __iter__(self):
        return self
        
    def __next__(self):
        while True:
            file = os.listdir(self.emb_path)[self.index]
            file_path = self.emb_path / file
            with open(file_path, 'r') as input_file:
                lines = input_file.readlines()

            raw_string = ''.join(lines).replace('\n', '').replace('[', '').replace(']', '')

            self.index += 1
            if 'None' not in raw_string:
                break            
        
        embedding = np.fromstring(raw_string, dtype=np.float, sep='   ')
        
        anon_file = self.file2anonfile[file]
        target = self.targets[anon_file]
        
        embedding = torch.FloatTensor(embedding)
        return embedding, target
        
    def __len__(self):
        return self.total_files
    
def get_raw_data(data_type):
    emb_path = EMBEDDINGS_PATH / data_type
    chunks = np.array_split(os.listdir(emb_path), mp.cpu_count())
    
    def process_chunk(files):
        data = {}
        
        for file in files:
            file_path = emb_path / file
            with open(file_path, 'r') as input_file:
                lines = input_file.readlines()

            raw_string = ''.join(lines).replace('\n', '').replace('[', '').replace(']', '')

            if 'None' not in raw_string:   
                embedding = np.fromstring(raw_string, dtype=np.float, sep='   ')
                #embedding = torch.FloatTensor(embedding)
            else:
                embedding = None
                
            data[file] = embedding
            
        return data

    with ProcessingPool(mp.cpu_count()) as pool:
        proc_chunks = list(pool.map(process_chunk, chunks))
    
    merged = {}
    for dict_chunk in proc_chunks:
        merged = {**merged, **dict_chunk}
    return merged


# In[5]:


class FullDataset(Dataset):
    
    def __init__(self, raw_data, data_type, target_metric):
        emb_path = EMBEDDINGS_PATH / data_type
        path = DATASET_PATH / data_type
        #self.file2anonfile = {}
        #for idx, file in enumerate(os.listdir(path)):
        #    self.file2anonfile[file] = f'{data_type}_{idx}'
        
        self.data = [(file, embedding) for file, embedding in raw_data.items() if embedding is not None]
        
        target_df = pd.read_csv(TARGETS_PATH / f'{data_type}_metrics.csv')
        orig_df = pd.read_csv(DATASET_PATH / f'{data_type}.csv')
        merged = pd.concat([orig_df, target_df], axis=1)[['filename', target_metric]]
        #file_metric_df = target_df[['id', target_metric]]
        self.targets = merged.set_index('filename').to_dict()[target_metric]
    
    def __getitem__(self, index):
        file, embedding = self.data[index]
        #anon_file = self.file2anonfile[file]
        target = self.targets[file]
        
        return embedding, target

    def __len__(self):
        return len(self.data)


def collect_data(dataset_full):
    X, y = [], []

    for index in range(len(dataset_full)):
        emb, target = dataset_full[index]
        
        if target != target: # nan check
            continue
        
        X.append(emb)
        y.append(target)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    #y = (y - np.mean(y)) / np.std(y)
    return X, y#torch.FloatTensor(X), torch.FloatTensor(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mlp_net = nn.Sequential(
            nn.Linear(384, 1), 
            nn.ReLU()
        )

    def forward(self, input_embedding):
        return self.mlp_net(input_embedding)

#print('importing cuml')
import cudf
from cuml import RandomForestRegressor as cumlRF
from cuml.dask.ensemble import RandomForestRegressor as distributed_cuml_Rf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait
import dask_cudf
from cuml.dask.common import utils as dask_utils


def create_xgb(X_train, y_train):
    start_time = datetime.now()
    final_params = {}
    final_params['objective'] = 'reg:squarederror'
    final_params['tree_method'] = 'gpu_hist'
    final_params['learning_rate'] = 0.01
    final_params['n_estimators'] = 500

    gbdt = xgb.XGBRegressor(**final_params)
    gbdt.fit(X_train, y_train, verbose=True)
    #print('xgb', datetime.now() - start_time)
    return gbdt

def create_rf(X_train, y_train):
    start_time = datetime.now()
    clf = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    clf.fit(X_train, y_train)
    #print('classic rf', datetime.now() - start_time)
    return clf

def create_lasso(X_train, y_train):
    start_time = datetime.now()
    clf = Lasso()
    clf.fit(X_train, y_train)
    #print('lasso', datetime.now() - start_time)
    return clf

def create_cuml_rf(X_train, y_train):
    start_time = datetime.now()
    model = cumlRF(n_estimators=500, n_streams=12)
    model.fit(X_train, y_train)
    #print('cuml rf', datetime.now() - start_time)
    return model

def create_cuml_distributed(X_train, y_train):
    start_time = datetime.now()
    print('init dask cluster')

    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)
    workers = client.has_what().keys()
    
    n_workers = len(workers)
    X_train_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
    y_train_cudf = cudf.Series(y_train)
    
    X_train_dask = dask_cudf.from_cudf(X_train_cudf, npartitions=n_workers)
    y_train_dask = dask_cudf.from_cudf(y_train_cudf, npartitions=n_workers)
    
    X_train_ddask, y_train_ddask = dask_utils.persist_across_workers(client, [X_train_dask, y_train_dask], workers=workers)
    print('cuml distributed initialized', datetime.now() - start_time)
    model = distributed_cuml_Rf(n_estimators=500, n_streams=64)
    model.fit(X_train, y_train)
    
    wait(model.rfs)
    print('cuml distributed finished', datetime.now() - start_time)
    client.close()
    cluster.close()
    return model

def evaluate_classic_model(model, str_model, features, targets, log_output, predict_model=False):
    if predict_model:
       # features = cudf.DataFrame.from_pandas(pd.DataFrame(features))
        output = model.predict(features, predict_model='GPU')
    else:
        output = model.predict(features)
    mse_error = mse(targets, output)
    r2_error = r2(targets, output)
        
    print(f'MSE {mse_error}', f'R2 {r2_error}')
    with open('test_results_others2.txt', 'a') as logs:
        logs.write(f'{str_model} {log_output} MSE {mse_error} R2 {r2_error}\n')
        
if __name__ == '__main__':
    train_metrics = pd.read_csv(TARGETS_PATH / 'train_metrics.csv')

    target_metrics = list(sorted(set(train_metrics.columns) - set(['id', 'shortMethodName'])))
    print('Target metrics', target_metrics)
    print('Total targets', len(target_metrics))

    print('get raw data')
    train_raw_data = get_raw_data('train')
    test_raw_data = get_raw_data('test')

    for target_metric in target_metrics:
        print("Evaluating", target_metric)
        #with open('test_results_others.txt', 'a') as logs:
        #    logs.write(f'Evaluating {target_metric}\n')
        train_full = FullDataset(train_raw_data, 'train', target_metric)
        #val_full = FullDataset('val', target_metric)
        test_full = FullDataset(test_raw_data, 'test', target_metric)

        X_train, y_train = collect_data(train_full)
        print(X_train.shape)
        #X_train, y_train = X_train[:10000], y_train[:10000]
        print('cuml rf')
        rf = create_cuml_rf(X_train, y_train)
        
        print('xgb')
        gbdt = create_xgb(X_train, y_train)
        
        print('lasso')
        lasso = create_lasso(X_train, y_train)
        
    #X_val, y_val = collect_data(val_full)
        X_test, y_test = collect_data(test_full)
#     xgb = create_xgb(X_train, y_train)
#     rf = create_rf(X_train, y_train)
#     lasso = create_lasso(X_train, y_train)
    
        print('testing rf')
        evaluate_classic_model(rf, 'rf', X_test, y_test, 'test', predict_model=True)
        for model, str_model in zip([gbdt, lasso], ['xgb', 'lasso']):
            print('testing', str_model)
            evaluate_classic_model(model, str_model, X_test, y_test, 'test')
        with open('test_results_others2.txt', 'a') as logs:
            logs.write(f'==================\n')
            
        #break
        

