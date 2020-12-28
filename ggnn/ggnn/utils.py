import hashlib
import json
import os
import pickle
from collections import defaultdict
from os import path
from typing import List, Dict
import math

import numpy as np
import torch
from numpy.random import RandomState

from .parsing.common import PathsLoader
from .parsing.utils import load_id_storage, Tokenizer, save_id_storage


def build_vocabulary(graphs_src: str, dicts_src: str, dst: str,
                     subtokens_bound: int, types_bound: int):
    tokens_dict = load_id_storage(path.join(dicts_src, "node_tokens.csv"))
    types_dict = load_id_storage(path.join(dicts_src, "var_types.csv"))
    edges_dict = load_id_storage(path.join(dicts_src, "edge_types.csv"))

    tokens_cnt = np.zeros(len(tokens_dict), dtype=np.int32)
    types_cnt = np.zeros(len(types_dict), dtype=np.int32)

    for relative_path, filename, abs_path in PathsLoader(graphs_src, file_pattern=r'.*\.pkl').load():
        with open(abs_path, 'rb') as f:
            nodes_tokens, nodes_types, _ = pickle.load(f)
            for i in nodes_tokens:
                tokens_cnt[i] += 1
            for i in nodes_types:
                types_cnt[i] += 1

    subtokens_index = defaultdict(int)
    tokenizer = Tokenizer()
    for key, value in tokens_dict.items():
        for subtoken in tokenizer.tokenize(key):
            subtoken = subtoken
            if subtoken:
                subtokens_index[subtoken] += tokens_cnt[value]

    subtokens_dict = {'': 0}
    subtokens_cnt = np.zeros(len(subtokens_index) + 1, dtype=np.int32)
    for subtoken in sorted(subtokens_index, key=subtokens_index.get, reverse=True):
        subtokens_cnt[len(subtokens_dict)] = subtokens_index[subtoken]
        subtokens_dict[subtoken] = len(subtokens_dict)

    subtokens_vocab = __filter_dict(subtokens_dict, subtokens_cnt, subtokens_bound)
    types_vocab = __filter_dict(types_dict, types_cnt, types_bound)
    os.makedirs(dst, exist_ok=True)
    save_id_storage(subtokens_vocab, path.join(dst, "subtokens.csv"))
    save_id_storage(types_vocab, path.join(dst, "node_types.csv"))
    save_id_storage(edges_dict, path.join(dst, "edges_types.csv"))
    __save_counts(tokens_dict, tokens_cnt, path.join(dst, "tokens_counts.csv"))
    __save_counts(types_dict, types_cnt, path.join(dst, "types_counts.csv"))
    __save_counts(subtokens_dict, subtokens_cnt, path.join(dst, "subtokens_counts.csv"))


def __load_labels(src, name):
    labels = {}
    with open(path.join(src, f'{name}_metrics.csv'), 'r', encoding='utf-8') as labels_file:
        labels_file.readline()
        for line in labels_file:
            sample_id, _, values = line.split(',', maxsplit=2)
            values = list(map(float, values.split(',')))
            if any(math.isnan(x) or math.isinf(x) for x in values):
                print('Skip')
                continue
            labels[sample_id] = torch.as_tensor(values, dtype=torch.float)
    return labels


def prepare_data(graphs_src: str, labels_src: str, dicts_src: str, dst: str,
                 n_subtokens=3, seed=0):
    subtokens_dict = load_id_storage(path.join(dicts_src, "subtokens.csv"))
    types_dict = load_id_storage(path.join(dicts_src, "node_types.csv"))
    edges_dict = load_id_storage(path.join(dicts_src, "edges_types.csv"))

    old_tokens_dict = load_id_storage(path.join(graphs_src, "node_tokens.csv"))
    tokens_mappings = __get_mappings(old_tokens_dict, subtokens_dict, n_subtokens)

    old_types_dict = load_id_storage(path.join(graphs_src, "var_types.csv"))
    types_mappings = __get_single_mappings(old_types_dict, types_dict)

    train_labels = __load_labels(labels_src, 'train')
    validation_labels = __load_labels(labels_src, 'val')
    test_labels = __load_labels(labels_src, 'test')

    __prepare_data(graphs_src, dst, 'test', tokens_mappings, types_mappings,
                   len(edges_dict), test_labels)
    __prepare_data(graphs_src, dst, 'val', tokens_mappings, types_mappings,
                   len(edges_dict), validation_labels)
    __prepare_data(graphs_src, dst, 'train', tokens_mappings, types_mappings,
                   len(edges_dict), train_labels)


def __save_counts(id_storage, counts, dst):
    with open(dst, 'w', encoding='utf-8') as file:
        file.write(f"token_id,count,token\n")
        for token, token_id in id_storage.items():
            file.write(f"{token_id},{counts[token_id]},\"{token}\"\n")


def __filter_dict(id_storage, counts, bound):
    result = {'': 0}
    for token, token_id in id_storage.items():
        if token and counts[token_id] >= bound:
            result[token] = len(result)
    return result


def __prepare_data(src, dst, name,
                   tokens_mappings, types_mappings, n_edges,
                   labels):
    id_list = []
    tokens_list = []
    var_types_list = []
    indexes_list = []
    tags_list = []
    for folder, _, files in os.walk(path.join(src, name)):
        for filename in files:
            sample_id = filename[:-4]
            if sample_id not in labels:
                continue
            tokens, types, indexes = __load_graph(path.join(folder, filename),
                                                  tokens_mappings, types_mappings, n_edges)
            id_list.append(path.join(folder, filename))
            tokens_list.append(tokens)
            var_types_list.append(types)
            indexes_list.append(indexes)
            tags_list.append(labels[sample_id])
            if indexes.max().item() >= 20 * len(tokens):
                print("ALARM!")
    batch_dst = os.path.join(dst, f"{name}")
    os.makedirs(batch_dst)
    torch.save(id_list, path.join(batch_dst, "ids.pkl"))
    torch.save(tokens_list, path.join(batch_dst, "tokens.pkl"))
    torch.save(var_types_list, path.join(batch_dst, "types.pkl"))
    torch.save(indexes_list, path.join(batch_dst, "indexes.pkl"))
    torch.save(torch.stack(tags_list),
               path.join(batch_dst, "labels.pkl"))


def __load_graph(src, tokens_mappings, types_mappings, n_edges):
    with open(src, 'rb') as file:
        old_tokens, old_types, edges_sparse = pickle.load(file)
        from_edge, to_edge, edge_type = edges_sparse
        return (
            torch.as_tensor(tokens_mappings[old_tokens], dtype=torch.long),
            torch.as_tensor(types_mappings[old_types], dtype=torch.long),
            torch.as_tensor([to_edge, from_edge.astype(np.int64) * n_edges + edge_type.astype(np.int64)],
                            dtype=torch.long)
        )


def __get_single_mappings(old_types: Dict[str, int], new_types: Dict[str, int]):
    types_mappings = np.zeros(len(old_types), dtype=np.int32)
    print(types_mappings.shape)
    for token, token_id in old_types.items():
        types_mappings[token_id] = new_types.get(token, 0)
    return types_mappings


def __get_mappings(old_tokens: Dict[str, int], new_subtokens: Dict[str, int], n_subtokens: int):
    tokens_mappings = np.zeros((len(old_tokens), n_subtokens), dtype=np.int32)
    print(tokens_mappings.shape)
    tokenizer = Tokenizer()
    for token, token_id in old_tokens.items():
        ptr = 0
        for subtoken in tokenizer.tokenize(token):
            new_id = new_subtokens.get(subtoken, 0)
            if new_id != 0:
                tokens_mappings[token_id][ptr] = new_id
                ptr += 1
                if ptr == n_subtokens:
                    break
    return tokens_mappings


def __tensor_from_sparse_indexes(sparse_list, n):
    values, indexes = sparse_list
    buffer = np.zeros(n, dtype=np.int64)
    buffer[indexes] = values
    return torch.as_tensor(buffer, dtype=torch.long)


def __select_batch(relative_path, files: List[str], seed, batch_id, n_batches):
    files.sort()
    local_seed = int(hashlib.sha1(relative_path.encode('utf-8')).hexdigest(), 16) % (2 ** 32)
    batch_ids = RandomState(seed ^ local_seed).randint(0, n_batches, len(files), dtype=np.int)
    return [file for file, i in zip(files, batch_ids) if i == batch_id]


def __produce_embeddings(src, dst, tokens_mappings, types_mappings, n_edges,
                         labels, solutions_limit, model, seed, device='cuda'):
    model.to(device)
    model.eval()
    id_list = []
    vectors_list = []
    sizes_list = []
    tags_list = []
    for folder, _, files in os.walk(src):
        relative_path = os.path.relpath(folder, src)
        if relative_path not in labels:
            continue
        tokens_list = []
        var_types_list = []
        indexes_list = []
        for filename in files:
            tokens, types, indexes = __load_graph(path.join(folder, filename),
                                                  tokens_mappings, types_mappings, n_edges)
            tokens_list.append(tokens)
            var_types_list.append(types)
            indexes_list.append(indexes)
        vectors = torch.zeros((solutions_limit, 128), dtype=torch.float)
        with torch.no_grad():
            for batch_id in range(0, 10):
                from_ind = 100 * batch_id
                to_ind = min(from_ind + 100, len(files))
                if to_ind <= from_ind:
                    break
                matrix, tokens, mask, types, heads = combine(indexes_list[from_ind:to_ind],
                                                             tokens_list[from_ind:to_ind],
                                                             var_types_list[from_ind:to_ind], 22)
                vectors[from_ind:to_ind] = model(types.to(device), tokens.to(device),
                                                 mask.to(device), matrix.to(device)).cpu()[heads, :]
        id_list.append(relative_path)
        tags_list.append(labels[relative_path])
        vectors_list.append(vectors)
        sizes_list.append(len(files))
        print(relative_path)
    os.makedirs(path.dirname(dst), exist_ok=True)
    torch.save((id_list,
                torch.stack(vectors_list),
                torch.as_tensor(tags_list, dtype=torch.float),
                torch.as_tensor(sizes_list)), dst)
