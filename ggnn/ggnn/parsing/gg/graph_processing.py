import pickle
from collections import defaultdict
from itertools import count
import os
from os import path

import numpy as np

from .common import GatedGraph
from .cpp_graph_parser import CppGatedGraphParser
from ..parallel_processing import DataSaver, DataProcessor
from ..utils import *


class SparseGraphSaver(DataSaver):

    def __init__(self, dst_folder):
        self.dst_folder = dst_folder
        self.tokens = defaultdict(count(1, 1).__next__, {'': 0})
        self.var_types = defaultdict(count(1, 1).__next__, {'': 0})
        self.edge_types = defaultdict(count(0, 1).__next__)
        self.normalization_regexp = re.compile('[^a-zA-Z0-9 .,_<>*&\\[\\]]')

    def save(self, data, relative_path: str, filename: str):
        tokens, types, edges = data
        nodes_tokens = np.zeros(len(tokens), dtype=np.int32)
        var_types = np.zeros(len(tokens), dtype=np.int32)
        for i in range(len(tokens)):
            nodes_tokens[i] = self.tokens[self.normalize(tokens[i])]
            var_types[i] = self.var_types[self.normalize(types[i])]

        from_ids, to_ids, edge_types = zip(*edges)

        os.makedirs(path.join(self.dst_folder, relative_path), exist_ok=True)
        dst_path = path.join(self.dst_folder, relative_path, filename)
        with open(dst_path, 'wb') as file:
            data = (
                nodes_tokens,
                var_types,
                (
                    np.array(from_ids, dtype=np.int16),
                    np.array(to_ids, dtype=np.int16),
                    np.array(list(map(self.edge_types.__getitem__, edge_types)), dtype=np.int8)
                )
            )
            pickle.dump(data, file)

    def flush(self):
        os.makedirs(path.join(self.dst_folder), exist_ok=True)
        save_id_storage(self.edge_types, path.join(self.dst_folder, "edge_types.csv"))
        save_id_storage(self.tokens, path.join(self.dst_folder, "tokens.csv"))
        save_id_storage(self.var_types, path.join(self.dst_folder, "var_types.csv"))

    def load_dicts(self, src_path=None):
        src_path = src_path if src_path is not None else self.dst_folder
        storage = load_id_storage(path.join(src_path, "edge_types.csv"))
        self.edge_types = defaultdict(count(len(storage), 1).__next__, storage)
        storage = load_id_storage(path.join(src_path, "tokens.csv"))
        self.tokens = defaultdict(count(len(storage), 1).__next__, storage)
        storage = load_id_storage(path.join(src_path, "var_types.csv"))
        self.var_types = defaultdict(count(len(storage), 1).__next__, storage)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()

    def normalize(self, token):
        return '' if token is None else self.normalization_regexp.sub('', token).strip()


class CppGatedGraphProcessor(DataProcessor):
    def process(self, content, meta):
        ast, graph = meta.parse(content)
        return graph

    def init_meta(self):
        return CppGatedGraphParser()
