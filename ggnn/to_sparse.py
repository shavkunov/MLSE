import os
import sys
from collections import defaultdict
from itertools import count
from os import path
import pickle
from typing import List, Dict
import numpy as np
import re
from ggnn.parsing.gg.common import GatedGraph
from ggnn.parsing.gg.graph_processing import SparseGraphSaver
from ggnn.parsing.parallel_processing import DataLoader, DataProcessor, DataSaver


def save_id_storage(storage: Dict[str, int], dst):
    with open(dst, 'w', encoding='utf-8') as file:
        file.write("id,value\n")
        for token, token_id in storage.items():
            file.write(f"{token_id},\"{token}\"\n")


def load_id_storage(src_path):
    result = dict()
    with open(src_path, 'r', encoding='utf-8') as file:
        file.readline()
        for line in file:
            line = line.strip()
            if line:
                token_id, token = line.split(',', maxsplit=1)
                result[token[1:-1]] = int(token_id)
    return result


class PathsLoader(DataLoader):

    def __init__(self, src, extension=''):
        self.src = src
        self.extension = extension

    def load(self):
        for folder, _, files in os.walk(self.src):
            relative_path = os.path.relpath(folder, self.src)
            print(relative_path)
            for filename in self.__select_files(relative_path, files):
                yield relative_path, filename, os.path.join(folder, filename)

    def __select_files(self, relative_path, files):
        return [file for file in files if self.__check_file(relative_path, file)]

    def __check_file(self, relative_path, filename):
        file_path = path.join(self.src, relative_path, filename)
        if not path.isfile(file_path):
            return False
        return filename.endswith(self.extension)

class GraphNode:
    def __init__(self, node_token, node_type):
        self.node_token = node_token
        self.node_type = node_type

class GraphProcessor(DataProcessor):
    def process(self, content, meta):
        with open(content, 'rb') as f:
            tokens, types, edges = pickle.load(f)
        return tokens, types, edges

    def init_meta(self):
        return None


def run_sequential(loader: DataLoader, processor: DataProcessor, saver: DataSaver):
    meta = processor.init_meta()
    with saver:
        for relative_path, filename, content in loader.load():
            try:
                result = processor.process(content, meta)
                if len(result[0]) > 1:
                    saver.save(result, relative_path, filename)
            except Exception as e:
                raise e


def main():
    src, dst = sys.argv[1:]
    loader = PathsLoader(src, '.pkl')
    saver = SparseGraphSaver(dst)
    processor = GraphProcessor()
    run_sequential(loader, processor, saver)


if __name__ == '__main__':
    main()
