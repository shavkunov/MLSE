import json
import os
import pickle
import re
import sys
import time
from os import path
from typing import List
import traceback

from ggnn.parsing.gg.common import GatedGraph
from ggnn.parsing.gg.java_graph_parser import JavaGatedGraphParser


def save_graph(graph, dst):
    nodes = {i: node for node, i in graph.nodes.items()}
    tokens = [nodes[i].token if nodes[i].is_leaf() else nodes[i].node_type for i in range(len(nodes))]
    types = [nodes[i].var_type for i in range(len(nodes))]
    with open(dst, 'wb') as f:
        pickle.dump((tokens, types, graph.edges), f)


def main():
    src, dst, dataset, i_from, i_to = sys.argv[1:]
    parser = JavaGatedGraphParser()

    os.makedirs(path.join(dst, dataset), exist_ok=True)

    for i in range(int(i_from), int(i_to)):
        try:
            with open(path.join(src, dataset, f'{dataset}_{i}.java'), 'r', encoding='utf-8') as f:
                content = 'class A {\n' + f.read() + '\n}'
            _, graph = parser.parse(content)
            save_graph(graph, path.join(dst, dataset, f'{dataset}_{i}.pkl'))
            if i % 1000 == 0:
                print(f'OK {i}')
        except Exception as e:
            print(f'Fail to process number {i}:', e)

if __name__ == '__main__':
    main()
