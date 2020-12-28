import traceback

from common import common
from extractor import Extractor
from vocabularies import VocabType
from config import Config
from interactive_predict import InteractivePredictor
from model_base import Code2VecModelBase

from pathlib import Path

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'
dir_types = ['train']
DATASET_PATH = Path('../samples')
import os
from collections import namedtuple
import numpy as np
import multiprocessing as mp
import pickle
from datetime import datetime
import shutil
#from tensorflow_model import Code2VecModel
# Modified code2vec + interactive predict files

config = Config(set_defaults=True, load_from_args=True, verify=False)
dir_type = dir_types[0]
dirname = DATASET_PATH / dir_type
#model = Code2VecModel(config)
#config.log('Done creating code2vec model')

path_extractor = Extractor(config, jar_path=JAR_PATH, max_path_length=MAX_PATH_LENGTH,max_path_width=MAX_PATH_WIDTH)

def extract(path):
    try:
        predict_lines, hash_to_string_dict = path_extractor.extract_paths(path)
    except:
        predict_lines, hash_to_string_dict = None, None
    return predict_lines, hash_to_string_dict

def extract_paths(args):
    paths, number = args
    print('extracting chunk', number)
    start_time_chunk = datetime.now()
    #print(paths[0])
    extracted = {}
    for path in paths:   
        extracted[path] = extract(path)
        
        #if path == paths[0]:
        #    print(type(extracted[path]))
    
    with open(DATASET_PATH / f'{dirname}_paths_{number}.pickle', 'wb') as output:
        pickle.dump(extracted, output, protocol=pickle.HIGHEST_PROTOCOL)
    time = datetime.now() - start_time_chunk
    print('extracting chunk completed', number, time)    
    del extracted

def get_lines2predict():
    files = os.listdir(dirname)
    paths = [dirname / file for file in files]
    total_chunks = 80
    chunks = np.array_split(paths, total_chunks)
    numbers = list(range(total_chunks))

    #total_extracted = {}
    #print(paths)
    #print(path_extractor.extract_paths(paths[0]))
    with mp.Pool(40) as pool:
        res = list(pool.map(extract_paths, zip(chunks, numbers)))


print(dir_type)
print(dirname)
start_time = datetime.now()
get_lines2predict()
print(datetime.now() - start_time)