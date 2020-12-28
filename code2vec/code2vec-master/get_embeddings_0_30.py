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

DATASET_PATH = Path('../samples')
#DATASET_WITH_ROLE = DATASET_PATH / dir_type
import os
from collections import namedtuple
import numpy as np
import pickle
from tensorflow_model import Code2VecModel
# Modified code2vec + interactive predict files

class InteractivePredictor:

    def __init__(self, config, model):
        model.predict([])
        self.model = model
        self.config = config
#         self.path_extractor = Extractor(config,
#                                         jar_path=JAR_PATH,
#                                         max_path_length=MAX_PATH_LENGTH,
#                                         max_path_width=MAX_PATH_WIDTH)

    def predict(self, predict_lines, hash_to_string_dict, target_method, top_attentions=3):
#         try:
#             predict_lines, hash_to_string_dict = self.path_extractor.extract_paths(input_filename)
#         except ValueError as e:
#             return None
        
        raw_prediction_results = self.model.predict(predict_lines)
        method_prediction_results = common.parse_prediction_results(
            raw_prediction_results, hash_to_string_dict,
            self.model.vocabs.target_vocab.special_words, topk=SHOW_TOP_CONTEXTS)
        for raw_prediction, method_prediction in zip(raw_prediction_results, method_prediction_results):
            #print('checking', target_method.lower(), method_prediction.original_name.replace('|', ''))
            if target_method.lower() == method_prediction.original_name.replace('|', ''):
                return raw_prediction.code_vector # Do we want to get a context as well? But how to use it?
    

# python3 get_embeddings.py --load 'models/java14_model/saved_model_iter8.release' --export_code_vectors   
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import pickle
from datetime import datetime
import shutil

EMBEDDINGS_PATH = DATASET_PATH / 'embeddings' / 'train'

model = None
predictor = None

def process_embeddings(path):
    start_time = datetime.now()
    #first_evs = 1000
    with open(path, 'rb') as input_pickle:
        paths_dict = pickle.load(input_pickle)
        
        #print('EVALUATING')
        for idx, file_path in enumerate(paths_dict):
            #print(paths_dict[file_path])
            raw_paths, hash_to_string_dict = paths_dict[file_path]
            
            if raw_paths is None:
                continue
            
            target_method = str(file_path).split('-')[-2]
            
            embedding = predictor.predict(raw_paths, hash_to_string_dict, target_method)
            
            with open(EMBEDDINGS_PATH / file_path.name, 'w') as output_emb:
                if embedding is not None:
                    output = np.array2string(embedding, precision=9)
                else:
                    output = "None"
                output_emb.write(output)
                
            #if idx + 1 == first_evs:
            #    break
    del paths_dict
    diff = datetime.now() - start_time
    print(path, 'total', diff)

if __name__ == '__main__':
    config = Config(set_defaults=True, load_from_args=True, verify=False)
    model = Code2VecModel(config)
    config.log('Done creating code2vec model')
    predictor = InteractivePredictor(config, model)
    
    start_time = datetime.now()
    total_chunks = 30
    #total_files = os.listdir(DATASET_PATH / 'test_files')
    
    process_paths = []
    for index in range(total_chunks):
        pickle_path = DATASET_PATH / f'train_paths_{index}.pickle'
        process_paths.append(pickle_path)
        process_embeddings(pickle_path)
        #break # remove
        
    #with mp.Pool(20) as pool:
    #    res = list(pool.map(process_embeddings, process_paths)
    
    
    
    
    
#     path_extractor = Extractor(config, jar_path=JAR_PATH, max_path_length=MAX_PATH_LENGTH,max_path_width=MAX_PATH_WIDTH)
#     def extract(path):
#         try:
#             predict_lines, hash_to_string_dict = path_extractor.extract_paths(path)
#         except ValueError as e:
#             predict_lines, hash_to_string_dict = None, None
#         return predict_lines, hash_to_string_dict

#     def extract_paths(total=10):
#         extracted = {}
#         cnt = 0
#         for file in os.listdir(DATASET_PATH / 'val'):
#             res = extract(DATASET_PATH / 'val' / file)
#             if res is not None:
#                 cnt +=1
#                 extracted[file] = res
#                 shutil.copyfile(DATASET_PATH / 'val' / file, DATASET_PATH / 'test_files' / file)
                
#             if cnt == total:
#                 break

#         return extracted
#     total_extracted = extract_paths()
#     with open(DATASET_PATH / f'test.pickle', 'wb') as output:
#         pickle.dump(total_extracted, output, protocol=pickle.HIGHEST_PROTOCOL)
        

    #model.close_session()
