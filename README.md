# MLSE
MLSE Project Repo

## Abstract

A lot of prior works used values of software metrics to build vector representation of code fragments. It’s an interesting task to see if we can predict those metrics values could be extracted by a neural model (GGNN, TreeLSTM, code2vec, and others). **For more information see our report** [here](https://github.com/shavkunov/MLSE/blob/master/HSE_ML4SE.pdf)

## Dataset sampling

Use `preprocess.ipynb` for create `samples` directory with train/val/test subdirectories with plain java files. Original dataset is from [refactoring paper](https://arxiv.org/pdf/2001.03338.pdf).

## Code2Vec instructions

Steps for evaluation:

1) follow the original code2vec repository instructions for downloading pretrained model

2) launch `extract_paths.py` script for java extract paths in a parallel mode. It's useful for testing different code2vec models on the same dataset

3) similar to `interactive_predict.py` launch `get_embeddings_0_30.py`

where `0_30` represents paths for model inference. Changing 109 line allows to evaluate model on a different workers

`embeddings` folder is created in the `samples` dataset.

4) after that, launch `cuml_test.py` for RandomForest, XGBoost, LASSO models prediction
and `metrics_prediction_nn.ipynb` for MLP prediction

all logs are saved to files

## Other

[Code2Vec embeddings](https://drive.google.com/file/d/1MJ1QzU7473m7c6ZI2c8rNp9kOtHbLx0r/view?usp=sharing)

Folder `embeddings` contains embeddings from pretrained java14_model.
Folder `emb_fine_tune` contains embeddings from the same pretrained model, which also post-trained on our dataset.
