# Tensor Decomp Embedding
Code for the paper [Word Embedding via Tensor Decomposition](https://arxiv.org/abs/1704.02686). 

This project is implemented in Python 3 only. 

## Training 
First, to set up the data, create a tokenized data file where one sentence is on each line and the words are separated by spaces. Then edit the absolute (or relative) file path in test_gensim.py in the sentences_generator function of GensimSandbox. (These functions are named as such because this repository was originally a fork of Gensim, and we were going to modify the existing code, but the purpose of this repository has since changed)

To train a new embedding, look up a valid embedding method in the Makefile, then type "make <embedding>". 
It will prompt you to type in an experiment name (if you have one), but if you are not running a specific experiment, just press enter. 
Assuming you have all dependencies properly installed, it will train, evaluate, and save the learned embedding type. 

After training, the program will save the embedding and its associated metadata to "runs/<embedding>/<num_sents>_<min_vocab_count>_<embedding_dim>/" for easy access and comparison.

## Evaluation
To compare a list of embeddings trained via the Makefile, modify the end of embedding_comparison.py to include the names of the embeddings you wish to compare, and then just run "python3 embedding_comparison.py <compairson_type>"

## CP Decomposition
A generic framework for online CP decomposition implemented in TensorFlow can be found in tensor_decomp.py. Included is also Joint Symmetric CP Decomposition, described in the paper. 

## BibTeX
TODO
