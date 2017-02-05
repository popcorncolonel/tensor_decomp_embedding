import itertools
import numpy as np
import os
import tensorflow as tf
from tensor_decomp import CPDecomp
import time


class TensorEmbedding(object):
    def __init__(self, vocab_model, embedding_dim, window_size=10, optimizer_type='adam', ndims=3):
        self.model = vocab_model
        self.embedding_dim = embedding_dim
        self.window_size = window_size  # 10 (|left context| + |right context|)
        self.optimizer_type = optimizer_type
        self.ndims = ndims
        if self.ndims > 3:
            raise ValueError('As of right now, ndims can be at most 3')

        self.vocab_len = len(self.model.vocab)
        # t-th batch tensor
        # contains all data for this minibatch. already summed/averaged/whatever it needs to be. 
        config = tf.ConfigProto(
            allow_soft_placement=True,
        )
        self.sess = tf.Session(config=config)
        with self.sess.as_default():
            # Goal: X_ijk == sum_{r=1}^{R} U_{ir} V_{jr} W_{kr}
            self.decomp_method = CPDecomp(
                shape=[self.vocab_len] * self.ndims,
                sess=self.sess,
                rank=embedding_dim,
                ndims=self.ndims,
                optimizer_type='adam',
            )

    def write_embedding_to_file(self, fname='vectors.txt'):
        vectors = {}
        model = self.model
        embedding = self.decomp_method.get_embedding_matrix()
        count = 0 # number of vects written
        for word in model.vocab:
            word_vocab = model.vocab[word]
            word_vect = embedding[word_vocab.index]
            vect_list = ['{:.3f}'.format(x) for x in word_vect]
            vectors[word] = ' '.join(vect_list)
        with open(fname, 'w') as f:
            for word in vectors:
                if not word:
                    continue
                try:
                    f.write(word.encode('utf-8') + ' ' + vectors[word] + '\n')
                    count += 1
                except TypeError:
                    f.write(word + ' ' + vectors[word] + '\n')
                    count += 1
                except:
                    pass
        with open(fname, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write('{} {}\n'.format(count, self.embedding_dim))  # write the number of vects
            f.write(content)

    def evaluate(self, rel_path='vectors.txt'):
        self.write_embedding_to_file(fname=rel_path)
        method = None
        method = self.optimizer_type
        out_fname = 'results_iter{}_{}.txt'.format(self.batch_num, method)
        os.system('time python3 embedding_benchmarks/scripts/evaluate_on_all.py -f /home/eric/code/gensim/{} -o /home/eric/code/gensim/results/{}'.format(rel_path, out_fname))
        print('done evaluating.')

    def get_embedding_matrix(self):
        embedding = self.decomp_method.U.eval(self.sess)
        return embedding

    def update_counts_with_sent_info(self, sent, counts):
        """
        `sent` is a tuple of tensors representing the word and the context.
            For example, ([74895, 1397, 2385, 23048, 9485, 58934, 2378, 51143, 35829, 34290], 15234)
        """
        context_list, word = sent
        if self.ndims == 2:
            for context_word in context_indices:
                context_index = (word, context_word)
                if context_index not in counts:
                    counts[context_index] = 1
                else:
                    counts[context_index] += 1
            return counts
        elif self.ndims == 3:
            context_indices = itertools.product(context_list, context_list)  # e.g., [(74895, 1397), (74895, 2385), ...]
            for context_word1, context_word2 in context_indices:
                context_index = (word, context_word1, context_word2)
                if context_index not in counts:
                    counts[context_index] = 1
                else:
                    counts[context_index] += 1
            return counts

    def convert_batches_to_sp_tensor(self, batches):
        for batch in batches:
            counts = {}
            for sent in batch:
                self.update_counts_with_sent_info(sent, counts)

            # https://www.tensorflow.org/api_docs/python/io_ops/placeholders#sparse_placeholder
            counts_iter = counts.items()
            if self.ndims == 2:
                sent_tensor = tf.SparseTensorValue(
                    indices=[pair for pair, _ in counts_iter], # e.g., [(15234, 74895), (15234, 2385), ...] 
                    values=[count for _, count in counts_iter],
                    shape=[self.vocab_len, self.vocab_len],
                )
            elif self.ndims == 3:
                sent_tensor = tf.SparseTensorValue(
                    indices=[triple for triple, _ in counts_iter], # e.g., [(15234, 74895, 1397), (15234, 74895, 2385), ...] 
                    values=[count for _, count in counts_iter],
                    shape=[self.vocab_len, self.vocab_len, self.vocab_len],
                )
                # this tensor takes about .12 seconds to make. Too slow? Since we're doing hundreds of thousands of batches
            yield sent_tensor

    def train(self, batches):
        converted_batch_tensors = self.convert_batches_to_sp_tensor(batches)
        self.decomp_method.train(converted_batch_tensors)

