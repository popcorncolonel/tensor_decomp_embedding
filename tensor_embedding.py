from collections import defaultdict
import itertools
import numpy as np
import os
import tensorflow as tf
from tensor_decomp import CPDecomp
import time
import scipy

from joblib import Parallel, delayed
def update_counts(self, batch):
    batch_counts = defaultdict(int)
    self.uni_counts = batch_counts
    # create own count here, return it, union them together
    for cooccurrence in self.get_indices(batch, update_uni_counts=True):
        batch_counts[cooccurrence] += 1
    num_samples = len(batch)
    return batch_counts, num_samples


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
                optimizer_type=self.optimizer_type,
            )

    def write_embedding_to_file(self, fname='vectors.txt'):
        vectors = {}
        model = self.model
        embedding = self.get_embedding_matrix()
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
        out_fname = 'results_iter_{}.txt'.format(method)
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


class PpmiSvdEmbedding(TensorEmbedding):
    def __init__(self, vocab_model, embedding_dim):
        super(PpmiSvdEmbedding, self).__init__(vocab_model, embedding_dim)
        self.optimizer_type = 'svd'

    def learn_embedding(self, ppmi_tensor):
        U,S,V = np.linalg.svd(ppmi_tensor)

        U_d = U[:,:self.embedding_dim]
        V_d = V[:self.embedding_dim, :]  # is this correct? (does it matter?)
        S_d = np.diag(S[:300])
        sqrt_S_d = scipy.linalg.sqrtm(S_d)
        self.embedding = np.matmul(U_d, sqrt_S_d)
        self.C_embedding = np.matmul(V_d.T, sqrt_S_d)

    def get_embedding_matrix(self):
        return self.embedding


class PMIGatherer(object):
    def __init__(self, vocab_model, n=2):
        self.model = vocab_model
        self.vocab_len = len(self.model.vocab)
        self.n = n

    def P(self, x):
        '''
        MLE for probabilities: #(x)/|D|

        '''
        # TODO: maybe do some smoothing?
        if isinstance(x, tuple):  # n-gram probability
            assert len(x) == self.n
            return self.n_counts[x] / self.num_samples
        else:  # unigram probability
            return self.uni_counts[x] / self.num_samples

    def PMI(self, *args):
        """
        For example, if n=2, PMI(x,y) = P(x,y) * |D| / (P(x)P(y))
        If n=3, PMI(x,y,z) = log(#(x,y,z) * |D|^2 / (#(x)#(y)#(z)))
                           = log(#(x,y,z)) + 2*log(|D|) - log(#(x)) - log(#(y)) - log(#(z))
        """
        log_num = np.log2(self.n_counts[args]) + (self.n - 1)*np.log2(self.num_samples)
        log_denom = 0.0
        for arg in args:
            log_denom += np.log2(self.uni_counts[arg])
        pmi = log_num - log_denom
        return pmi

    def populate_counts(self, batches):
        '''
        `batches` is a generator of (context, word) tuples,
            where `context` is like [98345, 2348975, 38239, 138492, 3829, 329] (indices into the vocab) 
            and `word` is like 3829 (index into the vocab)

        '''
        print('Gathering counts...')
        self.num_samples = 0
        self.uni_counts = defaultdict(int)
        self.n_counts = defaultdict(int)

        print('getting counts...')
        t = time.time()
        batch_counts, n_samples_per_batch = zip(*Parallel(n_jobs=50)(delayed(update_counts)(self, b) for b in batches))
        print('joining count dicts...')
        for d in batch_counts:
            for k in d.keys():
                if isinstance(k, tuple):
                    self.n_counts[k] += d[k]
                else:
                    self.uni_counts[k] += d[k]
        self.num_samples = sum(n_samples_per_batch)
        print('Gathering counts took {} secs'.format(time.time() - t))
        pass

    def get_indices(self, batch, update_uni_counts=False):
        # TODO: make it a set of indices, which are all ordered in ascending order, then just permute everything at lookup time?
        #       i.e. if indices = {(1,2,3)}, the indices gets expanded to [(1,2,3), (2,1,3), (3,2,1), ..., (1,3,2)] and the corresponding PMI vals get added
        #       This decreases computation time for PMI? But increases computation time for sorting everything according to (1,2,3)
        indices = []
        for context, word in batch:
            product_producer = itertools.product(context, repeat=self.n-1)
            if update_uni_counts:  # for efficiency -- only wanna loop through this once
                self.uni_counts[word] += 1
            for product_tuple in product_producer:
                index = (word, *product_tuple) 
                if len(set(index)) == self.n:  # only capture the "interesting" (non-repetitive) co-occurrences
                    indices.append(index)
        return indices

    def create_pmi_tensor(self, batch=None, positive=True, numpy_dense_tensor=False, debug=False):
        print('Creating Sparse PMI tensor...')
        if batch:
            indices = self.get_indices(batch)
        else:
            indices = list(self.n_counts.keys())

        print('filling values...')
        values = np.zeros(len(indices)) 
        t = time.time()
        for i in range(len(indices)):
            values[i] += self.PMI(*indices[i])  # NOTE: if this becomes unbearably slow, you are out of ram. decrease batch size. 
        print('pmi conversion took {} secs'.format(time.time() - t))
        # TODO: Why are all the PMI values positive? Is this bad?
        #print("done calculating PMI values")
        shape = (self.vocab_len,) * self.n
        if numpy_dense_tensor:
            ''' Probably not gonna wanna do this if you're bigger than 2 dimensions. '''
            ppmi_tensor = np.zeros(shape)
            for val, ix in zip(values, indices):
                ppmi_tensor[ix] += val
            return ppmi_tensor
        indices = np.asarray(indices)
        if debug:
            t = time.time()
            import heapq
            print('getting top 50...')
            top_k = heapq.nlargest(50, zip(values,indices), key=lambda x: x[0])
            top_k_pairs = list(set([tuple(ix) for (val,ix) in top_k]))
            top_k_pairs_words = [(self.model.index2word[x[0]], self.model.index2word[x[1]], self.model.index2word[x[2]]) for x in top_k_pairs]
            print(top_k_pairs_words)
            print('n largest took {} sec'.format(time.time() - t))
            import pdb; pdb.set_trace()
            import sys; sys.exit()
        num_total_vals = len(values)
        if positive:
            positive_args = np.argwhere(values > 0.0)
            indices = indices[positive_args]
            values = values[positive_args]
            print('{} nonzero pmi\'s out of {} total'.format(len(values), num_total_values))
        sparse_tensor = tf.SparseTensorValue(indices, values, shape)
        return sparse_tensor
        

