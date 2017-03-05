from collections import defaultdict
import itertools
import numpy as np
import os
import random
import tensorflow as tf
from tensor_decomp import CPDecomp
import time
import scipy

from joblib import Parallel, delayed


def update_counts(self, batch):
    ''' for parallel processing, of speed is an issue '''
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
            for context_word in context_list:
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
        print('getting svd of ppmi_tensor (shape: {})'.format(ppmi_tensor.shape))
        U,S,V = np.linalg.svd(ppmi_tensor)

        U_d = U[:,:self.embedding_dim]
        V_d = V[:self.embedding_dim, :]  # is this correct? (does it matter?)
        S_d = np.diag(S[:self.embedding_dim])
        sqrt_S_d = scipy.linalg.sqrtm(S_d)
        self.embedding = np.matmul(U_d, sqrt_S_d)
        self.C_embedding = np.matmul(V_d.T, sqrt_S_d)
        predicted = np.dot(self.embedding, self.C_embedding.T)
        print("RMSE: {}".format(np.sqrt(((ppmi_tensor - predicted) ** 2).mean())))
        predicted = np.dot(np.dot(U_d, S_d), V_d)
        print("RMSE (2): {}".format(np.sqrt(((ppmi_tensor - predicted) ** 2).mean())))
        import pdb; pdb.set_trace()
        pass

    def get_embedding_matrix(self):
        return self.embedding


class PMIGatherer(object):
    def __init__(self, vocab_model, n=2):
        self.model = vocab_model
        self.vocab_len = len(self.model.vocab)
        self.n = n
        self.debug = True

    def P(self, x):
        '''
        MLE for probabilities: #(x)/|D|

        '''
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
        if args not in self.valid_indices:
            # check in the set so we don't have to store a new tuple for each thing not in the defaultdict
            return 0.0
        log_num = np.log2(self.n_counts[args]) + (self.n - 1)*np.log2(self.num_samples)
        log_denom = 0.0
        for arg in args:
            log_denom += np.log2(self.uni_counts[arg])
        pmi = log_num - log_denom
        return pmi

    def kill_ncounts(self, p=0.5, m=1):
        '''
        kills `p` percent of the things with count <= m
        '''
        print(len(self.n_counts))
        print('killing {} of the count-{} n_counts...'.format(p, m))
        keys = [x for x in self.n_counts]
        for ix in keys:
            if self.n_counts[ix] <= m:
                if random.random() < p:
                    del self.n_counts[ix]
        print(len(self.n_counts))

    def populate_counts(self, batches, huge_vocab=True, min_count=1):
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
        if huge_vocab:  # memory is more important than time
            for i, batch in enumerate(batches):
                for ix in self.get_indices(batch, update_uni_counts=True):
                    self.n_counts[ix] += 1
                if len(self.n_counts) > 1e8:
                    self.kill_ncounts(0.5, 1)
        else:  # time is more impt than memory
            print('Populating count dicts (in parallel)...')
            batch_counts, n_samples_per_batch = zip(*Parallel(n_jobs=50)(delayed(update_counts)(self, b) for b in batches))
            print('joining count dicts...')
            for d in batch_counts:
                for k in d.keys():
                    if isinstance(k, tuple):
                        self.n_counts[k] += d[k]
                    else:
                        self.uni_counts[k] += d[k]
            self.num_samples = sum(n_samples_per_batch)
        print('Killing all n_counts with n < {}'.format(min_count))
        self.kill_ncounts(p=1.0, m=min_count)  # kill everything with a count of `min_count` - it's gonna have low PPMI anyway (since everything has a huge mincount). 
        self.valid_indices = {k for k in self.n_counts if self.n_counts[k] > 5}
        print('Gathering counts took {} secs'.format(time.time() - t))


    def get_indices(self, batch, update_uni_counts=False, return_set=False):
        '''
        We are assuming each sent chunk in each batch we want to keep the co-occurrence count of.
        '''
        # set of indices, which are all ordered in ascending order, then we just permute everything at lookup time
        #       i.e. if indices = {(1,2,3)}, the indices gets expanded to [(1,2,3), (2,1,3), (3,2,1), ..., (1,3,2)] and the corresponding PMI vals get added
        #       This decreases computation time (and ram) for PMI. But (negligibly) increases computation time for sorting everything according to (1,2,3)
        def get_sorted_sent_indices(sorted_sent, i, n):
            indices = set()
            if n == 2:
                for j in range(i+1, len(sorted_sent)):
                    indices.add((sorted_sent[i], sorted_sent[j]))
            else:
                for j in range(i+1, len(sorted_sent)):
                    for partial_index in get_sorted_sent_indices(sorted_sent, j, n-1):
                        indices.add((sorted_sent[i], partial_index[0], partial_index[1]))
            if return_set:
                return indices
            else:
                return list(indices)
                
        if return_set: 
            indices = set()
        else:
            indices = []
        for sent_chunk in batch:
            sent = sorted(list(set(sent_chunk)))
            if len(sent) < self.n:
                continue
            for i in range(len(sent) - self.n + 1):
                if self.n == 2:
                    if return_set:
                        indices = indices | get_sorted_sent_indices(sent, i, self.n)
                    else:
                        indices.extend(get_sorted_sent_indices(sent, i, self.n))
                else:
                    for j in range(i+1, len(sent)):
                        for k in range(j+1, len(sent)):
                            index = (sent[i], sent[j], sent[k]) 
                            if return_set:
                                indices.add(index)
                            else:
                                indices.append(index)
                if update_uni_counts:  # for efficiency -- only wanna loop through this once
                    self.uni_counts[sent[i]] += 1
            if update_uni_counts:  # since we don't loop through the last n-1 elements
                for i in range(-1, -self.n, -1):
                    self.uni_counts[sent[i]] += 1
                self.num_samples += len(sent)
            pass
        return indices

    def create_pmi_tensor(self, batch=None, positive=True, numpy_dense_tensor=False, debug=False, limit_large_vals=False, symmetric=False):
        print('Creating Sparse PMI tensor...')
        t = time.time()
        if batch:
            indices = self.get_indices(batch, return_set=True)
            indices = list(self.valid_indices.intersection(indices))
        else:
            indices = list(self.n_counts.keys())

        values = np.zeros(len(indices), dtype=np.float32) 
        for i in range(len(indices)):
            values[i] += self.PMI(*indices[i])  # NOTE: if this becomes unbearably slow, you are out of ram. decrease batch size. 
        shape = (self.vocab_len,) * self.n
        indices = np.asarray(indices, dtype=np.uint16)
        if limit_large_vals:
            new_indices = []
            new_values = []
            for val, ix in zip(values, indices):
                # exclude things like (0, 1, 2385) but include things like (2543, 13782, 3278) and (0, 43589, 3482)
                if len(np.where(ix < 15)[0]) <= 1:
                    new_indices.append(ix)
                    new_values.append(val)
            indices = np.array(new_indices)
            values = np.array(new_values)

        num_total_vals = len(values)
        if positive:
            positive_args = np.argwhere(values > 0.0)
            indices = np.squeeze(indices[positive_args])  # squeeze to get rid of the 1-dimension columns (resulting from the indices[positive_args])
            values = np.squeeze(values[positive_args])
            #print('{} nonzero pmi\'s out of {} total (=> {} total entries)'.format(len(values), num_total_vals, 6*len(values)))
        if debug and self.debug:
            # self.debug so we can turn it off (in pdb) whenever we want
            t = time.time()
            import heapq
            print('Getting top 200 PMIs...')
            top_k = heapq.nlargest(200, zip(values,indices), key=lambda x: x[0])
            top_k_pairs = sorted(list(set([(tuple(ix), val) for (val,ix) in top_k])), key=lambda x: x[1])
            top_k_pairs_words = [(tuple(self.model.index2word[x[i]] for i in range(self.n)), val) for (x, val) in top_k_pairs]
            print(top_k_pairs_words)
            print('n largest took {} sec'.format(time.time() - t))
            #import pdb; pdb.set_trace()
            pass
        if not symmetric:
            n_fact = int(scipy.misc.factorial(self.n))
            indices_extended = np.zeros((n_fact*len(indices), self.n), dtype=np.int16)
            values_extended = np.zeros((n_fact*len(indices),), dtype=np.float32)
            for i in range(len(indices)):
                tup = indices[i]
                j = 0
                for perm in itertools.permutations(range(self.n)):
                    for k, sigma in enumerate(perm):
                        indices_extended[n_fact*i+j][perm[k]] = tup[k]
                        values_extended[n_fact*i+j] = values[i]
                    j += 1
            indices = indices_extended
            values = values_extended
        if numpy_dense_tensor:
            ''' Probably not gonna wanna do this if you're bigger than 2 dimensions. '''
            ppmi_tensor = np.zeros(shape)
            for val, ix in zip(values, indices):
                ppmi_tensor[tuple(ix)] += val
            return ppmi_tensor
        print('total #values: {}...'.format(len(indices)), end='')
        print('took {} secs'.format(int(time.time() - t)))
        return (indices, values)
        

