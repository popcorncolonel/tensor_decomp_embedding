import _pickle as pickle  # python 3's cPickle
import datetime
import gensim
import gensim.utils
import os
import pdb
import sys
import time
import tensorflow as tf

from gensim_utils import batch_generator, batch_generator2, bigram_batch_generator
from tensor_embedding import TensorEmbedding, PMIGatherer, PpmiSvdEmbedding
from tensor_decomp import CPDecomp
from nltk.corpus import stopwords


stopwords = set(stopwords.words('english'))
grammar_stopwords = {',', "''", '``', '.', 'the'}
stopwords = stopwords.union(grammar_stopwords)


class GensimSandbox(object):
    def __init__(self, method, num_sents=1e6, embedding_dim=300, min_count=100):
        assert method in ('tt', 'subspace', 'cbow', 'cp_decomp', 'bigram-cbow')
        self.method = method
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.num_sents = num_sents
        if '--buildvocab' in sys.argv:
            self.buildvocab = True
        else:
            self.buildvocab = False

        # To be assigned later
        self.model = None
        self.sess = None
        self.embedding = None

    def sentences_generator_old(self, num_sents=None, remove_stopwords=True):
        if num_sents is None:
            num_sents = self.num_sents
        tokenized_wiki = '/home/eric/code/wiki_complete_dump_2008.txt.tokenized'
        count = 0
        n_tokens = 0
        with gensim.utils.smart_open(tokenized_wiki, 'r') as f:
            for line in f:
                if count < num_sents:
                    if 'olela' in line or 'lakou' in line:  # some hawaiian snuck into this wiki dump...
                        continue
                    count += 1
                    sent = line.rstrip().split()
                    if remove_stopwords:
                        sent = [w for w in sent if w not in stopwords]
                    n_tokens += len(sent)
                    yield sent
                else:
                    print("{} total tokens".format(n_tokens))
                    raise StopIteration

    def sentences_generator(self, num_sents=None):
        if num_sents is None:
            num_sents = self.num_sents
        tokenized_wiki = '/home/eric/code/wikidump_2008.txt.randomized'  # also has stopwords and hawaiian removed
        count = 0
        n_tokens = 0
        with gensim.utils.smart_open(tokenized_wiki, 'r') as f:
            for line in f:
                if count % int(num_sents / 10) == 0 and count > 0:
                    print("Just hit sentence {} out of {} ({}%)".format(count, num_sents, 100*count / num_sents))
                if count < num_sents:
                    count += 1
                    sent = line.rstrip().split()
                    n_tokens += len(sent)
                    yield sent
                else:
                    print("{} total tokens".format(n_tokens))
                    raise StopIteration

    def get_model_with_vocab(self, fname='wikimodel'):
        model = gensim.models.Word2Vec(
            iter=1,
            max_vocab_size=None,
            negative=128,
            size=self.embedding_dim,
            min_count=self.min_count,
        )
        if self.method == 'tt':
            model.tt = 1
        elif self.method == 'subspace':
            model.subspace = 1
        elif self.method == 'cbow':
            model.cbow = 1
        elif self.method == 'bigram-cbow':
            model.bigramcbow = 1
        if self.buildvocab:
            print('building vocab...')
            model.build_vocab(self.sentences_generator())
            if self.method == 'bigram-cbow':
                print('building bigram vocab...')
                model.build_bigram_vocab(self.sentences_generator())
            with open(fname, 'wb') as f:
                pickle.dump(model, f)
        else:
            print('depickling model...')
            with open(fname, 'rb') as f:
                model = pickle.load(f)
        print('Finished building vocab. length of vocab: {}'.format(len(model.vocab)))
        self.model = model
        return self.model

    def evaluate_embedding(self):
        rel_path = 'vectors.txt'
        self.write_embedding_to_file()
        out_fname = 'results_{}.txt'.format(self.method)
        os.system('time python3 embedding_benchmarks/scripts/evaluate_on_all.py -f /home/eric/code/gensim/{} -o /home/eric/code/gensim/results/{}'.format(rel_path, out_fname))
        print('done evaluating.')

    def list_vars_in_checkpoint(self, dirname):
        ''' Just for tf debugging.  '''
        from tensorflow.contrib.framework.python.framework.checkpoint_utils import list_variables
        abspath = os.path.abspath(dirname)
        return list_variables(abspath)

    def write_embedding_to_file(self, fname='vectors.txt'):
        vectors = {}
        for word in self.model.vocab:
            word_vocab = self.model.vocab[word]
            word_vect = self.embedding[word_vocab.index]
            vect_list = ['{:.3f}'.format(x) for x in word_vect]
            vectors[word] = ' '.join(vect_list)
        count = 0
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

    def train_gensim_embedding(self):
        print('training...')
        if self.method == 'bigram-cbow':
            self.model.window -= 1  # don't need 5 bigrams around our word - 4 will do
            batches = bigram_batch_generator(self.model, self.sentences_generator(), batch_size=128)
        else:
            batches = batch_generator(self.model, self.sentences_generator(), batch_size=128)
        self.model.train(sentences=None, batches=batches)
        print('finished training!')

        print("most similar to king - man + woman: {}".format(self.model.most_similar(
            positive=['king', 'woman'], negative=['man'],
            topn=5,
        )))
        print("most similar to king: {}".format(self.model.most_similar(
            positive=['king'],
            topn=5,
        )))
        self.embedding = self.model.syn0

    def train_tensor_decomp_embedding(self):
        gatherer = None
        count_sents = self.num_sents
        batches = batch_generator2(self.model, self.sentences_generator(num_sents=count_sents), batch_size=1000)  # batch_size doesn't matter. But higher is probably better (in terms of threading & speed)
        if os.path.exists('gatherer_{}.pkl'.format(count_sents)):
            with open('gatherer_{}.pkl'.format(count_sents), 'rb') as f:
                t = time.time()
                import gc; gc.disable()
                gatherer = pickle.load(f)
                gc.enable()
                print('Loading gatherer took {} secs'.format(time.time() - t))
        if not gatherer:
            gatherer = PMIGatherer(self.model, n=3)
            if count_sents <= 1e6:
                gatherer.populate_counts(batches, huge_vocab=False)
            else:
                gatherer.populate_counts(batches, huge_vocab=True, min_count=5)

            with open('gatherer_{}.pkl'.format(count_sents), 'wb') as f:
                t = time.time()
                import gc; gc.disable()
                pickle.dump(gatherer, f)
                gc.enable()
                print('Dumping gatherer took {} secs'.format(time.time() - t))

        #gatherer.valid_indices = {k for k in gatherer.n_counts if gatherer.n_counts[k] > 5}
        def sparse_tensor_batches(batch_size=20000, debug=False):
            ''' because we are using batch_generator2, batches carry much much more information. (and we get through `sentences` much more quickly) '''
            batches = batch_generator2(self.model, self.sentences_generator(), batch_size=batch_size)
            for batch in batches:
                sparse_ppmi_tensor = gatherer.create_pmi_tensor(batch=batch, positive=True, debug=debug)
                yield sparse_ppmi_tensor

        config = tf.ConfigProto(
            allow_soft_placement=True,
        )
        self.sess = tf.Session(config=config)
        with self.sess.as_default():
            decomp_method = CPDecomp(shape=(len(self.model.vocab),)*3, rank=self.embedding_dim, sess=self.sess, optimizer_type='2sgd', reg_param=0)
        print('Starting CP Decomp training')
        decomp_method.train(sparse_tensor_batches(), checkpoint_every=100)
        self.embedding = decomp_method.U.eval()

        self.evaluate_embedding()

        print('creating saver...')
        saver = tf.train.Saver()
        '''
        saver = tf.train.Saver({
            'selfU': decomp_method.U,
            'selfV': decomp_method.V,
            'selfW': decomp_method.W,
        })
        '''

        timestamp = str(datetime.datetime.now())
        LOG_DIR = 'tf_logs'
        LOG_DIR = os.path.join(LOG_DIR, timestamp)
        print('saving matrices to model.ckpt...')
        saver.save(self.sess, os.path.join(LOG_DIR, "model.ckpt"), decomp_method.global_step)
        f = open(LOG_DIR + '/metadata.tsv', 'w')
        for i in range(len(self.model.vocab)): f.write(self.model.index2word[i] + '\n')
        f.close()
        from tensorflow.contrib.tensorboard.plugins import projector
        print('Adding projector config...')
        summary_writer = tf.summary.FileWriter(LOG_DIR)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = 'U'
        embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

        projector.visualize_embeddings(summary_writer, config)
        # REMEMBER TO CLOSE tensor_decomp/train_summary_writer
        import pdb; pdb.set_trace()
        pass
        
    def train_svd_embedding(self):
        batches = batch_generator(self.model, self.sentences_generator(), batch_size=10000, fixed_size=False)
        gatherer = PMIGatherer(self.model, n=2)
        gatherer.populate_counts(batches)
        sparse_ppmi_tensor = gatherer.create_pmi_tensor(positive=True, numpy_dense_tensor=True, debug=False)

        embedding_model = PpmiSvdEmbedding(self.model, embedding_dim=self.embedding_dim)
        print("calculating SVD...")
        t = time.time()
        embedding_model.learn_embedding(sparse_ppmi_tensor)
        total_svd_time = time.time() - t
        print("SVD on {}x{} took {}s".format(len(self.model.vocab), len(self.model.vocab), total_svd_time))
        self.embedding = embedding_model.evaluate()

    def train(self):
        self.get_model_with_vocab()
        if self.method in ['cp_decomp']:
            self.train_tensor_decomp_embedding()
        elif self.method in ['bigram-cbow', 'cbow', 'tt', 'subspace']:
            self.train_gensim_embedding()
        self.evaluate_embedding()

def main():
    method = 'cp_decomp'
    num_sents = 5e5
    min_count = 10
    for arg in sys.argv:
        if arg.startswith('--method='):
            method = arg.split('--method=')[1]
        if arg.startswith('--num_sents='):
            num_sents = float(arg.split('--num_sents=')[1])
        if arg.startswith('--min_count='):
            min_count = float(arg.split('--min_count=')[1])
    print('Creating sandbox with method {}, num_sents {} and min_count {}.'.format(method, num_sents, min_count))

    sandbox = GensimSandbox(
        method=method,
        num_sents=num_sents,
        embedding_dim=300,
        min_count=min_count,
    )
    sandbox.train()
    print('vocab len: {}'.format(len(sandbox.model.vocab)))

if __name__ == '__main__':
    main()

