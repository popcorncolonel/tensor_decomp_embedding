import datetime
import dill
import gensim
import gensim.utils
import numpy as np
import os
import pdb
import scipy
import scipy.io
import select
import shutil
import sys
import time
import tensorflow as tf

#from embedding_evaluation import write_embedding_to_file, evaluate, EmbeddingTaskEvaluator
from embedding_evaluation import write_embedding_to_file, EmbeddingTaskEvaluator
from gensim_utils import batch_generator, batch_generator2
from tensor_embedding import PMIGatherer, PpmiSvdEmbedding
from tensor_decomp import CPDecomp, SymmetricCPDecomp, JointSymmetricCPDecomp
from nltk.corpus import stopwords


stopwords = set(stopwords.words('english'))
grammar_stopwords = {',', "''", '``', '.', 'the'}
stopwords = stopwords.union(grammar_stopwords)


class GensimSandbox(object):
    def __init__(self, method, embedding_dim, num_articles, min_count, gpu=True):
        self.method = method
        self.embedding_dim = int(embedding_dim)
        self.min_count = int(min_count)
        self.num_articles = int(num_articles)
        self.gpu = gpu

        # To be assigned later
        self.model = None
        self.sess = None
        self.embedding = None
        self.to_save = {}

    def sentences_generator(self, num_articles=None):
        if num_articles is None:
            num_articles = self.num_articles
        gzipped_wiki = '../enwiki-latest-pages-articles.xml.bz2'
        wiki = gensim.corpora.wikicorpus.WikiCorpus(gzipped_wiki, dictionary={})
        articles = wiki.get_texts()
        n_tokens = 0
        count = 0
        for article in articles:
            if count % int(num_articles / 10) == 0:
                print("Just hit article {} out of {} ({}%)".format(count, num_articles, 100*count / num_articles))
            article = [x.decode('utf8') for x in article if x not in stopwords]
            if count < num_articles:
                n_tokens += len(article)
                count += 1
                yield article
            else:
                break
        print("avg article word length: {}".format(n_tokens / count))
        print("{} total tokens".format(n_tokens))
        print("num articles: {}".format(count))
        raise StopIteration

    def get_model_with_vocab(self, fname='wikimodel'):
        fname += '_{}_{}'.format(self.num_articles, self.min_count)
        model = gensim.models.Word2Vec(
            iter=1,
            max_vocab_size=None,
            negative=128,
            size=self.embedding_dim,
            min_count=self.min_count,
        )
        if not os.path.exists(fname):
            print('building vocab...')
            model.build_vocab(self.sentences_generator())
            with open(fname, 'wb') as f:
                dill.dump(model, f)
        else:
            print('depickling model...')
            with open(fname, 'rb') as f:
                model = dill.load(f)
        model.tt = 0
        model.cbow = 0
        model.cnn = 0
        model.subspace = 0
        model.vector_size = self.embedding_dim
        if self.method == 'tt':
            model.tt = 1
        elif self.method == 'subspace':
            model.subspace = 1
        elif self.method == 'cbow':
            model.cbow = 1
        elif self.method == 'cnn':
            model.cnn = 1
        print('Finished building vocab. length of vocab: {}'.format(len(model.vocab)))
        self.model = model
        return self.model

    def create_embedding_visualization(self):
        config = tf.ConfigProto(
            allow_soft_placement=True,
        )
        sess = tf.Session(config=config)
        with sess.as_default():
            U = tf.Variable(self.embedding, name='U')
            print('creating saver for embedding viz...')
            saver = tf.train.Saver()

        LOG_DIR = 'tf_logs/{}_{}_{}'.format(self.method, self.num_articles, self.min_count)
        LOG_DIR = os.path.join(LOG_DIR, 'embedding_viz')
        print('Saving embeddings to {}...'.format(LOG_DIR))
        if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
        global_step = tf.Variable(0)
        sess.run(tf.global_variables_initializer())
        saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step)
        f = open(LOG_DIR + '/metadata.tsv', 'w')
        for i in range(len(self.model.vocab)): f.write(self.model.index2word[i] + '\n')
        f.close()
        from tensorflow.contrib.tensorboard.plugins import projector
        print('Adding projector config...')
        summary_writer = tf.summary.FileWriter(LOG_DIR)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = 'U'
        embedding.metadata_path = 'metadata.tsv'

        projector.visualize_embeddings(summary_writer, config)

    def train_gensim_embedding(self):
        print('training...')
        batches = batch_generator(self.model, self.sentences_generator(), batch_size=128)
        self.model.train(sentences=None, batches=batches, gpu=self.gpu)
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

    def get_pmi_gatherer(self, n):
        gatherer = None
        if os.path.exists('gatherer_{}_{}_{}.pkl'.format(self.num_articles, self.min_count, n)):
            with open('gatherer_{}_{}_{}.pkl'.format(self.num_articles, self.min_count, n), 'rb') as f:
                t = time.time()
                import gc; gc.disable()
                gatherer = dill.load(f)
                gc.enable()
                print('Loading gatherer took {} secs'.format(time.time() - t))
        else:
            # batch_size doesn't matter. But higher is probably better (in terms of threading & speed)
            batches = batch_generator2(self.model, self.sentences_generator(num_articles=self.num_articles), batch_size=1000)
            gatherer = PMIGatherer(self.model, n=n)
            if self.num_articles <= 1e4:
                gatherer.populate_counts(batches, huge_vocab=False)
            else:
                gatherer.populate_counts(batches, huge_vocab=True, min_count=5)

            with open('gatherer_{}_{}_{}.pkl'.format(self.num_articles, self.min_count, n), 'wb') as f:
                t = time.time()
                import gc; gc.disable()
                dill.dump(gatherer, f)
                gc.enable()
                print('Dumping gatherer took {} secs'.format(time.time() - t))
        return gatherer

    def train_joint_online_cp_embedding(self, dimlist: list, dimweights: list, nonneg: bool):
        gatherers = [self.get_pmi_gatherer(dim) for dim in dimlist]
        exp_shifts = [1., 1.]
        shifts = [-np.log2(s) for s in exp_shifts]

        def sparse_tensor_batches(batch_size=1000):
            batches = batch_generator2(self.model, self.sentences_generator(num_articles=self.num_articles), batch_size=batch_size)
            for batch in batches:
                pairlist = [
                    gatherer.create_pmi_tensor(
                        batch=batch,
                        positive=True,
                        debug=False,
                        symmetric=True,
                        log_info=False,
                        neg_sample_percent=0.0,
                        pmi=True,
                        shift=shift,
                    )
                    for (shift, gatherer) in zip(shifts, gatherers)
                ]
                yield ([x[0] for x in pairlist], [x[1] for x in pairlist])

        config = tf.ConfigProto(
            allow_soft_placement=True,
        )
        self.sess = tf.Session(config=config)
        with self.sess.as_default():
            reg_param = 1e-6
            self.to_save['reg_param'] = reg_param
            print('reg_param: {}'.format(reg_param))
            decomp_method = JointSymmetricCPDecomp(
                size=len(self.model.vocab),
                dimlist=dimlist,
                dimweights=dimweights,
                rank=self.embedding_dim,
                sess=self.sess,
                reg_param=reg_param,
                nonneg=nonneg,
                gpu=True,
            )
        print('Starting JOINT CP Decomp training')
        decomp_method.train(sparse_tensor_batches())

        U = decomp_method.U.eval(self.sess)
        if nonneg:
            sparse_embedding = U.clip(min=0.0)
            self.embedding = sparse_embedding
        else:
            self.embedding = U.copy()

    def train_online_cp_embedding(self, ndims: int, symmetric: bool, nonneg: bool):
        gatherer = self.get_pmi_gatherer(ndims)
        if nonneg:
            shift = 0.
        else:
            shift = -np.log2(15.)

        def sparse_tensor_batches(batch_size=1000, symmetric=symmetric):
            batches = batch_generator2(self.model, self.sentences_generator(), batch_size=batch_size)
            for batch in batches:
                sparse_ppmi_tensor_pair = gatherer.create_pmi_tensor(
                    batch=batch,
                    positive=True,
                    debug=False,
                    symmetric=symmetric,
                    log_info=False,
                    neg_sample_percent=0.0 if nonneg else 0.05,
                    pmi=True,
                    shift=shift,
                )
                yield sparse_ppmi_tensor_pair

        (indices, values) = None, None  # to be filled in later
        config = tf.ConfigProto(
            allow_soft_placement=True,
        )
        self.sess = tf.Session(config=config)
        with self.sess.as_default():
            if symmetric:
                print('getting full PMI tensor...')
                (indices, values) = gatherer.create_pmi_tensor(positive=True, debug=False, symmetric=symmetric, shift=shift)
                mean_value = np.mean(values)
                print('mean tensor value: {}'.format(mean_value))

                n_entries = len(self.model.vocab) * self.embedding_dim
                # reg_param should be set so that initial reg. loss is about 1.0
                # random init: mean=(1. / self.embedding_dim) * (mu ** (1/ndims)). There will be ~|V|*k of these values.
                mean = (1. / self.embedding_dim) * (mean_value ** (1/ndims))
                #reg_param = mean / 100.
                reg_param = 0
                self.to_save['reg_param'] = reg_param
                print('reg_param: {}'.format(reg_param))
                decomp_method = SymmetricCPDecomp(
                    dim=len(self.model.vocab),
                    ndims=ndims,
                    rank=self.embedding_dim,
                    sess=self.sess,
                    optimizer_type='adam',
                    reg_param=reg_param,
                    nonneg=nonneg,
                    gpu=True,
                    mean_value=mean_value,
                )
            else:
                decomp_method = CPDecomp(
                    shape=(len(self.model.vocab),)*ndims,
                    rank=self.embedding_dim,
                    sess=self.sess,
                    optimizer_type='adam',
                    reg_param=0.0,
                )
        print('Starting CP Decomp training')
        decomp_method.train(sparse_tensor_batches())

        U = decomp_method.U.eval(self.sess)
        if not symmetric: 
            V = decomp_method.V.eval(self.sess)
            W = decomp_method.W.eval(self.sess)

            lambdaU = np.linalg.norm(U, axis=1)
            lambdaV = np.linalg.norm(V, axis=1)
            lambdaW = np.linalg.norm(W, axis=1)
            embedding = U / lambdaU[:, None]
            C1 = V / lambdaV[:, None]
            C2 = W / lambdaW[:, None]

            lambda_ = lambdaU * lambdaV * lambdaW
            D = np.diag(lambda_ ** (1/3))
            self.C1 = np.dot(C1, D)
            self.C2 = np.dot(C2, D)
            self.embedding = np.dot(embedding, D)
            pred_val = lambda x: (self.embedding[x[0]] * self.C1[x[1]] * self.C2[x[2]]).sum()
        else:
            if nonneg:
                sparse_embedding = U.clip(min=0.0)
                self.embedding = sparse_embedding
            else:
                self.embedding = U.copy()

            def mse(embedding_mat):
                total_err = 0.0
                for ix, val in zip(indices, values):
                    prod = embedding_mat[ix[0]].copy()
                    for i in range(1, ndims):
                        prod *= embedding_mat[ix[i]]
                    pred_val = prod.sum()
                    total_err += (pred_val - val) ** 2
                return total_err / len(indices)

            err = mse(self.embedding)
            print("RMSE: {:.3f}".format(np.sqrt(err)))
            self.to_save['RMSE'] = np.sqrt(err)
        #self.embedding /= np.linalg.norm(self.embedding, axis=1)[:, None]  # normalize vectors to unit lengths
        self.to_save['indices'] = indices
        self.to_save['values'] = values

    def train_random_embedding(self, gauss=False):
        if gauss:
            self.embedding = np.random.normal(0, .5, size=(len(self.model.vocab), self.embedding_dim))
        else:
            self.embedding = (np.random.rand(len(self.model.vocab), self.embedding_dim) - .5) * 2

    def train_save_sp_tensor(self, pmi=True):
        gatherer = self.get_pmi_gatherer(3)
        if pmi:
            print('creating PPMI tensor...')
        else:
            print('creating sparse count tensor...')
        indices, values = gatherer.create_pmi_tensor(positive=True, debug=True, symmetric=False, pmi=pmi, shift=-np.log2(15.))
        matfile_name = 'sp_tensor_{}_{}_log15.mat'.format(self.num_articles, self.min_count)
        scipy.io.savemat(matfile_name, {'indices': indices, 'values': values})
        print('saved {}. exiting.'.format(matfile_name))
        sys.exit()

    def restore_from_ckpt(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            U = tf.Variable(tf.random_uniform(
                shape=[len(self.model.vocab), self.embedding_dim],
                minval=-1.0,
                maxval=1.0,
            ), name="U")
            saver = tf.train.Saver({'U': U})
            dirname = '2017-03-16 02:40:19.675237'
            saver.restore(sess, tf.train.latest_checkpoint('tf_logs/{}/checkpoints/'.format(dirname)))
            U = U.eval()

        sparse_embedding = U.clip(min=0)
        self.embedding = U
        import pdb; pdb.set_trace()
        self.save_metadata()
        self.evaluate_embedding()
        pass

    def loadmatlab(self):
        #d = scipy.io.loadmat('../matlab/sp_tensor.mat')
        d = scipy.io.loadmat('sp_tensor_10000000_2000_log15.mat')
        values = d['values'].T
        indices = d['indices']

        d = scipy.io.loadmat('../matlab/UVW_300_10e6_2000_log15.mat')
        U = d['U']
        V = d['V']
        W = d['W']
        lambda_ = np.squeeze(d['lambda'])
        embedding  = np.dot(U, np.diag(lambda_ ** (1. / 3.)))
        C1 = np.dot(V, np.diag(lambda_ ** (1. / 3.)))
        C2 = np.dot(W, np.diag(lambda_ ** (1. / 3.)))
        def pred_xijk(i, j, k):
            hadamard = embedding[i] * C1[j] * C2[k] 
            return np.sum(hadamard)
        def mse(embedding=embedding, C1=C1, C2=C2):
            err = 0.0
            cnt = 0.0
            smallest_se = float('inf')  # se = squared error
            for val, ix in zip(values, indices):
                val = val[0]
                if cnt > 5e6:
                    return np.sqrt(err / cnt)
                pred_val = pred_xijk(ix[0], ix[1], ix[2])
                se = (val - pred_val)**2 
                #print('|{} - {}|^2 = {}'.format(val, pred_val, se))
                if se < smallest_se:
                    smallest_se = se
                err += se
                cnt += 1
            return err / cnt
        print("MSE: {}".format(mse()))
        self.embedding = embedding

    def train_svd_embedding(self):
        gatherer = self.get_pmi_gatherer(2)

        print('Making PPMI tensor for SVD...')
        dense_ppmi_tensor = gatherer.create_pmi_tensor(positive=True, numpy_dense_tensor=True, debug=True)
        del gatherer

        embedding_model = PpmiSvdEmbedding(self.model, embedding_dim=self.embedding_dim)
        print("calculating SVD on {0}x{0}...".format(len(self.model.vocab)))
        t = time.time()
        embedding_model.learn_embedding(dense_ppmi_tensor)
        total_svd_time = time.time() - t
        print("SVD on {}x{} took {}s".format(len(self.model.vocab), len(self.model.vocab), total_svd_time))
        self.embedding = embedding_model.get_embedding_matrix()

    def evaluate_embedding(self):
        #evaluate(self.embedding, self.method, self.model)
        evaluator = EmbeddingTaskEvaluator(self.method)
        evaluator.word_classification_tasks(print_score=True)
        evaluator.sentiment_analysis_tasks(print_score=True)
        evaluator.outlier_detection()
        evaluator.analogy_tasks()

    def save_metadata(self):
        grandparent_dir = os.path.abspath('runs/{}'.format(self.method))
        parent_dir = grandparent_dir + '/' + '{}_{}_{}'.format(self.num_articles, self.min_count, self.embedding_dim)
        if not os.path.exists(grandparent_dir):
            os.mkdir(grandparent_dir)
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        timestamp = str(datetime.datetime.now())
        with open(parent_dir + '/metadata.txt', 'w') as f:
            f.write('Evaluation time: {}\n'.format(timestamp))
            f.write('Vocab size: {}\n'.format(len(self.model.vocab)))
            f.write('Elapsed training time: {}\n'.format(time.time() - self.start_time))
            print('Elapsed training time: {}\n'.format(time.time() - self.start_time))
        write_embedding_to_file(self.embedding, self.model, parent_dir + '/vectors.txt')
        try:
            shutil.copyfile('../results/results_{}.txt'.format(self.method), parent_dir + '/results.txt')
            shutil.copyfile('../results/results_{}.xlsx'.format(self.method), parent_dir + '/results.xlsx')
        except Exception as e:
            print('caught exception while trying to copy results: {}'.format(e))
            import pdb; pdb.set_trace()
        try:
            shutil.copyfile('../results/outlier_det_{}.txt'.format(self.method), parent_dir + '/results_outlier_det.txt')
            shutil.copyfile('../results/word_class_{}.txt'.format(self.method), parent_dir + '/results_word_class.txt')
        except Exception as e:
            print('caught exception while trying to copy results: {}'.format(e))
            import pdb; pdb.set_trace()
        with open(parent_dir + '/embedding.pkl', 'wb') as f:
            dill.dump(self.embedding, f)
        try:
            with open(parent_dir + '/model.pkl', 'wb') as f:
                dill.dump(self.model, f)
        except Exception as e:
            print(e)
            print('caught exception trying to dump model. wooops. carrying on...')
        for name, obj in self.to_save.items():
            fname = parent_dir + '/' + name + '.pkl'
            with open(fname, 'wb') as f:
                dill.dump(obj, f)
        self.create_embedding_visualization()

    def train(self, experiment=''):
        self.get_model_with_vocab()
        self.start_time = time.time()
        if experiment != '':
            experiment = '_' + experiment.replace(' ', '_')
            print("experiment name: {}".format(experiment[1:]))

        if self.method in ['random']:
            self.method += experiment
            self.train_random_embedding(gauss=True)
        elif self.method in ['cp']:  # Basic CP Decomp (from matlab)
            self.method += experiment
            self.loadmatlab()
        elif self.method in ['cp-s']:  # Symmetric CP Decomp experiments
            self.method += experiment
            self.train_online_cp_embedding(ndims=3, symmetric=True, nonneg=False)
        elif self.method in ['cp-sn']:
            self.method += experiment
            self.train_online_cp_embedding(ndims=3, symmetric=True, nonneg=True)
        elif self.method in ['cp-s_4d']:
            self.method += experiment
            self.train_online_cp_embedding(ndims=4, symmetric=True, nonneg=False)
        elif self.method in ['cp-sn_4d']:
            self.method += experiment
            self.train_online_cp_embedding(ndims=4, symmetric=True, nonneg=True)
        elif self.method in ['jcp-s']:  # Joint Symmetric CP Decomp experiments
            self.method += experiment
            self.train_joint_online_cp_embedding(dimlist=[2,3], dimweights=[1., 1.,], nonneg=False)
        elif self.method in ['jcp-s_432']:
            self.method += experiment
            self.train_joint_online_cp_embedding(dimlist=[2,3,4], dimweights=[2., .4, .1], nonneg=False)
        elif self.method in ['nnse']:
            self.method += experiment
            self.train_online_cp_embedding(ndims=2, symmetric=True, nonneg=True)
        elif self.method in ['cnn', 'cbow', 'tt', 'subspace']:
            self.method += experiment
            self.train_gensim_embedding()
        elif self.method in ['svd']:
            self.method += experiment
            self.train_svd_embedding()
        elif self.method in ['matlab']:
            self.method += experiment
            self.train_save_sp_tensor()
        elif self.method in ['restore_ckpt']:
            self.method += experiment
            self.restore_from_ckpt()
        else:
            raise ValueError('undefined method {}'.format(self.method))
        for i, vec in enumerate(self.embedding):
            if np.linalg.norm(vec) == 0.0:
                self.embedding[i][0] += 1e-4

        self.evaluate_embedding()
        self.save_metadata()
        print('All done training and evaluating {}!'.format(self.method))


def list_vars_in_checkpoint(dirname):
    ''' Just for tf debugging.  '''
    from tensorflow.contrib.framework.python.framework.checkpoint_utils import list_variables
    abspath = os.path.abspath(dirname)
    return list_variables(abspath)


def input_with_timeout(prompt, timeout):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [],[], timeout)
    if ready:
        return sys.stdin.readline().rstrip('\n') # expect stdin to be line-buffered
    return ''


def main():
    method = None
    num_articles = None
    min_count = None
    embedding_dim = None
    for arg in sys.argv:
        if arg.startswith('--method='):
            method = arg.split('--method=')[1]
        if arg.startswith('--num_articles='):
            num_articles = float(arg.split('--num_articles=')[1])
        if arg.startswith('--min_count='):
            min_count = float(arg.split('--min_count=')[1])
        if arg.startswith('--embedding_dim='):
            embedding_dim = int(arg.split('--embedding_dim=')[1])
    assert all([method, num_articles, min_count, embedding_dim]), 'Please supply all necessary parameters'

    print('Creating sandbox with method {}, num_articles {} and min_count {}.'.format(method, num_articles, min_count))

    sandbox = GensimSandbox(
        method=method,
        num_articles=num_articles,
        embedding_dim=embedding_dim,
        min_count=min_count,
    )
    sandbox.train(experiment='')

if __name__ == '__main__':
    main()

