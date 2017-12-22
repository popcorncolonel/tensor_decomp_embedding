import dill
import numpy as np
import random
import os
import sklearn
import sys
import tensorflow as tf
import time

from gensim.models import word2vec
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from functools import lru_cache

def write_embedding_to_file(embedding, model, fname='vectors.txt'):
    vectors = {}
    for word in model.vocab:
        word_vocab = model.vocab[word]
        word_vect = embedding[word_vocab.index]
        vect_list = ['{:.7f}'.format(x) for x in word_vect]
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
        f.write('{} {}\n'.format(count, embedding.shape[1]))  # write the number of vects
        f.write(content)

def evaluate(embedding, method, model):
    rel_path = 'vectors_{}.txt'.format(method)
    write_embedding_to_file(embedding, model, rel_path)
    out_fname = 'results_{}.txt'.format(method)
    evaluate_vectors_from_path(rel_path, out_fname)
    model.clear_sims()
    model.syn0 = embedding
    print("most similar to king - man + woman: {}".format(model.most_similar(
        positive=['king', 'woman'], negative=['man'],
        topn=5,
    )))
    print("most similar to king: {}".format(model.most_similar(
        positive=['king'],
        topn=5,
    )))
    print('done evaluating {}.'.format(method))

def evaluate_vectors_from_path(vector_path, results_path):
    os.system('python3 embedding_benchmarks/scripts/evaluate_on_all.py -f {} -o results/{}'.format(vector_path, results_path))


class EmbeddingTaskEvaluator(object):
    def __init__(self, method: str, fname: str=None, normalize_vects: bool=True, nonneg: bool=False, seed_bump=0, embedding_format='normal'):
        '''
        `fname` is the name of an embedding vectors file 
        '''
        self.embedding_dict = {}
        if fname is None:
            fname = 'vectors_{}.txt'.format(method)
        self.fname = fname
        with open(fname, 'r') as f:
            for i, line in enumerate(list(f)):
                line = line.strip()
                if i != 0:  # (skip header)
                    [word, vectstring] = line.split(maxsplit=1)
                    self.embedding_dict[word] = np.fromstring(vectstring, dtype=np.float32, sep=' ')
                    if nonneg:
                        self.embedding_dict[word] = self.embedding_dict[word].clip(min=0.0)
                    if normalize_vects:
                        self.embedding_dict[word] /= np.linalg.norm(self.embedding_dict[word])
                    self.embedding_dim = len(self.embedding_dict[word])
        self.normalize_vects = normalize_vects
        self.method = method
        self.seed_bump = seed_bump
        random.seed(42 + self.seed_bump)
        self._setup_analogy_graph()

    def get_word_classification_data_pos(self, split_type='train'):
        words_and_POSs = []
        with open('evaluation_data/pos.txt') as f:
            all_words = set()
            for line in list(f):
                line = line.strip()
                [word, pos] = line.split(maxsplit=1)
                if word in self.embedding_dict and word not in all_words:
                    words_and_POSs.append((word, pos))
                    all_words.add(word)
        random.seed(42 + self.seed_bump)
        random.shuffle(words_and_POSs)
        deterministic_words = [(self.embedding_dict[word], pos) for (word, pos) in words_and_POSs if ' ' not in pos]  # Words with only one possible POS
        num_words = len(deterministic_words)
        split_point = int(.85 * num_words)
        if split_type == 'train':
            data = deterministic_words[:split_point]
        elif split_type == 'test':
            data = deterministic_words[split_point:]
        else:
            raise ValueError('Unrecognized split type {}'.format(split_type))
        X = np.array([x for (x,y_) in data])
        y = np.array([y_ for (x,y_) in data])
        if self.normalize_vects:
            X = sklearn.preprocessing.normalize(X)
        return X, y

    def get_word_classification_data_emotions(self, split_type='train'):
        words_and_emotions = []
        with open('evaluation_data/emotions.txt') as f:
            for line in list(f):
                line = line.strip()
                [word, emotion] = line.split(maxsplit=1)
                if word in self.embedding_dict:
                    words_and_emotions.append((word, emotion))
        random.seed(42 + self.seed_bump)
        random.shuffle(words_and_emotions)
        deterministic_words = [(self.embedding_dict[word], cls) for (word, cls) in words_and_emotions if ' ' not in cls]  # Words with only one possible class
        pos_neg_words = [(vec, cls) for (vec, cls) in deterministic_words if cls in ['emotion_positive', 'emotion_negative']]
        deterministic_words = pos_neg_words
        num_words = len(deterministic_words)
        split_point = int(.85 * num_words)
        if split_type == 'train':
            data = deterministic_words[:split_point]
        elif split_type == 'test':
            data = deterministic_words[split_point:]
        else:
            raise ValueError('Unrecognized split type {}'.format(split_type))
        X = np.array([x for (x,y_) in data])
        y = np.array([y_ for (x,y_) in data])
        if self.normalize_vects:
            X = sklearn.preprocessing.normalize(X)
        return X, y

    def word_classification_tasks(self, print_score=False, classification_problem='PoS', train_pct=1.0):
        if classification_problem == 'PoS':
            X, y = self.get_word_classification_data_pos('train')
            X_test, y_test = self.get_word_classification_data_pos('test')
        elif classification_problem == 'emotions':
            X, y = self.get_word_classification_data_emotions('train')
            X_test, y_test = self.get_word_classification_data_emotions('test')
        else:
            raise ValueError(classification_problem + ' not defined')
        X = X[:int(len(X) * train_pct)]
        y = y[:int(len(y) * train_pct)]

        classifier = LogisticRegression()
        classifier.fit(X, y)
        score = classifier.score(X_test, y_test)
        if print_score:
            print('Word classification ({}, {}%) score: {}'.format(classification_problem, int(train_pct*100), score))
        return score

    def get_analogy_data(self, split_type='train', seed=0, is_sem_only=False):
        from embedding_benchmarks.scripts.web.datasets.analogy import fetch_google_analogy
        analogy = fetch_google_analogy()
        X = analogy['X']
        y = analogy['y']
        categories = analogy['category_high_level']
        parallel_lists = list(zip(X,y,categories))
        if is_sem_only:
            parallel_lists = [_ for _ in parallel_lists if _[2] == 'semantic']
        random.seed(42 + seed)
        random.shuffle(parallel_lists)
        X = [x[0] for x in parallel_lists]
        y = [x[1] for x in parallel_lists]
        categories = [x[2] for x in parallel_lists]
        valid_pairs = []
        query_words = []
        answer_words = []
        valid_categories = []
        for triple, answer, cat in zip(X, y, categories):
            if all([x in self.embedding_dict for x in triple]) and answer in self.embedding_dict:
                triple_embedded = np.array([self.embedding_dict[x] for x in triple])
                if self.normalize_vects:
                    triple_embedded = sklearn.preprocessing.normalize(triple_embedded)
                answer_embedded = np.array(self.embedding_dict[answer])
                triple = [x for x in triple] 
                if triple in query_words and answer in answer_words:
                    continue
                valid_pairs.append((triple_embedded, answer_embedded))
                query_words.append(triple)
                answer_words.append(answer)
                valid_categories.append(cat)

        if split_type == 'train':
            print('{} valid analogy questions out of {} total ({}%)'.format(len(valid_pairs), len(X), 100.0*len(valid_pairs) / len(X)))
        num_words = len(valid_pairs)
        split_point = int(.85 * num_words)
        if split_type == 'train':
            data = valid_pairs[:split_point]
            query_data = query_words[:split_point]
            answer_data = answer_words[:split_point]
            category_data = valid_categories[:split_point]
        elif split_type == 'test':
            data = valid_pairs[split_point:]
            query_data = query_words[split_point:]
            answer_data = answer_words[split_point:]
            category_data = valid_categories[split_point:]
        else:
            raise ValueError('Unrecognized split type {}'.format(split_type))
        x1s = np.array([x[0][0] for x in data])
        x2s = np.array([x[0][1] for x in data])
        x3s = np.array([x[0][2] for x in data])
        y = np.array([x[1] for x in data])

        x1s = sklearn.preprocessing.normalize(x1s)
        x2s = sklearn.preprocessing.normalize(x2s)
        x3s = sklearn.preprocessing.normalize(x3s)
        y = sklearn.preprocessing.normalize(y)
        return x1s, x2s, x3s, y, query_data, answer_data, category_data

    def _setup_analogy_graph(self, reg_param=0.005):
        v1, v2, v3, v4 = (None,) * 4
        v4_hat = None
        train_op = None
        loss = None
        with tf.device('/cpu:0'):
            W1 = tf.Variable(initial_value=np.identity(self.embedding_dim), name='W1', dtype=tf.float64)
            W2 = tf.Variable(initial_value=np.identity(self.embedding_dim), name='W2', dtype=tf.float64)
            W3 = tf.Variable(initial_value=np.identity(self.embedding_dim), name='W3', dtype=tf.float64)
            b = tf.Variable(tf.zeros([self.embedding_dim], dtype=tf.float64), name='b')
            v1 = tf.placeholder(tf.float64, shape=[None, self.embedding_dim], name='v1')
            v2 = tf.placeholder(tf.float64, shape=[None, self.embedding_dim], name='v2')
            v3 = tf.placeholder(tf.float64, shape=[None, self.embedding_dim], name='v3')
            v4 = tf.placeholder(tf.float64, shape=[None, self.embedding_dim], name='v4')

            v1_e = tf.expand_dims(v1, -1)  # [?, 300] -> [?, 300, 1]
            v2_e = tf.expand_dims(v2, -1)
            v3_e = tf.expand_dims(v3, -1)
            matmul1s = tf.scan(lambda _, v: tf.matmul(W1, v), v1_e)
            matmul2s = tf.scan(lambda _, v: tf.matmul(W2, v), v2_e)
            matmul3s = tf.scan(lambda _, v: tf.matmul(W3, v), v3_e)
            if True:  # if add non-linearities
                matmul1s = tf.tanh(matmul1s)
                matmul2s = tf.tanh(matmul2s)
                matmul3s = tf.tanh(matmul3s)
            pred_value = -matmul1s + matmul2s + matmul3s
            pred_value = tf.squeeze(pred_value)
            pred_value += b
            v4_hat = pred_value
            v4_hat /= tf.sqrt(tf.nn.l2_loss(pred_value) * 2)  # [?, 300]

            losses = tf.reduce_sum(tf.squared_difference(v4, v4_hat), axis=1)
            self.prediction_loss = tf.reduce_mean(losses)
            self.loss = self.prediction_loss
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

            self.v1 = v1
            self.v2 = v2
            self.v3 = v3
            self.v4 = v4
            self.W1 = W1
            self.W2 = W2
            self.W3 = W3
            self._create_analogy_ops(reg_param)

    def _create_analogy_ops(self, reg_param, regularize_all=False): 
            # regularization
            self.reg_param = reg_param
            if regularize_all:
                self.reg_loss = (1/3) * reg_param * (tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2) + tf.nn.l2_loss(self.W3)) 
            else:
                self.reg_loss = reg_param * tf.nn.l2_loss(self.W3)
            self.loss += self.reg_loss
            self.train_op = self.optimizer.minimize(self.loss, self.global_step)

    def _train_analogy_NN(self, x1s, x2s, x3s, y, verbose=False, iter_pct=1.0):
        def chunker(seq, size):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        config = tf.ConfigProto(
            allow_soft_placement=True,
        )
        sess = tf.Session(config=config)
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            n_iters = 1
            n_iters = max(1, int(n_iters * iter_pct))
            print('n_iters = {}'.format(n_iters))
            for _ in range(n_iters):
                if verbose:
                    print('running batches...')
                for x1s_batch, x2s_batch, x3s_batch, y_batch in zip(chunker(x1s, 25), chunker(x2s, 25), chunker(x3s, 25), chunker(y, 25)):
                    _, loss_val, step, p_loss, r_loss = sess.run([
                        self.train_op,
                        self.loss,
                        self.global_step,
                        self.prediction_loss,
                        self.reg_loss,
                    ], feed_dict={
                        self.v1: x1s_batch,
                        self.v2: x2s_batch,
                        self.v3: x3s_batch,
                        self.v4: y_batch,
                    })
                    if True:  # verbose:
                        if step % 5 == 0:
                            print('loss at step {}: {:.3f} ({:.3f},{:.3f} = pred,reg loss)'.format(step, loss_val, p_loss, r_loss))
            print('Prediction loss: {:.2f}, Regularization loss: {:.2f} (at end of training)'.format(p_loss, r_loss))
        return self.W1.eval(sess), self.W2.eval(sess), self.W3.eval(sess)

    def analogy_tasks(self, train_pct=1.0, verbose=True, reg_param=.001, is_sem_only=False, iter_pct=1.0, regularize_all=False):
        x1s, x2s, x3s, y, word_X_train, word_y_train, cats_train = self.get_analogy_data(
            'train',
            seed=self.seed_bump,
            is_sem_only=is_sem_only,
        )
        x1s_test, x2s_test, x3s_test, y_test, word_X_test, word_y_test, categories = self.get_analogy_data(
            'test',
            seed=self.seed_bump,
            is_sem_only=is_sem_only,
        )
        print('train_pct: {}'.format(train_pct * 100))
        same_analogies = [(trip, ans) for (trip, ans) in zip(word_X_test, word_y_test) if trip in word_X_train and ans in word_y_train]
        assert len(same_analogies) == 0
        x1s = x1s[:int(train_pct * len(x1s))]
        x2s = x2s[:int(train_pct * len(x2s))]
        x3s = x3s[:int(train_pct * len(x3s))]
        y = y[:int(train_pct * len(y))]
        if verbose:
            print("{} training words".format(len(x1s)))
            print("{} testing words".format(len(x1s_test)))
        if reg_param != self.reg_param:
            print('creating a new set of analogy ops due to change in reg_param')
            print('new reg_param: {}'.format(reg_param))
            print('regularize_all: {}'.format(regularize_all))
            self._create_analogy_ops(reg_param, regularize_all=regularize_all)
        W1, W2, W3 = self._train_analogy_NN(x1s, x2s, x3s, y, iter_pct=iter_pct)
        print('learned NN. evaluating...')

        correct_syn = 0
        total_syn = 0
        correct_sem = 0
        total_sem = 0
        def get_closest_vocab_word(predicted, query):
            best_word = None
            best_vect = None
            best_dist = float('-inf')
            for word, vect in self.embedding_dict.items():
                if word not in query:
                    dist = np.dot(vect, predicted)  # cosine similarity
                    dist /= (np.linalg.norm(vect) * np.linalg.norm(predicted))
                    if dist > best_dist:
                        best_word = word
                        best_vect = vect
                        best_dist = dist
            return best_word, best_vect

        ordered_embedding_words = []
        embedding_mat = []
        for word, vect in self.embedding_dict.items():
            ordered_embedding_words.append(word)
            embedding_mat.append(vect)
        embedding_mat = np.array(embedding_mat)  # |V| x k
        P1 = x1s_test
        P2 = x2s_test
        P3 = x3s_test
        predictions = -np.dot(W1, P1.T) + np.dot(W2, P2.T) + np.dot(W3, P3.T)
        predictions = sklearn.preprocessing.normalize(predictions)
        dots = np.dot(embedding_mat, predictions)
        argmaxes = np.argmax(dots, axis=0)
        predicted_words = [ordered_embedding_words[i] for i in argmaxes]
        for predicted_word, correct_word, cat in zip(predicted_words, word_y_test, categories):
            if cat == 'syntactic':
                if predicted_word == correct_word:
                    correct_syn += 1
                total_syn += 1
            elif cat == 'semantic':
                if predicted_word == correct_word:
                    correct_sem += 1
                total_sem += 1
            else:
                raise ValueError('unrecognized category')

        sem_score = 0.
        syn_score = 0.
        if total_sem != 0:
            sem_score = correct_sem / total_sem
            print('Semantic Analogy Accuracy: {}'.format(sem_score))
        if total_syn != 0:
            syn_score = correct_syn / total_syn
            print('Syntactic Analogy Accuracy: {}'.format(syn_score))
        return sem_score, syn_score

    def get_sent_class_data_old(self, split_type='train'):
        pos_Xy = []
        neg_Xy = []
        pos_dir = 'evaluation_data/sentiment/pos/'
        for fname in os.listdir(pos_dir):
            with open(pos_dir + fname, 'r') as f:
                contents = [x.strip() for x in f]
            all_words = ' '.join(contents)
            pos_Xy.append((all_words, True))
        neg_dir = 'evaluation_data/sentiment/neg/'
        for fname in os.listdir(neg_dir):
            with open(neg_dir + fname, 'r') as f:
                contents = [x.strip() for x in f]
            all_words = ' '.join(contents)
            neg_Xy.append((all_words, False))
        all_data = pos_Xy + neg_Xy
        random.seed(42 + self.seed_bump)
        random.shuffle(all_data)

        split_point = int(.85 * len(all_data))
        if split_type == 'train':
            data = all_data[:split_point]
        elif split_type == 'test':
            data = all_data[split_point:]
        else:
            raise ValueError('Unrecognized split type {}'.format(split_type))
        tokenized_X = [x[0].split() for x in data]
        X_data = [np.array([self.embedding_dict[w] for w in sent if w in self.embedding_dict]) for sent in tokenized_X]
        y_data = [x[1] for x in data]
        return X_data, y_data

    def get_sent_class_data(self, split_type='train'):
        pos_Xy = []
        neg_Xy = []
        if split_type in ['train', 'test']:
            fix_sent = lambda s: s.replace('<br />', '').replace(',', '').replace('.', '').replace('"', '').replace("'", '').replace('!', '').replace('?', '').replace('/', ' ').replace('(', '').replace(')', '')
            pos_dir = 'aclImdb/{}/pos/'.format(split_type)
            for fname in os.listdir(pos_dir):
                with open(pos_dir + fname, 'r') as f:
                    contents = [x.strip().lower() for x in f]
                all_words = ' '.join(contents)
                all_words = fix_sent(all_words)
                pos_Xy.append((all_words, True))
            neg_dir = 'aclImdb/{}/neg/'.format(split_type)
            for fname in os.listdir(neg_dir):
                with open(neg_dir + fname, 'r') as f:
                    contents = [x.strip().lower() for x in f]
                all_words = ' '.join(contents)
                all_words = fix_sent(all_words)
                neg_Xy.append((all_words, False))
        else:
            raise ValueError('Unrecognized split type {}'.format(split_type))
        all_data = pos_Xy + neg_Xy
        random.seed(42 + self.seed_bump)
        random.shuffle(all_data)

        tokenized_X = [x[0].split() for x in all_data]
        X_data = [np.array([self.embedding_dict[w] for w in sent if w in self.embedding_dict]) for sent in tokenized_X]
        y_data = [x[1] for x in all_data]
        return X_data, y_data

    def sentiment_analysis_tasks(self, print_score=False, train_pct=1.0):
        X, y = self.get_sent_class_data('train')
        X_test, y_test = self.get_sent_class_data('test')
        X = np.array([x.sum(axis=0) for x in X])[:int(train_pct*len(X))]
        y = y[:int(train_pct*len(y))]
        X_test = np.array([x.sum(axis=0) for x in X_test])


        classifier = LogisticRegression()
        classifier.fit(X, y)
        score = classifier.score(X_test, y_test)
        if print_score:
            print('Sentiment classification score: {}'.format(score))
        return score

    @lru_cache(maxsize=32)
    def outlier_detection(self, verbose=True, n=3):
        from wikisem500.src.evaluator import Evaluator
        from wikisem500.src.embeddings import WrappedEmbedding
        from wikisem500.src.outlier_test_group import TestGroup
        from wikisem500.src.utils import scandir

        def read_dataset_directory(d):
            for f in scandir(d):
                if f.name.endswith('.txt') and f.is_file():
                    yield TestGroup.from_file(f.path)

        def score_embedding(embedding, groups):
            evaluator = Evaluator(groups)
            evaluator.evaluate(embedding, n=n)
            if verbose:
                print("   RESULTS")
                print("==============")
                print("OPP score: %f" % evaluator.opp)
                print("Accuracy: %f" % evaluator.accuracy)
                print("---------------------------------")
                print("Total number of test groups: %d" % evaluator.num_total_groups)
                print("Number of filtered test groups: %d (%f%%)" % (evaluator.num_filtered_groups, evaluator.percent_filtered_groups))
                print("Total number of non-OOV test cases: %d" % evaluator.num_cases)
                print("Number of filtered cluster entities: %d/%d (mean per %% cluster: %f%%)" % (evaluator.num_filtered_cluster_items, evaluator.num_total_cluster_items, evaluator.percent_filtered_cluster_items))
                print("Number of filtered outlier entities: %d/%d (mean per %% cluster: %f%%)" % (evaluator.num_filtered_outliers, evaluator.num_total_outliers, evaluator.percent_filtered_outliers))
            return (evaluator.opp, evaluator.accuracy)

        embedding = WrappedEmbedding.from_word2vec(self.fname, binary=False)
        dataset = list(read_dataset_directory('wikisem500/dataset/en/'))
        if verbose:
            print("Scoring...")
        opp, accuracy = score_embedding(embedding, dataset)
        return opp, accuracy


if __name__ == '__main__':
    method = 'cbow'
    if False:
        evaluator = EmbeddingTaskEvaluator(method)
        evaluator.word_classification_tasks(print_score=True)
        evaluator.analogy_tasks()
        sys.exit()
    else:
        fname = 'wikimodel_{}_{}'.format(int(1e5), 1000)
        model = dill.load(fname)
        with open('runs/{}/{}_{}_300'.format(method, int(1e5), 1000), 'rb') as f:
            embedding = dill.load(f)
        evaluate(embedding, method, model)

