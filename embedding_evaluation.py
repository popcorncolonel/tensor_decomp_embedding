import numpy as np
import random
import os
import sklearn
import sys
import tensorflow as tf
import time

from functools import lru_cache
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


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
    os.system('python3 embedding_benchmarks/scripts/evaluate_on_all.py -f /home/eric/code/gensim/{} -o /home/eric/code/gensim/results/{}'.format(vector_path, results_path))


class EmbeddingTaskEvaluator(object):
    def __init__(self, method: str, fname: str=None, normalize_vects: bool=True, nonneg: bool=False, seed_bump=0):
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

    @lru_cache()
    def get_word_classification_data_old(self, split_type='train'):
        words_and_POSs = []
        with open('evaluation_data/pos.txt') as f:
            for line in list(f):
                line = line.strip()
                [word, pos] = line.split(maxsplit=1)
                if word in self.embedding_dict:
                    words_and_POSs.append((word, pos))
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

    @lru_cache()
    def get_word_classification_data(self, split_type='train'):
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

    def word_classification_tasks(self, print_score=False):
        X, y = self.get_word_classification_data('train')
        X_test, y_test = self.get_word_classification_data('test')

        #classifier = LogisticRegression()
        classifier = MLPClassifier(hidden_layer_sizes=(100, 42))
        classifier.fit(X, y)
        score = classifier.score(X_test, y_test)
        if print_score:
            print('Word classification score: {}'.format(score))
        with open('results/word_class_{}.txt'.format(self.method), 'w') as f:
            print('Score: {}'.format(score), file=f)
        return score

    @lru_cache()
    def get_analogy_data(self, split_type='train'):
        from embedding_benchmarks.scripts.web.datasets.analogy import fetch_google_analogy
        analogy = fetch_google_analogy()
        X = analogy['X']
        y = analogy['y']
        categories = analogy['category_high_level']
        parallel_lists = list(zip(X,y,categories))
        random.seed(42 + self.seed_bump)
        random.shuffle(parallel_lists)
        X = [x[0] for x in parallel_lists]
        y = [x[1] for x in parallel_lists]
        categories = [x[2] for x in parallel_lists]
        valid_pairs = []
        word_pairs = []
        valid_categories = []
        for triple, answer, cat in zip(X, y, categories):
            if all([x in self.embedding_dict for x in triple]) and answer in self.embedding_dict:
                triple_embedded = np.array([self.embedding_dict[x] for x in triple])
                if self.normalize_vects:
                    triple_embedded = sklearn.preprocessing.normalize(triple_embedded)
                answer_embedded = np.array(self.embedding_dict[answer])
                valid_pairs.append((triple_embedded, answer_embedded))
                word_pairs.append((triple, answer))
                valid_categories.append(cat)

        if split_type == 'train':
            print('{} valid analogy questions out of {} total ({}%)'.format(len(valid_pairs), len(X), 100.0*len(valid_pairs) / len(X)))
        num_words = len(valid_pairs)
        split_point = int(.85 * num_words)
        if split_type == 'train':
            data = valid_pairs[:split_point]
            word_data = word_pairs[:split_point]
            category_data = valid_categories[:split_point]
        elif split_type == 'test':
            data = valid_pairs[split_point:]
            word_data = word_pairs[split_point:]
            category_data = valid_categories[split_point:]
        else:
            raise ValueError('Unrecognized split type {}'.format(split_type))
        X = np.array([x[0] for x in data])
        y = np.array([x[1] for x in data])
        word_X = np.array([x[0] for x in word_data])
        word_y = np.array([x[1] for x in word_data])
        if self.normalize_vects:
            y = sklearn.preprocessing.normalize(y)
        return X, y, word_X, word_y, category_data

    def _analogy_train_NN(self, X, y, verbose=False):
        config = tf.ConfigProto(
            allow_soft_placement=True,
        )
        sess = tf.Session(config=config)
        v1, v2, v3, v4 = (None,) * 4
        v4_hat = None
        train_op = None
        loss = None
        with sess.as_default():
            with tf.device('/gpu:0'):
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
                pred_value = -matmul1s + matmul2s + matmul3s
                pred_value = tf.squeeze(pred_value)
                v4_hat = pred_value / tf.sqrt(tf.nn.l2_loss(pred_value) * 2)  # [?, 300]

                losses = tf.reduce_sum(tf.squared_difference(v4, v4_hat), axis=1)
                loss = tf.reduce_mean(losses)

                global_step = tf.Variable(0, name='global_step', trainable=False)
                optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
                #grads_and_vars = optimizer.compute_gradients(loss)
                #train_op = optimizer.apply_gradients(grads_and_vars, global_step)
                train_op = optimizer.minimize(loss, global_step)

                sess.run(tf.global_variables_initializer())

                def chunker(seq, size):
                    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
                n_iters = 3
                for _ in range(n_iters):
                    if verbose:
                        print('running batches...')
                    for X_batch, y_batch in zip(chunker(X, 25), chunker(y, 25)):
                        _, loss_val, step = sess.run([train_op, loss, global_step], feed_dict={
                            v1: X_batch[:, 0],
                            v2: X_batch[:, 1],
                            v3: X_batch[:, 2],
                            v4: y_batch,
                        })
                        if verbose:
                            if step % 20 == 0:
                                print('loss at step {}: {}'.format(step, loss_val))
        return sess, W1.eval(sess), W2.eval(sess), W3.eval(sess)

    def analogy_tasks(self, verbose=True):
        '''
        Currently not working for any embedding. 
        '''
        X, y, word_X_train, word_y_train, cats_train = self.get_analogy_data('train')
        X_test, y_test, word_X_test, word_y_test, categories = self.get_analogy_data('test')
        if verbose:
            print("{} training words".format(len(X)))
            print("{} testing words".format(len(X_test)))
        sess, W1, W2, W3 = self._analogy_train_NN(X, y)

        correct_syn = 0
        total_syn = 0
        correct_sem = 0
        total_sem = 0
        for p, query, correct_word, cat in zip(X_test, word_X_test, word_y_test, categories):
            def get_closest_vocab_word(predicted):
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

            predicted = -np.dot(W1, p[0]) + np.dot(W2, p[1]) + np.dot(W3, p[2])
            predicted /= np.linalg.norm(predicted)
            predicted = np.squeeze(predicted)

            best_word, best_vect = get_closest_vocab_word(predicted)
            if cat == 'syntactic':
                if best_word == correct_word:
                    correct_syn += 1
                total_syn += 1
            elif cat == 'semantic':
                if best_word == correct_word:
                    correct_sem += 1
                total_sem += 1
            else:
                raise ValueError('unrecognized category')

        print('Semantic Analogy Accuracy: {}'.format(correct_sem / total_sem))
        print('Syntactic Analogy Accuracy: {}'.format(correct_syn / total_syn))
        return (correct_sem / total_sem, correct_syn / total_syn)

    @lru_cache()
    def get_sent_class_data(self, split_type='train'):
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

    def sentiment_classification_tasks(self, print_score=False):
        X, y = self.get_sent_class_data('train')
        X_test, y_test = self.get_sent_class_data('test')
        X = np.array([x.sum(axis=0) for x in X])
        X_test = np.array([x.sum(axis=0) for x in X_test])

        classifier = LogisticRegression()
        t = time.time()
        classifier.fit(X, y)
        score = classifier.score(X_test, y_test)
        if print_score:
            print('Sentiment classification score: {}'.format(score))
        with open('results/sentiment_class_{}.txt'.format(self.method), 'w') as f:
            print('Score: {}'.format(score), file=f)
        return score

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
        with open('results/outlier_det_{}.txt'.format(self.method), 'w') as f:
            print('OPP: {}'.format(opp), file=f)
            print('Accuracy: {}'.format(accuracy), file=f)
        return opp, accuracy


if __name__ == '__main__':
    method = 'nnse'
    evaluator = EmbeddingTaskEvaluator(method)
    evaluator.word_classification_tasks(print_score=True)
    sys.exit()
    #score = evaluator.outlier_detection()
    evaluator.sentiment_classification_tasks(print_score=True)

