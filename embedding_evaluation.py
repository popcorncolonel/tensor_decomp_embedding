import numpy as np
import random
import os
import sklearn

from sklearn.linear_model import LogisticRegression


def write_embedding_to_file(embedding, model, fname='vectors.txt'):
    vectors = {}
    for word in model.vocab:
        word_vocab = model.vocab[word]
        word_vect = embedding[word_vocab.index]
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
        f.write('{} {}\n'.format(count, embedding.shape[1]))  # write the number of vects
        f.write(content)


def evaluate(embedding, method, model):
    rel_path = 'vectors_{}.txt'.format(method)
    write_embedding_to_file(embedding, model, rel_path)
    out_fname = 'results_{}.txt'.format(method)
    # WORD EMBEDDING BENCHMARKS - word similarity, categorization
    os.system('time python3 embedding_benchmarks/scripts/evaluate_on_all.py -f /home/eric/code/gensim/{} -o /home/eric/code/gensim/results/{}'.format(rel_path, out_fname))
    model.syn0 = embedding
    print("most similar to king - man + woman: {}".format(model.most_similar(
        positive=['king', 'woman'], negative=['man'],
        topn=5,
    )))
    print("most similar to king: {}".format(model.most_similar(
        positive=['king'],
        topn=5,
    )))
    print('done evaluating.')


class EmbeddingTaskEvaluator(object):
    def __init__(self, method: str, normalize_vects: bool = True):
        '''
        `self.embedding_dict` is a dict mapping words (strings) to their embedding
        '''
        self.embedding_dict = {}
        fname = 'vectors_{}.txt'.format(method)
        #fname = 'vectors_random.txt'
        #fname = 'runs/loadmatlab/30000000_5000_100/vectors.txt'
        with open(fname, 'r') as f:
            for i, line in enumerate(list(f)):
                line = line.strip()
                if i != 0:  # (skip header)
                    [word, vectstring] = line.split(maxsplit=1)
                    self.embedding_dict[word] = np.fromstring(vectstring, dtype=np.float32, sep=' ')
                    self.embedding_dim = len(self.embedding_dict[word])
        self.normalize_vects = normalize_vects
        self.method = method
        random.seed(42)

    def get_word_classification_data(self, split_type='train'):
        words_and_POSs = []
        with open('evaluation_data/pos.txt') as f:
            for line in list(f):
                line = line.strip()
                [word, pos] = line.split(maxsplit=1)
                if word in self.embedding_dict:
                    words_and_POSs.append((word, pos))
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

    def word_classification_tasks(self):
        X, y = self.get_word_classification_data('train')
        print("{} training words".format(len(X)))
        
        X_test, y_test = self.get_word_classification_data('test')
        print("{} testing words".format(len(X_test)))

        LR = LogisticRegression()
        LR.fit(X, y)
        score = LR.score(X_test, y_test)
        print('Score: {}'.format(score))
        import pdb; pdb.set_trace()
        return score

    def get_sent_class_data(self, split_type='train'):
        from sklearn.datasets import fetch_20newsgroups
        if split_type == 'train':
            data = fetch_20newsgroups(subset='train')
        elif split_type == 'test':
            data = fetch_20newsgroups(subset='test')
        else:
            raise ValueError('Unrecognized split type {}'.format(split_type))
        import pdb; pdb.set_trace()
        return data

    def sent_classification_tasks(self):
        data = self.get_sent_class_data()

    def get_analogy_data(self, split_type='train'):
        from embedding_benchmarks.scripts.web.datasets.analogy import fetch_google_analogy
        analogy = fetch_google_analogy()
        X = analogy['X']
        y = analogy['y']
        parallel_lists = list(zip(X,y))
        random.shuffle(parallel_lists)
        X = [x[0] for x in parallel_lists]
        y = [x[1] for x in parallel_lists]
        word_pairs = []
        valid_pairs = []
        for triple, answer in zip(X, y):
            if all([x in self.embedding_dict for x in triple]) and answer in self.embedding_dict:
                triple_embedded = np.array([self.embedding_dict[x] for x in triple])
                if self.normalize_vects:
                    triple_embedded = sklearn.preprocessing.normalize(triple_embedded)
                answer_embedded = np.array(self.embedding_dict[answer])
                valid_pairs.append((triple_embedded, answer_embedded))
                word_pairs.append((triple, answer))

        if split_type == 'train':
            print('{} valid analogy questions out of {} total ({}%)'.format(len(valid_pairs), len(X), 100.0*len(valid_pairs) / len(X)))
        num_words = len(valid_pairs)
        split_point = int(.85 * num_words)
        if split_type == 'train':
            data = valid_pairs[:split_point]
            word_data = word_pairs[:split_point]
        elif split_type == 'test':
            data = valid_pairs[split_point:]
            word_data = word_pairs[split_point:]
        else:
            raise ValueError('Unrecognized split type {}'.format(split_type))
        X = np.array([x[0] for x in data])
        y = np.array([x[1] for x in data])
        word_X = np.array([x[0] for x in word_data])
        word_y = np.array([x[1] for x in word_data])
        if self.normalize_vects:
            y = sklearn.preprocessing.normalize(y)
        return X, y, word_X, word_y

    def _analogy_train_NN(self, X, y):
        import tensorflow as tf
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
                n_iters = 1
                for _ in range(n_iters):
                    print('running batches...')
                    for X_batch, y_batch in zip(chunker(X, 25), chunker(y, 25)):
                        _, loss_val, step = sess.run([train_op, loss, global_step], feed_dict={
                            v1: X_batch[:, 0],
                            v2: X_batch[:, 1],
                            v3: X_batch[:, 2],
                            v4: y_batch,
                        })
                        if step % 20 == 0:
                            print('loss at step {}: {}'.format(step, loss_val))
        return sess, v1, v2, v3, v4_hat

    def analogy_tasks(self):
        '''
        Currently not working for any embedding. 
        '''
        X, y, _, _ = self.get_analogy_data('train')
        X_test, y_test, word_X_test, word_y_test = self.get_analogy_data('test')
        print("{} training words".format(len(X)))
        print("{} testing words".format(len(X_test)))
        print('fitting NN...')
        sess, v1, v2, v3, v4_hat = self._analogy_train_NN(X, y)

        correct = 0
        for p, answer, query, correct_word in zip(X_test, y_test, word_X_test, word_y_test):
            [predicted] = sess.run([v4_hat], feed_dict={
                v1: np.expand_dims(p[0], 0),
                v2: np.expand_dims(p[1], 0),
                v3: np.expand_dims(p[2], 0),
            })
            predicted = np.squeeze(predicted)

            def get_closest_vocab_word(predicted):
                best_word = None
                best_vect = None
                best_dist = float('-inf')
                for word, vect in self.embedding_dict.items():
                    if word not in query:
                        dist = np.dot(vect, predicted)
                        if dist > best_dist:
                            best_word = word
                            best_vect = vect
                            best_dist = dist
                return best_word, best_vect

            best_word, best_vect = get_closest_vocab_word(predicted)
            if np.allclose(best_vect, answer):
                correct += 1

            ##########################
            v1_ = self.embedding_dict[query[0]]
            v2_ = self.embedding_dict[query[1]]
            v3_ = self.embedding_dict[query[2]]
            arithmetic_soln = -v1_ + v2_ + v3_
            best_word2, best_vect2 = get_closest_vocab_word(predicted)
            ##########################

            print('query: {}'.format(query))
            print('predicted word: {}'.format(best_word))
            print('predicted word (by non-NN): {}'.format(best_word2))
            print('correct word: {}'.format(correct_word))
            pass
        [predicted] = sess.run([v4_hat], feed_dict={
            v1: X_test[:, 0],
            v2: X_test[:, 1],
            v3: X_test[:, 2],
        })
        predicted = np.squeeze(predicted)
        print('getting accuracy...')
        from embedding_benchmarks.scripts.web.datasets.analogy import fetch_google_analogy
        for predicted_vect, answer in zip(predicted[:100], y_test[:100]):
            best_word = None
            best_vect = None
            best_dist = float('inf')
            for word, vect in self.embedding_dict.items():
                dot_prod = np.dot(vect, predicted_vect)
                if dot_prod < best_dist:
                    best_word = word
                    best_vect = vect
            if np.allclose(best_vect, answer):
                correct += 1
            print('predicted word: {}'.format(best_word))
        print('accuracy: {}/{}={}%'.format(correct, len(X_test), 100.0*correct / float(len(X_test))))
        import pdb; pdb.set_trace()
        pass

