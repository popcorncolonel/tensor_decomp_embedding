import datetime
import tensorflow as tf
import numpy as np
import time
import os

from collections import defaultdict
import embedding_benchmarks


class WordEmbedding(object):
    def __init__(
        self,
        vocab_model,
        embedding_size,
        context_size,  # = 2 * vocab_model.window (= 10)
        method=None,
        gpu=True,
    ):
        self.vocab_model = vocab_model
        self.vocab = vocab_model.vocab
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.method = method
        self.gpu = gpu

        config = tf.ConfigProto(
            allow_soft_placement=True,
        )
        self.sess = tf.Session(config=config)
        with self.sess.as_default():
            with tf.device('/{}'.format('gpu:0' if gpu else 'cpu:0')):
                self.input_x = tf.placeholder(tf.int32, [None, context_size], name='input_x')
                self.input_y = tf.placeholder(tf.int64, [None, 1], name='input_y') # Index of correct word. (list for minibatching)
                self.create_embedding_layer()
                self.create_fully_connected_layer_and_loss_fn(vocab_model)
        self.write_graph()

    def write_graph(self):
        tf.summary.FileWriter('tf/graphs', graph=self.sess.graph)

    def write_embedding_to_file(self, fname='vectors.txt'):
        vectors = {}
        model = self.vocab_model
        embedding = self.get_embedding_matrix2()
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
            f.write('{} {}\n'.format(count, self.embedding_size))  # write the number of vects
            f.write(content)

    def embed_with_sgns(self, embedded_target):
        '''
        Given the embedding of the input, embeds the hidden layer with the formulation in CBOW
        '''
        self.h = embedded_target

    def embed_with_cbow(self, embedded_chars):
        '''
        Given the embedding of the input, embeds the hidden layer with the formulation in CBOW
        '''
        self.embedding_b = tf.Variable(
            tf.constant(0., shape=[self.embedding_size], dtype=tf.float32),
            name='b',
            dtype=tf.float32,
        )
        #self.h = tf.reduce_mean(embedded_chars, 1) + self.embedding_b
        self.h = tf.reduce_mean(embedded_chars, 1)

    def embed_with_cnn(self, embedded_chars):
        with tf.name_scope('embedding'):
            # embedded_chars_expanded is of shape [None (minibatch size), context_size, embedding_size, 1 (#channel maps)]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        filter_sizes = [3,4,5]  # TODO: 1?
        num_filters = 50

        pooled_outputs = []
        num_features_total = 0
        for filter_size in filter_sizes:
            with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                filter_shape = [filter_size, self.embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                    input=self.embedded_chars_expanded,
                    filter=W,
                    strides=[1,1,1,1],
                    padding='VALID',
                    use_cudnn_on_gpu=True,
                    name='conv',
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                stride = int(self.context_size / 3) # int((context_size - filter_size + 1) / 3)
                pooled = tf.nn.max_pool(
                    h,
                    ksize = [1, stride, 1, 1],
                    strides=[1, stride, 1, 1],
                    padding='VALID',
                    name='pooling',
                )
                num_features_total += num_filters*int(np.math.ceil(float(int(h._shape[1]) - stride + 1) / float(stride))) # "VALID" equation for output height. See https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#convolution
                pooled_outputs.append(pooled)
        # concatenate the pooled outputs into a single tensor (by their dimension 3, which is all the different filter outputs. All others dims will be the same.)
        # so h_pool is of shape [batch_size, num_features_total, 1, num_filters]
        h_pool = tf.concat(1, pooled_outputs)
        # -1 flattens into 1D. So h_pool_flat is of shape [batch_size, num_filters_total].
        assert num_features_total == h_pool._shape[1] * h_pool._shape[3] == self.embedding_size
        self.h_pool_flat = tf.reshape(h_pool, [-1, num_features_total], name='h_pool_flat')
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, 0.5)
        self.h = self.h_drop

    def embed_with_subspace_proj(self, embedded_chars):
        '''
        Given the embedding of the input, embeds the context vector as the projection of the correct word
        onto the subspace spanned by the context words.
        '''
        context_T = embedded_chars  # W (Each matrix in the batch is of shape (|C|, k))
        context_matrix = tf.transpose(context_T, perm=[0, 2, 1])  # Since these are in batches, we need to transpose each matrix in the batch. Now of shape (k, |C|). (perm=[0,2,1] because we keep the first index in place, which represents each batch)
        lambda_ = 100
        identity = tf.constant(value=lambda_*np.identity(self.context_size), dtype=tf.float32)

        inv = tf.matrix_inverse(tf.batch_matmul(context_T, context_matrix) + identity)  #  (W^T * W)^-1   
        proj_matrix = tf.batch_matmul(tf.batch_matmul(context_matrix, inv), context_T)  # W * ((W^T * W)^-1) * W^T
        word_T = tf.nn.embedding_lookup(self.word_embedding, self.input_y)
        word = tf.transpose(word_T, perm=[0, 2, 1])  # again, we keep the 0 axis in line because of minibatching. Embedding lookup returns a 1xk-dimensional matrix, and we want a k-dimensional one. 
        self.h = tf.batch_matmul(proj_matrix, word)  # W * ((W^T * W)^-1) * W^T * v_i
        self.h = tf.unstack(self.h, axis=2)[0]  # turn the hidden output from a (?,k,1) tensor into a (?,k) tensor

    def embed_with_tensor_train(self, r):
        '''
        `self.h` := U_1(i_1,:)*U_2(:,:,i_2)*U_3(:,:,i_3)*...*U_|C|(:,:,i_|C|)
        where U_1 of shape [|V|, 1, r], U_i (1 < i < |C|) is of shape [|V|, r, r], U_|C| of shape [|V|, r, d], so the entire product is
        of shape [1, d], as desired. (embedding lookup is done by first dimension)
        '''
        U_1 = tf.Variable(
            tf.random_uniform([len(self.vocab), 1,  r], minval=-1, maxval=1),
            name='tt_layer_1'
        )
        layers = [U_1]
        for i in range(1, self.context_size-1):
            U_i = tf.Variable(
                tf.random_uniform([len(self.vocab), r, r], minval=-1, maxval=1),
                name='tt_layer_{}'.format(i+1)
            )
            layers.append(U_i)
        final_layer = tf.Variable(
            tf.random_uniform([len(self.vocab), r, self.embedding_size], minval=-1, maxval=1),
            name='tt_layer_{}'.format(self.context_size)
        )
        layers.append(final_layer)

        def lookup(layers_index, vocab_index):
            return tf.nn.embedding_lookup(layers[layers_index], vocab_index)

        h = lookup(0, self.input_x[:, 0])
        for i in range(1, self.context_size):
            h = tf.batch_matmul(h, lookup(i, self.input_x[:, i]))
        h = tf.unstack(h, axis=1)[0]  # turn the hidden output from a (?,1,k) tensor into a (?,k) tensor
        self.h = h

    def create_embedding_layer(self):
        with tf.name_scope('embedding'):
            # |V| x d embedding matrix
            C = tf.Variable(
                tf.random_uniform([len(self.vocab), self.embedding_size], minval=-1, maxval=1),
                name='embedding_matrix'
            )
            self.context_embedding = C  # aka the first embedding matrix in the NN. If SGNS, this is not actually the context embedding!
            # Embed the input. The embedding lookup takes in a list of numbers (not one-hot vectors).
            self.embedded_chars = tf.nn.embedding_lookup(C, self.input_x)

            if self.method == 'subspace':  # Subspace
                print("Embedding the context via subspace projection.")
                self.embed_with_subspace_proj(self.embedded_chars)
            elif self.method == 'tensor':  # Tensor Train
                print("Embedding the context via Tensor Train.")
                self.embed_with_tensor_train(r=15)
            elif self.method == 'cnn':  # CNN
                print("Embedding the context with CNN")
                self.embed_with_cnn(self.embedded_chars)
            elif self.method == 'sgns':  # SGNS
                print("Embedding the Target word (not the context) (SGNS).")
                embedded_target = tf.nn.embedding_lookup(C, self.input_y)
                self.embed_with_sgns(embedded_target)
            else:  # CBOW
                print("Embedding the context with simple averaging (CBOW).")
                self.embed_with_cbow(self.embedded_chars)

    def create_fully_connected_layer_and_loss_fn(self, vocab_model):
        with tf.name_scope('fully_connected'):
            '''
            self.fc_W = self.context_embedding
            self.embedding_b = tf.Variable(
                tf.constant(0., shape=[self.embedding_size]),
                name='b'
            )
            self.fc_b = self.embedding_b
            '''
            self.fc_W = tf.Variable(
                tf.truncated_normal(
                    shape=[len(self.vocab), self.embedding_size],
                    stddev=0.01,
                    dtype=tf.float32,
                ),
                name='W',
                dtype=tf.float32,
            )
            self.fc_b = tf.Variable(
                tf.constant(0., shape=[len(self.vocab)], dtype=tf.float32),
                name='b',
                dtype=tf.float32,
            )
        self.create_loss_fn(self.fc_W, self.fc_b, vocab_model)

    def create_loss_fn(self, fc_W, fc_b, vocab_model):
        with tf.name_scope('loss'), tf.device('/{}'.format('gpu:0' if self.gpu else 'cpu:0')):
            sampled_candidates, true_expected_count, sampled_expected_count = tf.nn.learned_unigram_candidate_sampler(
                true_classes=self.input_y,
                num_true=1,
                num_sampled=self.vocab_model.negative,
                unique=True,
                range_max=len(vocab_model.vocab),
                name="unigram_sampler",
            )
            sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)

            if self.method != 'sgns':
                losses = tf.nn.nce_loss(
                #losses=tf.nn.sampled_softmax_loss(
                    weights=fc_W,
                    biases=fc_b,
                    inputs=tf.cast(self.h, tf.float32),
                    labels=self.input_y,
                    num_sampled=self.vocab_model.negative,
                    num_classes=len(vocab_model.vocab),
                    remove_accidental_hits=True,
                    sampled_values=sampled_values,
                )
            else:
                losses = tf.nn.nce_loss(
                    weights=fc_W,
                    biases=fc_b,
                    inputs=tf.squeeze(tf.cast(self.h, tf.float32), axis=1),
                    labels=self.input_x,
                    num_sampled=self.vocab_model.negative,
                    num_classes=len(vocab_model.vocab),
                    num_true=self.context_size,
                    remove_accidental_hits=True,
                    sampled_values=sampled_values,
                )
            self.loss = tf.reduce_mean(losses)
            #self.loss += .01 * (tf.nn.l2_loss(self.context_embedding) + tf.nn.l2_loss(fc_W))  # regularization

    def get_embedding_matrix1(self):
        embedding = self.context_embedding.eval(self.sess)
        return embedding

    def get_embedding_matrix2(self):
        embedding = self.fc_W.eval(self.sess)
        return embedding

    def set_vocab_model_embedding_matrix(self):
        if self.method in ('subspace', 'tensor', 'cbow', 'cnn'):
            embedding = self.get_embedding_matrix2()
        else:
            embedding = self.get_embedding_matrix1()
        self.vocab_model.syn0 = embedding

    def train_step(self, x_batch, y_batch, print_every=50):
        y_batch = np.reshape(y_batch, (len(y_batch), 1))
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
        }
        _, step, summaries, loss = self.sess.run(
            [
                self.train_op,
                self.global_step,
                self.loss_summary,
                self.loss,
            ],
            feed_dict=feed_dict
        )
        if step % print_every == 0:
            print("{}: loss {:g}".format(step, loss))
        self.train_summary_writer.add_summary(summaries, step)

    def dev_step(self, x_batch, y_batch):
        y_batch = np.reshape(y_batch, (len(y_batch), 1))
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
        }
        summaries, loss = self.sess.run(
            [
                self.loss_summary,
                self.loss,
            ],
            feed_dict=feed_dict,
        )
        time_str = datetime.datetime.now().isoformat()
        print("(dev) {}: step {}, loss {:g}".format(time_str, self.step, loss))
        self.dev_summary_writer.add_summary(summaries, self.step)
        #self.evaluate()

    def evaluate(self, rel_path='vectors.txt'):
        self.write_embedding_to_file(fname=rel_path)
        out_fname = 'results_iter{}_{}.txt'.format(self.step, self.method)
        os.system('time python3 embedding_benchmarks/scripts/evaluate_on_all.py -f /home/eric/code/gensim/{} -o /home/eric/code/gensim/results/{}'.format(rel_path, out_fname))
        print('done evaluating.')

    def train(self, batches):
        with self.sess.as_default(), tf.device('/cpu:0'):
            ######## Misc housekeeping: logs, summaries, tensorboard, checkpoints ###########
            timestamp = str(datetime.datetime.now())
            out_dir = os.path.abspath(os.path.join(os.path.curdir, 'tf', timestamp))
            print('Writing summaries to {}.'.format(out_dir))

            self.loss_summary = tf.summary.scalar('loss', self.loss)
            self.train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, 'summaries', 'train'), self.sess.graph)
            self.dev_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, 'summaries', 'dev'), self.sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
            ######## /Misc housekeeping ###########

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.step = 0

        with tf.device('/{}'.format('gpu:0' if self.gpu else 'cpu:0')):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            grads_and_vars = optimizer.compute_gradients(self.loss)

            self.train_op = optimizer.apply_gradients(grads_and_vars, self.global_step)

        self.sess.run(tf.global_variables_initializer())
        # TODO: look into using pre-trained values for our first word embedding.

        self.word_index = 0
        self.sent_index = 0
        self.start_time = None
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            y_batch = np.reshape(y_batch, (len(y_batch), 1))
            if self.step % 5000 == 1:
                self.dev_step(x_batch, y_batch)
            #if self.step > 0 and self.step % 50000 == 0:
            #    print('Saving checkpoint at step {}...'.format(self.step))
            #    path = self.saver.save(self.sess, checkpoint_prefix, global_step=self.step)
            #    print('Saved model checkpoint to {}'.format(path))
            self.train_step(x_batch, y_batch)
            current_step = tf.train.global_step(self.sess, self.global_step)
        path = self.saver.save(self.sess, checkpoint_prefix, global_step=tf.train.global_step(self.sess, self.global_step))
        print('Saved FINAL model checkpoint to {}'.format(path))
        #self.evaluate()

