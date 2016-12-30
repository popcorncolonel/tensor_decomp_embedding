import datetime
import tensorflow as tf
import numpy as np
import time
import os

from collections import defaultdict


class SubspaceProjEmbedding(object):
    """
      Methods:
    __init__,
    write_graph,
    create_embedding_layer,
    create_fully_connected_layer_and_loss_fn,
    create_loss_fn,
    get_embedding_matrix, 
    set_vocab_model_embedding_matrix, 
    set_accuracy,
    train_set,
    dev_set,
    train,
    """

    def __init__(
        self,
        vocab_model,
        embedding_size,
        context_size,  # = 2 * vocab_model.window (= 10)
    ):
        self.vocab_model = vocab_model
        self.vocab = vocab_model.vocab
        self.embedding_size = embedding_size
        self.context_size = context_size

        config = tf.ConfigProto(
            allow_soft_placement=True,
        )
        self.sess = tf.Session(config=config)
        with self.sess.as_default():
            with tf.device('/gpu:0'):
                self.input_x = tf.placeholder(tf.int32, [None, context_size], name='input_x')
                self.input_y = tf.placeholder(tf.int64, [None, 1], name='input_y') # Index of correct word. (list for minibatching)
                self.create_embedding_layer(embedding_size)
                self.create_fully_connected_layer_and_loss_fn(vocab_model)
        self.write_graph()

    def write_graph(self):
        tf.train.SummaryWriter('tf/graphs', graph=self.sess.graph)

    def create_embedding_layer(self, embedding_size):
        with tf.name_scope('embedding'):
            W = tf.Variable(
                # |V| x d embedding matrix
                tf.random_uniform([len(self.vocab), embedding_size], minval=-1, maxval=1),
                name='embedding_matrix'
            )
            self.word_embedding = W
            # Embed the input. The embedding lookup takes in a list of numbers (not one-hot vectors).
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # Instead of the mean, it is the projection (of the real word) onto the subspace spanned by the context words. 
            context_T = self.embedded_chars  # W (Each matrix in the batch is of shape (|C|, k))
            context_matrix = tf.transpose(context_T, perm=[0, 2, 1])  # Since these are in batches, we need to transpose each matrix in the batch. Now of shape (k, |C|). (perm=[0,2,1] because we keep the first index in place, which represents each batch)
            lambda_ = .01
            identity = tf.constant(value=lambda_*np.identity(self.context_size), dtype=tf.float32)

            inv = tf.matrix_inverse(tf.batch_matmul(context_T, context_matrix) + identity)  #  (W^T * W)^-1   
            proj_matrix = tf.batch_matmul(tf.batch_matmul(context_matrix, inv), context_T)  # W * ((W^T * W)^-1) * W^T
            word_T = tf.nn.embedding_lookup(W, self.input_y)
            word = tf.transpose(word_T, perm=[0, 2, 1])  # again, we keep the 0 axis in line because of minibatching. Embedding lookup returns a 1x300-dimensional matrix, and we want a 300x1-dimensional one. 
            self.h = tf.batch_matmul(proj_matrix, word)  # W * ((W^T * W)^-1) * W^T * v_i
            self.h = tf.unstack(self.h, axis=2)[0]  # turn the hidden output from a (?,300,1) tensor into a (?,300) tensor
            '''
            b = tf.Variable(
                tf.constant(0., shape=[self.embedding_size]),
                name='b'
            )
            self.h = self.h + b
            '''

    def create_fully_connected_layer_and_loss_fn(self, vocab_model):
        W = tf.Variable(
            tf.truncated_normal(
                shape=[len(self.vocab), 300],
                stddev=0.01
            ),
            name='W',
        )
        b = tf.Variable(
            tf.constant(0., shape=[len(self.vocab)]),
            name='b'
        )
        self.create_loss_fn(W, b, vocab_model)

    def create_loss_fn(self, fc_W, fc_b, vocab_model):
        with tf.name_scope('loss'), tf.device('/cpu:0'): # neg. sampling not implemented on GPU yet
            sampled_candidates, true_expected_count, sampled_expected_count = tf.nn.learned_unigram_candidate_sampler(
                true_classes=self.input_y,
                num_true=1,
                num_sampled=self.vocab_model.negative,
                unique=True,
                range_max=len(vocab_model.vocab),
                name="unigram_sampler",
            )
            sampled_values = (sampled_candidates, true_expected_count, sampled_expected_count)
            losses = tf.nn.nce_loss(
            #losses=tf.nn.sampled_softmax_loss(
                weights=fc_W,
                biases=fc_b,
                inputs=self.h,
                labels=self.input_y,
                num_sampled=self.vocab_model.negative,
                num_classes=len(vocab_model.vocab),
                remove_accidental_hits=True,
                sampled_values=sampled_values,
            )
            self.loss = tf.reduce_mean(losses)
        with tf.name_scope('accuracy'), tf.device('/cpu:0'):
            self.accuracy = tf.Variable(0.0)

    def get_embedding_matrix(self):
        embedding = self.word_embedding.eval(self.sess)
        return embedding

    def set_vocab_model_embedding_matrix(self):
        embedding = self.get_embedding_matrix()
        self.vocab_model.syn0 = embedding
        # We must delete the syn0norm of the vocab in order to compute accuracy.
        # Because if it already has a syn0norm, it will keep using that value and not use the new embedding.
        self.vocab_model.clear_sims()

    def set_accuracy(self, *args, **kwargs):
        self.set_vocab_model_embedding_matrix()
        accuracy = self.vocab_model.accuracy('~/code/w2v_eval/questions-words.txt')
        correct = defaultdict(float)
        totals = defaultdict(float)
        for d in accuracy:
            if d['section'] == 'total':
                correct['total'] += len(d['correct'])
                totals['total'] += len(d['correct'])
                totals['total'] += len(d['incorrect'])
            elif d['section'].startswith('gram'):
                correct['syn'] += len(d['correct'])
                totals['syn'] += len(d['correct'])
                totals['syn'] += len(d['incorrect'])
            else:
                correct['sem'] += len(d['correct'])
                totals['sem'] += len(d['correct'])
                totals['sem'] += len(d['incorrect'])
        print('sem accuracy: {}/{} = {}'.format(correct['sem'], totals['sem'], correct['sem'] / max(1, totals['sem'])))
        print('syn accuracy: {}/{} = {}'.format(correct['syn'], totals['syn'], correct['syn'] / max(1, totals['syn'])))
        return self.accuracy.assign(correct['total'] / float(max(1, totals['total'])))

    def train_step(self, x_batch, y_batch, print_every=100):
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
        time_str = datetime.datetime.now().isoformat()
        if step % print_every == 0:
            print("{}: step {}, loss {:g}".format(time_str, step, loss))
        self.train_summary_writer.add_summary(summaries, step)

    def dev_step(self, x_batch, y_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
        }
        set_acc_op = self.set_accuracy()
        step, summaries, loss, _, _, acc_summ = self.sess.run(
            [
                self.global_step,
                self.loss_summary,
                self.loss,
                set_acc_op,
                self.accuracy,
                self.acc_summary,
            ],
            feed_dict=feed_dict
        )
        time_str = datetime.datetime.now().isoformat()
        print("(dev) {}: step {}, loss {:g}".format(time_str, step, loss))
        self.dev_summary_writer.add_summary(summaries, step)
        self.dev_summary_writer.add_summary(acc_summ, step)

    def train(self, batches):
        with self.sess.as_default(), tf.device('/cpu:0'):
            ######## Misc housekeeping: logs, summaries, tensorboard, checkpoints ###########
            timestamp = str(datetime.datetime.now())
            out_dir = os.path.abspath(os.path.join(os.path.curdir, 'tf', timestamp))
            print('Writing summaries to {}.'.format(out_dir))

            self.loss_summary = tf.scalar_summary('loss', self.loss)
            self.acc_summary = tf.scalar_summary('accuracy', self.accuracy)
            self.train_summary_writer = tf.train.SummaryWriter(os.path.join(out_dir, 'summaries', 'train'), self.sess.graph)
            self.dev_summary_writer = tf.train.SummaryWriter(os.path.join(out_dir, 'summaries', 'dev'), self.sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self.saver = tf.train.Saver(tf.all_variables(), write_version=tf.train.SaverDef.V2)
            ######## /Misc housekeeping ###########

            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            grads_and_vars = optimizer.compute_gradients(self.loss)

        with tf.device('/gpu:0'):
            self.train_op = optimizer.apply_gradients(grads_and_vars, self.global_step)

            self.sess.run(tf.initialize_all_variables())
            # TODO: look into using pre-trained values for our first word embedding.

            self.word_index = 0
            self.sent_index = 0
            self.start_time = None
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                y_batch = np.reshape(y_batch, (len(y_batch), 1))
                self.train_step(x_batch, y_batch)
                current_step = tf.train.global_step(self.sess, self.global_step)
                #if current_step % 500 == 1:
                #    print("\nEvaluation: ")
                #    self.dev_step(x_batch, y_batch)
                if current_step % 10000 == 0:
                    path = self.saver.save(self.sess, checkpoint_prefix, global_step=current_step)
                    print('Saved model checkpoint to {}'.format(path))
                if current_step % 10000 == 0:
                    self.dev_step(x_batch, y_batch)
            path = self.saver.save(self.sess, checkpoint_prefix, global_step=tf.train.global_step(self.sess, self.global_step))
            print('Saved FINAL model checkpoint to {}'.format(path))
            self.set_accuracy()

