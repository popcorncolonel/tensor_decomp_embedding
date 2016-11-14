import tensorflow as tf
import numpy as np


class EmbeddingCNN(object):
    '''
    Input is a C x |V| bag-of-words embedding of the context.
    Starts with a |V| x d fully connected embedding layer (or multiple: TODO).
    The resulting C x d embedded context matrix is passed into a CNN with one convolutional and one max pooling layer.
    Then, we use a softmax layer to get a probability distribution over the output.

    The cost function is calculated via negative sampling.
    '''
    def __init__(
        self,
        vocab_model,
        embedding_size,
        filter_sizes,
        num_filters,
        context_size,  # 2 * vocab_model.window
    ):
        with tf.Graph().as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                self.input_x = tf.placeholder(tf.int32, [None, context_size], name='input_x')
                self.input_y = tf.placeholder(tf.float32, [None, len(vocab_model.vocab)], name='input_y')
                self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
                self.create_embedding_layer(vocab_model, embedding_size)
                self.create_conv_pooling_layer(embedding_size, filter_sizes, num_filters, context_size)
                self.create_fully_connected_layer(num_filters_total=num_filters * len(filter_sizes), vocab_model=vocab_model)
        writer = tf.train.SummaryWriter('/tmp/tf_logs', graph=self.sess.graph)

    def create_embedding_layer(self, vocab_model, embedding_size):
        with tf.name_scope('embedding'):
            W = tf.Variable(
                # |V| x d embedding matrix
                tf.random_uniform([len(vocab_model.vocab), embedding_size])
            )
            # Embed the input. TODO: the embedding lookup might just take in a list of numbers rather than one-hot vectors.
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # embedded_chars_expanded is of shape [None (minibatch size), context_size, embedding_size, 1 (#channel maps)]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

    def create_conv_pooling_layer(self, embedding_size, filter_sizes, num_filters, context_size):
        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                filter_shape = [filter_size, embedding_size, 1, num_filters]  # TODO: understand the dimensionality. Why is the third size 1?
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
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, context_size - filter_size + 1, 1, 1],
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='pooling',
                )
                pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        # concatenate the pooled outputs into a single tensor (by their dimension 3, which is all the different filter outputs. All others dims will be the same.)
        # so h_pool is of shape [batch_size, 1, 1, num_filters_total]
        h_pool = tf.concat(3, pooled_outputs)
        # -1 flattens into 1D. So h_pool_flat is of shape [batch_size, num_filters_total].
        self.h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total], name='h_pool_flat')

    def create_fully_connected_layer(self, num_filters_total, vocab_model):
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope('output'):
            W = tf.Variable(
                tf.truncated_normal(
                    shape=[num_filters_total, len(vocab_model.vocab)],
                    stddev=0.1
                ),
                name='W',
            )
            b = tf.Variable(
                tf.constant(0.1, shape=[len(vocab_model.vocab)]),
                name='b'
            )
            self.scores = tf.matmul(self.h_drop, W) + b
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

    def loss(self):
        pass

    def accuracy(self):
        pass

