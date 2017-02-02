import itertools
import tensorflow as tf

class TensorEmbedding(object):
    def __init__(self, vocab_model, window=5):
        self.model = vocab_model
        self.window = window  # 5

        vocab_len = len(self.model)
        self.cooccurence = tf.SparseTensor(
            indices=[],
            values=[],
            dense_shape=[vocab_len, vocab_len, vocab_len],
        )

    def create_sent_tensor(self, sent):
        """
        `sent` is a tuple of tensors representing the word and the context.
            For example, ([15234], [74895, 1397, 2385, 23048, 9485, 58934, 2378, 51143, 35829, 34290])
        """
        sent_tensor = tf.SparseTensor(
            indices=[],
            values=[],
            dense_shape=[vocab_len, vocab_len, vocab_len],
        )
        

    def train_on_batch(self, batch):
        """
        `batch` is a list of tuples of tensors representing the word and the context.
            For example, [
                ([15234], [74895, 1397, 2385, 23048, 9485, 58934, 2378, 51143, 35829, 34290]), 
                ...,
            ]
        """
        # update the online version of the CP decomp, given a batch of words and contexts


