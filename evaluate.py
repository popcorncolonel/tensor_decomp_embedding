# Reads in k word embedding files to dicts of numpy arrays (1 per file)
# Runs various accuracy tasks: 
#   - lcl.uniroma1.it/outlier-detection
#   - Downstream task directly: download yoonkim's implementation of sentence classification
#   - Analogy accuracy?


from collections import defaultdict
import numpy as np


class EmbeddingEvaluation(object):
    def __init__(self, filenames):
        self.embeddings = defaultdict(dict)
        for filename in filenames:
            self.load_filename(filename)

    def load_filename(self, filename):
        with open(filename, 'r') as f:
            # Ex. if filename = '/tmp/embeddings/cnn.txt', embedding_name = 'cnn'.
            embedding_name = filename.split('/')[-1].split('.')[0]
            assert not self.embeddings[embedding_name]
            for line in f:
                line = line.strip().split()
                word = line[0]
                vects = np.asarray([float(x) for x in line[1:]])
                self.embeddings[embedding_name][word] = vects

    def eval_analogy_comparison(self):
        pass

    def eval_sent_classification(self):
        pass

    def eval_outliers(self):
        pass


if __name__ == '__main__':
    eval = EmbeddingEvaluation(['/home/eric/Desktop/cnn.txt', '/home/eric/Desktop/w2v.txt'])
    print('SUP')
