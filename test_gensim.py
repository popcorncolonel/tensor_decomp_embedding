import os
import gensim
import pickle
import sys
import tensorflow as tf
import numpy as np
import gensim.utils 
from collections import defaultdict
from gensim.models.word2vec import Text8Corpus
from gensim.models.word2vec import BrownCorpus
from gensim.corpora.wikicorpus import WikiCorpus


#brown_loc = '/home/eric/nltk_data/corpora/brown'
#wiki_loc = '/home/eric/Downloads/enwiki-latest-pages-articles.xml.bz2'
text9_loc = '/cluster/home/ebaile01/data/enwik9'
text8_loc = '/cluster/home/ebaile01/data/enwik8'
tokenized_wiki = '/cluster/shared/ebaile01/wiki_complete_dump_2008.txt.tokenized'

corpus = 'wiki'

def sentences_for_tokenized_wiki(max_sents=3.5e6):
    count = 0
    n_tokens = 0
    with gensim.utils.smart_open(tokenized_wiki, 'r') as f:
        for line in f:
            count += 1
            if count < max_sents:
                sent = line.rstrip().split()
                n_tokens += len(sent)
                yield sent
            else:
                print("{} total tokens".format(n_tokens))
                raise StopIteration

g = sentences_for_tokenized_wiki()

corpus_dict = {
    #'brown': (BrownCorpus(brown_loc, False), 10),
    #'wiki': (WikiCorpus(wiki_loc), 5),
    'text9': (Text8Corpus(text9_loc), 1),
    'text8': (Text8Corpus(text8_loc), 1),
    'wiki': (sentences_for_tokenized_wiki(), 1),
}

sentences, iters = corpus_dict[corpus]
max_vocab_size = None
min_count = 100
embedding_dim = 300

tt = 1
subspace = 0
cbow = 0

def get_model_with_vocab(fname=corpus+'model', load=False):
    model = gensim.models.Word2Vec(
        iter=iters,
        max_vocab_size=max_vocab_size,
        negative=128,
        size=embedding_dim,
        min_count=min_count,
        tt=tt,
        subspace=subspace,
        cbow=cbow,
    )
    if load:
        print('depickling model...')
        with open(fname, 'rb') as f:
            model = pickle.load(f)
    else:
        print('building vocab...')
        model.build_vocab(sentences)
        with open(fname, 'wb') as f:
            pickle.dump(model, f)
    print('finished building vocab. length of vocab: {}'.format(len(model.vocab)))
    return model


def print_accuracy(model):
    accuracy = model.accuracy('~/code/w2v_eval/questions-words.txt')
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
    print('sem accuracy: {}/{} = {}'.format(correct['sem'], totals['sem'], correct['sem']/max(1,totals['sem'])))
    print('syn accuracy: {}/{} = {}'.format(correct['syn'], totals['syn'], correct['syn']/max(1,totals['syn'])))
    print('total accuracy: {}/{} = {}'.format(correct['total'], totals['total'], correct['total']/max(1,totals['total'])))


def evaluate_embedding(embedding):
    model = get_model_with_vocab(load=True)
    model.syn0 = embedding
    # We must delete the syn0norm of the vocab in order to compute accuracy.
    # Because if it already has a syn0norm, it will keep using that value and not use the new embedding.
    model.clear_sims()
    print_accuracy(model)


def list_vars_in_checkpoint(dirname):
    from tensorflow.contrib.framework.python.framework.checkpoint_utils import list_variables
    abspath = os.path.abspath(dirname)
    return list_variables(abspath)


def write_embedding_to_file(model, embedding):
    vectors = {}
    for word in model.vocab:
        word_vocab = model.vocab[word]
        word_vect = embedding[word_vocab.index]
        vect_list = ['{:.3f}'.format(x) for x in word_vect]
        vectors[word] = ' '.join(vect_list)
    with open('vectors.txt', 'w') as f:
        for word in vectors:
            if not word:
                continue
            try:
                f.write(word.encode('utf-8') + ' ' + vectors[word] + '\n')
            except TypeError:
                f.write(word + ' ' + vectors[word] + '\n')
            except:
                pass


def main():
    if '--load' in sys.argv:
        model = get_model_with_vocab(corpus+'model', load=True)
        embedding = tf.Variable(tf.random_uniform([len(model.vocab), 100]), name="embedding/embedding_matrix")
        embedding2 = tf.Variable(tf.random_uniform([len(model.vocab), 100]), name="fully_connected/W")
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        with tf.Session() as sess:
            saver.restore(sess, 'tf/text8_subspace100/checkpoints/model-163478')
            print(list_vars_in_checkpoint('tf/text8_subspace100/checkpoints/model-163478'))
            #print(list_vars_in_checkpoint('tf/2017-01-01 12:14:59.907021/checkpoints/model-163478'))
            embedding = embedding.eval(sess)
            embedding2 = embedding2.eval(sess)

            print('evaluating...')
        evaluate_embedding(embedding)
        evaluate_embedding(embedding2)
        def save_sent_jpeg(sentence, fname):
            sentence = sentence.split()
            model.syn0 = embedding
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab]
            word_indices = np.asarray([v.index for v in word_vocabs])
            matrix = np.asarray([embedding[i] for i in word_indices])
            matrix = np.expand_dims(matrix, axis=2)
            min_matrix = -np.min(matrix)
            matrix = np.add(min_matrix, matrix)
            matrix = np.multiply(128. / np.max(matrix), matrix)
            matrix = tf.cast(matrix, tf.uint8)
            image = tf.image.encode_jpeg(matrix, quality=100)
            with tf.Session() as sess:
                with open(fname, 'w') as f:
                    jpeg = sess.run(image)
                    f.write(jpeg)

        write_embedding_to_file(model, embedding)

        sys.exit()

    else:
        if '--buildvocab' in sys.argv:
            model = get_model_with_vocab()
            sys.exit()
        else:
            with open(corpus+'model', 'rb') as f:
                model = pickle.load(f)

        print('training...')
        model.train(sentences)
        print('finished training!')

        # model.save(corpus+'model')

    print("most similar to king - man + woman: {}".format(model.most_similar(
        positive=['king', 'woman'], negative=['man'],
        topn=5,
    )))
    print("most similar to king: {}".format(model.most_similar(
        positive=['king'],
        topn=5,
    )))

    print_accuracy(model)
    print('Corpus: {} ({})'.format(sentences, corpus))
    print('iters: {}'.format(iters))
    print('vocab len: {}'.format(len(model.vocab)))
    embedding = model.syn0
    write_embedding_to_file(model, embedding)
    print("Wrote vectors to vectors.txt")

if __name__ == '__main__':
    main()

