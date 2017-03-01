import os


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

