def get_context_matrix(model, word_vocabs, word_index, fixed_size=True, padding_words=False):
    """
    `word_index` is the index in word_vocabs where the target word appears.
    # word_vocabs is a list of vocabs corresponding to sentence indices

    if `fixed_size` is false, it will not put anything in the list for things out of range - it will simply no-op.
    """
    start = word_index - model.window
    window_pos = []
    for i in range(start, word_index + model.window + 1):
        if i == word_index:
            continue
        if 0 <= i < len(word_vocabs):
            window_pos.append(word_vocabs[i].index)
            assert word_vocabs[i] != len(model.vocab)
            assert word_vocabs[i] != len(model.vocab) + 1
        elif i < 0: # before sentence
            if fixed_size and padding_words:
                window_pos.append(len(model.vocab)) # this is a "padding" vector (<S> token)
        else: # after sentence
            if fixed_size and padding_words:
                window_pos.append(len(model.vocab) + 1) # this is a "padding" vector (</S> token)

    #sent = [model.index2word[x] for x in window_pos]
    return window_pos


def get_target_y(word_vocabs, word_index):
    return word_vocabs[word_index].index


def batch_generator(model, sentences, batch_size=512, n_iters=None, fixed_size=True, randomize=False):
    '''
    if `fixed_size` is True, sentences will only include words and contexts in the middle of sentences
        (because the first word in the sentence doesn't have 5 words before it)
    otherwise, it will include all words in the sentence. In that case, the context will
        range from min{len(sentence)-1, 5} (usually 5) to model.window (usually 10)
    '''
    if not n_iters:
        n_iters = model.iter
    batch = []
    for i in range(n_iters):
        print('STARTING NEW TRAINING SET ITER!!!!\nITER {}\n'.format(i))
        for sentence in sentences:
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab]
            for pos, word in enumerate(word_vocabs):
                if fixed_size:
                    if pos < model.window:
                        continue
                    if pos + model.window >= len(word_vocabs):
                        break
                # `word` is the word we're trying to predict
                word_matrix = get_context_matrix(model, word_vocabs, pos, fixed_size=fixed_size)
                target_y = get_target_y(word_vocabs, pos)
                batch.append((word_matrix, target_y))
                if len(batch) == batch_size:
                    if randomize:
                        random.shuffle(batch)
                    #import pdb; pdb.set_trace()
                    yield batch
                    batch = []

