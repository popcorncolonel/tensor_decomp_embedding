from nltk import bigrams

def get_context_matrix(model, word_vocabs, word_index, fixed_size=True, padding_words=False):
    """
    `word_index` is the index in word_vocabs where the target word appears.
    # word_vocabs is a list of vocabs corresponding to sentence indices

    if `fixed_size` is false, it will not put anything in the list for things out of range - it will simply no-op.
    """
    start = word_index - model.window
    context_matrix = []
    for i in range(start, word_index + model.window + 1):
        if i == word_index:
            continue
        if 0 <= i < len(word_vocabs):
            context_matrix.append(word_vocabs[i].index)
            #assert word_vocabs[i] != len(model.vocab)
            #assert word_vocabs[i] != len(model.vocab) + 1
        elif i < 0: # before sentence
            if fixed_size and padding_words:
                context_matrix.append(len(model.vocab)) # this is a "padding" vector (<S> token)
        else: # after sentence
            if fixed_size and padding_words:
                context_matrix.append(len(model.vocab) + 1) # this is a "padding" vector (</S> token)

    #sent = [model.index2word[x] for x in context_matrix]
    return context_matrix


def get_target_y(word_vocabs, word_index):
    return word_vocabs[word_index].index


def batch_generator(model, sentences, batch_size=512, n_iters=1, fixed_size=True, randomize=False):
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
        #print('STARTING NEW TRAINING SET ITER!!!!\nITER {}\n'.format(i))
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
            if len(batch) >= batch_size:
                if randomize:
                    random.shuffle(batch)
                yield batch
                batch = []
        if batch:
            yield batch

def batch_generator2(model, sentences, batch_size=512, randomize=False):
    '''
    Outputs sentences in chunks of 11. No word/context pairs or anything. 
    '''
    batch = []
    def append_chunks(l, n):
        for i in range(0, len(l), n):
            batch.append(l[i:i+n])
    for sentence in sentences:
        words = [model.vocab[w].index for w in sentence if w in model.vocab]
        append_chunks(words, 1 + 2*model.window)
        if len(batch) >= batch_size:
            if randomize:
                random.shuffle(batch)
            yield batch
            batch = []
    if batch:
        yield batch


############### BIGRAMS ##############

def get_bigram_context_matrix(model, words, word_index, fixed_size=True, padding_words=False):
    """
    `word_index` is the index in words where the target word appears.

    if `fixed_size` is false, it will not put anything in the list for things out of range - it will simply no-op.
    """
    start = word_index - model.window - 1
    context_matrix = []
    for i in range(start, word_index + model.window + 1):
        ''' `i` is the start word of the bigram '''
        if i == word_index - 1 or i == word_index:
            continue
        if 0 <= i < len(words):
            assert (words[i], words[i+1]) in model.bigram_vocab, "bigram vocab built incorrectly - unseen bigram observed"
            context_matrix.append(model.bigram_vocab[(words[i], words[i+1])])
        elif i < 0: # before sentence
            if fixed_size and padding_words:
                context_matrix.append(len(model.vocab)) # this is a "padding" vector (<S> token)
        else: # after sentence
            if fixed_size and padding_words:
                context_matrix.append(len(model.vocab) + 1) # this is a "padding" vector (</S> token)

    #sent = [model.index2word[x] for x in context_matrix]
    return context_matrix


def bigram_batch_generator(model, sentences, batch_size=512, fixed_size=True, randomize=False):
    '''
    if `fixed_size` is True, sentences will only include words and contexts in the middle of sentences
        (because the first word in the sentence doesn't have 5 words before it)
    otherwise, it will include all words in the sentence. In that case, the context will
        range from min{len(sentence)-1, 5} (usually 5) to model.window (usually 10)
    '''
    batch = []
    for sentence in sentences:
        words = [w for w in sentence if w in model.vocab]
        for pos, word in enumerate(words):
            if fixed_size:
                if pos < model.window + 1:
                    continue
                if pos + model.window + 2 >= len(words):
                    break
            # `word` is the word we're trying to predict
            word_matrix = get_bigram_context_matrix(model, words, pos, fixed_size=fixed_size)
            batch.append((word_matrix, model.vocab[word].index))
        if len(batch) >= batch_size:
            if randomize:
                random.shuffle(batch)
            yield batch
            batch = []
    if batch:
        yield batch

