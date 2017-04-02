#
# Copyright 2016 Basis Technology Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import abc
import numpy as np
import operator
from .lib.polyglot import mapping as polyglot_mapping
import re

from six.moves import xrange, reduce

try:
    from functools import lru_cache
except ImportError:
    # lru_cache is a no-op for versions which predate it
    def lru_cache(*args, **kwargs):
        return lambda x: x

NUM_MATCHER = re.compile("[0-9]{2,}")

def phrase_gen(toks):
    """Returns all sequences of phrases in the given list which start at the beginning.
    Example:
      phrase_gen(["foo", "bar", "baz"]) -> "foo", "foo_bar", "foo_bar_baz"
    """
    for i in xrange(1, len(toks) + 1):
        yield (i, '_'.join(toks[:i]))


class Embedding(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self.supports_phrases = kwargs.get("supports_phrases", False)
        self.google_news_normalize = kwargs.get("google_news_normalize", False)
        self.case_sensitive = kwargs.get("case_sensitive", False)
        if "case_sensitive" not in kwargs:
            for word in self.vocabulary:
                if word[0].isupper() and (word[0] != word[0].lower()):
                    self.case_sensitive = True
                    break

    def __contains__(self, w):
        return self[w] is not None

    def __getitem__(self, text):
        # Don't bother trying phrase lookups on single words
        return self.lookup_phrase(text) if '_' in text else self.get_vector(self.get_normalized(text))

    @abc.abstractmethod
    def get_vector(self, word):
        """Looks up the given word in this embedding.
        If the given word is out-of-vocabulary, None is returned."""
        pass

    @abc.abstractmethod
    def in_vocabulary(self, word):
        """Returns true if the given word/phrase is in this embedding's vocabulary"""
        pass

    def get_normalized(self, text):
        if not self.case_sensitive:
            text = text.lower()
        if not self.google_news_normalize:
            return text
        exploded = list(text)
        for m in NUM_MATCHER.finditer(text):
            start, end = m.span()
            exploded[start:end] = '#' * (end - start)
        return ''.join(exploded)

    # Disable normalization by default, since it doesn't affect cosine similarity
    @lru_cache(maxsize=64)
    def lookup_phrase(self, phrase, normalize_vec=False):
        """Looks up the given phrase in this embedding"""
        phrase = self.get_normalized(phrase)
        vectors = []
        split = phrase.split('_')
        if self.supports_phrases:
            i = 0
            while i < len(split):
                # Yes, this is O(n^2). Since phrases are all short in
                # outlier detection, this is still tractable, but it
                # could be better.
                best_match = [(l, x) for (l, x) in phrase_gen(split[i:]) if self.in_vocabulary(x)]
                if best_match:
                    i += best_match[-1][0]
                    vectors.append(np.asarray(self.get_vector(best_match[-1][1])))
                else:
                    i += 1
        else:
            for word in (w for w in split if self.in_vocabulary(w)):
                vectors.append(np.asarray(self.get_vector(word)))
        if len(vectors) == 0:
            #print("OOV: " + phrase + " (split: " + str(split) + ")")
            return None
        else:
            summed = reduce(operator.add, vectors)
            average = summed / len(vectors)
            if normalize_vec:
                averaged = average / np.linalg.norm(average)
            return average

            
class WrappedEmbedding(Embedding, polyglot_mapping.Embedding):
    """Convenience wrapper around polyglot_mapping.Embedding"""
    def __init__(self, vocabulary, vectors, **kwargs):
        polyglot_mapping.Embedding.__init__(self, vocabulary, vectors)
        Embedding.__init__(self, **kwargs)

    def in_vocabulary(self, w):
        return self.get_vector(w) is not None

    def get_vector(self, w):
        vec = self.vocabulary.get(w)
        if vec is None:
            return None
        return self.vectors[vec]

    @staticmethod
    def __wrap(e, **kwargs):
        return WrappedEmbedding(e.vocabulary, e.vectors, **kwargs)

    @staticmethod
    def from_gensim(*args, **kwargs):
        return WrappedEmbedding.__wrap(super(WrappedEmbedding, WrappedEmbedding).from_gensim(*args), **kwargs)

    @staticmethod
    def from_word2vec(*args, **kwargs):
        fvocab = kwargs['fvocab'] if 'fvocab' in kwargs else None
        binary = kwargs['binary'] if 'binary' in kwargs else False
        return WrappedEmbedding.__wrap(super(WrappedEmbedding, WrappedEmbedding).from_word2vec(*args, fvocab=fvocab, binary=binary), **kwargs)

    @staticmethod
    def from_glove(*args, **kwargs):
        return WrappedEmbedding.__wrap(super(WrappedEmbedding, WrappedEmbedding).from_glove(*args), **kwargs)

    @staticmethod
    def load(*args, **kwargs):
        return WrappedEmbedding.__wrap(super(WrappedEmbedding, WrappedEmbedding).load(*args), **kwargs)

