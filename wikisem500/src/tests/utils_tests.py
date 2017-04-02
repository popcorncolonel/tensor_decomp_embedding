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
import unittest
import numpy as np
from six import PY2
from six.moves import StringIO

from ..embeddings import WrappedEmbedding

__all__ = ['EmbeddingTestCase']

vector_file_contents = u"""
7 3
dog 1.0 0.0 0.6 
cat 1.0 0.0 0.7 
hamster 1.0 0.0 0.5
fish 1.0 0.0 0.2
cat_dog 0.0 1.0 0.0
obama 0.0 0.2 -0.1
NIKON_#### -0.2 0.5 -0.8
""".strip()


# Hack in order to make Python 2 StringIO usable with `with`
# (adapted from: http://stackoverflow.com/a/39337422/1451908)
def nop(self, *args):
    return self

if PY2:
    StringIO.__exit__ = nop
    StringIO.__enter__ = nop


def get_test_vectors():
    """Returns a test vector file."""
    return StringIO(vector_file_contents)


class EmbeddingTestCase(unittest.TestCase):
    def build_test_vectors(self, *args, **kwargs):
        self.vectors = WrappedEmbedding.from_word2vec(get_test_vectors(),
                                                      *args, **kwargs)

    def assertArrayEqual(self, a1, a2):
        self.assertEqual(np.asarray(a1).all(), np.asarray(a2).all())

    def assertArrayNotEqual(self, a1, a2):
        self.assertEqual(np.asarray(a1).all(), np.asarray(a2).all())
