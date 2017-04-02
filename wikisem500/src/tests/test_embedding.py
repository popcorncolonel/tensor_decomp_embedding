#!/usr/bin/env python3
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
# Modelled off of polyglot's embeddings tests
import unittest

from ..embeddings import WrappedEmbedding

import numpy as np
from .utils_tests import EmbeddingTestCase


class EmbeddingTest(EmbeddingTestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_vectors_work(self):
        self.build_test_vectors()
        self.assertArrayEqual(self.vectors['dog'], np.array([1.0, 0.0, 0.6]))
        self.assertArrayEqual(self.vectors['cat'], np.array([1.0, 0.0, 0.7]))

    def test_multiword_lookup(self):
        self.build_test_vectors()
        self.assertArrayNotEqual(self.vectors['cat'], self.vectors['cat_dog'])
        self.assertArrayNotEqual(self.vectors['dog'], self.vectors['cat_dog'])
        averaged = self.vectors['cat_dog']
        self.vectors.supports_phrases = True
        self.assertArrayNotEqual(self.vectors['cat'], self.vectors['cat_dog'])
        self.assertArrayNotEqual(self.vectors['dog'], self.vectors['cat_dog'])
        self.assertArrayNotEqual(self.vectors['cat_dog'], averaged)

    def test_normalization(self):
        self.build_test_vectors(supports_phrases=True)
        self.assertFalse(self.vectors.google_news_normalize)
        self.assertEqual(self.vectors.get_normalized('NIKON_2002'), 'NIKON_2002')
        self.build_test_vectors(google_news_normalize=True, supports_phrases=True)
        self.assertTrue(self.vectors.google_news_normalize)
        self.assertEqual(self.vectors.get_normalized('NIKON_2002'), 'NIKON_####')

    def test_normalized_lookup(self):
        self.build_test_vectors(supports_phrases=True)
        self.assertFalse(self.vectors.google_news_normalize)
        self.assertFalse('NIKON_2002' in self.vectors)
        self.build_test_vectors(google_news_normalize=True, supports_phrases=True)
        self.assertTrue(self.vectors.google_news_normalize)
        self.assertTrue('NIKON_2002' in self.vectors)

suite = unittest.TestLoader().loadTestsFromTestCase(EmbeddingTest)

if __name__ == "__main__":
    unittest.main()
