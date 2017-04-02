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

from ..evaluator import Evaluator
from ..outlier_test_group import TestGroup

from .utils_tests import EmbeddingTestCase


class EvaluatorTest(EmbeddingTestCase):
    def setUp(self):
        self.group = TestGroup("tgroup", ["dog", "cat", "hamster", "fish"], ["cat_dog", "obama"])
        self.build_test_vectors(supports_phrases=True, google_news_normalize=True)
        self.resolved = self.group.resolve(self.vectors)
        self.evaluator = Evaluator(self.group)

    def tearDown(self):
        pass

    def test_outlier_detected(self):
        test_case = [x for x in self.resolved][0] # cat_dog is outlier
        op, detected = self.evaluator.score_test_case(test_case)
        self.assertEquals(op, 1.0)
        self.assertTrue(detected)

suite = unittest.TestLoader().loadTestsFromTestCase(EvaluatorTest)

if __name__ == "__main__":
    unittest.main()
