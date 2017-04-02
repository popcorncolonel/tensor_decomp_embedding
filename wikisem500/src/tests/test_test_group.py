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

from ..outlier_test_group import TestGroup, ResolvedTestGroup


from .utils_tests import EmbeddingTestCase


class TestGroupTest(EmbeddingTestCase):
    def setUp(self):
        self.group = TestGroup("tgroup", ["dog", "cat", "hamster", "fish"], ["cat_dog", "obama"])
        self.build_test_vectors(supports_phrases=True, google_news_normalize=True)
        self.resolved = self.group.resolve(self.vectors)

    def tearDown(self):
        pass

    def test_is_valid(self):
        self.assertTrue(self.group.is_valid())
        self.assertTrue(self.resolved.is_valid())

    def test_iter(self):
        self.assertEqual(len([x for x in self.resolved]), 2)
        self.assertEqual([o[0] for c,o in self.resolved], ["cat_dog", "obama"])

suite = unittest.TestLoader().loadTestsFromTestCase(TestGroupTest)

if __name__ == "__main__":
    unittest.main()
