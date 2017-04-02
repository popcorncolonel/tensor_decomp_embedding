#!/usr/bin/env python
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

from src.tests import test_embedding, test_evaluator, test_test_group
from src.lib.polyglot.mapping.tests import test_expansion, test_embeddings

tests = unittest.TestSuite([test_embedding.suite, test_evaluator.suite, test_test_group.suite, test_embeddings.suite] + test_expansion.suites)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(tests)
