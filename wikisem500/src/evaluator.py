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
from six.moves import map


class Evaluator(object):
    def __init__(self, test_groups):
        self.reset()
        self.test_groups = list(test_groups)

    def reset(self):
        self.num_total_groups = 0
        self.num_filtered_groups = 0
        self.num_filtered_outliers = 0
        self.num_filtered_cluster_items = 0
        self.num_total_cluster_items = 0
        self.num_total_outliers = 0
        self.num_non_oov_outliers = 0
        self.num_cases = 0
        self.sum_opp = 0.0
        self.sum_acc = 0.0
        self.sum_pct_filtered_cluster_items = 0.0
        self.sum_pct_filtered_outliers = 0.0

    def extract_op(self, test_case):
        """Returns the Outlier Position in the given test case and the test case's size"""
        cluster_items, outlier = test_case
        combined = [outlier] + cluster_items
        #combined = sorted(combined, key=lambda x: x[2], reverse=True)
        combined = sorted(combined, key=lambda x: x[2])
        #print([(x[0], x[2]) for x in combined])
        #import pdb; pdb.set_trace()
        # Hack to make numpy happy
        combined = list(map(lambda t: (t[0], t[1]), combined))
        return (combined.index((outlier[0], outlier[1])), len(cluster_items))

    def score_test_case(self, test_case):
        """Returns the percentile of the outlier in the given test case, along with whether the outlier was detected."""
        op, size = self.extract_op(test_case)
        return (float(op) / size, op == size)

    def score_test_cases(self, test_group, embedding, n):
        resolved = test_group.resolve(embedding, n=n)

        self.num_total_cluster_items += len(test_group.cluster)
        self.num_total_outliers += len(test_group.outliers)
        self.num_non_oov_outliers += len(resolved.outliers)
        self.num_filtered_cluster_items += len(test_group.cluster) - len(resolved.cluster)
        self.num_filtered_outliers += len(test_group.outliers) - len(resolved.outliers)

        self.sum_pct_filtered_cluster_items += float(len(test_group.cluster) - len(resolved.cluster)) / len(test_group.cluster)
        self.sum_pct_filtered_outliers += float(len(test_group.outliers) - len(resolved.outliers)) / len(test_group.outliers)

        if not resolved.is_valid():
            self.num_filtered_groups += 1
            return

        for test_case in resolved:
            self.num_cases += 1
            opp, acc = self.score_test_case(test_case)
            self.sum_opp += opp
            self.sum_acc += acc

    def evaluate(self, embedding, n):
        self.reset()
        for test_group in self.test_groups:
            self.num_total_groups += 1
            self.score_test_cases(test_group, embedding, n=n)

    @property
    def opp(self):
        return (self.sum_opp / self.num_cases) * 100

    @property
    def accuracy(self):
        return (self.sum_acc / self.num_cases) * 100

    @property
    def percent_filtered_cluster_items(self):
        return (self.sum_pct_filtered_cluster_items / self.num_total_groups) * 100

    @property
    def percent_filtered_outliers(self):
        return (self.sum_pct_filtered_outliers / self.num_total_groups) * 100

    @property
    def percent_filtered_groups(self):
        return (self.num_filtered_groups / self.num_total_groups) * 100
