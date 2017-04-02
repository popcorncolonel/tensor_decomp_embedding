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
import os.path
from collections import defaultdict
from .utils import decode, similarity, similarity3

class TestGroup(object):
    def __init__(self, name, cluster, outliers):
        self.name = name
        self.cluster = cluster
        self.outliers = outliers

    def __iter__(self):
        for o in self.outliers:
            yield (self.cluster, o)

    def __len__(self):
        return len(self.outliers)

    def __str__(self):
        return self.name + "\n" + "\n".join(self.cluster) + "\n\n" + "\n".join(self.outliers)

    def is_valid(self):
        ''' We need at least n words in the cluster to get the outlier compactness '''
        return len(self.cluster) >= self.n and len(self.outliers) > 0

    def resolve(self, embedding, n=2):
        # Keep original name as well for possible debugging information
        filtered_cluster = [(c, embedding[c]) for c in self.cluster if c in embedding]
        filtered_outliers = [(o, embedding[o]) for o in self.outliers if o in embedding]
        return ResolvedTestGroup(embedding, self.name, filtered_cluster, filtered_outliers, n=n)

    @staticmethod
    def from_file(filename):
        clustername = os.path.splitext(os.path.basename(filename))[0]
        with open(filename, "rb") as f:
            cluster = []
            outliers = []
            active = cluster
            for line in f:
                line = decode(line).strip()
                if line == "":
                    active = outliers
                    continue
                active.append(line)
            return TestGroup(clustername, cluster, outliers)


class ResolvedTestGroup(TestGroup):
    def __init__(self, embedding, *args, n=2, **kwargs):
        TestGroup.__init__(self, *args, **kwargs)
        self.embedding = embedding
        self.compactness = {}
        self.n = n

    def resolve(self):
        raise RuntimeError("Cannot resolve resolved test group")

    def __iter__(self):
        """Yields tuples of the following form:
        ([(cluster-item-name, cluster-item-vec, cluster-item-compactness) ...],
          (outlier-item-name, outlier-item-vec, outlier-item-compactness))"""
        for o in self.outliers:
            if self.n == 2:
                with_similarities = []

                l = len(self.cluster)
                for i in range(l):
                   e1 = self.cluster[i]
                   compactness = 0.0
                   for j in range(l):
                       if i == j: continue
                       e2 = self.cluster[j]
                       #outlier_compactness += similarity(e1[1], e2[1])
                       compactness += similarity(e2[1], o[1])
                       for k in range(j+1, l):
                           if k == i:
                               continue
                           e3 = self.cluster[k]
                           compactness += similarity(e2[1], e3[1])
                   with_similarities.append((e1[0], e1[1], compactness)) # try replaceing one of the cluster words with the outlier, see if the predicted word remains the same

                words_seen = set()
                outlier_compactness = 0.0
                for i, e1 in enumerate(self.cluster):
                   for j in range(i+1, l):
                       e2 = self.cluster[j]
                       outlier_compactness += similarity(e1[1], e2[1])
                yield (with_similarities, (o[0], o[1], outlier_compactness))
            elif self.n == 3:
                with_similarities = []
                l = len(self.cluster)
                for i in range(l):
                   e1 = self.cluster[i]
                   compactness = 0.0
                   for j in range(l):
                       if i == j: continue
                       e2 = self.cluster[j]
                       for k in range(j+1, l):
                           if k == i:
                               continue
                           e3 = self.cluster[k]
                           compactness += similarity3(e2[1], e3[1], o[1])
                           for m in range(k+1, l):
                               if m == i:
                                   continue
                               e4 = self.cluster[m]
                               compactness += similarity3(e2[1], e3[1], e4[1])
                   with_similarities.append((e1[0], e1[1], compactness)) # try replaceing one of the cluster words with the outlier, see if the predicted word remains the same

                outlier_compactness = 0.0
                for i, e1 in enumerate(self.cluster):
                   for j in range(i+1, l):
                       e2 = self.cluster[j]
                       for k in range(j+1, l):
                           e3 = self.cluster[k]
                           outlier_compactness += similarity3(e1[1], e2[1], e3[1])
                yield (with_similarities, (o[0], o[1], outlier_compactness))

