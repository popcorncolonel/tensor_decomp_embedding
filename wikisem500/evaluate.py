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
import sys

from src.evaluator import Evaluator
from src.embeddings import Embedding, WrappedEmbedding
from src.outlier_test_group import TestGroup
from src.utils import scandir


def read_dataset_directory(d):
    for f in scandir(d):
        if f.name.endswith('.txt') and f.is_file():
            yield TestGroup.from_file(f.path)


def score_embedding(embedding, groups):
    evaluator = Evaluator(groups)
    evaluator.evaluate(embedding)
    print("   RESULTS")
    print("==============")
    print("OPP score: %f" % evaluator.opp)
    print("Accuracy: %f" % evaluator.accuracy)
    print("---------------------------------")
    print("Total number of test groups: %d" % evaluator.num_total_groups)
    print("Number of filtered test groups: %d (%f%%)" % (evaluator.num_filtered_groups, evaluator.percent_filtered_groups))
    print("Total number of non-OOV test cases: %d" % evaluator.num_cases)
    print("Number of filtered cluster entities: %d/%d (mean per %% cluster: %f%%)" % (evaluator.num_filtered_cluster_items, evaluator.num_total_cluster_items, evaluator.percent_filtered_cluster_items))
    print("Number of filtered outlier entities: %d/%d (mean per %% cluster: %f%%)" % (evaluator.num_filtered_outliers, evaluator.num_total_outliers, evaluator.percent_filtered_outliers))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scoring script for outlier detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-w2v', '--word2vec', type=str, help="Specify word2vec embedding file")
    group.add_argument('-gv', '--glove', type=str, help="Specify GloVe embedding file")
    group.add_argument('-gs', '--gensim', type=str, help="Specify Gensim embedding file")
    parser.add_argument('-d', '--dataset', type=str, help="Path to outlier dataset", required=True)

    parser.add_argument('-b', '--binary', action="store_true", help="Indicates that the embedding file is binary (ignored for GloVe files)")
    parser.add_argument('-p', '--phrases', action="store_true", help="Indicates that the embedding file supports phrases")
    parser.add_argument('-goog', '--google-news', action="store_true", help="Indicates that the embeddings have been normalized in the same fashion as the Google News word2vec embeddings")
    parser.add_argument('-ci', '--case-insensitive', type=bool, help="Indicates whether the embeddings are all lowercased")

    args = parser.parse_args()

    sys.stdout.write("Loading embeddings...")
    sys.stdout.flush()
    kwargs = {'supports_phrases': args.phrases,
              'google_news_normalize': args.google_news}
    if args.case_insensitive is not None:
        kwargs['case_sensitive'] = not args.case_insensitive
    if args.word2vec:
        embedding = WrappedEmbedding.from_word2vec(args.word2vec, binary=args.binary, **kwargs)
    elif args.glove:
        embedding = WrappedEmbedding.from_glove(args.glove, **kwargs)
    else:
        embedding = WrappedEmbedding.from_gensim(args.gensim, **kwargs)

    bool_text = lambda b: ("Yes" if b else "No")

    print("loaded.")
    print("Vector Metadata:")
    print(" +-- Supports Phrases: " + bool_text(embedding.supports_phrases))
    print(" +-- Google News-normalized: " + bool_text(embedding.google_news_normalize))
    print(" +-- Case Sensitive: " + bool_text(embedding.case_sensitive))
    sys.stdout.write("Loading dataset...")
    sys.stdout.flush()
    dataset = list(read_dataset_directory(args.dataset))
    print("loaded.")
    print("Scoring...")
    score_embedding(embedding, dataset)
