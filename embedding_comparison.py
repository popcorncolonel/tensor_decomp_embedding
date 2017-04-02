import dill
from embedding_benchmarks.scripts.web.embeddings import load_embedding
from embedding_benchmarks.scripts.web.evaluate import evaluate_on_all
import heapq
import numpy as np
import pandas as pd
import random
import sys

from embedding_evaluation import EmbeddingTaskEvaluator, evaluate_vectors_from_path

class EmbeddingComparison(object):
    def __init__(self, num_sents, min_count, methods, comparison_name, embedding_dim=None, embedding_dim_list=None, normalize=False):
        '''
        `methods` is a list of strings
        `num_sents`, `embedding_dim`, and `min_count` are passed in uniformly for a more fair comparison (we should be evaluating on the exact same test suite, the
            vocabulary of which is determined by those parameters)
        '''
        self.evaluators = []
        num_sents = int(num_sents)
        min_count = int(min_count)
        if embedding_dim is not None:
            embedding_dim = int(embedding_dim)
            embedding_dim_list = [embedding_dim for _ in methods]
        self.num_sents = num_sents
        self.min_count = min_count
        self.embedding_dim_list = embedding_dim_list
        self.comparison_name = comparison_name

        diff_dims = len(set(embedding_dim_list)) != 1  # if we are comparing embeddings of multiple dimensions, add the dimension to the method (for keeping track)
        for method, dim in zip(methods, embedding_dim_list):
            fname = 'runs/{method}/{num_sents}_{min_count}_{dim}/vectors.txt'.format(**locals())
            if diff_dims:
                method += '_{}'.format(dim)
            self.evaluators.append(EmbeddingTaskEvaluator(method=method, fname=fname, nonneg=True))
        self.vocab_set = set(self.evaluators[0].embedding_dict.keys())

    def print_method(self, method):
        print()
        print("===={}====".format(method))

    def compare_word_dimensions(self, words):
        print("\n==================================")
        '''
        Qualitative evaluation. gets the top 5 dimensions of each word in `words`, then prints the top 3 words in those dimensions.
        `words` is a list of strings
        '''
        for evaluator in self.evaluators:
            self.print_method(evaluator.method)
            for word in words:
                vec = evaluator.embedding_dict[word]
                k = 4
                n = 4
                top_n_dims = vec.argsort()[-n:][::-1]
                print('Word: {}'.format(word))
                for i in range(n):
                    print('top words in dimension {}: {}'.format(top_n_dims[i], ','.join([w for (w,v) in heapq.nlargest(k+1, evaluator.embedding_dict.items(), key=lambda x: x[1][top_n_dims[i]]) if w != word][:k])))

    def compare_nearest_neighbors(self, words):
        print("\n==================================")
        '''
        Qualitative evaluation. Prints the nearest vectors to each word in `words`
        `words` is a list of strings
        '''
        def closest_neighbors(evaluator, word, n=5):
            embedding_dict = evaluator.embedding_dict
            cos_sims = []
            if word not in embedding_dict:
                print('{} not in vocab'.format(word))
                return []
            vec = embedding_dict[word]
            for word2, vec2 in embedding_dict.items():
                if word == word2:
                    continue
                cos_sim = np.dot(vec, vec2) / (np.linalg.norm(vec) * np.linalg.norm(vec2))
                cos_sims.append((cos_sim, word2))
            nlargest = heapq.nlargest(n, cos_sims)
            nlargest = [word for (sim, word) in nlargest]
            return nlargest

        for evaluator in self.evaluators:
            self.print_method(evaluator.method)
            for word in words:
                nlargest = closest_neighbors(evaluator, word)
                if not nlargest:
                    continue
                print('Closest words to {}: {}'.format(word, ', '.join(nlargest)))

    def compare_web(self, normalize=True):
        print("\n==================================")
        frames = []
        for evaluator in self.evaluators:
            self.print_method(evaluator.method)
            vecpath = evaluator.fname
            w = load_embedding(vecpath, format='word2vec', normalize=normalize, lower=True, clean_words=False, load_kwargs={})
            results = evaluate_on_all(w)
            results.index = [evaluator.method]
            frames.append(results)
            print(results)
            #results_dirname = 'results_{}.txt'.format(evaluator.method)
            #evaluate_vectors_from_path(fname, results_dirname)
        all_results = pd.concat(frames)
        return all_results

    def compare_word_classification(self):
        print("\n==================================")
        score_dict = {}
        for evaluator in self.evaluators:
            self.print_method(evaluator.method)
            score = evaluator.word_classification_tasks()
            print("Word classification score: {}".format(score))
            method = evaluator.method
            score_dict[method] = score
        return score_dict

    def compare_sentiment_classification(self):
        print("\n==================================")
        # TODO: multithread this
        score_dict = {}
        for evaluator in self.evaluators:
            self.print_method(evaluator.method)
            score = evaluator.sentiment_classification_tasks()
            print("Sentiment classification score: {}".format(score))
            method = evaluator.method
            score_dict[method] = score
        return score_dict

    def compare_analogy(self):
        print("\n==================================")
        # TODO: multithread this
        sem_dict = {}
        syn_dict = {}
        for evaluator in self.evaluators:
            self.print_method(evaluator.method)
            (sem_score, syn_score) = evaluator.analogy_tasks()
            print("Analogy sem/syn scores: {}".format((sem_score, syn_score)))
            method = evaluator.method
            sem_dict[method] = sem_score
            syn_dict[method] = syn_score
        return sem_dict, syn_dict

    def compare_outlier_detection(self, n=3):
        print("\n==================================")
        opp_dict = {}
        acc_dict = {}
        print("COMPARING OUTLIER DETECTION WITH N={}".format(n))
        for evaluator in self.evaluators:
            method = evaluator.method
            opp, accuracy = evaluator.outlier_detection(verbose=False, n=n)
            self.print_method(evaluator.method)
            print("OD{} OPP: {}".format(n, opp))
            print("OD{} accuracy: {}".format(n, accuracy))
            opp_dict[method] = opp / 100.0
            acc_dict[method] = accuracy / 100.0
        return opp_dict, acc_dict

    def compare_coherency(self, n):
        def get_outlier(evaluator, dim): 
            ''' Returns a word in the bottom half of this dimension that is also in the top 10% of another dimension. '''
            embedding_dict = evaluator.embedding_dict
            num_words = len(embedding_dict)
            dim_values = []
            for word, vec in embedding_dict.items():
                dim_values.append((vec[dim], word))
            bottom_n = heapq.nlargest(num_words // 2, dim_values, key=lambda x: -x[0])  # bottom n words
            while True:
                rand_word = random.choice(bottom_n)[1]
                for dim in range(evaluator.embedding_dim):
                    dim_values = []
                    for word, vec in embedding_dict.items():
                        dim_values.append((vec[dim], word))
                    top_10pct_for_dim = heapq.nlargest(num_words // 10, dim_values)
                    if rand_word in [word for (val, word) in top_10pct_for_dim]:
                        return rand_word
            
        def top_n_words_for_dim(evaluator, dim, n=5):
            embedding_dict = evaluator.embedding_dict
            dim_values = []
            for word, vec in embedding_dict.items():
                dim_values.append((vec[dim], word))
            topn = heapq.nlargest(n, dim_values)
            return [word for (val, word) in topn]

        for evaluator in self.evaluators:
            rand_dim = np.random.randint(0, evaluator.embedding_dim)
            self.print_method(evaluator.method)
            print("Highest words in dim {}: {}, with outlier '{}'".format(
                rand_dim,
                top_n_words_for_dim(evaluator, rand_dim, n=n),
                get_outlier(evaluator, rand_dim),
            ))

    def compare_all(self, num_runs=1):
        all_dfs = []  # allow for random resets
        for run_index in range(num_runs):
            for evaluator in self.evaluators:
                evaluator.seed_bump = run_index

            self.compare_coherency(n=3)
            words = random.sample(list(self.vocab_set), 3)
            self.compare_nearest_neighbors(words)
            self.compare_word_dimensions(words)

            # this should be fastest to slowest
            sentiment_classification_results = self.compare_sentiment_classification()
            word_class_results = self.compare_word_classification()
            outlier_det2_opps, outlier_det2_accs = self.compare_outlier_detection(n=2)
            outlier_det3_opps, outlier_det3_accs = self.compare_outlier_detection(n=3)
            analogy_sem_results, analogy_syn_results = self.compare_analogy()
            result_name_pairs = [
                (analogy_sem_results, 'Analogy (sem)'),
                (analogy_syn_results, 'Analogy (syn)'),
                (sentiment_classification_results, 'Sentiment analysis'), 
                (word_class_results, 'Word classification'), 
                (outlier_det2_opps, 'OD2 OPP'), 
                (outlier_det2_accs, 'OD2 acc'), 
                (outlier_det3_opps, 'OD3 OPP'), 
                (outlier_det3_accs, 'OD3 acc'), 
            ]
            df = pd.DataFrame([ d for (d, name) in result_name_pairs ])
            df.index = [ name for (d, name) in result_name_pairs ]
            df = df.transpose()

            web_results = self.compare_web()
            all_df = web_results.join(df)
            all_df = all_df.transpose()  # excel likes it better this way

            all_dfs.append(all_df)

        df_sum = all_dfs[0]
        for df in all_dfs[1:]:
            df_sum += df
        all_df = (1. / num_runs) * df_sum
        # write to excel file
        excel_fname = 'comparison_{}_{}_{}.xlsx'.format(self.num_sents, self.min_count, self.comparison_name)
        writer = pd.ExcelWriter(excel_fname)
        all_df.to_excel(writer)
        writer.save()
        print('Saved to {}'.format(excel_fname))
        print('Num runs: {}'.format(num_runs))

        return all_df


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('Please specify the name of the comparison')
    comparison_name = sys.argv[1]
    embedding_dim = None
    embedding_dim_list = None
    if comparison_name == '10e6':  # 10E6 SENTS
        num_sents = int(10e6)
        methods = ['random', 'cbow', 'nnse', 'cp', 'cp-s', 'cp-sn', 'jcp-s']
        embedding_dim = 300
    elif comparison_name == 'jcp-s_dims':  # JCP-S EMBEDDING DIM COMPARISON
        num_sents = int(10e6)
        embedding_dim_list = [100, 200, 300, 400, 500, 1000]
        methods = ['jcp-s' for _ in embedding_dim_list]
    elif comparison_name == '30e6':  # 30E6 SENTS
        num_sents = int(30e6)
        methods = ['random', 'cbow', 'nnse', 'cp', 'cp-s', 'cp-sn', 'jcp-s', 'jcp-s_1e-8_reg']
        embedding_dim = 300
    elif comparison_name == 'test':
        num_sents = int(10e6)
        methods = ['cp-s', 'cp-sn', 'jcp-s']
        embedding_dim_list = [300]
    min_count = 2000
    comparator = EmbeddingComparison(
        methods=methods,
        num_sents=num_sents,
        min_count=min_count,
        embedding_dim=embedding_dim,
        embedding_dim_list=embedding_dim_list,
        comparison_name=comparison_name,
    )

    # Don't really need num runs because most of these methods aren't even stochastic (wordsim, etc)
    comparator.compare_all()

