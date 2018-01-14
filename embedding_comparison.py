import dill
from embedding_benchmarks.scripts.web.embeddings import load_embedding
from embedding_benchmarks.scripts.web.evaluate import evaluate_on_all
import heapq
import numpy as np
import pandas as pd
import random
import sys
import time

from embedding_evaluation import EmbeddingTaskEvaluator, evaluate_vectors_from_path

class EmbeddingComparison(object):
    def __init__(self, num_sents, min_count, methods, comparison_name, embedding_dim=None, embedding_dim_list=None, normalize=True, fname=None):
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
        self.methods = methods
        self.num_sents = num_sents
        self.min_count = min_count
        self.embedding_dim_list = embedding_dim_list
        self.comparison_name = comparison_name

        diff_dims = len(set(embedding_dim_list)) != 1  # if we are comparing embeddings of multiple dimensions, add the dimension to the method (for keeping track)
        for method, dim in zip(methods, embedding_dim_list):
            if method == 'word2vec':
                fname = '../word2vec.txt'
            else:
                fname = 'runs/{}/{}_{}_{}/vectors.txt'.format(method, num_sents, min_count, dim)
            if diff_dims:
                method += '_{}'.format(dim)
            self.evaluators.append(EmbeddingTaskEvaluator(method=method, fname=fname, normalize_vects=normalize))
        print('intersecting vocabs...')
        vocab_sets = [set(evaluator.embedding_dict.keys()) for evaluator in self.evaluators]
        self.vocab_set = set.intersection(*vocab_sets)
        print('intersected vocab len: {}'.format(len(self.vocab_set)))
        print('intersecting embedding dicts...')
        for evaluator in self.evaluators:
            d = evaluator.embedding_dict
            keys = [x for x in d.keys()]
            for k in keys:
                if k not in self.vocab_set:
                    d.pop(k, None)
        print('done initializing!')

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
                k = 3
                n = 2
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
            results = evaluate_on_all(w, categorization=False)
            results.index = [evaluator.method]
            frames.append(results)
            print(results)
        all_results = pd.concat(frames)
        return all_results

    def compare_word_classification(self, train_pct=1.0):
        print("\n==================================")
        score_dict = {}
        for evaluator in self.evaluators:
            self.print_method(evaluator.method)
            score = evaluator.word_classification_tasks(classification_problem='PoS', train_pct=train_pct)
            print("PoS classification ({}) score: {}".format(int(train_pct*100), score))
            method = evaluator.method
            score_dict[method] = score
        return score_dict

    def compare_sentiment_analysis(self, train_pct=1.0):
        print("\n==================================")
        score_dict = {}
        for evaluator in self.evaluators:
            self.print_method(evaluator.method)
            score = evaluator.sentiment_analysis_tasks(train_pct=train_pct)
            print("Sentiment classification ({}%) score: {}".format(train_pct*100.0, score))
            method = evaluator.method
            score_dict[method] = score
        return score_dict

    def compare_analogy(self, train_pct, iter_pct=1.0, is_sem_only=False, reg_param=0.001, regularize_all=False,
                        multiplicative=False):
        print("\n==================================")
        sem_dict = {}
        syn_dict = {}
        for evaluator in self.evaluators:
            self.print_method(evaluator.method)
            (sem_score, syn_score) = evaluator.analogy_tasks(train_pct=train_pct,
                iter_pct=iter_pct,
                is_sem_only=is_sem_only,
                reg_param=reg_param,
                regularize_all=regularize_all,
                multiplicative=multiplicative,
            )
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

            print('qualitative:')
            self.compare_coherency(n=3)
            words = random.sample(list(self.vocab_set), 5)
            #self.compare_word_dimensions(words)
            self.compare_nearest_neighbors(words)

            print('quantitative:')
            # this should be fastest to slowest
            sentiment_analysis_10_results = self.compare_sentiment_analysis(train_pct=.1)
            sentiment_analysis_30_results = self.compare_sentiment_analysis(train_pct=.3)
            sentiment_analysis_50_results = self.compare_sentiment_analysis(train_pct=.5)
            sentiment_analysis_results = self.compare_sentiment_analysis()

            word_class_10_results = self.compare_word_classification(train_pct=.1)
            word_class_30_results = self.compare_word_classification(train_pct=.3)
            word_class_50_results = self.compare_word_classification(train_pct=.5)
            word_class_results = self.compare_word_classification(train_pct=1.0)

            outlier_det2_opps, outlier_det2_accs = self.compare_outlier_detection(n=2)
            outlier_det3_opps, outlier_det3_accs = self.compare_outlier_detection(n=3)

            #analogy_sem_results_10, analogy_syn_results_10 = self.compare_analogy(.1)
            #analogy_sem_results_30, analogy_syn_results_30 = self.compare_analogy(.3)
            #analogy_sem_results_50, analogy_syn_results_50 = self.compare_analogy(.5)
            #analogy_sem_results, analogy_syn_results = self.compare_analogy(1.0)

            result_name_pairs = [
                #(analogy_sem_results_10, 'Analogy 10% (sem)'),
                #(analogy_sem_results_30, 'Analogy 30% (sem)'),
                #(analogy_sem_results_50, 'Analogy 50% (sem)'),
                #(analogy_sem_results, 'Analogy 100% (sem)'),

                #(analogy_syn_results_10, 'Analogy 10% (syn)'),
                #(analogy_syn_results_30, 'Analogy 30% (syn)'),
                #(analogy_syn_results_50, 'Analogy 50% (syn)'),
                #(analogy_syn_results, 'Analogy 100% (syn)'),

                (sentiment_analysis_10_results, 'Sentiment analysis (10%)'), 
                (sentiment_analysis_30_results, 'Sentiment analysis (30%)'), 
                (sentiment_analysis_50_results, 'Sentiment analysis (50%)'), 
                (sentiment_analysis_results, 'Sentiment analysis (100%)'), 

                (word_class_10_results, 'PoS classification (10%)'), 
                (word_class_30_results, 'PoS classification (30%)'), 
                (word_class_50_results, 'PoS classification (50%)'), 
                (word_class_results, 'PoS classification (100%)'), 

                (outlier_det2_opps, 'OD2 OPP'), 
                (outlier_det2_accs, 'OD2 acc'), 
                (outlier_det3_opps, 'OD3 OPP'), 
                (outlier_det3_accs, 'OD3 acc'), 
            ]
            df = pd.DataFrame([ d for (d, name) in result_name_pairs ])
            df.index = [ name for (d, name) in result_name_pairs ]
            df = df.transpose()

            if True:
                web_results = self.compare_web()
                all_df = web_results.join(df)
                all_df = all_df.transpose()  # excel likes it better this way
            else:
                all_df = df.transpose()

            all_df = all_df[self.methods]  # Reorder columns to be in initial order
            all_dfs.append(all_df)

        df_sum = all_dfs[0]
        for df in all_dfs[1:]:
            df_sum += df
        avg_df = (1. / num_runs) * df_sum
        # write to excel file
        excel_fname = 'comparison_{}_{}_{}.xlsx'.format(self.num_sents, self.min_count, self.comparison_name)
        writer = pd.ExcelWriter(excel_fname)
        avg_df.to_excel(writer)
        writer.save()
        print(avg_df)
        print('Saved to {}'.format(excel_fname))
        print('Num runs: {}'.format(num_runs))
        import pdb; pdb.set_trace()
        print(np.std(all_dfs, axis=2))

        return avg_df


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('Please specify the name of the comparison')
    comparison_name = sys.argv[1]
    embedding_dim = None
    embedding_dim_list = None
    min_count = 1000
    if comparison_name == '1e5':
        num_sents = int(float(comparison_name))
        methods = ['random', 'cbow', 'sgns', 'glove_sym', 'nnse', 'cp-s', 'jcp-s', 'cp-s_best']
        embedding_dim = 300
    elif comparison_name == 'web':
        num_sents = int(1e5)
        methods = ['random', 'cbow', 'sgns', 'glove', 'nnse', 'cp-s', 'jcp-s']
        embedding_dim = 300
    elif comparison_name == 'glove':
        num_sents = int(1e5)
        methods = ['random', 'glove', 'glove_sym']
        embedding_dim = 300
    elif comparison_name == 'hosg':
        num_sents = int(1e5)
        methods = ['random', 'cp-s', 'jcp-s', 'hosg']
        embedding_dim = 300
    elif comparison_name == 'learn_nn_hparams':
        num_sents = int(1e5)
        methods = ['random', 'glove', 'nnse', 'cp-s', 'jcp-s']
        embedding_dim = 300
    comparator = EmbeddingComparison(
        methods=methods,
        num_sents=num_sents,
        min_count=min_count,
        embedding_dim=embedding_dim,
        embedding_dim_list=embedding_dim_list,
        comparison_name=comparison_name,
    )

    if comparison_name == 'web':
        comparator.compare_web(normalize=True)
    elif comparison_name != 'learn_nn_hparams':
        comparator.compare_all(num_runs=10)
    else:
        t = time.time()
        results_dict = dict()
        n_random_trials = 1
        def loguniform(low=1e-6, high=1, size=None):
            return np.exp(np.random.uniform(np.log(low), np.log(high), size))
        for _ in range(10):
            reg_param = loguniform(.001, 0.01)  # regulate W3
            #reg_param = 0.0030505989722323123  # pretty high reg param
            iter_pct = int(np.random.uniform(2, 10))
            for regularize_all in [False]:
                multiplicative = True
                mean_scores = {method: np.array([0., 0., 0., 0.], dtype=np.float64) for method in methods}
                for _ in range(n_random_trials):
                    for evaluator in comparator.evaluators:
                        evaluator.seed_bump += 1
                    kwargs = dict(
                            is_sem_only=False,
                            iter_pct=iter_pct,
                            reg_param=reg_param,
                            regularize_all=regularize_all,
                            multiplicative=multiplicative,
                    )
                    scores1 = comparator.compare_analogy(0.1, **kwargs)[0]
                    scores2 = comparator.compare_analogy(0.3, **kwargs)[0]
                    scores3 = comparator.compare_analogy(0.5, **kwargs)[0]
                    scores5 = comparator.compare_analogy(1.0, **kwargs)[0]
                    for method in methods:
                        mean_scores[method] += np.array([
                            scores1[method], scores2[method], scores3[method], scores5[method]
                        ]) / n_random_trials
                for method in methods:
                    mean_scores[method] = ["{:.4f}".format(x) for x in mean_scores[method]]
                results_dict[(iter_pct, regularize_all, reg_param)] = mean_scores
                with open('hparams/{}_{}_{:.4f}_multiplicative.txt'.format(iter_pct, regularize_all, reg_param), 'a') as f:
                    for k in mean_scores:
                        print("{}: {}\n".format(k, mean_scores[k]), file=f)
            print('time so far: {}'.format(time.time() - t))
        for k in results_dict:
            print("{}: {}".format(k, results_dict[k]))
        print('welp. that took {:.4f} seconds'.format(time.time() - t))
        print("with biases and with no tanh, {} random restarts".format(n_random_trials))
        print("high iter_pct's, fixed biases")
        print("MULTIPLICATIVE")
        import pdb; pdb.set_trace()
        sys.exit()

