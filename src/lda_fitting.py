import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import time
import scipy
from sklearn.decomposition import NMF, LatentDirichletAllocation
import src.configuration as config



def index_slice_list(lst, indices):
    """
    Slice list by list of indices
    Args:
        lst (list): list to be split by indices
        indicies (list): list of positions with which to filter lst
    Returns:
        list
    """
    list_slice = operator.itemgetter(*indices)(lst)
    if len(indices) == 1:
        return [list_slice]
    else:
        return list(list_slice)



def print_timestamp_message(message, timestamp_format = '%Y-%m-%d %H:%M:%S'):
    """
    Print formatted timestamp followed by custom message
    Args:
        message (str): string to concatenate with timestamp
        timestamp_format (str): format for datetime string. defaults to '%Y-%m-%d %H:%M:%S'
    """
    ts_string = datetime.datetime.fromtimestamp(time.time()).strftime(timestamp_format)
    print(f'{ts_string}: {message}')



def show_lda_topics(lda_model, feature_names, n_top_words):
    """
    Return top n words for each topic from a fit LatentDirichletAllocation object
    Args:
        lda_model (LatentDirichletAllocation): fit LatentDirichletAllocation object
        feature_names (list): list of feature names generated from tfid sklearn.feature_extraction.text.TfidfVectorizer().get_feature_names() call
        n_top_words (int): number of top words per topic to display
    Returns:
        list
    """
    str_list = []
    for i, topic in enumerate(lda_model.components_):
        sorted_components = topic.argsort()[:-n_top_words - 1:-1]
        top_components = " ".join(["'" + feature_names[i] + "'" for i in sorted_components])
        str_list.append("Topic %d:" % (i + 1) + top_components)
    return str_list 



def tfid_kfold_split(tfid_vector, k = 10):
    """
    Split sparse tfid vector into k-chunks (not randomly shuffled)
    Args:
        tfid_vector (sparse matrix): object created with sklearn.feature_extraction.text.TfidfVectorizer call
        k (int): number of splits to apply to tfid_vector
    Returns:
        list
    """
    indices = range(tfid_vector.shape[0])
    len_to_split = [len(indices) // k] * (len(indices) // k)
    if len(indices) > sum(len_to_split):
        len_to_split[-1] += len(indices) - sum(len_to_split)
    return [tfid_vector[x - y: x] for x, y in zip(itertools.accumulate(len_to_split), len_to_split)]



class LDATopicFinder:
    def __init__(self, tfid_vector,
                 tfid_vector_test = None,
                 kfolds = 5,
                 learning_method = 'online',
                 max_n_topics = 20,
                 max_iter = 5,
                 min_n_topics = 2,
                 print_kfold_plot = True,
                 random_state = 7302020,
                 use_n_topics = 6):
        self.tfid_vector = tfid_vector
        self.tfid_vector_test = tfid_vector_test
        self.kfolds = kfolds
        self.learning_method = learning_method
        self.max_n_topics = max_n_topics
        self.max_iter = max_iter
        self.min_n_topics = min_n_topics
        self.print_kfold_plot = print_kfold_plot
        self.random_state = random_state
        self.use_n_topics = use_n_topics
    
    def run_perplexity_grid_search(self):
        i_counter = 1
        n_topic_range = range(self.min_n_topics, (self.max_n_topics + 1))
        n_iterations = len(n_topic_range)
        perplexity_list = []
        for i in n_topic_range:
            print_timestamp_message(f'Starting lda fit iteration {i_counter} of {n_iterations}')
            fit_lda = LatentDirichletAllocation(n_components = i,
                                                max_iter = self.max_iter,
                                                learning_method = self.learning_method,
                                                random_state = self.random_state).fit(self.tfid_vector)
            perplexity_list.append(fit_lda.perplexity(self.tfid_vector))
            i_counter += 1
        output_df = pd.DataFrame({'n_topics' : list(n_topic_range),
                                  'perplexity' : perplexity_list})
        return output_df
    
    
    def run_kfold_perplexity_grid(self):
        # Define No. Topics & Iterations, Split Data into Folds
        n_topic_range = range(self.min_n_topics, (self.max_n_topics + 1))
        n_iterations = len(n_topic_range)
        fold_tfid_vector = tfid_kfold_split(tfid_vector = self.tfid_vector, k = self.kfolds)
        n_topic_list = []
        perplexity_list = []
        uncertainty_list = []
        kfold_list = []
        
        # Loop Over K-Folds & No. Topics
        for k in range(self.kfolds):
            train_k = [x for x in range(self.kfolds) if x != k]
            train_tfid_vector = scipy.sparse.vstack(index_slice_list(fold_tfid_vector, train_k))
            i_counter = 1
            fold_counter = k + 1
            for i in n_topic_range:
                print_timestamp_message(f'Fold {fold_counter} of {self.kfolds}: iteration {i_counter} of {n_iterations}')
                fit_lda = LatentDirichletAllocation(n_components = i,
                                                    max_iter = self.max_iter,
                                                    learning_method = self.learning_method,
                                                    random_state = self.random_state).fit(train_tfid_vector)
                # Score for Uncertainty Calculation
                scores = fit_lda.transform(fold_tfid_vector[k])
                topic_probs = scores.max(axis = 1)
                
                # Score and Append Results
                n_topic_list.append(i)
                perplexity_list.append(fit_lda.perplexity(fold_tfid_vector[k]))
                uncertainty_list.append(np.mean(1 - topic_probs))
                kfold_list.append(k)
                i_counter += 1
        output_df = pd.DataFrame({'n_topics' : n_topic_list,
                                  'k_fold' : kfold_list,
                                  'uncertainty' : uncertainty_list,
                                  'perplexity' : perplexity_list})
        # Average of Fold Results
        mean_perplexity_grid_results = output_df[['n_topics', 'perplexity', 'uncertainty']].\
        groupby(['n_topics'], as_index = False).\
        agg({'perplexity' : 'mean', 'uncertainty' : 'mean'})
        
        # Print Plot
        if self.print_kfold_plot:
            fig, axs = plt.subplots(2, 1, constrained_layout=True)
            fig.suptitle(f'Mean Out of Sample {self.kfolds}-Fold Results', fontsize=16)
            axs[0].plot(mean_perplexity_grid_results['n_topics'], mean_perplexity_grid_results['perplexity'], '--',
                        mean_perplexity_grid_results['n_topics'], mean_perplexity_grid_results['perplexity'], 'o')
            axs[0].set_title(f'Perplexity')
            axs[0].set_xlabel('No. Topics')
            axs[0].set_ylabel('Mean Perplexity')
            
            axs[1].plot(mean_perplexity_grid_results['n_topics'], mean_perplexity_grid_results['uncertainty'], '--',
                        mean_perplexity_grid_results['n_topics'], mean_perplexity_grid_results['uncertainty'], 'o')
            axs[1].set_title(f'Uncertainty (1 - highest topic probability)')
            axs[1].set_xlabel('No. Topics')
            axs[1].set_ylabel('Mean Uncertainty')
            
        return mean_perplexity_grid_results
    
    
    def fit_lda(self):
        lda = LatentDirichletAllocation(n_components = self.use_n_topics,
                                        max_iter = self.max_iter,
                                        learning_method = self.learning_method,
                                        random_state = self.random_state).fit(self.tfid_vector)
        return lda
    
    
    def fit_and_score_train(self):
        # Fit Model & Transform Test Set
        fitted_model = self.fit_lda()
        train_set_pred = fitted_model.transform(self.tfid_vector)
        
        # Assign Labels
        max_prob_values = np.amax(train_set_pred, axis = 1)
        max_prob_index = list(np.argmax(train_set_pred, axis = 1))
            
        # Return Data.Frame() Object
        score_df = pd.DataFrame({'predicted_topic' : [i + 1 for i in list(max_prob_index)],
                                 'predicted_probability' : list(max_prob_values)})
        
        return fitted_model, score_df
    
    
    def fit_and_score_test(self):
        # Fit Model & Transform Test Set
        fitted_model = self.fit_lda()
        test_set_pred = fitted_model.transform(self.tfid_vector_test)
        
        # Assign Labels
        max_prob_values = np.amax(test_set_pred, axis = 1)
        max_prob_index = list(np.argmax(test_set_pred, axis = 1))
            
        # Return Data.Frame() Object
        score_df = pd.DataFrame({'predicted_topic' : [i + 1 for i in list(max_prob_index)],
                                 'predicted_probability' : list(max_prob_values)})
        
        return fitted_model, score_df

        









