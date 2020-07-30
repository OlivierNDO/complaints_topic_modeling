import datetime
import pandas as pd
import time
from sklearn.decomposition import NMF, LatentDirichletAllocation
import src.configuration as config



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
        str_list.append("Topic %d:" % (i) + top_components)
    return str_list 



class LDATopicFinder:
    def __init__(self, tfid_vector,
                 learning_method = 'online',
                 max_n_topics = 20,
                 max_iter = 5,
                 min_n_topics = 2,
                 random_state = 7302020):
        self.tfid_vector = tfid_vector
        self.learning_method = learning_method
        self.max_n_topics = max_n_topics
        self.max_iter = max_iter
        self.min_n_topics = min_n_topics
        self.random_state = random_state
        
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



