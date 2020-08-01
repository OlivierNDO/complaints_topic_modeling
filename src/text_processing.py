import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.configuration as config
import re
import sklearn



def count_each_unique(lst):
    """
    Count occurences of each unique element of a list
    Args:
        lst (list): list or other iterable object
    Returns:
        pandas.DataFrame() object with fields 'element', 'count', and 'percent'
    """
    unique_values = list(set(lst))
    uv_counts = [len([l for l in lst if l == uv]) for uv in unique_values]
    uv_percent_counts = [uvc / sum(uv_counts) for uvc in uv_counts]
    output_df = pd.DataFrame({'element' : unique_values,
                              'count' : uv_counts,
                              'percent' : uv_percent_counts})
    return output_df



def remove_stopwords(input_string, word_delimiter = ' ',
                     use_lowercase = True, stopword_list = config.all_stopwords):
   """
   Remove stopwords from a string
   Args:
       input_string (str): string
       word_delimiter (str): delimiter used in .split() to separate words in a string. defaults to ' '.
       use_lowercase (bool): boolean value indicating whether or not to convert characters to lowercase. defaults to True.
       stopword_list (list): list of stopwords to omit from string
   Returns:
       string
   """
   return word_delimiter.join([w for w in input_string.lower().split(word_delimiter) if w not in stopword_list])


def remove_punctuation(input_string, replace_punct_with = ' ', punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
   """
   Remove punctuation from a string
   Args:
       input_string (str): string
       replace_punct_with (str): string to replace punctuation with. defaults to ' '.
       punctuation (str): string of concatenated punctuation marks to remove
   Returns:
       str
   """
   punct_regex = re.compile('[%s]' % re.escape(punctuation))
   return punct_regex.sub(replace_punct_with, input_string)


def remove_numerics(input_string):
   """
   Remove numeric characters  from a string
   Args:
       input_string (str): string
   Returns:
       str
   """
   return ''.join([i for i in input_string if not i.isdigit()])


def remove_multiple_substrings(input_string, substring_removals, use_lowercase = True):
    """
    Remove list of substrings from a string
    Args:
        input_string (string): string
        substring_removals (list): list of substrings to remove
    Returns:
        string
    """
    your_new_string = input_string
    replace_dict = dict(zip(substring_removals, ['' for ssr in substring_removals]))
    for removal, blank in replace_dict.items():
        your_new_string = your_new_string.replace(removal, blank)
    return your_new_string


def stem_string_porter(input_string, word_delimiter = ' '):
   """
   Use NLTKs PorterStemmer() class to stem every word in a string
   Args:
       input_string (str): string
       word_delimiter (str): delimiter used in .split() to separate words in a string. defaults to ' '.
   Returns:
       string
   """
   stemmer_object = nltk.stem.PorterStemmer()
   return word_delimiter.join([stemmer_object.stem(w) for w in input_string.split(word_delimiter)])


def wordnet_lemmatize_string(input_string, word_delimiter = ' ', pos = 'n'):
   """
   Use NLTKs PorterStemmer() class to stem every word in a string
   Args:
       input_string (str): string
       word_delimiter (str): delimiter used in .split() to separate words in a string. defaults to ' '.
       pos (str): part of speech fed into wordnet lemmatization class. defaults to 'n' (noun)
   Returns:
       string
   """
   lemmatizer_object = nltk.stem.WordNetLemmatizer()
   return word_delimiter.join([lemmatizer_object.lemmatize(w, pos = pos) for w in input_string.split(word_delimiter)])


class TextProcessingPipeline:
    def __init__(self, string_list,
                 test_string_list = None,
                 max_df = 0.7,
                 min_df = 10,
                 max_features = 1000,
                 ngram_range = (1,3),
                 punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
                 replace_punct_with = ' ',
                 stopword_list = config.all_stopwords,
                 substring_removals = config.substrings_to_remove,
                 use_lowercase = True,
                 word_delimiter = ' '):
        self.string_list = string_list
        self.test_string_list = test_string_list
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.punctuation = punctuation
        self.replace_punct_with = replace_punct_with
        self.stopword_list = stopword_list
        self.substring_removals = substring_removals
        self.use_lowercase = use_lowercase
        self.word_delimiter = word_delimiter
        
    def get_cleaned_train_text(self):
        sl_rm_punct = [remove_punctuation(s, self.replace_punct_with, self.punctuation) for s in self.string_list]
        sl_rm_nums = [remove_numerics(s) for s in sl_rm_punct]
        sl_rm_stopwords = [remove_stopwords(s, self.word_delimiter, self.use_lowercase, self.stopword_list) for s in sl_rm_nums]
        sl_rm_substrings = [remove_multiple_substrings(s, self.substring_removals, self.use_lowercase) for s in sl_rm_stopwords]
        return sl_rm_substrings
    
    def get_cleaned_test_text(self):
        sl_rm_punct = [remove_punctuation(s, self.replace_punct_with, self.punctuation) for s in self.test_string_list]
        sl_rm_nums = [remove_numerics(s) for s in sl_rm_punct]
        sl_rm_stopwords = [remove_stopwords(s, self.word_delimiter, self.use_lowercase, self.stopword_list) for s in sl_rm_nums]
        sl_rm_substrings = [remove_multiple_substrings(s, self.substring_removals, self.use_lowercase) for s in sl_rm_stopwords]
        return sl_rm_substrings
        
    def get_vectorized_text_and_feature_names(self):
        clean_text = self.get_cleaned_text()
        vectorizer_object = sklearn.feature_extraction.text.TfidfVectorizer(max_df = self.max_df,
                                                                            min_df = self.min_df,
                                                                            max_features = self.max_features,
                                                                            ngram_range = self.ngram_range)
        return vectorizer_object.fit_transform(clean_text), vectorizer_object.get_feature_names()
        
    def get_vectorized_text_and_feature_names_train_test(self):
        clean_text = self.get_cleaned_train_text()
        clean_text_test = self.get_cleaned_train_text()
        vectorizer_object = sklearn.feature_extraction.text.TfidfVectorizer(max_df = self.max_df,
                                                                            min_df = self.min_df,
                                                                            max_features = self.max_features,
                                                                            ngram_range = self.ngram_range)
        
        train_vec = vectorizer_object.fit_transform(clean_text)
        test_vec = vectorizer_object.fit_transform(clean_text_test)
        feat_names = vectorizer_object.get_feature_names()
        
        return train_vec, test_vec, feat_names













