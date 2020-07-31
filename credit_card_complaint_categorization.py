### Configuration
########################################################################################################
# System Path Insert
import sys
sys.path.insert(1, 'D:/complaints_topic_modeling/')

# Python Module Imports
import csv
import itertools
import nltk
import numpy as np
import operator
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import NMF, LatentDirichletAllocation
import random
import re

# Local Module Imports
import src.configuration as config
import src.text_processing as txt_proc
import src.lda_fitting as ldaf



### File Reading
########################################################################################################
# Read Csv File from CFPB
complaint_df = pd.read_csv(config.complaints_file, encoding = config.complaints_file_encoding)
complaint_df = complaint_df[complaint_df[config.complaints_narrative_column].notna()]

# Subset Credit Card/Prepaid Card Product
card_complaint_df = complaint_df[complaint_df.Product == 'Credit card or prepaid card']



### Transformation
########################################################################################################
# Use Pipeline Function
text_pipeline = txt_proc.TextProcessingPipeline(string_list = card_complaint_df[config.complaints_narrative_column],
                                                max_df = 0.6,
                                                min_df = 20,
                                                max_features = 650,
                                                ngram_range = (1,2))
text_vector, feature_names = text_pipeline.get_vectorized_text_and_feature_names()



### Model Fitting
########################################################################################################
# Perplexity Grid Search
lda_finder = ldaf.LDATopicFinder(text_vector, min_n_topics = 3, max_n_topics = 6)
perplexity_grid_results = lda_finder.run_kfold_perplexity_grid()


