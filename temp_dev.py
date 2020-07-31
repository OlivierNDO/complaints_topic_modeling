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
temp = complaint_df.head(20000)



### Transformation
########################################################################################################
# Use Pipeline Function
text_pipeline = txt_proc.TextProcessingPipeline(string_list = temp[config.complaints_narrative_column])
text_vector, feature_names = text_pipeline.get_vectorized_text_and_feature_names()



### Model Fitting
########################################################################################################
# Perplexity Grid Search
lda_finder = ldaf.LDATopicFinder(text_vector, min_n_topics = 3, max_n_topics = 6)
perplexity_grid_results = lda_finder.run_kfold_perplexity_grid()
perplexity_grid_results = lda_finder.run_perplexity_grid_search()

# Scoring
lda = LatentDirichletAllocation(n_components = 7, max_iter = 5, learning_method = 'online', learning_offset = 50.,random_state = 0).fit(text_vector)
lda.perplexity(text_vector)
scores = lda.transform(text_vector[2:10])
topic_probs = scores.max(axis = 1)
topic_labels = scores.argmax(axis = 1)
topic_uncertainty = 1 - topic_probs








