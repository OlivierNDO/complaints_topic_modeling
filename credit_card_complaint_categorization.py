### Configuration
########################################################################################################
# System Path Insert
import sys
sys.path.insert(1, 'D:/complaints_topic_modeling/')

# Python Module Imports
import pandas as pd
import sklearn

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

# Train and Test
card_complaints_train, card_complaints_test = sklearn.model_selection.train_test_split(card_complaint_df, test_size = 0.2)



### Transformation
########################################################################################################
# Call Pipeline Class
text_pipeline = txt_proc.TextProcessingPipeline(string_list = card_complaints_train[config.complaints_narrative_column],
                                                test_string_list = card_complaints_test[config.complaints_narrative_column],
                                                max_df = 0.6,
                                                min_df = 20,
                                                max_features = 850,
                                                ngram_range = (1,3))
# Generate Train, Test Vectors & Feature Names
train_vec, test_vec, feat_names = text_pipeline.get_vectorized_text_and_feature_names_train_test()



### Model Fitting
########################################################################################################
# Call TopicFinder Class
lda_finder = ldaf.LDATopicFinder(tfid_vector = train_vec, tfid_vector_test = test_vec,
                                 min_n_topics = 3, max_n_topics = 20, max_iter = 10)


# Perplexity & Uncertainty Grid Search
perplexity_grid_results = lda_finder.run_kfold_perplexity_grid()

# Fit on Selected Number of Topics
lda_finder.use_n_topics = 9
lda_model, scored_train = lda_finder.fit_and_score_train()

# Look at & Label Topics
topics = ldaf.show_lda_topics(lda_model, feat_names, n_top_words = 70)

# Label Topics
topic_labels = []



### Transform and Score Test
########################################################################################################

# Score Test Set
