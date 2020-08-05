### Configuration
########################################################################################################
# System Path Insert
import sys
sys.path.insert(1, 'D:/complaints_topic_modeling/')

# Python Module Imports
import numpy as np
import pandas as pd
import pickle
import sklearn

# Local Module Imports
import src.configuration as config
import src.eda as eda
import src.text_processing as txt_proc
import src.lda_fitting as ldaf

# Random State
np.random.seed(8042020)



### File Reading
########################################################################################################
# Read Csv File from CFPB
complaint_df = pd.read_csv(config.complaints_file, encoding = config.complaints_file_encoding)
complaint_df = complaint_df[complaint_df[config.complaints_narrative_column].notna()].drop_duplicates(subset = config.complaints_narrative_column, keep = 'first')

# Subset by Product
card_complaint_df = complaint_df[(complaint_df.Product == 'Credit card or prepaid card') & (complaint_df['Sub-product'] == 'General-purpose credit card or charge card')]

# Train and Test
card_complaints_train, card_complaints_test = sklearn.model_selection.train_test_split(card_complaint_df, test_size = 0.2)



### Transformation 1
########################################################################################################
# Call Pipeline Class
text_pipeline = txt_proc.TextProcessingPipeline(string_list = card_complaints_train[config.complaints_narrative_column],
                                                test_string_list = card_complaints_test[config.complaints_narrative_column],
                                                max_df = 0.5,
                                                min_df = 10,
                                                max_features = 1200,
                                                ngram_range = (1,2))

# Generate Train, Test Vectors & Feature Names
train_vec, test_vec, feat_names = text_pipeline.get_vectorized_text_and_feature_names_train_test()


### Model Fitting 1
########################################################################################################
# Call TopicFinder Class
lda_finder = ldaf.LDATopicFinder(tfid_vector = train_vec, tfid_vector_test = test_vec,
                                 min_n_topics = 4, max_n_topics = 20, max_iter = 10)

# Perplexity & Uncertainty Grid Search
perplexity_grid_results = lda_finder.run_kfold_perplexity_grid()

# Fit on Selected Number of Topics
lda_finder.use_n_topics = 12
lda_model, scored_train = lda_finder.fit_and_score_train()
scored_train['clean_text'] = text_pipeline.get_cleaned_train_text()
scored_train['text'] = list(card_complaints_train[config.complaints_narrative_column])

# Plot Topic Frequency
eda.plot_frequency_counts(scored_train['predicted_topic'], title = 'Training Set Topic Frequency')

# Look at & Label Topics
topics = ldaf.show_lda_topics(lda_model, feat_names, n_top_words = 70)

# Label Topics
topic_labels = ['Merchant Disputes',
                'Closures/Credit Limits',
                'Other',
                'Payments & Billing',
                'Other',
                'Rewards/Promos',
                'Other',
                'Other',
                'Other',
                'Fees/Interest',
                'Other',
                'Other']

topic_label_dict = dict(zip(list(range(1, len(topic_labels) + 1)), topic_labels))

# Plot Topic Distribution After Labeling
scored_train['predicted_topic_label'] = [topic_label_dict.get(x) for x in scored_train.predicted_topic]
eda.plot_frequency_counts(scored_train['predicted_topic_label'],
                          title = 'Credit Card Complaints - Modeled Topics',
                          color = 'darkblue',
                          xlab = 'Topic')


# Save First Model (lda_model1_20200804_212126.pkl)
save_name_first_model = ldaf.get_timestamp_save_name('lda_model1', '.pkl') 
with open(f'{config.model_save_folder}{save_name_first_model}', 'wb') as file:
    pickle.dump(lda_model, file)




### Transform and Score Test
########################################################################################################
     
# Predict on Test Set
scored_test = ldaf.fit_and_score_dframe(lda_model, test_vec)
scored_test['text'] = list(card_complaints_test[config.complaints_narrative_column])
scored_test['clean_text'] = text_pipeline.get_cleaned_test_text()
scored_test['predicted_topic_label'] = [topic_label_dict.get(x) for x in scored_test.predicted_topic]









