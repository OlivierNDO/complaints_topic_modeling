# File Path to Data Sourced from https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data
complaints_file = 'D:/complaint_data/complaints.csv'
complaints_file_encoding = 'cp850'
complaints_narrative_column = 'Consumer complaint narrative'

# Text Processing Configuration
punctuation_str = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
general_stopwords = ["again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't",
                     "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
                     "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does",
                     "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for",
                     "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven",
                     "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
                     "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself",
                     "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't",
                     "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on",
                     "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re",
                     "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn",
                     "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their",
                     "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
                     "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't",
                     "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who",
                     "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you",
                     "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

custom_stopwords = ['\n', 'bank of america', 'chase', 'wells fargo', 'citi', 'citigroup', 'citibank',
                    'american express', 'epress', 'wells', 'synchrony', 'discover', 'usaa',
                    'capital one', 'barclay', 'express', 'american', 'amex', 'paypal', 'bank',
                    'fargo']

year_stopwords = [str(y) for y in range(2000,2050)]

all_stopwords = general_stopwords + custom_stopwords + year_stopwords

substrings_to_remove = ['\n'] + ['x' * n for n in range(2,25)] + [w for w in all_stopwords if len(w.split(' ')) > 1]
