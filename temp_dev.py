
# System Path Insert
import sys
sys.path.insert(1, 'D:/complaints_topic_modeling/')

# Python Module Imports
import csv
import pandas as pd

# Local Module Imports
import src.configuration as config


temp = pd.read_csv(config.config_complaints_file)


with open(config.config_complaints_file, 'r', encoding = 'cp850') as f:
    reader_object = csv.reader(f, delimiter = ',')
    row_list = [r for r in reader_object]
    for row in reader_object:
        

('cp1252', 'cp850','utf-8','utf8')

file1 = open(config.config_complaints_file, 'r')
t = file1.read()
file1.close()


