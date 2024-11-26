import os
import sys

# from preprocess import process_twitterstream
# from merge import merge_data, merge_all_years
# from tokenizer import process_tweets
from clean import clean_data
from classifier import vectorize_train
from sentiment import run_sentiment

# Set the current working directory to the folder where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.append(current_dir)

# for year in range(2019, 2023):
#     process_twitterstream(year)
#     merge_data(year)

# for year in range(2019, 2023):
#     input_file = f'../preprocess/data_merged/{year}_merged.csv'
#     output_path = f'../preprocess/data_tokenized/{year}_tokenized.csv'
#     process_tweets(input_file=input_file, output_path=output_path, sample_frac=.2, base_chunk_size=100000)

# merge_all_years()   # Merge all years into one file

# Run only this for replication
clean_data()        # Remove porn content from data
vectorize_train()   # Train classifier
run_sentiment()     # Run sentiment analysis