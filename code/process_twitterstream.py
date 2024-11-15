"""
In this code, I open tarfiles of the Twitter Stream Archive, 
keep relevant information and output results in a dataframe.
"""

import bz2
import glob 
import json
import random
import re
import tarfile
import zipfile
from datetime import datetime as dt
import os

import carmen
import pandas as pd
from tqdm import tqdm

def load_bz2_json(filename):
    """
    This function loads a .bz2 compressed JSON file and yields each tweet in the file as a Python dictionary.
    If an error occurs while loading a tweet, it prints the error message and continues with the next tweet.

    Parameters:
    filename (str): The path to the .bz2 file.

    Yields:
    dict: The next tweet in the file.
    """
    try:
        # Ensure the file object is correctly handled for bz2 decompression
        with bz2.BZ2File(filename) as handle:
            for line in handle:
                try:
                    tweet = json.loads(line.decode('utf-8'))  # Ensure proper decoding
                    yield tweet
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                except Exception as e:
                    print(f"Unexpected error loading tweet: {e}")
    except Exception as e:
        print(f"Unexpected error opening file: {e}")

def clean(tweets, percentage, seed):
    """
    This function takes a generator of tweets, selects a random sample of the tweets based on the provided percentage,
    resolves the location of each tweet using the provided resolver, and yields the cleaned tweets.
    Each cleaned tweet is a dictionary with keys 'id', 'text', 'created_at', 'country', 'state', 'county', 'city', 
    'known', 'latitude', and 'longitude'.

    This version is modified to only yield tweets posted on the first day of any month.

    Parameters:
    tweets (generator): The list of tweets to clean.
    percentage (float): The percentage of tweets to sample.
    seed (int): The seed for the random number generator used for sampling tweets.

    Yields:
    dict: The next cleaned tweet.
    """
    tweets = list(tweets)
    num_samples = int(len(tweets) * percentage)
    if len(tweets) >= num_samples:
        random.seed(seed)
        tweets = random.sample(tweets, num_samples)
    for tweet in tweets:
        try:
            tweet_date = dt.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y').date()
            yield {
                'id': tweet['id'],
                'text': tweet['text'],
                'created_at': tweet_date.strftime('%Y-%m-%d')
            }
        except KeyError:
            continue  # Do nothing if required keys are missing

def process_twitterstream_other_days(year, percentage=1, seed=0, csv=False):
    """
    This function processes a Twitter stream data for a given year. It reads the data from .tar or .zip files, 
    extracts the tweets, selects a random sample of the tweets based on the provided percentage, 
    resolves the location of each tweet using the Carmen library, and saves the processed data in a .json file. 
    If csv is set to True, it also saves the data in a .csv file.

    Parameters:
    year (int): The year of the Twitter stream data to process.
    percentage (float, optional): The percentage of tweets to sample from each file. Default is 1, which means all tweets are used.
    seed (int, optional): The seed for the random number generator used for sampling tweets. Default is 0.
    csv (bool, optional): Whether to save the processed data in a .csv file. Default is False.

    Returns:
    None

    """

    filenames = glob.glob(os.path.join('main', 'preprocess', 'data', str(year), '*'))

    for filename in filenames:
        print('Starting processing...')
        print(filename)

        output_filename = os.path.basename(filename)
        output_filename = re.sub('.tar', '', output_filename)
        output_filename = re.sub('.zip', '', output_filename)

        output_dir = os.path.join('main', 'preprocess', 'data_located', str(year))
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, f'{output_filename}.json'), 'a+', encoding='utf-8') as output:
            if filename.endswith('.tar'):
                with tarfile.open(filename) as tar:
                    for member in tqdm(tar.getmembers()):
                        file = tar.extractfile(member)
                        tweets = clean(load_bz2_json(file), percentage, seed)
                        for tweet in tweets:
                            json.dump(tweet, output)
                            output.write('\n')
            elif filename.endswith('.zip'):
                with zipfile.ZipFile(filename) as zipf:
                    for member in tqdm(zipf.namelist()):
                        if member.endswith('.bz2'):
                            with zipf.open(member) as file:
                                tweets = clean(load_bz2_json(file), percentage, seed)
                                for tweet in tweets:
                                    json.dump(tweet, output)
                                    output.write('\n')

        all_tweets = []
        with open(os.path.join(output_dir, f'{output_filename}.json'), 'r', encoding='utf-8') as f:
            for line in f:
                content = json.loads(line)
                all_tweets.append(content)

        df = pd.DataFrame(all_tweets, columns=['id', 'text', 'created_at'])

        if csv:
            df.to_csv(os.path.join(output_dir, f'{output_filename}.csv'))