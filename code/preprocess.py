"""
Dataset Preprocess Script

This script processes the Twitter Stream Archive by extracting relevant information from compressed files and outputting the results in a DataFrame.

"""

import bz2
import gzip
import glob 
import json
import random
import re
import tarfile
import zipfile
from datetime import datetime as dt
import os

import pandas as pd
from tqdm import tqdm

def load_compressed_json(file, compression_type, year):
    """
    Load compressed JSON files and handle both modular (with 'data', 'includes') 
    and flat tweet structures.

    Parameters:
    file (file-like object): The compressed file to load.
    compression_type (str): The type of compression ('bz2', 'gz', 'json').
    year (int): The year of the Twitter stream data.

    Yields:
    dict: Parsed tweet data.
    """
    try:
        if compression_type == 'bz2':
            with bz2.BZ2File(file) as handle:
                for line in handle:
                    yield parse_tweet(line, year)
        elif compression_type == 'gz':
            with gzip.GzipFile(fileobj=file) as handle:
                for line in handle:
                    yield parse_tweet(line, year)
        elif compression_type == 'json':
            for line in file:
                yield parse_tweet(line, year)
    except Exception as e:
        print(f"Error opening or processing file: {e}")

def parse_tweet(line, year):
    """
    Parse individual tweet data based on format differences.

    Parameters:
    line (bytes): A line of JSON data.
    year (int): The year of the Twitter stream data.

    Returns:
    dict: Parsed tweet data.
    """
    try:
        tweet = json.loads(line.decode('utf-8'))  # Decode if compressed
        # First format with 'data' and modular structure
        if 'data' in tweet:
            tweet = tweet['data']
        # Extract text and creation date for modular structure
        parsed_tweet = {
            'id': tweet.get('id', None),
            'text': tweet.get('text', None),
            'created_at': tweet.get('created_at', None)
        }
        return parsed_tweet
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"Unexpected error parsing tweet: {e}")
    return None

def clean(tweets, percentage, seed):
    """
    Clean the tweets by selecting a random sample and formatting the data.

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
            tweet_date = dt.strptime(tweet['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ').date()
            yield {
                'id': tweet['id'],
                'text': tweet['text'],
                'created_at': tweet_date.strftime('%Y-%m-%d')
            }
        except KeyError:
            continue  # Do nothing if required keys are missing

def filter_by_dates(df, dates):
    """
    Filter the DataFrame to include only rows with 'created_at' in the specified dates.

    Parameters:
    df (pd.DataFrame): The DataFrame to filter.
    dates (list of str): List of dates to keep. Format should be 'YYYY-MM-DD'.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    df['created_at'] = pd.to_datetime(df['created_at']).dt.date
    dates = pd.to_datetime(dates).date
    return df[df['created_at'].isin(dates)]

def process_twitterstream(year, percentage=1, seed=0, dates=None):
    """
    Process Twitter stream data for a given year, extracting and cleaning tweets from compressed files.

    This function searches for compressed files (.tar or .zip) in the specified directory, extracts
    .bz2, .json.gz, or .json files, loads the tweets, cleans them, and saves the results to CSV files.

    Parameters:
    year (int): The year of the Twitter stream data to process.
    percentage (float): The percentage of tweets to sample. Default is 1 (100%).
    seed (int): The seed for the random number generator used for sampling tweets. Default is 0.
    dates (list of str, optional): List of dates to filter the tweets. Format should be 'YYYY-MM-DD'.

    Returns:
    None
    """
    filenames = glob.glob(os.path.join('..', 'preprocess', 'data', str(year), '*'))

    if not filenames:
        print("No files found for the given year.")
        return

    for filename in filenames:
        print('Starting processing...')
        print(filename)

        output_filename = os.path.basename(filename)
        output_filename = re.sub('.tar', '', output_filename)
        output_filename = re.sub('.zip', '', output_filename)

        output_dir = os.path.join('..', 'preprocess', 'data_preprocessed', str(year))
        os.makedirs(output_dir, exist_ok=True)

        all_tweets = []

        if filename.endswith('.tar'):
            with tarfile.open(filename) as tar:
                for member in tqdm(tar.getmembers()):
                    if member.isfile() and (member.name.endswith('.bz2') or member.name.endswith('.json.gz') or member.name.endswith('.json')):
                        print(f"Extracting file from tar: {member.name}")
                        file = tar.extractfile(member)
                        if file:
                            compression_type = 'bz2' if member.name.endswith('.bz2') else 'gz' if member.name.endswith('.json.gz') else 'json'
                            tweets = clean(load_compressed_json(file, compression_type, year), percentage, seed)
                            all_tweets.extend(tweets)
                        else:
                            print(f"Could not extract file from tar: {member.name}")
                    else:
                        print(f"Skipping non-bz2/non-gz/non-json file: {member.name}")
        elif filename.endswith('.zip'):
            with zipfile.ZipFile(filename) as zipf:
                for member in tqdm(zipf.namelist()):
                    if member.endswith('.bz2') or member.endswith('.json.gz') or member.endswith('.json'):
                        print(f"Extracting file from zip: {member}")
                        with zipf.open(member) as file:
                            compression_type = 'bz2' if member.endswith('.bz2') else 'gz' if member.endswith('.json.gz') else 'json'
                            tweets = clean(load_compressed_json(file, compression_type, year), percentage, seed)
                            all_tweets.extend(tweets)
                    else:
                        print(f"Skipping non-bz2/non-gz/non-json file: {member}")

        print(f"Total tweets extracted: {len(all_tweets)}")

        if not all_tweets:
            print("No tweets extracted, skipping file.")
            continue

        df = pd.DataFrame(all_tweets, columns=['id', 'text', 'created_at'])

        if dates:
            df = filter_by_dates(df, dates)
            print(f"Total tweets after date filtering: {len(df)}")

        if df.empty:
            print("DataFrame is empty after processing, skipping CSV writing.")
            continue

        df.to_csv(os.path.join(output_dir, f'{output_filename}.csv'), index=False)
        print(f"CSV written: {os.path.join(output_dir, f'{output_filename}.csv')}")

if __name__ == "__main__":
    for year in range(2019, 2023):
        process_twitterstream(year)