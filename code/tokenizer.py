"""
Dataset Tokenizer Script

This script processes the dataset by tokenizing the text data and saving the tokenized data.

"""

import pandas as pd
import os
import spacy
from tqdm import tqdm
import nltk
from nltk.corpus import words
from langdetect import detect, LangDetectException
import time
import psutil

# Download the words corpus if not already downloaded
nltk.download('words')
english_words = set(words.words())

def filter_pos(doc):
    """
    Filter the tokens in the document to include only nouns, verbs, and adjectives that are English words.

    Parameters:
    doc (spacy.tokens.doc.Doc): The spaCy document containing the tokens.

    Returns:
    str: The filtered text.
    """
    filtered_words = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and token.text.lower() in english_words]
    return ' '.join(filtered_words)

def add_label_column(df):
    """
    Add a label column to the DataFrame based on the presence of AI-related keywords.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data.

    Returns:
    pd.DataFrame: The DataFrame with an added 'label' column.
    """
    ai_keywords = (
        r'\bai\b|\bartificial intelligence\b|\bmachine learning\b|\bml\b|\bdeep learning\b|\bdl\b|'
        r'\bneural network\b|\bnn\b|\bnatural language processing\b|\bnlp\b|\bcomputer vision\b|\bcv\b|'
        r'\bA\.I\.\b|\bartificial-intelligence\b'
    )
    df['label'] = df['text'].str.contains(ai_keywords, case=False, regex=True).fillna(False).astype(int)
    return df

def is_english(text):
    """
    Check if the given text is in English.

    Parameters:
    text (str): The text to check.

    Returns:
    bool: True if the text is in English, False otherwise.
    """
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def process_chunk(chunk, nlp, text_column='text'):
    """
    Process a chunk of the dataset by filtering, tokenizing, and cleaning the text data.

    Parameters:
    chunk (pd.DataFrame): The chunk of the dataset to process.
    nlp (spacy.lang.en.English): The spaCy language model.
    text_column (str): The name of the text column in the DataFrame.

    Returns:
    pd.DataFrame: The processed chunk.
    """
    chunk[text_column] = chunk[text_column].fillna('').astype(str)
    english_mask = chunk[text_column].apply(is_english)
    chunk = chunk.iloc[english_mask.values]
    chunk.loc[:, 'processed_text'] = chunk[text_column].str.lower().str.replace(r'\bai\b|\bartificial intelligence\b', '', regex=True)
    chunk.loc[:, 'processed_text'] = chunk['processed_text'].str.findall(r'\w{3,}').str.join(' ')
    chunk['processed_text'] = chunk['processed_text'].fillna('')
    texts = chunk['processed_text'].tolist()
    docs = list(tqdm(nlp.pipe(texts, batch_size=50), total=len(texts), desc="spaCy processing"))
    chunk.loc[:, 'processed_text'] = [filter_pos(doc) for doc in tqdm(docs, desc="Filtering POS")]
    chunk = chunk[chunk['processed_text'].str.strip() != '']
    chunk = chunk.drop(columns=[text_column])
    chunk = chunk.rename(columns={'processed_text': 'text'})
    return chunk

def get_last_processed_id(output_path):
    """
    Get the ID of the last processed row from the output file.

    Parameters:
    output_path (str): The path to the output CSV file.

    Returns:
    str or None: The ID of the last processed row, or None if the file does not exist or is empty.
    """
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        if not df.empty:
            return df['id'].iloc[-1]  # Assuming 'id' column exists and is sorted
    return None

def process_and_write_chunk(chunk, nlp, sample_frac, random_state, output_path):
    """
    Process a chunk of the dataset and write the processed chunk to the output file.

    Parameters:
    chunk (pd.DataFrame): The chunk of the dataset to process.
    nlp (spacy.lang.en.English): The spaCy language model.
    sample_frac (float): The fraction of the chunk to sample.
    random_state (int): The random seed for sampling.
    output_path (str): The path to the output CSV file.

    Returns:
    pd.DataFrame: The processed chunk.
    """
    if sample_frac < 1:
        chunk = chunk.sample(frac=sample_frac, random_state=random_state)
    chunk = add_label_column(chunk)
    chunk = process_chunk(chunk, nlp)
    if output_path:  # Ensure output_path is not None
        if not os.path.isfile(output_path):
            chunk.to_csv(output_path, index=False, mode='w')
            print(f"Created new file and wrote chunk to {output_path}")
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
            print(f"Appended chunk to {output_path}")
    return chunk

def get_dynamic_chunk_size(base_chunk_size, memory_threshold=0.9):
    """
    Get the dynamic chunk size based on available memory.

    Parameters:
    base_chunk_size (int): The base chunk size.
    memory_threshold (float): The memory usage threshold.

    Returns:
    int: The dynamic chunk size.
    """
    available_memory = psutil.virtual_memory().available
    total_memory = psutil.virtual_memory().total
    memory_usage_ratio = available_memory / total_memory

    if memory_usage_ratio < memory_threshold:
        return max(int(base_chunk_size * memory_usage_ratio), 1)
    return base_chunk_size

def process_tweets(input_file='../preprocess/data_merged/merged.csv', output_path='../preprocess/data_tokenized/tokenized.csv', sample_frac=1, random_state=42, base_chunk_size=500000, encoding='utf-8'):
    """
    Process the tweets by tokenizing the text data and saving the tokenized data.

    Parameters:
    input_file (str): The path to the input CSV file.
    output_path (str): The path to the output CSV file.
    sample_frac (float): The fraction of the dataset to sample. Default is 1 (100%).
    random_state (int): The random seed for sampling. Default is 42.
    base_chunk_size (int): The base chunk size for processing. Default is 500000.
    encoding (str): The encoding of the input file. Default is 'utf-8'.

    Returns:
    None
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing file: {input_file}")
    try:
        start_time = time.time()
        
        # Load spaCy model
        nlp = spacy.load('en_core_web_sm')
        print("Loaded spaCy model 'en_core_web_sm'")
        
        end_time = time.time()
        print(f"Initialization time: {end_time - start_time:.2f} seconds")

        # Calculate total number of chunks
        total_rows = sum(1 for _ in open(input_file, encoding=encoding)) - 1
        total_chunks = (total_rows // base_chunk_size) + 1

        last_processed_id = get_last_processed_id(output_path)
        skip_rows = 0
        if last_processed_id is not None:
            print(f"Resuming from last processed ID: {last_processed_id}")
            # Find the row number of the last processed ID
            for chunk in pd.read_csv(input_file, chunksize=base_chunk_size, encoding=encoding):
                if last_processed_id in chunk['id'].values:
                    skip_rows += chunk.index[chunk['id'] == last_processed_id].tolist()[0] + 1
                    break
                skip_rows += len(chunk)

        for chunk in tqdm(pd.read_csv(input_file, chunksize=get_dynamic_chunk_size(base_chunk_size), encoding=encoding, skiprows=range(1, skip_rows)), total=total_chunks, desc="Processing chunks"):
            process_and_write_chunk(chunk, nlp, sample_frac, random_state, output_path)

    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        return

    print(f"CSV written: {output_path}")

if __name__ == "__main__":
    process_tweets(input_file='../preprocess/data_merged/merged.csv', output_path='../preprocess/data_tokenized/tokenized.csv', sample_frac=.1, base_chunk_size=100000, encoding='utf-8')