""" In this script, I tokenize Reddit posts from a CSV file.
I then save the processed records to a new CSV file. """

import pandas as pd
import os
import spacy
from tqdm import tqdm

def filter_pos(doc):
    """
    Filters nouns, verbs, and adjectives from a spaCy doc.

    Args:
        doc (spacy.tokens.doc.Doc): The spaCy doc to filter.

    Returns:
        str: A string of filtered tokens.
    """
    return ' '.join([token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']])

def add_label_column(df):
    """
    Adds a binary label column based on AI keywords.

    Args:
        df (pd.DataFrame): The DataFrame to add the label column to.

    Returns:
        pd.DataFrame: The DataFrame with the added label column.
    """
    df['label'] = df['title'].str.contains(r'\bai\b|\bartificial intelligence\b', case=False, regex=True).fillna(False).astype(int)
    return df

def preprocess_posts(df, text_column='title'):
    """
    Preprocesses the text data in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        text_column (str, optional): The column containing the text data. Defaults to 'title'.

    Returns:
        pd.DataFrame: The DataFrame with the processed text data.
    """
    df['processed_text'] = df[text_column].str.lower().str.replace(r'\bai\b|\bartificial intelligence\b', '', regex=True)
    df['processed_text'] = df['processed_text'].str.findall(r'\w{3,}').str.join(' ')
    df['processed_text'] = df['processed_text'].fillna('')
    
    texts = df['processed_text'].tolist()
    docs = list(tqdm(nlp.pipe(texts, batch_size=50), total=len(texts), desc="spaCy processing"))
    df['processed_text'] = [filter_pos(doc) for doc in tqdm(docs, desc="Filtering POS")]
    return df

def tokenize(input_path, sampling_frac=1.0):
    """
    Main function to execute the tokenization and preprocessing.

    Args:
        input_path (str): Path to the input CSV file.
    """
    # Define the output path
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    output_path = os.path.join(output_dir, 'tokenized_posts.csv')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load and shuffle the data
    df = pd.read_csv(input_path).sample(frac=sampling_frac, random_state=42)

    # Convert 'created_utc' to datetime and filter posts after 2017
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    df = df[df['created_utc'].dt.year >= 2017]

    # Load spaCy model for processing
    global nlp
    nlp = spacy.load('en_core_web_sm')

    # Apply label addition and preprocessing
    df_processed = add_label_column(df)
    df_processed = preprocess_posts(df)

    # Save processed data
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python tokenize.py <input_csv_path>")
        sys.exit(1)
    input_csv_path = sys.argv[1]
    tokenize(input_csv_path)