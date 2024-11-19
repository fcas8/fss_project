import pandas as pd
import os
import spacy
from tqdm import tqdm
import nltk
from nltk.corpus import words

# Download the words corpus if not already downloaded
nltk.download('words')
english_words = set(words.words())

def filter_pos(doc):
    """
    Filters out words that are not nouns, verbs, or adjectives using spaCy,
    and keeps only English words.
    """
    filtered_words = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and token.text.lower() in english_words]
    return ' '.join(filtered_words)

def add_label_column(df):
    """
    Adds a binary label column to a pandas DataFrame based on the presence of AI-related keywords in a text column.
    Handles NaN values in the 'text' column by treating them as not containing the keyword.
    """
    print("Adding label column based on AI-related keywords...")
    ai_keywords = (
        r'\bai\b|\bartificial intelligence\b|\bmachine learning\b|\bml\b|\bdeep learning\b|\bdl\b|'
        r'\bneural network\b|\bnn\b|\bnatural language processing\b|\bnlp\b|\bcomputer vision\b|\bcv\b|'
        r'\bA\.I\.\b|\bartificial-intelligence\b'
    )
    df['label'] = df['text'].str.contains(ai_keywords, case=False, regex=True).fillna(False).astype(int)
    print("Label column added.")
    return df

def process_chunk(chunk, text_column='text'):
    """
    Preprocesses a chunk of tweets by cleaning and tokenizing the text data, including the removal of AI-related keywords.
    """
    print("Starting preprocessing of chunk...")

    # Ensure all values in the text column are strings and fill NaN values with empty strings
    chunk[text_column] = chunk[text_column].fillna('').astype(str)

    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')

    # Lowercase conversion and removal of AI-related keywords
    print("Lowercasing and removing AI-related keywords...")
    chunk['processed_text'] = chunk[text_column].str.lower().str.replace(r'\bai\b|\bartificial intelligence\b', '', regex=True)
    
    # Tokenize and remove short tokens
    print("Tokenizing and removing short tokens...")
    chunk['processed_text'] = chunk['processed_text'].str.findall(r'\w{3,}').str.join(' ')
    
    # Remove NaN values
    print("Removing NaN values...")
    chunk['processed_text'] = chunk['processed_text'].fillna('')
    
    # Process texts in batches using spaCy's pipe method
    print("Processing texts with spaCy...")
    texts = chunk['processed_text'].tolist()
    docs = list(tqdm(nlp.pipe(texts, batch_size=50), total=len(texts), desc="spaCy processing"))
    chunk['processed_text'] = [filter_pos(doc) for doc in tqdm(docs, desc="Filtering POS")]

    # Drop rows where 'processed_text' is empty
    chunk = chunk[chunk['processed_text'].str.strip() != '']

    # Drop the original 'text' column and rename 'processed_text' to 'text'
    chunk = chunk.drop(columns=[text_column])
    chunk = chunk.rename(columns={'processed_text': 'text'})

    print(f"Preprocessing complete. Chunk size: {chunk.shape}")
    return chunk

def process_tweets(input_file='../preprocess/data_merged/merged.csv', output_path='../preprocess/data_tokenized/tokenized_posts.csv', sample_frac=1, random_state=42, chunk_size=100000, encoding='utf-8'):
    """
    Main function to process tweets in chunks.
    """
    # Set the current working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Delete the existing output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Deleted existing file: {output_path}")

    print(f"Processing file: {input_file}")
    try:
        # Get the total number of rows in the input file
        total_rows = sum(1 for _ in open(input_file, encoding=encoding)) - 1  # Subtract 1 for the header
        total_chunks = (total_rows // chunk_size) + 1

        # Process the data in chunks with a progress bar
        for chunk in tqdm(pd.read_csv(input_file, chunksize=chunk_size, encoding=encoding), total=total_chunks, desc="Processing chunks"):
            # Sample the data
            if sample_frac < 1:
                chunk = chunk.sample(frac=sample_frac, random_state=random_state)
            
            chunk = add_label_column(chunk)
            chunk = process_chunk(chunk)

            # Append the processed chunk to the output file
            if not os.path.isfile(output_path):
                chunk.to_csv(output_path, index=False, mode='w')
                print(f"Created new file and wrote chunk to {output_path}")
            else:
                chunk.to_csv(output_path, index=False, mode='a', header=False)
                print(f"Appended chunk to {output_path}")
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        return

    print(f"CSV written: {output_path}")

if __name__ == "__main__":
    process_tweets(input_file='../preprocess/data_merged/merged.csv', output_path='../preprocess/data_tokenized/tokenized_posts.csv', sample_frac=1, chunk_size=100000, encoding='utf-8')