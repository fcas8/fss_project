"""
Dataset Clean Script

This script cleans the dataset by removing explicit content and saving the cleaned data.

"""

import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils import resample
import shutil

def add_label_column(df):
    """
    Add a label column to the DataFrame based on the presence of explicit content keywords.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data.

    Returns:
    pd.DataFrame: The DataFrame with an added 'label_p' column.
    """
    # Extended list of keywords commonly associated with explicit content
    porn_keywords = (
        r'\bporn\b|\bpornography\b|\bsex\b|\bsexy\b|\berotic\b|\bnude\b|\bnudes\b|\bxxx\b|'
        r'\bx-rated\b|\badult\b|\bfetish\b|\blust\b|\bhentai\b|\bcamgirl\b|\bonlyfans\b|'
        r'\bpornhub\b|\bnsfw\b|\bmature\b|\bstrip\b|\bescort\b|\bplayboy\b|\bboobs\b|'
        r'\bbutts\b|\btwerk\b|\bexotic\b|\bsmut\b|\bdirty\b|\bexplicit\b|\bthirsty\b|'
        r'\bhot\b|\bbabes\b|\bbare\b|\bintimate\b|\bdildo\b|\bplaymate\b|\bprovocative\b|'
        r'\bskin\b|\bbusty\b|\bsensual\b|\btease\b|\blingerie\b|\bdaddy\b|\bfantasy\b|'
        r'\bracy\b|\bobscene\b|\btitty\b|\btitties\b|\bpussy\b|\bcum\b|\bcumshot\b|\blustful\b|'
        r'\bdominatrix\b|\bspicy\b|\broleplay\b|\bfurry\b|\bsubmission\b|\bsubmissive\b|'
        r'\bprovocative\b|\bbdsm\b|\bwhip\b|\bslave\b|\btaboo\b|\bbooty\b|\bpenetration\b|'
        r'\b69\b|\bg-spot\b|\bintimacy\b|\bseduct\b|\bprovocative\b|\bwet\b|\brimming\b'
    )
    # Create a new 'label' column: 1 if porn content is detected, 0 otherwise
    df['label_p'] = df['text'].str.contains(porn_keywords, case=False, regex=True).fillna(False).astype(int)
    return df

def load_data(input_path):
    """
    Load and clean the dataset.

    Parameters:
    input_path (str): The path to the input CSV file.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    print("Loading data...")
    df = pd.read_csv(input_path)
    df = df.dropna()
    print(f"Data loaded. Shape: {df.shape}")

    # Add label column
    df = add_label_column(df)

    return df

def split_and_vectorize(df):
    """
    Split the data into training and test sets, balance the training set, and vectorize the text data.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data and labels.

    Returns:
    tuple: Tuple containing the vectorized training and test features, training and test labels, and the vectorizer.
    """
    print("Splitting and vectorizing data...")

    # Split data into features and labels
    X = df['text']
    Y = df['label_p']

    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Combine training data back for resampling
    train_data = pd.concat([X_train, Y_train], axis=1)

    # Separate majority and minority classes
    class_0 = train_data[train_data['label_p'] == 0]
    class_1 = train_data[train_data['label_p'] == 1]

    # Downsample majority class
    class_0_downsampled = resample(class_0, 
                                   replace=False,    # sample without replacement
                                   n_samples=len(class_1),     # to match minority class
                                   random_state=42) # reproducible results

    # Combine minority class with downsampled majority class
    train_data_balanced = pd.concat([class_0_downsampled, class_1])

    # Splitting the features and label variable for balanced data
    X_train_balanced = train_data_balanced.drop('label_p', axis=1)
    Y_train_balanced = train_data_balanced['label_p']

    # Vectorize the text data using built-in stop words
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_df=0.9999, min_df=0.0001, stop_words='english')

    # Vectorize the balanced training features
    X_train_balanced_counts = vectorizer.fit_transform(X_train_balanced.squeeze())

    # Vectorize the test features
    X_test_counts = vectorizer.transform(X_test)

    print(f"Data split and vectorized. Number of features: {X_train_balanced_counts.shape[1]}")
    return X_train_balanced_counts, X_test_counts, Y_train_balanced, Y_test, vectorizer

def train_classifier(X_train_balanced_counts, Y_train_balanced):
    """
    Train a Logistic Regression CV classifier.

    Parameters:
    X_train_balanced_counts (scipy.sparse.csr.csr_matrix): The vectorized training features.
    Y_train_balanced (pd.Series): The training labels.

    Returns:
    LogisticRegressionCV: The trained classifier.
    """
    print("Training classifier...")
    clf = LogisticRegressionCV(cv=5, max_iter=1000)
    clf.fit(X_train_balanced_counts, Y_train_balanced)
    print("Classifier trained.")
    return clf

def add_predictions(df, clf, vectorizer):
    """
    Add predictions to the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data.
    clf (LogisticRegressionCV): The trained classifier.
    vectorizer (CountVectorizer): The vectorizer used to transform the text data.

    Returns:
    pd.DataFrame: The DataFrame with an added 'prediction' column.
    """
    print("Adding predictions...")
    X_counts = vectorizer.transform(df['text'])
    df['prediction'] = clf.predict_proba(X_counts)[:, 1]
    return df

def clean_data(input_path='../preprocess/data_tokenized/tokenized.csv', output_dir='../preprocess/data_cleaned'):
    """
    Clean the dataset by removing explicit content and saving the cleaned data.

    Parameters:
    input_path (str): The path to the input CSV file.
    output_dir (str): The directory to save the cleaned data.

    Returns:
    None
    """
    # Ensure the output directory exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'cleaned.csv')
    
    # Remove the output file if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Existing file {output_file} removed.")
    
    df = load_data(input_path)
    X_train_balanced_counts, X_test_counts, Y_train_balanced, Y_test, vectorizer = split_and_vectorize(df)
    clf = train_classifier(X_train_balanced_counts, Y_train_balanced)
    
    # Add predictions to the DataFrame
    df = add_predictions(df, clf, vectorizer)
    
    # Delete tweets with a prediction greater than 0.8
    df_cleaned = df[df['prediction'] <= 0.8]
    
    # Drop the 'label_p' and 'prediction' columns
    df_cleaned = df_cleaned.drop(columns=['label_p', 'prediction'])
    
    # Save the cleaned DataFrame
    df_cleaned.to_csv(output_file, index=False)
    print("Cleaned data saved.")

if __name__ == "__main__":
    clean_data()