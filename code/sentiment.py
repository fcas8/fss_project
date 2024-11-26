"""
Sentiment Analysis Script

This script performs sentiment analysis on the dataset, filters AI-related tweets, and generates various sentiment-related outputs.

"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import joblib
import contextlib

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def load_data(input_file):
    """
    Load the dataset from a CSV file and sort by date.

    Parameters:
    input_file (str): The path to the input CSV file.

    Returns:
    pd.DataFrame: The loaded and sorted DataFrame.
    """
    print("Loading data...")
    df = pd.read_csv(input_file)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.sort_values(by='created_at').reset_index(drop=True)
    print(f"Data loaded and sorted by date. Shape: {df.shape}")
    return df

def filter_ai_tweets(df, classifier, vectorizer):
    """
    Filter the DataFrame to include only AI-related tweets with prediction > 0.8.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data.
    classifier (LogisticRegressionCV): The trained classifier.
    vectorizer (CountVectorizer): The vectorizer used to transform the text data.

    Returns:
    pd.DataFrame: The filtered DataFrame containing only AI-related tweets.
    """
    print("Filtering AI-related tweets...")
    df['text'] = df['text'].fillna('').astype(str)
    X_counts = vectorizer.transform(df['text'])
    predictions = classifier.predict_proba(X_counts)[:, 1]
    df_ai = df[predictions > 0.8].copy()
    print(f"Filtered AI-related tweets. Shape: {df_ai.shape}")
    return df_ai

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text using VADER.

    Parameters:
    text (str): The text to analyze.

    Returns:
    dict: The sentiment scores.
    """
    return sia.polarity_scores(text)

def add_sentiment_scores(df_ai):
    """
    Add sentiment scores to the DataFrame.

    Parameters:
    df_ai (pd.DataFrame): The DataFrame containing AI-related tweets.

    Returns:
    pd.DataFrame: The DataFrame with added sentiment scores.
    """
    print("Adding sentiment scores...")
    df_ai['sentiment'] = df_ai['text'].apply(analyze_sentiment)

    # Extract sentiment scores into separate columns
    df_ai['neg'] = df_ai['sentiment'].apply(lambda x: x['neg'] if x else None)
    df_ai['neu'] = df_ai['sentiment'].apply(lambda x: x['neu'] if x else None)
    df_ai['pos'] = df_ai['sentiment'].apply(lambda x: x['pos'] if x else None)
    df_ai['compound'] = df_ai['sentiment'].apply(lambda x: x['compound'] if x else None)

    # Remove the sentiment column
    df_ai = df_ai.drop(columns=['sentiment'])

    print(f"Sentiment scores added. Shape: {df_ai.shape}")
    return df_ai

def save_data(df_ai, output_file):
    """
    Save the DataFrame with sentiment scores to a CSV file.

    Parameters:
    df_ai (pd.DataFrame): The DataFrame containing AI-related tweets with sentiment scores.
    output_file (str): The path to the output CSV file.

    Returns:
    None
    """
    df_ai.to_csv(output_file, index=False)
    print(f"Data with sentiment scores saved to {output_file}")

def assign_sentiment_label(compound):
    """
    Assign a sentiment label based on the compound score.

    Parameters:
    compound (float): The compound sentiment score.

    Returns:
    str: The sentiment label ('positive', 'negative', or 'neutral').
    """
    if compound > 0.05:
        return 'positive'
    elif compound < -0.05:
        return 'negative'
    else:
        return 'neutral'

def add_sentiment_labels(df_ai):
    """
    Add sentiment labels to the DataFrame.

    Parameters:
    df_ai (pd.DataFrame): The DataFrame containing AI-related tweets with sentiment scores.

    Returns:
    pd.DataFrame: The DataFrame with added sentiment labels.
    """
    print("Adding sentiment labels...")
    df_ai['sentiment_label'] = df_ai['compound'].apply(assign_sentiment_label)
    print(f"Sentiment labels added. Shape: {df_ai.shape}")
    return df_ai

def calculate_sentiment_fractions(df_ai):
    """
    Calculate the fractions of positive, negative, and neutral tweets.

    Parameters:
    df_ai (pd.DataFrame): The DataFrame containing AI-related tweets with sentiment labels.

    Returns:
    pd.Series: The fractions of each sentiment label.
    """
    print("Calculating sentiment fractions...")
    sentiment_counts = df_ai['sentiment_label'].value_counts()
    total_tweets = len(df_ai)
    sentiment_fractions = sentiment_counts / total_tweets
    print("Positive, negative, and neutral shares:")
    print(sentiment_fractions)
    return sentiment_fractions

def analyze_sentiment_by_year(df_ai):
    """
    Analyze sentiment distribution by year.

    Parameters:
    df_ai (pd.DataFrame): The DataFrame containing AI-related tweets with sentiment labels.

    Returns:
    pd.DataFrame: The sentiment fractions by year.
    """
    print("Analyzing sentiment by year...")
    df_ai['created_at'] = pd.to_datetime(df_ai['created_at'], errors='coerce')
    df_ai = df_ai.dropna(subset=['created_at'])  # Drop rows with invalid dates
    df_ai.loc[:, 'year'] = df_ai['created_at'].dt.year.astype(str)
    sentiment_fractions_by_year = (
        df_ai.groupby('year')['sentiment_label']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    print("Sentiment fractions by year:")
    print(sentiment_fractions_by_year)
    return sentiment_fractions_by_year

def plot_sentiment_distribution_by_year(sentiment_fractions_by_year, output_dir):
    """
    Plot the sentiment distribution by year.

    Parameters:
    sentiment_fractions_by_year (pd.DataFrame): The sentiment fractions by year.
    output_dir (str): The directory to save the plot.

    Returns:
    None
    """
    print("Plotting sentiment distribution by year...")
    output_file = os.path.join(output_dir, 'sentiment_distribution_by_year.png')
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Existing file {output_file} removed.")
    
    # Convert fractions to percentages
    sentiment_fractions_by_year = sentiment_fractions_by_year * 100
    
    ax = sentiment_fractions_by_year.plot(
        kind='bar',
        figsize=(12, 6),
        colormap='coolwarm',
        width=0.8
    )
    ax.set_xticks(np.arange(len(sentiment_fractions_by_year.index)))
    ax.set_xticklabels(sentiment_fractions_by_year.index, rotation=45, fontsize=16)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Proportion (%)', fontsize=16)
    plt.legend(title='Sentiment', labels=['Negative', 'Neutral', 'Positive'], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Sentiment distribution plot saved to {output_file}")

def plot_average_sentiment(df_ai, output_dir):
    """
    Plot the average sentiment per month-year.

    Parameters:
    df_ai (pd.DataFrame): The DataFrame containing the sentiment data.
    output_dir (str): The directory to save the plot.

    Returns:
    None
    """
    print("Plotting average sentiment per month-year...")

    output_file = os.path.join(output_dir, 'average_sentiment_per_month_year.png')
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Existing file {output_file} removed.")
    
    df_ai['created_at'] = pd.to_datetime(df_ai['created_at'], errors='coerce')
    df_ai = df_ai.dropna(subset=['created_at'])  # Drop rows with invalid dates
    df_ai.loc[:, 'year_month'] = df_ai['created_at'].dt.to_period('M').astype(str)
    
    # Calculate the average sentiment
    monthly_avg_sentiment = df_ai.groupby('year_month')['compound'].mean()
    
    plt.figure(figsize=(12, 10))
    plt.plot(monthly_avg_sentiment.index, monthly_avg_sentiment, label='Average Sentiment')
    
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Average sentiment (compound score)', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(output_file)
    plt.close()
    print(f"Average sentiment plot saved to {output_file}")

def generate_wordcloud_from_frequencies(frequencies, output_file):
    """
    Generate and save a word cloud image from word frequencies.

    Parameters:
    frequencies (dict): The word frequencies.
    output_file (str): The path to the output image file.

    Returns:
    None
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequencies)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Word cloud saved to {output_file}")

def plot_wordclouds(df_ai, output_dir):
    """
    Plot word clouds with words that make tweets positive or negative.

    Parameters:
    df_ai (pd.DataFrame): The DataFrame containing AI-related tweets.
    output_dir (str): The directory to save the word clouds.

    Returns:
    None
    """
    print("Generating word clouds for words that make tweets positive or negative...")

    # Initialize dictionaries to hold word frequencies
    positive_words = {}
    negative_words = {}

    # Analyze each tweet
    for text in df_ai['text']:
        if isinstance(text, str):
            words = text.split()
            for word in words:
                sentiment = sia.polarity_scores(word)
                if sentiment['compound'] > 0.05:
                    if word in positive_words:
                        positive_words[word] += 1
                    else:
                        positive_words[word] = 1
                elif sentiment['compound'] < -0.05:
                    if word in negative_words:
                        negative_words[word] += 1
                    else:
                        negative_words[word] = 1

    # Generate and save word clouds
    positive_wordcloud_file = os.path.join(output_dir, 'positive_wordcloud.png')
    negative_wordcloud_file = os.path.join(output_dir, 'negative_wordcloud.png')

    generate_wordcloud_from_frequencies(positive_words, positive_wordcloud_file)
    generate_wordcloud_from_frequencies(negative_words, negative_wordcloud_file)

    print("Word clouds generated and saved.")

def run_sentiment(input_file = "../preprocess/data_cleaned/cleaned.csv",
                  output_file = '../results/sentiment/sentiment.csv', 
                  classifier_path = '../results/classifier/trained_model_logistic.joblib',
                  vectorizer_path = '../results/classifier/vectorizer.joblib',
                  output_dir = '../results/sentiment',
                  sentiment_path=None):
    """
    Run the sentiment analysis pipeline.

    Parameters:
    input_file (str): The path to the input CSV file.
    output_file (str): The path to the output CSV file.
    classifier_path (str): The path to the trained classifier model.
    vectorizer_path (str): The path to the vectorizer.
    output_dir (str): The directory to save the results.
    sentiment_path (str, optional): The path to a precomputed sentiment CSV file.

    Returns:
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    if sentiment_path:
        df_ai = pd.read_csv(sentiment_path)
        print(f"Loaded sentiment data from {sentiment_path}")
        
    else:
        
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"Existing file {output_file} removed.")
        
        df = load_data(input_file)
        classifier = joblib.load(classifier_path)
        vectorizer = joblib.load(vectorizer_path)
        df_ai = filter_ai_tweets(df, classifier, vectorizer)
        df_ai = add_sentiment_scores(df_ai)
        save_data(df_ai, output_file)
    
    df_ai = add_sentiment_labels(df_ai)
    calculate_sentiment_fractions(df_ai)
    sentiment_fractions_by_year = analyze_sentiment_by_year(df_ai)
    plot_sentiment_distribution_by_year(sentiment_fractions_by_year, output_dir)
    plot_average_sentiment(df_ai, output_dir)
    plot_wordclouds(df_ai, output_dir)

if __name__ == "__main__":
    with open('output_log.txt', 'w') as f:
        with contextlib.redirect_stdout(f):
            run_sentiment()