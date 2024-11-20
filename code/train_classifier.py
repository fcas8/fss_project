import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from wordcloud import WordCloud
from tqdm import tqdm
import joblib

# Set the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_data(input_path):
    """
    Load and clean the dataset.
    """
    print("Loading data...")
    df = pd.read_csv(input_path)
    df = df.dropna()
    print(f"Data loaded. Shape: {df.shape}")
    return df

def split_and_vectorize(df):
    """
    Split the data into training and test sets, balance the training set, and vectorize the text data.
    """
    print("Splitting and vectorizing data...")

    # Stopwords for vectorizer
    stopwords = [
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
        "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could",
        "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
        "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
        "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
        "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
        "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
        "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
        "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
        "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
        "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
        "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
        "yourselves"
    ]

    # Split data into features and labels
    X = df['text']
    Y = df['label']

    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Combine training data back for resampling
    train_data = pd.concat([X_train, Y_train], axis=1)

    # Separate majority and minority classes
    class_0 = train_data[train_data['label'] == 0]
    class_1 = train_data[train_data['label'] == 1]

    # Downsample majority class
    class_0_downsampled = resample(class_0, 
                                   replace=False,    # sample without replacement
                                   n_samples=len(class_1),     # to match minority class
                                   random_state=42) # reproducible results

    # Combine minority class with downsampled majority class
    train_data_balanced = pd.concat([class_0_downsampled, class_1])

    # Splitting the features and label variable for balanced data
    X_train_balanced = train_data_balanced.drop('label', axis=1)
    Y_train_balanced = train_data_balanced['label']

    # Vectorize the text data
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_df=0.9999, min_df=0.0001, stop_words=stopwords)

    # Vectorize the balanced training features
    X_train_balanced_counts = vectorizer.fit_transform(X_train_balanced.squeeze())

    # Vectorize the test features
    X_test_counts = vectorizer.transform(X_test)

    print("Data split and vectorized.")
    return X_train_balanced_counts, X_test_counts, Y_train_balanced, Y_test, vectorizer

def train_classifier(X_train_balanced_counts, Y_train_balanced):
    """
    Train a Logistic Regression CV classifier.
    """
    print("Training classifier...")
    clf = LogisticRegressionCV(cv=5, max_iter=1000)
    clf.fit(X_train_balanced_counts, Y_train_balanced)
    print("Classifier trained.")
    return clf

def evaluate_classifier(clf, X_test_counts, Y_test):
    """
    Evaluate the classifier and print the accuracy.
    """
    print("Evaluating classifier...")
    Y_pred = clf.predict(X_test_counts)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'Accuracy: {accuracy}')

def get_top_features(clf, vectorizer, top_n=50):
    """
    Get the top N features with the highest absolute coefficients.
    """
    ai_class_index = 1

    # Get the coefficients of the features for the class of interest
    feature_coeffs = clf.coef_[0] if clf.coef_.shape[0] == 1 else clf.coef_[ai_class_index]

    # Get the indices of the top N features with the highest absolute coefficients
    top_n_indices = np.argsort(np.abs(feature_coeffs))[-top_n:][::-1]

    # Get the corresponding feature names and their scores
    top_feature_names = vectorizer.get_feature_names_out()[top_n_indices]
    top_feature_scores = feature_coeffs[top_n_indices]

    return top_feature_names, top_feature_scores

def save_top_features(top_feature_names, top_feature_scores, output_dir):
    """
    Save the top features to a CSV file.
    """
    print("Saving top features...")
    top_features_df = pd.DataFrame({
        'Feature': top_feature_names,
        'Score': top_feature_scores
    })
    top_features_df.to_csv(os.path.join(output_dir, 'top_features.csv'), index=False)
    print("Top features saved.")

def generate_wordcloud(top_feature_names, top_feature_scores, output_dir):
    """
    Generate and save a word cloud image of the top features.
    """
    print("Generating word cloud...")
    positive_features = {name: score for name, score in zip(top_feature_names, top_feature_scores) if score > 0}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(positive_features)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, 'wordcloud.png'))
    plt.close()
    print("Word cloud generated and saved.")

def plot_probabilities(df, clf, vectorizer, output_dir):
    """
    Plot and save the average probability of posts belonging to the AI class over time.
    """
    print("Plotting probabilities...")

    # Ensure 'created_at' is in datetime format
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Extract the month and year part
    df['year_month'] = df['created_at'].dt.to_period('M')

    # Predict probabilities for class 1
    X_counts = vectorizer.transform(df['text'])  # Vectorize the entire dataset
    probabilities = clf.predict_proba(X_counts)[:, 1]  # Getting the probability for class 1
    df['probability'] = probabilities

    # Group by year and month and calculate the monthly average probabilities
    monthly_average_probabilities = df.groupby('year_month').probability.mean()

    # Plot the results
    plt.figure(figsize=(12, 6))
    monthly_average_probabilities.plot(kind='line')
    plt.title('Average Probability of Posts Belonging to Class "AI" - CV Logistic Regression')
    plt.xlabel('Month')
    plt.ylabel('Average Probability')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'probability_plot.png'))
    plt.close()
    print("Probabilities plotted and saved.")

def vectorize_train(output_dir='../results'):
    input_path = '../preprocess/data_tokenized/tokenized.csv'
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data(input_path)
    X_train_balanced_counts, X_test_counts, Y_train_balanced, Y_test, vectorizer = split_and_vectorize(df)
    clf = train_classifier(X_train_balanced_counts, Y_train_balanced)
    evaluate_classifier(clf, X_test_counts, Y_test)
    top_feature_names, top_feature_scores = get_top_features(clf, vectorizer)
    save_top_features(top_feature_names, top_feature_scores, output_dir)
    generate_wordcloud(top_feature_names, top_feature_scores, output_dir)
    plot_probabilities(df, clf, vectorizer, output_dir)
    
    # Save the trained model and vectorizer
    model_path = os.path.join(output_dir, 'trained_model.joblib')
    vectorizer_path = os.path.join(output_dir, 'vectorizer.joblib')
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    vectorize_train()