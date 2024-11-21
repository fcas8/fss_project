import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from wordcloud import WordCloud
from tqdm import tqdm
import joblib
import shutil

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

    # Vectorize the text data using built-in stop words
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_df=0.9999, min_df=0.0001, stop_words='english')

    # Vectorize the balanced training features
    X_train_balanced_counts = vectorizer.fit_transform(X_train_balanced.squeeze())

    # Vectorize the test features
    X_test_counts = vectorizer.transform(X_test)

    print(f"Data split and vectorized. Number of features: {X_train_balanced_counts.shape[1]}")
    return X_train_balanced_counts, X_test_counts, Y_train_balanced, Y_test, vectorizer

def train_classifier(X_train_balanced_counts, Y_train_balanced, model_type='logistic'):
    """
    Train a classifier.
    """
    print(f"Training {model_type} classifier...")
    if model_type == 'logistic':
        clf = LogisticRegressionCV(cv=5, max_iter=1000)
    elif model_type == 'elasticnet':
        clf = ElasticNetCV(cv=5, max_iter=1000)
    elif model_type == 'lasso':
        clf = LassoCV(cv=5, max_iter=1000)
    clf.fit(X_train_balanced_counts, Y_train_balanced)
    print(f"{model_type.capitalize()} classifier trained.")
    return clf

def evaluate_classifier(clf, X_test_counts, Y_test, model_type='logistic'):
    """
    Evaluate the classifier and print the accuracy.
    """
    print(f"Evaluating {model_type} classifier...")
    Y_pred = clf.predict(X_test_counts)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'{model_type.capitalize()} Accuracy: {accuracy}')

def get_top_features(clf, vectorizer, top_n=300):
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

def save_top_features(top_feature_names, top_feature_scores, output_dir, model_type='logistic'):
    """
    Save the top features to a CSV file.
    """
    print(f"Saving top features for {model_type}...")
    top_features_df = pd.DataFrame({
        'Feature': top_feature_names,
        'Score': top_feature_scores
    })
    top_features_df.to_csv(os.path.join(output_dir, f'top_features_{model_type}.csv'), index=False)
    print(f"Top features for {model_type} saved.")

def generate_wordcloud(top_feature_names, top_feature_scores, output_dir, model_type='logistic'):
    """
    Generate and save a word cloud image of the top features.
    """
    print(f"Generating word cloud for {model_type}...")
    positive_features = {name: score for name, score in zip(top_feature_names, top_feature_scores) if score > 0}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(positive_features)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, f'wordcloud_{model_type}.png'))
    plt.close()
    print(f"Word cloud for {model_type} generated and saved.")

def plot_probabilities(df, clfs, vectorizer, output_dir):
    """
    Plot and save the average probability of posts belonging to the AI class over time for all models.
    """
    print("Plotting probabilities...")

    # Ensure 'created_at' is in datetime format
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Extract the month and year part
    df['year_month'] = df['created_at'].dt.to_period('M')

    plt.figure(figsize=(12, 6))

    for model_type, clf in clfs.items():
        # Predict probabilities for class 1
        X_counts = vectorizer.transform(df['text'])  # Vectorize the entire dataset
        probabilities = clf.predict_proba(X_counts)[:, 1]  # Getting the probability for class 1
        df[f'probability_{model_type}'] = probabilities

        # Group by year and month and calculate the monthly average probabilities
        monthly_average_probabilities = df.groupby('year_month')[f'probability_{model_type}'].mean()

        # Plot the results
        monthly_average_probabilities.plot(kind='line', label=f'{model_type.capitalize()}')

    plt.title('Average Probability of Posts Belonging to Class "AI"')
    plt.xlabel('Month')
    plt.ylabel('Average Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'probability_plot.png'))
    plt.close()
    print("Probabilities plotted and saved.")

def vectorize_train(input_path = '../preprocess/data_tokenized/tokenized.csv', output_dir='../results'):
    
    # Ensure the output directory exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data(input_path)
    X_train_balanced_counts, X_test_counts, Y_train_balanced, Y_test, vectorizer = split_and_vectorize(df)
    
    # Train and evaluate Logistic Regression model
    clf_logistic = train_classifier(X_train_balanced_counts, Y_train_balanced, model_type='logistic')
    evaluate_classifier(clf_logistic, X_test_counts, Y_test, model_type='logistic')
    top_feature_names, top_feature_scores = get_top_features(clf_logistic, vectorizer)
    save_top_features(top_feature_names, top_feature_scores, output_dir, model_type='logistic')
    generate_wordcloud(top_feature_names, top_feature_scores, output_dir, model_type='logistic')
    
    # Train and evaluate Elastic Net model
    clf_elasticnet = train_classifier(X_train_balanced_counts, Y_train_balanced, model_type='elasticnet')
    evaluate_classifier(clf_elasticnet, X_test_counts, Y_test, model_type='elasticnet')
    top_feature_names, top_feature_scores = get_top_features(clf_elasticnet, vectorizer)
    save_top_features(top_feature_names, top_feature_scores, output_dir, model_type='elasticnet')
    generate_wordcloud(top_feature_names, top_feature_scores, output_dir, model_type='elasticnet')
    
    # Train and evaluate Lasso model
    clf_lasso = train_classifier(X_train_balanced_counts, Y_train_balanced, model_type='lasso')
    evaluate_classifier(clf_lasso, X_test_counts, Y_test, model_type='lasso')
    top_feature_names, top_feature_scores = get_top_features(clf_lasso, vectorizer)
    save_top_features(top_feature_names, top_feature_scores, output_dir, model_type='lasso')
    generate_wordcloud(top_feature_names, top_feature_scores, output_dir, model_type='lasso')
    
    # Plot probabilities for all models
    clfs = {'logistic': clf_logistic, 'elasticnet': clf_elasticnet, 'lasso': clf_lasso}
    plot_probabilities(df, clfs, vectorizer, output_dir)
    
    # Save the trained models and vectorizer
    for model_type, clf in clfs.items():
        model_path = os.path.join(output_dir, f'trained_model_{model_type}.joblib')
        joblib.dump(clf, model_path)
        print(f"Model saved to {model_path}")
    
    vectorizer_path = os.path.join(output_dir, 'vectorizer.joblib')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    vectorize_train()