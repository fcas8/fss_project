"""
Classifier Training Script

This script trains classifiers on text data by vectorizing features and balancing classes. 
It supports model training, evaluation, and visualization of results, including top features and temporal trends.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample
from wordcloud import WordCloud
import joblib
import shutil
import contextlib

def load_data(input_path):
    """
    Load and clean the dataset.

    Parameters:
    input_path (str): The path to the input CSV file.

    Returns:
    pd.DataFrame: The cleaned DataFrame containing only relevant rows for classification.
    """
    print("Loading data...")
    df = pd.read_csv(input_path)
    df = df.dropna()
    print(f"Data loaded. Shape: {df.shape}")

    # Convert the 'label' column to numeric, setting errors='coerce' to convert non-numeric values to NaN
    df['label_numeric'] = pd.to_numeric(df['label'], errors='coerce')

    # Filter the dataframe to get rows where 'label_numeric' is 0 or 1
    df = df[df['label_numeric'].isin([0, 1])]

    # Drop rows with NaN values
    df = df.dropna()
    print(f"Data filtered. Shape: {df.shape}")

    # Use the numeric labels for further processing
    df['label'] = df['label_numeric']
    df = df.drop(columns=['label_numeric'])

    return df

def split_and_vectorize(df):
    """
    Split the data into training and test sets, balance the training set, and vectorize the text data.

    Parameters:
    df (pd.DataFrame): The DataFrame containing text data and labels.

    Returns:
    tuple: Contains the vectorized training features, test features, training labels, test labels, and the vectorizer.
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

    # Configure tf-idf vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),  
        max_df=0.9999,      
        min_df=0.0001,     
        stop_words='english'
    )

    X_train_balanced_counts = vectorizer.fit_transform(X_train_balanced.squeeze())
    X_test_counts = vectorizer.transform(X_test)

    print(f"Data split and vectorized. Number of features: {X_train_balanced_counts.shape[1]}")
    return X_train_balanced_counts, X_test_counts, Y_train_balanced, Y_test, vectorizer

def train_classifier(X_train_balanced_counts, Y_train_balanced, model_type='logistic'):
    """
    Train a classifier on the given dataset.

    Parameters:
    X_train_balanced_counts (scipy.sparse.csr.csr_matrix): The vectorized training features.
    Y_train_balanced (pd.Series): The training labels.
    model_type (str): The type of classifier to train. Options include 'logistic', 'random forest', 'svm', etc.

    Returns:
    sklearn.base.BaseEstimator: The trained classifier.
    """
    print(f"Training {model_type} classifier...")
    if model_type == 'logistic':
        clf = LogisticRegressionCV(cv=5, max_iter=1000)
    elif model_type == 'random forest':
        clf = RandomForestClassifier(n_estimators=100)
    elif model_type == 'svm':
        clf = SVC(kernel='linear', probability=True)
    elif model_type == 'naive bayes':
        clf = MultinomialNB()
    elif model_type == 'lasso':
        clf = LogisticRegressionCV(cv=5, max_iter=1000, penalty='l1', solver='saga')
    elif model_type == 'elastic net':
        clf = LogisticRegressionCV(cv=5, max_iter=1000, penalty='elasticnet', solver='saga', l1_ratios=[0.8])
    clf.fit(X_train_balanced_counts, Y_train_balanced)
    print(f"{model_type.capitalize()} classifier trained.")
    return clf

def evaluate_classifier(clf, X_test_counts, Y_test, model_type='logistic'):
    """
    Evaluate the performance of the classifier.

    Parameters:
    clf (sklearn.base.BaseEstimator): The trained classifier.
    X_test_counts (scipy.sparse.csr.csr_matrix): The vectorized test features.
    Y_test (pd.Series): The test labels.
    model_type (str): The type of classifier being evaluated.

    Returns:
    float: The accuracy of the classifier.
    """
    print(f"Evaluating {model_type} classifier...")
    Y_pred = clf.predict(X_test_counts)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'Accuracy: {accuracy}')
    return accuracy

def get_top_features(clf, vectorizer, top_n=300):
    """
    Retrieve the top N features based on importance scores from the trained classifier.

    Parameters:
    clf (sklearn.base.BaseEstimator): The trained classifier.
    vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): The vectorizer used for feature extraction.
    top_n (int): The number of top features to retrieve.

    Returns:
    tuple: Contains two lists - top feature names and their corresponding importance scores.
    """
    print(f"Extracting top features for model: {type(clf).__name__}")

    # For linear models (e.g., Logistic Regression, SVM with linear kernel)
    if hasattr(clf, "coef_"):
        ai_class_index = 1  # Class index to focus on (binary classification)
        feature_coeffs = clf.coef_[0] if clf.coef_.shape[0] == 1 else clf.coef_[ai_class_index]
        sorted_indices = np.argsort(np.abs(feature_coeffs))[-top_n:][::-1]
        top_feature_names = vectorizer.get_feature_names_out()[sorted_indices]
        top_feature_scores = feature_coeffs[sorted_indices]

    # For tree-based models (e.g., Random Forest)
    elif hasattr(clf, "feature_importances_"):
        feature_importances = clf.feature_importances_
        sorted_indices = np.argsort(feature_importances)[-top_n:][::-1]
        top_feature_names = vectorizer.get_feature_names_out()[sorted_indices]
        top_feature_scores = feature_importances[sorted_indices]

    # For Naive Bayes models (e.g., MultinomialNB)
    elif hasattr(clf, "feature_log_prob_"):
        feature_log_prob = clf.feature_log_prob_
        class_diff = feature_log_prob[1] - feature_log_prob[0]  # Difference between classes
        sorted_indices = np.argsort(np.abs(class_diff))[-top_n:][::-1]
        top_feature_names = vectorizer.get_feature_names_out()[sorted_indices]
        top_feature_scores = class_diff[sorted_indices]

    else:
        raise ValueError(f"Feature extraction not supported for classifier: {type(clf).__name__}")

    return top_feature_names, top_feature_scores

def save_top_features(top_feature_names, top_feature_scores, output_dir, model_type='logistic'):
    """
    Save the top features to a CSV file.

    Parameters:
    top_feature_names (list): The names of the top features.
    top_feature_scores (list): The importance scores of the top features.
    output_dir (str): The directory to save the file.
    model_type (str): The type of model for which the features are being saved.

    Returns:
    None
    """
    print("Saving top features...")
    top_features_df = pd.DataFrame({
        'Feature': top_feature_names,
        'Score': top_feature_scores
    })
    top_features_df.to_csv(os.path.join(output_dir, f'top_features_{model_type}.csv'), index=False)
    print("Top features saved.")

def generate_wordcloud(top_feature_names, top_feature_scores, output_dir, model_type='logistic'):
    """
    Generate and save a word cloud visualization for the top features.

    Parameters:
    top_feature_names (list): The names of the top features.
    top_feature_scores (list): The importance scores of the top features.
    output_dir (str): The directory to save the word cloud image.
    model_type (str): The type of model used to determine the features.

    Returns:
    None
    """
    print("Generating word cloud...")
    positive_features = {name: score for name, score in zip(top_feature_names, top_feature_scores) if score > 0}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(positive_features)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'wordcloud_{model_type}.png'))
    plt.close()
    print("Word cloud generated and saved.")

def plot_probabilities(df, clfs, vectorizer, output_dir):
    """
    Plot the average probability of posts belonging to the AI class over time for all classifiers.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data and timestamps.
    clfs (dict): Dictionary of trained classifiers, keyed by model type.
    vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): The vectorizer used for feature extraction.
    output_dir (str): The directory to save the probability plot.

    Returns:
    None
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

    plt.xlabel('Month')
    plt.ylabel('Average Probability')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'probability_plot.png'))
    plt.close()
    print("Probabilities plotted and saved.")

def plot_roc_curves(clfs, X_test_counts, Y_test, output_dir):
    """
    Plot the ROC curves for all classifiers.

    Parameters:
    clfs (dict): Dictionary of trained classifiers, keyed by model type.
    X_test_counts (scipy.sparse.csr.csr_matrix): The vectorized test features.
    Y_test (pd.Series): The test labels.
    output_dir (str): The directory to save the ROC curve plot.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))

    for model_type, clf in clfs.items():
        # Predict probabilities for class 1
        y_pred = clf.predict_proba(X_test_counts)[:, 1]
        fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
        auc = roc_auc_score(Y_test, y_pred)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{model_type.capitalize()} (AUC = {auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random chance diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()
    print("ROC curves plotted and saved.")

def print_confusion_matrix(clf, X_test_counts, Y_test, output_dir, model_type='logistic'):
    """
    Generate and save the confusion matrix for the specified classifier.

    Parameters:
    clf (sklearn.base.BaseEstimator): The trained classifier.
    X_test_counts (scipy.sparse.csr.csr_matrix): The vectorized test features.
    Y_test (pd.Series): The test labels.
    output_dir (str): The directory to save the confusion matrix.
    model_type (str): The type of classifier being evaluated.

    Returns:
    None
    """
    print(f"Confusion matrix for {model_type} classifier:")
    y_pred = clf.predict(X_test_counts)
    cm = confusion_matrix(Y_test, y_pred)
    
    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='Blues')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_type}.png'))
    plt.close()
    print(f"Confusion matrix for {model_type} saved.")

def plot_high_probability_percentage(df, clfs, vectorizer, output_dir):
    """
    Plot the percentage of tweets with high probabilities of belonging to the AI class over time.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data and timestamps.
    clfs (dict): Dictionary of trained classifiers, keyed by model type.
    vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): The vectorizer used for feature extraction.
    output_dir (str): The directory to save the high-probability percentage plot.

    Returns:
    None
    """
    print("Plotting high probability percentage...")

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

        # Calculate the percentage of tweets with probability > 80%
        df[f'high_prob_{model_type}'] = df[f'probability_{model_type}'] > 0.8
        monthly_high_prob = df.groupby('year_month')[f'high_prob_{model_type}'].mean() * 100

        # Plot the results
        plt.plot(monthly_high_prob.index.astype(str), monthly_high_prob, label=f'{model_type.capitalize()}')

    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Proportion of AI-related tweets (%)', fontsize=16)
    plt.grid(True)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'high_probability_percentage_plot.png'))
    plt.close()
    print("High probability percentage plotted and saved.")

def vectorize_train(input_path='../preprocess/data_cleaned/cleaned.csv', output_dir='../results/classifier/'):
    """
    Execute the complete pipeline: load data, vectorize text, train classifiers, and generate outputs.

    Parameters:
    input_path (str): Path to the preprocessed and cleaned data file.
    output_dir (str): Directory to save the trained models, vectorizer, and other outputs.

    Returns:
    None
    """
    
    # Ensure the output directory exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data(input_path)
    X_train_balanced_counts, X_test_counts, Y_train_balanced, Y_test, vectorizer = split_and_vectorize(df)
    
    classifiers = {
        'logistic': 'logistic',
        # 'naive bayes': 'naive bayes',
        # 'elastic net': 'elastic net', Not converging
        # 'lasso': 'lasso', Not converging
        # 'random forest': 'random forest', Low AUC
        # 'svm': 'svm' Low AUC
    }
    trained_classifiers = {}
    results = {}
    results_auc = {}

    for clf_type in classifiers:
        clf = train_classifier(X_train_balanced_counts, Y_train_balanced, model_type=clf_type)
        trained_classifiers[clf_type] = clf 
        accuracy = evaluate_classifier(clf, X_test_counts, Y_test, model_type=clf_type)
        results[clf_type] = accuracy
        y_pred = clf.predict_proba(X_test_counts)[:, 1]
        fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
        auc = roc_auc_score(Y_test, y_pred)
        results_auc[clf_type] = auc
        print(f'AUC: {auc:.2f}')

    # Find the best model
    best_classifier_type = max(results_auc, key=results_auc.get)
    best_classifier = trained_classifiers[best_classifier_type]
    best_auc = results_auc[best_classifier_type]
    
    # Generate and save top feature names and word cloud with the best classifier
    top_feature_names, top_feature_scores = get_top_features(best_classifier, vectorizer)
    save_top_features(top_feature_names, top_feature_scores, output_dir, model_type=best_classifier_type)
    generate_wordcloud(top_feature_names, top_feature_scores, output_dir, model_type=best_classifier_type)
    
    # # Plot ROC curves for all models
    # plot_roc_curves(trained_classifiers, X_test_counts, Y_test, output_dir)

    # # Print and save confusion matrix for the best classifier
    # print_confusion_matrix(best_classifier, X_test_counts, Y_test, output_dir, model_type=best_classifier_type)
    
    # # Plot probabilities for all models
    # plot_probabilities(df, trained_classifiers, vectorizer, output_dir)

    # Plot high probability percentage for all models
    plot_high_probability_percentage(df, trained_classifiers, vectorizer, output_dir)

    # Save the trained models and vectorizer
    for model_type, clf in trained_classifiers.items():
        model_path = os.path.join(output_dir, f'trained_model_{model_type}.joblib')
        joblib.dump(clf, model_path)
        print(f"Model saved to {model_path}")
    
    vectorizer_path = os.path.join(output_dir, 'vectorizer.joblib')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    with open('output_log.txt', 'w') as f:
        with contextlib.redirect_stdout(f):
            vectorize_train()