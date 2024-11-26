# Replication Package for "The growing positivity toward AI: a multi-year sentiment analysis of Twitter posts"

## Overview

This repository contains the replication materials for the paper, *"The Growing Positivity Toward AI: A Multi-Year Sentiment Analysis of Twitter Posts"*. The study analyzes Twitter data from January 2019 to November 2022 to track public sentiment toward artificial intelligence (AI). It uses a cross-validated logistic regression classifier and dictionary-based sentiment analysis to classify tweets as positive, neutral, or negative and examines temporal trends in public opinion.

## Dataset

The dataset, sourced from [The Twitter Stream Archive](https://archive.org/details/twitterstream), is a collection of Twitter data in JSON format grabbed from the general twitter stream. For replication purposes, as mentioned in the setup section, we give out the tokenized dataset[here](https://we.tl/t-LN64E9jzeA).

## Project Goals

The project aims to:
1. Identify tweets discussing AI topics.
2. Use a logistic regression cross-validated classifier to determine the volume of posts discussing AI-related topics over time.
3. Conduct sentiment analysis to capture public perception and attitudes towards AI over time.

## Setup for Replication

1. Clone the repository:
   ```bash
   git clone <https://github.com/fcas8/fss_project>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. For replication purposes, download the tokenized dataset [here](https://we.tl/t-LN64E9jzeA) and place it in the `preprocess/data_tokenized/` folder.[^1]

[^1]: If instead you would like to replicate the entire pipeline, download compressed files from the Twitter Stream Archive (use of torrents is suggested) and place them in the `preprocess/data/` folder. Then, uncomment the commented part of the `main.py` file and execute.

## Project Structure and Pipeline

* `code/main.py`: Main script to run the entire pipeline.
* `code/preprocess.py`: Prepares the raw dataset by filtering out only the necessary records.
* `code/merge.py`: Merges datasets from different days/years.
* `code/tokenizer.py`: Tokenizes the text data.
* `code/clean.py`: Cleans the dataset by removing unwanted content.
* `code/classifier.py`: Contains functions to vectorize the text data and train the classifier.
* `code/sentiment.py`: Contains functions for performing sentiment analysis.

## Running the Analysis

Simply run the `main.py` script, which will execute the replication pipeline:

```bash
python code/main.py
```

## Results

The results of the analysis, including trained models, vectorizers, and sentiment analysis outputs, are stored in the `results/` directory:
* `results/classifier/`: Contains the trained classifier models and vectorizers.
* `results/sentiment/`: Contains the sentiment analysis results.

## Contributing

Feel free to submit issues or pull requests to improve the project.

## Contact

For any questions or inquiries, please contact Federico Casotto at federico.casotto@studbocconi.it.