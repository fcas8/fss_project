# AI Sentiment Analysis in Political Discussions

This project aims to analyze public sentiment surrounding AI-related topics within the subreddit r/PoliticalDiscussion. Using a dataset containing all posts and comments, we plan to extract, process, and interpret user sentiment toward AI.

## Dataset

The dataset, sourced from [Academic Torrents](https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10), includes historical data from the r/PoliticalDiscussion subreddit. It contains posts and comments that will serve as the foundation for this sentiment analysis.

## Project Goals

The project will attempt to:
1. Identify posts and comments discussing AI topics.
2. Establish, through a logistic regression cross-validated classifier, the volume of posts discussing AI-related topics over time.
3. Conduct sentiment analysis to capture public perception and attitudes towards AI.
4. Explore trends or shifts in sentiment over time, as available in the dataset.

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
2. Install dependencies:
3. Download and prepare the dataset as instructed on Academic Torrents.
...

## Project Structure

* `preprocess.py`: Prepares the raw dataset by filtering out only the records needed.
* `logisticregression.py`: Contains functions to train the classifier.
* `sentiment_analysis.py`: Contains functions for performing sentiment analysis.
* `utils.py`: Additional utilities for data handling and cleaning.

## Contributing

Feel free to submit issues or pull requests to improve the project.
