"""
Dataset Merger Script

This script provides utilities to load datasets from various sources, filter the content, merge them,
and return it in a standardized format for further processing.

"""

import os
import pandas as pd
import glob
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def process_and_save_chunk(file_path, output_path, filters=None, chunk_size=100000):
    """
    Process a single file in chunks and append the results to the output file.

    Parameters:
    file_path (str): Path to the input CSV file.
    output_path (str): Path to the output CSV file.
    filters (dict, optional): Filters to apply to the dataset.
    chunk_size (int, optional): Number of rows per chunk.

    Returns:
    None
    """
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        if filters:
            chunk = filter_dataset(chunk, filters)
        if not os.path.isfile(output_path):
            chunk.to_csv(output_path, index=False, mode='w')
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)

def filter_dataset(df, filters=None):
    """
    Filter the dataset based on specified conditions.

    Parameters:
    df (pd.DataFrame): The dataset to filter.
    filters (dict, optional): A dictionary of column names and tuples to filter by.
                              Tuples should be in the form (operator, value) where operator is one of
                              '==', '!=', '>', '>=', '<', '<='.

    Returns:
    pd.DataFrame: The filtered dataset.
    """
    if filters is None:
        filters = {}

    try:
        for column, condition in filters.items():
            if column in df.columns:
                if isinstance(condition, tuple) and len(condition) == 2:
                    operator, value = condition
                    if operator == '==':
                        df = df[df[column] == value]
                    elif operator == '!=':
                        df = df[df[column] != value]
                    elif operator == '>':
                        df = df[df[column] > value]
                    elif operator == '>=':
                        df = df[df[column] >= value]
                    elif operator == '<':
                        df = df[df[column] < value]
                    elif operator == '<=':
                        df = df[df[column] <= value]
                    else:
                        raise ValueError(f"Unsupported operator: {operator}")
                else:
                    df = df[df[column] == condition]
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to filter dataset. Error: {e}")

def merge_data(year, filters=None):
    """
    Full pipeline: Load preprocessed datasets for a given year, merge them, apply filters, and save the output.

    Parameters:
    year (int): The year of the datasets to merge.
    filters (dict, optional): Filters to apply to the dataset.

    Returns:
    None
    """
    # Define the input and output directories
    input_dir = os.path.join('..', 'preprocess', 'data_preprocessed', str(year))
    output_dir = os.path.join('..', 'preprocess', 'data_merged')
    output_path = os.path.join(output_dir, f'{year}_merged.csv')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all CSV files in the input directory
    file_paths = glob.glob(os.path.join(input_dir, '*.csv'))

    if not file_paths:
        raise FileNotFoundError(f"No files found for the year {year} in {input_dir}.")

    # Process and save each file in chunks
    for file_path in tqdm(file_paths, desc="Processing files"):
        process_and_save_chunk(file_path, output_path, filters)

    print(f"Datasets for year {year} merged and saved successfully to {output_path}.")

def merge_all_years(output_path=None, chunk_size=100000):
    """
    Merge all yearly merged files from the data_merged folder into a single dataset.

    Parameters:
    output_path (str, optional): Path to save the final merged dataset. Defaults to 'data_merged/merged.csv'.
    chunk_size (int, optional): Number of rows per chunk.

    Returns:
    pd.DataFrame: The final merged dataset.
    """
    # Define the input directory
    input_dir = os.path.join('..', 'preprocess', 'data_merged')

    # Set default output path if not specified
    if output_path is None:
        output_path = os.path.join(input_dir, 'merged.csv')

    # Get all CSV files in the input directory that start with a year
    file_paths = glob.glob(os.path.join(input_dir, '[0-9][0-9][0-9][0-9]_merged.csv'))

    if not file_paths:
        raise FileNotFoundError(f"No merged files found in {input_dir}.")

    # Process and save each file in chunks
    for file_path in tqdm(file_paths, desc="Merging yearly files"):
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            if not os.path.isfile(output_path):
                chunk.to_csv(output_path, index=False, mode='w')
            else:
                chunk.to_csv(output_path, index=False, mode='a', header=False)

    print(f"Final merged dataset saved to {output_path}")

    # Load the final merged dataset to return it
    final_merged_df = pd.read_csv(output_path)
    return final_merged_df