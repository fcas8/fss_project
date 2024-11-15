"""
Dataset Merger Script

This script provides utilities to load datasets from various sources, filter the content, merge them,
and return it in a standardized format for further processing.

"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def load_datasets(file_paths=None):
    """
    Load datasets from multiple CSV files. If no paths are provided, opens a file dialog for user selection.

    Parameters:
    file_paths (list of str, optional): List of paths to the dataset files.

    Returns:
    pd.DataFrame: Merged dataset.
    """
    if file_paths is None:
        root = tk.Tk()
        file_paths = filedialog.askopenfilenames(title="Select dataset files",
                                                 filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if not file_paths:
            raise FileNotFoundError("No files were selected.")

    try:
        dfs = [pd.read_csv(file_path) for file_path in file_paths]
        merged_df = pd.concat(dfs, ignore_index=True)
        print(f"Datasets loaded and merged successfully. Shape: {merged_df.shape}")
        return merged_df
    except Exception as e:
        raise RuntimeError(f"Failed to load and merge datasets. Error: {e}")

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
        print(f"Dataset filtered successfully. Shape after filtering: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to filter dataset. Error: {e}")

def save_dataset(df, output_path):
    """
    Save the dataset to a specified CSV file.

    Parameters:
    df (pd.DataFrame): The dataset to save.
    output_path (str): Path to save the dataset.

    Returns:
    None
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset saved successfully to {output_path}.")
    except Exception as e:
        raise RuntimeError(f"Failed to save dataset to {output_path}. Error: {e}")

def merge_data(file_paths=None, filters=None, output_path=None):
    """
    Full pipeline: Load datasets, merge them, apply filters, and save the output.

    Parameters:
    file_paths (list of str, optional): List of paths to the dataset files.
    filters (dict, optional): Filters to apply to the dataset.
    output_path (str, optional): Path to save the filtered dataset.

    Returns:
    pd.DataFrame: The filtered dataset.
    """
    df = load_datasets(file_paths)
    if filters:
        df = filter_dataset(df, filters)
    if output_path:
        save_dataset(df, output_path)
    return df