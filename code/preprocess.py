""" In this script, I preprocess the zst file and filter out the records that I need.
I then save the filtered records to a CSV file. """

import zstandard as zstd
import json
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import os

def process_zst_file_to_csv(input_path, output_path, filter_func, chunk_size=1024*1024):
    """
    Processes a .zst file and converts it to a CSV file.

    Args:
        input_path (str): Path to the input .zst file.
        output_path (str): Path to the output CSV file.
        filter_func (function): A function to filter records. Should return True for records to keep.
        chunk_size (int, optional): Size of chunks to read from the .zst file. Defaults to 1MB.

    Returns:
        None
    """
    records = []
    total_size = os.path.getsize(input_path)
    print(f"Processing file: {input_path}")
    print(f"Total file size: {total_size / (1024 * 1024):.2f} MB")

    with open(input_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            buffer = b""
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Processing") as pbar:
                while True:
                    chunk = reader.read(chunk_size)
                    if not chunk:
                        break
                    buffer += chunk
                    pbar.update(len(chunk))
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        try:
                            record = json.loads(line)
                            if filter_func(record):
                                filtered_record = {
                                    'id': record.get('id'),
                                    'subreddit': record.get('subreddit'),
                                    'author': record.get('author'),
                                    'title': record.get('title'),
                                    'selftext': record.get('selftext'),
                                    'created_utc': record.get('created_utc')
                                }
                                records.append(filtered_record)
                        except json.JSONDecodeError:
                            continue

    print(f"Writing {len(records)} records to CSV file: {output_path}")
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print("Processing complete.")

def select_file():
    """
    Opens a file dialog to select a file.

    Returns:
        str: Path to the selected file.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

# Define the input and output path
input_path = select_file()
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
output_path = os.path.join(output_dir, 'processed_posts.csv')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Preprocess the zst file
input_path = select_file()
process_zst_file_to_csv(input_path, output_path, lambda x: True)