import zstandard as zstd
import json
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import os

os.chdir('D:\\reddit\\reddit\\code\\')

def process_zst_file_to_csv(input_path, output_path, filter_func, chunk_size=1024*1024):
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
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

# Example usage
input_path = select_file()
output_path = 'processed_posts.csv'
process_zst_file_to_csv(input_path, output_path, lambda x: True)