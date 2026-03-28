from torch.utils.data import Dataset
import torch
import pandas as pd
import glob
import os

# use this function to convert entire train/test tokenized datasets from json to parquet for faster loading in training loop
def convert_json_to_parquet(json_folder, output_file):
    # 1. Grab all JSON file paths
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    
    # 2. Read them into a list of dictionaries
    # (Using a list comprehension is faster for thousands of files)
    data_list = [pd.read_json(f, typ='series') for f in json_files]
    
    # 3. Create a DataFrame
    df = pd.DataFrame(data_list)
    
    # 4. Save as Parquet with Snappy compression (very fast)
    df.to_parquet(output_file, engine='pyarrow', compression='snappy')
    print(f"Success! Converted {len(json_files)} files into {output_file}")

# Usage
#convert_json_to_parquet("./my_json_data", "dataset.parquet")
class MidiDataset(Dataset):
    def __init__(self, file_paths, labels, preload_to_ram=False):
        self.file_paths = file_paths
        self.labels = labels
        self.preload_to_ram = preload_to_ram
        self.midis = []

        if self.preload_to_ram:
            print("Pre-loading dataset to RAM... (Ensure you have enough memory!)")
            for path in self.file_paths:
                # Load and convert to tensor immediately
                pass
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Pull from RAM if available
        if self.preload_to_ram:
            image = self.images[idx]
        # 2. Otherwise, pull "File-by-File" from Disk
        else:
            path = self.file_paths[idx]

        label = self.labels[idx]
        return image, label