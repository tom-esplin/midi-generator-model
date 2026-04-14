from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import glob
import os
import orjson
import numpy as np  
def convert_json_to_parquet(json_folder, output_file):
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    data_list = []
    
    # 1. Use orjson for reading (much faster than pd.read_json for thousands of files)
    for f in json_files:
        with open(f, 'rb') as file:
            data = orjson.loads(file.read())
            data_list.append(data["ids"])
            
    # 2. Store as a single column to prevent Pandas from making thousands of NaN-padded columns
    df = pd.DataFrame({'ids': data_list})
    
    # 3. Save as Parquet
    df.to_parquet(output_file, engine='pyarrow', compression='snappy')
    print(f"Success! Converted {len(json_files)} files into {output_file}")


class MidiDataset(Dataset):
    def __init__(self, folder_path, preload_to_ram=False, parquet_path=None,chunk_size=1000, pad_token=0):
        self.file_paths = glob.glob(os.path.join(folder_path, "*.json"))
        self.folder_path = folder_path
        self.preload_to_ram = preload_to_ram
        self.parquet_path = parquet_path
        self.chunk_size = chunk_size + 1
        self.pad_token = pad_token
        self.midis = None
        
        if self.parquet_path and os.path.exists(self.parquet_path):
            print(f"Loading dataset from Parquet: {self.parquet_path}")
            # Extract the single column to a pure Python list of lists
            self.midis = pd.read_parquet(self.parquet_path)['ids'].tolist()
            self.preload_to_ram = True
            print(f"\nMIDI data length for item 0: {len(self.midis[0])}")
            
        elif self.preload_to_ram:
            print("Pre-loading dataset to RAM... (Ensure you have enough memory!)")
            pq_path = os.path.join(folder_path, "dataset.parquet")
            if not os.path.exists(pq_path):
                convert_json_to_parquet(folder_path, pq_path)
            self.midis = pd.read_parquet(pq_path)['ids'].tolist()

    def __len__(self):
        # Base length on RAM data if available, otherwise file paths
        if self.preload_to_ram and self.midis is not None:
            return len(self.midis)
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Pull from RAM if available
        if self.preload_to_ram:
            raw_data = self.midis[idx]
            
            # Parquet/Pandas wrapped our data in an extra list/object array. 
            # If the data looks like [array([...])], extract the inner array:
            if isinstance(raw_data, (list, np.ndarray)) and len(raw_data) == 1:
                raw_data = raw_data[0]
                
            # Force the data into a clean NumPy integer array to remove the 'object_' type
            clean_data = np.array(raw_data, dtype=np.int64)
            
            # Now PyTorch will gladly accept it
            midi = torch.tensor(clean_data, dtype=torch.long)
        else:
            path = self.file_paths[idx]
            with open(path, 'rb') as f:
                json_data = orjson.loads(f.read())
            midi = torch.tensor(json_data['ids'], dtype=torch.long)

        midi = midi.flatten()
        seq_len = len(midi)
        
        if seq_len > self.chunk_size:
            # If song is longer than chunk size, grab a random slice
            # This is great for training as it shows the model different parts of the song every epoch
            max_start = seq_len - self.chunk_size
            start_idx = torch.randint(0, max_start + 1, (1,)).item()
            midi_chunk = midi[start_idx : start_idx + self.chunk_size]
            
        elif seq_len < self.chunk_size:
            # If song is shorter, pad the end with the pad_token
            padding_len = self.chunk_size - seq_len
            padding = torch.full((padding_len,), self.pad_token, dtype=torch.long)
            midi_chunk = torch.cat([midi, padding])
            
        else:
            # If it happens to be exactly the right size
            midi_chunk = midi
        # 2. Otherwise, pull "File-by-File" from Disk
        
        return midi_chunk

class ContinuousMidiDataset(Dataset):
    def __init__(self, folder_path, chunk_size=2048, eos_token=2): # Assuming 1024 is your EOS token
        self.chunk_size = chunk_size
        self.eos_token = eos_token
        
        print("Loading and flattening entire dataset into a single stream...")
        pq_path = os.path.join(folder_path, "dataset.parquet")
        raw_midis = pd.read_parquet(pq_path)['ids'].tolist()
        
        all_tokens = []
        for raw_data in raw_midis:
            if isinstance(raw_data, (list, np.ndarray)) and len(raw_data) == 1:
                raw_data = raw_data[0]
                
            # 1. Add the song tokens
            all_tokens.extend(np.array(raw_data, dtype=np.int64))
            # 2. Add the EOS delimiter immediately after the song ends!
            all_tokens.append(self.eos_token)
            
        self.data_stream = torch.tensor(all_tokens, dtype=torch.long)
        self.num_chunks = len(self.data_stream) // self.chunk_size
        self.data_stream = self.data_stream[:self.num_chunks * self.chunk_size]
    def __len__(self):
            return self.num_chunks

    def __getitem__(self, idx):
        # Grab the exact, non-overlapping chunk
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        return self.data_stream[start_idx:end_idx]

# --- Execution ---
# Note: Ensure all your sequences are the exact same length, otherwise DataLoader will crash!
# If they are not, you will need to add a custom `collate_fn` to the DataLoader to pad them.
if __name__ == "__main__":
    midi_dataset = MidiDataset(
        folder_path="tokenization/saved_tokens/jazz-0-30-03-2026_18-56-02/test", 
        preload_to_ram=True
    )

    midi_loader = DataLoader(midi_dataset, batch_size=32, shuffle=True)

    for batch in midi_loader:
        print("\nBatch shape:", batch.shape)
        break # Just testing the first batch

    midi_dataset_continuous = ContinuousMidiDataset(
        folder_path="tokenization/saved_tokens/jazz-0-30-03-2026_18-56-02/test",
        chunk_size=2048)
    midi_loader_continuous = DataLoader(midi_dataset_continuous, batch_size=32, shuffle=True)  

    for batch in midi_loader_continuous:
        print("\nContinuous Batch shape:", batch.shape)
        break # Just testing the first batch