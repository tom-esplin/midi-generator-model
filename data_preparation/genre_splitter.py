# This script splits the dataset into separate JSON files based on genres.

import pathlib
import os
import json
import shutil
import tqdm
METADATA_PATH = os.path.join("raw_data", 'metadata.json')
METADATA_JSON = None 
with open(METADATA_PATH, 'r') as f:
        METADATA_JSON = json.load(f)
def split_by_genres():
# Get all subdirectories and sort them
    genre_dict = {} 
    dirs = sorted([d for d in pathlib.Path('raw_data').iterdir() if d.is_dir()])
    for directory in dirs:
        filenames = [entry.name for entry in pathlib.Path(directory).iterdir() if entry.is_file()]
        for f in filenames:
           id = int(f.split("_")[0])
           file_genre = METADATA_JSON[f"{id}"]["metadata"].get("genre")
           if file_genre is None:
                file_genre = "UNK"
           if file_genre in genre_dict:
                genre_dict[file_genre].append(os.path.join("raw_data",directory.name,f))
           else:
                genre_dict[file_genre] = [os.path.join("raw_data",directory.name,f)]
    for genre, files in tqdm.tqdm(genre_dict.items()):
        output_path = os.path.join("prepared_data", genre)
        os.makedirs(output_path,exist_ok=True)
        for f in tqdm.tqdm(files):
            shutil.copy(f,output_path)
split_by_genres()
"""
import pathlib
# Get all subdirectories and sort them
dirs = sorted([d for d in pathlib.Path('.').iterdir() if d.is_dir()])
for directory in dirs:
    print(directory.name)

I could reverse this process and iterate through each midi file read its index from the file name then get the
json and boom I have the genre and I split off into the proper subfolder
Lookup should be constant because its a dictionary rather than logn for a search each time a significant difference

Then I count files and do test train eval splits

Easy right?
""" 