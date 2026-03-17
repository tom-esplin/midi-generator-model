# This script splits the dataset into separate JSON files based on genres.

def split_by_genres(data_path, output_dir):
    import os
    import json
    json_path = os.path.join(data_path, 'metadata.json')
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    genre_dict = {}
    for item in json_data:
        genres = item.get('genre', [])
        for genre in genres:
            if genre not in genre_dict:
                genre_dict[genre] = []
            genre_dict[genre].append(item)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for genre, items in genre_dict.items():
        output_path = os.path.join(output_dir, f'{genre}.json')
        with open(output_path, 'w') as f:
            json.dump(items, f, indent=4)

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