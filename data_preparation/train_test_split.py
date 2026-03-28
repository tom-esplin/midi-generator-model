from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
def split_train_test(midi_type: str = "classical", test_size: float = 0.2, random_state: int = 42):
    search_path = os.path.join("prepared_data", midi_type)
    if not os.path.exists(search_path):
        raise RuntimeError("Make sure to run from the top level of directory")
    midis = [f for f in os.listdir(search_path) if f.endswith(".mid")]
    train_files, test_files = train_test_split(midis, test_size=test_size, random_state=random_state)
    train_path = os.path.join("prepared_data", midi_type, "train")
    test_path = os.path.join("prepared_data", midi_type, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    for f in tqdm(train_files):
        os.rename(os.path.join(search_path, f), os.path.join(train_path, f))
    for f in tqdm(test_files):
        os.rename(os.path.join(search_path, f), os.path.join(test_path, f))