from miditok import PerTok,TokenizerConfig
from copy import deepcopy
from pathlib import Path
import os
import random
from datetime import datetime
import json

from tqdm import tqdm
def train_tokenizer(sample_size: int = 0,midi_genre: str = "classical", vocab_size: int = 30000 , tokenizer_model: str = "BPE", pitch_range: tuple[int,int] = (21,109),use_chords:bool = False,tick_division: int = 16):
    search_path = Path("prepared_data", midi_genre,"train")
    print(f"Checking in: {search_path.absolute()}")
    print(f"Directory exists: {search_path.exists()}")
    if not search_path.exists():
        raise RuntimeError("Make sure to run from the top level of directory")
    midis = list(Path("prepared_data",midi_genre).glob("*.mid"))
    time_stamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    tokenizer_exp_name = f"{midi_genre}-{sample_size}-{time_stamp}"
    if sample_size != 0:
        midis = random.sample(midis,sample_size)
    tokenizer = PerTok(TokenizerConfig(pitch_range=pitch_range,use_chords=use_chords,ticks_per_quarter=tick_division,beat_res={(0, 4): 8, (0, 4): 12},use_microtiming=True,max_microtiming_shift=0.5,num_microtiming_bins = 50))
    exp_path = Path("tokenization","saved_tokens",tokenizer_exp_name)
    os.makedirs(exp_path)
    pre_tokens_path = Path("tokenization","saved_tokens",tokenizer_exp_name,"pre_tokens")
    os.mkdir(pre_tokens_path)
    tokenizer.tokenize_midi_dataset(midis,pre_tokens_path)
    token_paths = list((pre_tokens_path).glob("*.json"))
    tokenizer.train(
        vocab_size=vocab_size,
        model= tokenizer_model,
        files_paths=token_paths
    )
    tokenizer.save(Path(exp_path,"tokenizer.json"))
    return exp_path

def tokenize_genre(exp_path: str, midi_genre: str):
    test_search_path = Path("prepared_data", midi_genre,"test")
    train_search_path = Path("prepared_data", midi_genre,"train")
    if not test_search_path.exists() or not train_search_path.exists():
        raise RuntimeError("Make sure to run from the top level of directory")
    tokenizer = PerTok()
    tokenizer.load(Path(exp_path,"tokenizer.json"))
    for path in [test_search_path,train_search_path]:
        midis = list(path.glob("*.mid"))
        tokenized_output_path = Path("tokenization","saved_tokens",exp_path,path.name)
        os.makedirs(tokenized_output_path,exist_ok=True)
        for midi in tqdm(midis, desc=f"Tokenizing {path.name}"):
            tokens = tokenizer.tokenize_midi(midi)
            output_file = tokenized_output_path / f"{midi.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(tokens, f)