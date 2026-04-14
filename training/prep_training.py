from data_preparation.train_test_split import split_train_test
from pathlib import Path
from tokenization.tokenize_genre import train_tokenizer, tokenize_genre
def prepare_genres(genres: list[str]):
    for genre in genres:
        #split_train_test(genre,0.2,42)
        #exp_path = train_tokenizer(0,genre)
        exp_path = Path("tokenization","saved_tokens","soundtrack-0-31-03-2026_13-26-17")
        tokenize_genre(exp_path,genre)
if __name__ == "__main__":
    genres = ["soundtrack"]
    #genres = ["classical","jazz","soundtrack","UNK"]
    prepare_genres(genres)
        