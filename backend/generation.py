"""OptimizedGru inference: tokenize seed MIDI, generate tokens, decode to notes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from miditok import PerTok, TokSequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.midi_convert import notes_to_midi_file, score_to_notes
from models.gru import OptimizedGru

VALID_GENRES = frozenset({"jazz", "classical", "soundtrack"})
WEIGHTS_DIR = ROOT / "models" / "model_weights"
TOKENS_ROOT = ROOT / "tokenization" / "saved_tokens"

CHUNK_SIZE = 1000
NUM_LAYERS = 5
HIDDEN_DIM = 512
EMBEDDING_DIM = 512
TOKENS_PER_MEASURE = 64
DEFAULT_TEMPERATURE = 0.8

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_cache: dict[str, OptimizedGru] = {}
_tokenizer_cache: dict[str, PerTok] = {}


def _resolve_weights_path(genre: str) -> Path:
    patterns = [
        f"optimized_gru_per_song_{genre}_zip_*.pt",
        f"optimized_gru_per_song_{genre}-*.pt",
        f"optimized_gru_per_song_{genre}_*.pt",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(WEIGHTS_DIR.glob(pattern))
    matches = sorted(set(matches), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"No OptimizedGru weights for genre '{genre}' in {WEIGHTS_DIR}. "
            f"Expected optimized_gru_per_song_{genre}_zip_<timestamp>.pt"
        )
    return matches[-1]


def _resolve_tokenizer_path(genre: str) -> Path:
    candidates = sorted(TOKENS_ROOT.glob(f"{genre}*/tokenizer.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No tokenizer.json for genre '{genre}' under {TOKENS_ROOT}."
        )
    return candidates[-1]


def get_tokenizer(genre: str) -> PerTok:
    if genre not in _tokenizer_cache:
        path = _resolve_tokenizer_path(genre)
        _tokenizer_cache[genre] = PerTok(params=path)
    return _tokenizer_cache[genre]


def get_model(genre: str) -> OptimizedGru:
    if genre not in _model_cache:
        tokenizer = get_tokenizer(genre)
        model = OptimizedGru(
            tokenizer.vocab_size,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            NUM_LAYERS,
        )
        weights_path = _resolve_weights_path(genre)
        try:
            state = torch.load(weights_path, map_location=_device, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=_device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.to(_device)
        model.eval()
        _model_cache[genre] = model
        print(f"[generation] loaded {genre} from {weights_path.name} on {_device}")
    return _model_cache[genre]


def tokenize_notes(notes: list[dict[str, Any]], tempo: float, genre: str) -> list[int]:
    tokenizer = get_tokenizer(genre)
    midi_path = notes_to_midi_file(notes, tempo)
    try:
        tok_seq = tokenizer(midi_path)[0]
        return list(tok_seq.ids)
    finally:
        midi_path.unlink(missing_ok=True)


def generate_tokens(
    context_ids: list[int],
    genre: str,
    pred_length: int,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[int]:
    """Autoregressive sampling with full-sequence warmup (OptimizedGru)."""
    model = get_model(genre)
    if not context_ids:
        raise ValueError("Context must contain at least one token")

    context = context_ids[-CHUNK_SIZE:]
    x = torch.tensor([context], dtype=torch.long, device=_device)
    hidden = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_DIM, device=_device)

    with torch.no_grad():
        logits, hidden = model(x, hidden)
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        current = torch.multinomial(probs, num_samples=1)
        generated = [current]

        for _ in range(pred_length):
            logits, hidden = model(current, hidden)
            probs = F.softmax(logits / temperature, dim=-1)
            current = torch.multinomial(probs, num_samples=1)
            generated.append(current)

    new_tokens = torch.cat(generated, dim=1).squeeze(0).tolist()
    return context + new_tokens


def decode_token_ids(token_ids: list[int], genre: str):
    tokenizer = get_tokenizer(genre)
    return tokenizer.decode([TokSequence(ids=token_ids)])


def generate_continuation(
    notes: list[dict[str, Any]],
    tempo: float,
    genre: str,
    length_measures: int,
    start_measure: int | None = None,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[dict[str, Any]]:
    """Full pipeline used by the Flask API."""
    genre = genre.lower().strip()
    if genre not in VALID_GENRES:
        raise ValueError(
            f"Unsupported genre '{genre}'. Choose from: {', '.join(sorted(VALID_GENRES))}"
        )
    if not notes:
        raise ValueError("No notes provided")

    beat_duration = 60.0 / tempo
    measure_duration = beat_duration * 4

    if start_measure is not None:
        gen_start = (start_measure - 1) * measure_duration
    else:
        gen_start = max(n["startTime"] + n["duration"] for n in notes)

    seed_ids = tokenize_notes(notes, tempo, genre)
    seed_end = max(n["startTime"] + n["duration"] for n in notes)
    pred_length = max(32, int(length_measures) * TOKENS_PER_MEASURE)

    all_ids = generate_tokens(seed_ids, genre, pred_length, temperature=temperature)
    decoded = decode_token_ids(all_ids, genre)

    continuation = score_to_notes(decoded, tempo, min_start_time=seed_end - 1e-3)

    if not continuation:
        continuation = score_to_notes(decoded, tempo, min_start_time=0.0)
        for n in continuation:
            n["startTime"] = round(n["startTime"] + gen_start, 6)

    return continuation
