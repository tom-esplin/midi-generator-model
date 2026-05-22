"""Smoke tests for the generation pipeline (run from project root)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.generation import (  # noqa: E402
    VALID_GENRES,
    WEIGHTS_DIR,
    _resolve_tokenizer_path,
    _resolve_weights_path,
    decode_token_ids,
    generate_continuation,
    generate_tokens,
    get_tokenizer,
    tokenize_notes,
)
from backend.midi_convert import notes_to_midi_file  # noqa: E402

SAMPLE_NOTES = [
    {"pitch": 60, "velocity": 90, "startTime": 0.0, "duration": 0.5},
    {"pitch": 64, "velocity": 85, "startTime": 0.5, "duration": 0.5},
    {"pitch": 67, "velocity": 80, "startTime": 1.0, "duration": 0.5},
    {"pitch": 72, "velocity": 95, "startTime": 1.5, "duration": 1.0},
]


def test_assets_exist():
    print("=== Asset check ===")
    for genre in sorted(VALID_GENRES):
        tok = _resolve_tokenizer_path(genre)
        print(f"  {genre}: tokenizer OK -> {tok.parent.name}")
        try:
            weights = _resolve_weights_path(genre)
            print(f"  {genre}: weights OK -> {weights.name}")
        except FileNotFoundError as exc:
            print(f"  {genre}: weights MISSING ({exc})")


def test_tokenize_roundtrip(genre: str = "jazz"):
    print(f"\n=== Tokenize roundtrip ({genre}) ===")
    ids = tokenize_notes(SAMPLE_NOTES, tempo=120, genre=genre)
    assert len(ids) > 0, "expected non-empty token sequence"
    decoded = decode_token_ids(ids, genre)
    out_path = Path(__file__).parent / f"output_roundtrip_{genre}.mid"
    decoded.dump_midi(str(out_path))
    print(f"  {len(ids)} tokens -> {out_path}")


def test_generate_from_midi_file(genre: str = "jazz", pred_length: int = 128):
    print(f"\n=== Generate from prepared_data ({genre}) ===")
    prepared = ROOT / "prepared_data" / genre / "test"
    if not prepared.exists():
        print(f"  skip: {prepared} not found")
        return

    midis = sorted(prepared.glob("*.mid"))
    if not midis:
        print("  skip: no test midis")
        return

    try:
        _resolve_weights_path(genre)
    except FileNotFoundError as exc:
        print(f"  skip: {exc}")
        return

    tokenizer = get_tokenizer(genre)
    tok_seq = tokenizer(midis[0])[0]
    seed_ids = list(tok_seq.ids)[-512:]
    print(f"  seed from {midis[0].name}, {len(seed_ids)} context tokens")

    all_ids = generate_tokens(seed_ids, genre, pred_length)
    decoded = decode_token_ids(all_ids, genre)
    out_path = Path(__file__).parent / f"output_test_{genre}.mid"
    decoded.dump_midi(str(out_path))
    print(f"  generated {pred_length} tokens -> {out_path}")


def test_api_pipeline(genre: str = "jazz"):
    print(f"\n=== API pipeline ({genre}) ===")
    try:
        _resolve_weights_path(genre)
    except FileNotFoundError as exc:
        print(f"  skip: {exc}")
        return

    notes = generate_continuation(
        SAMPLE_NOTES,
        tempo=120,
        genre=genre,
        length_measures=2,
        start_measure=5,
    )
    assert notes, "expected non-empty continuation"
    print(f"  returned {len(notes)} notes, first at t={notes[0]['startTime']}")


def main():
    test_assets_exist()
    for genre in sorted(VALID_GENRES):
        try:
            test_tokenize_roundtrip(genre)
        except Exception as exc:
            print(f"  FAILED roundtrip {genre}: {exc}")

    for genre in ("jazz",):
        try:
            test_generate_from_midi_file(genre)
            test_api_pipeline(genre)
        except Exception as exc:
            print(f"  FAILED generation {genre}: {exc}")

    # Also verify notes->midi without tokenizer
    p = notes_to_midi_file(SAMPLE_NOTES, 120, Path(__file__).parent / "output_sample_seed.mid")
    print(f"\n=== Wrote sample seed MIDI -> {p} ===")
    print("\nDone.")


if __name__ == "__main__":
    main()
