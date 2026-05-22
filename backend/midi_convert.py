"""Convert between frontend note dicts and symusic MIDI (PerTok-compatible timing)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from symusic import Note, Score, Tempo, Track

TICKS_PER_QUARTER = 16
PITCH_MIN = 21
PITCH_MAX = 109


def seconds_to_ticks(seconds: float, tempo: float) -> int:
    """Map wall-clock seconds to MIDI ticks at the given BPM."""
    return max(0, round(seconds * (tempo / 60.0) * TICKS_PER_QUARTER))


def ticks_to_seconds(ticks: int, tempo: float) -> float:
    return ticks * 60.0 / (TICKS_PER_QUARTER * tempo)


def notes_to_score(notes: list[dict[str, Any]], tempo: float) -> Score:
    """Build a single-track symusic Score from API note objects."""
    score = Score()
    score.ticks_per_quarter = TICKS_PER_QUARTER
    score.tempos.append(Tempo(time=0, qpm=float(tempo)))

    track = Track()
    for n in notes:
        pitch = int(n["pitch"])
        if pitch < PITCH_MIN or pitch > PITCH_MAX:
            continue
        start_tick = seconds_to_ticks(float(n["startTime"]), tempo)
        dur_ticks = max(1, seconds_to_ticks(float(n["duration"]), tempo))
        velocity = max(1, min(127, int(n.get("velocity", 80))))
        track.notes.append(
            Note(pitch=pitch, velocity=velocity, time=start_tick, duration=dur_ticks)
        )

    track.notes.sort(key=lambda note: note.time)
    score.tracks.append(track)
    return score


def notes_to_midi_file(notes: list[dict[str, Any]], tempo: float, path: Path | None = None) -> Path:
    score = notes_to_score(notes, tempo)
    if path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        path = Path(tmp.name)
        tmp.close()
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(str(path))
    return path


def score_to_notes(
    score: Score,
    tempo: float,
    *,
    min_start_time: float = 0.0,
) -> list[dict[str, Any]]:
    """Extract API note dicts from a symusic Score."""
    out: list[dict[str, Any]] = []
    for track in score.tracks:
        for note in track.notes:
            start = ticks_to_seconds(int(note.time), tempo)
            if start < min_start_time - 1e-6:
                continue
            duration = ticks_to_seconds(max(1, int(note.duration)), tempo)
            out.append(
                {
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                    "startTime": round(start, 6),
                    "duration": round(duration, 6),
                }
            )
    out.sort(key=lambda n: n["startTime"])
    return out
