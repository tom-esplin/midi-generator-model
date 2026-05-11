const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export function midiToNoteName(midi: number): string {
  const octave = Math.floor(midi / 12) - 1;
  const note = NOTE_NAMES[midi % 12];
  return `${note}${octave}`;
}

export function noteNameToMidi(name: string): number {
  const match = name.match(/^([A-G]#?)(-?\d)$/);
  if (!match) return -1;
  const [, note, octStr] = match;
  const octave = parseInt(octStr, 10);
  const idx = NOTE_NAMES.indexOf(note);
  if (idx === -1) return -1;
  return (octave + 1) * 12 + idx;
}

export function midiToVexKey(midi: number): string {
  const octave = Math.floor(midi / 12) - 1;
  const note = NOTE_NAMES[midi % 12].toLowerCase();
  return `${note}/${octave}`;
}

export function isBlackKey(midi: number): boolean {
  const n = midi % 12;
  return [1, 3, 6, 8, 10].includes(n);
}

/**
 * Ableton-style keyboard layout.
 * Middle row = white keys starting at C.
 * Top row = black keys (sharps).
 * Values are semitone offsets from the base octave note (C).
 */
export const ABLETON_KEY_MAP: Record<string, number> = {
  // White keys (middle row): A=C, S=D, D=E, F=F, G=G, H=A, J=B, K=C+1, L=D+1, Semicolon=E+1
  KeyA: 0,   // C
  KeyS: 2,   // D
  KeyD: 4,   // E
  KeyF: 5,   // F
  KeyG: 7,   // G
  KeyH: 9,   // A
  KeyJ: 11,  // B
  KeyK: 12,  // C (next octave)
  KeyL: 14,  // D (next octave)
  Semicolon: 16, // E (next octave)

  // Black keys (top row): W=C#, E=D#, T=F#, Y=G#, U=A#, O=C#+1, P=D#+1
  KeyW: 1,   // C#
  KeyE: 3,   // D#
  KeyT: 6,   // F#
  KeyY: 8,   // G#
  KeyU: 10,  // A#
  KeyO: 13,  // C# (next octave)
  KeyP: 15,  // D# (next octave)
};

export const DEFAULT_BASE_OCTAVE = 3; // C3 = MIDI 48
export const MIN_OCTAVE = -1;
export const MAX_OCTAVE = 8;
export const VELOCITY_STEP = 20;
export const DEFAULT_VELOCITY = 100;
export const MIN_VELOCITY = 20;
export const MAX_VELOCITY = 127;

export function abletonKeyToMidi(code: string, baseOctave: number): number | null {
  const offset = ABLETON_KEY_MAP[code];
  if (offset == null) return null;
  return (baseOctave + 1) * 12 + offset;
}

/**
 * Get a human-readable label for an Ableton key mapping entry,
 * given the current base octave.
 */
export function getAbletonKeyLabel(code: string): string {
  const labels: Record<string, string> = {
    KeyA: 'A', KeyW: 'W', KeyS: 'S', KeyE: 'E', KeyD: 'D',
    KeyF: 'F', KeyT: 'T', KeyG: 'G', KeyY: 'Y', KeyH: 'H',
    KeyU: 'U', KeyJ: 'J', KeyK: 'K', KeyO: 'O', KeyL: 'L',
    KeyP: 'P', Semicolon: ';',
  };
  return labels[code] ?? code;
}
