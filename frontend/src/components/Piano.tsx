import { useCallback, useState, useEffect, useMemo } from 'react';
import { useNoteStore } from '../lib/noteStore';
import {
  ABLETON_KEY_MAP,
  getAbletonKeyLabel,
  isBlackKey,
  midiToNoteName,
} from '../lib/midiHelpers';

const FULL_FIRST = 48;  // C3
const FULL_LAST = 84;   // C6
const SMALL_FIRST = 60; // C4
const SMALL_LAST = 72;  // C5
const BREAKPOINT = 640;

interface PianoKeyProps {
  midi: number;
  isActive: boolean;
  helpText: string;
  onDown: (midi: number) => void;
  onUp: (midi: number) => void;
}

function PianoKey({ midi, isActive, helpText, onDown, onUp }: PianoKeyProps) {
  const black = isBlackKey(midi);
  const baseClass = black ? 'piano-key black' : 'piano-key white';
  const activeClass = isActive ? ' active' : '';

  return (
    <div
      className={baseClass + activeClass}
      onPointerDown={(e) => {
        e.preventDefault();
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
        onDown(midi);
      }}
      onPointerUp={() => onUp(midi)}
      onPointerLeave={() => onUp(midi)}
      data-midi={midi}
      data-help-title={midiToNoteName(midi)}
      data-help={helpText}
    />
  );
}

interface PianoProps {
  baseOctave: number;
}

export default function Piano({ baseOctave }: PianoProps) {
  const activeNotes = useNoteStore((s) => s.activeNotes);
  const noteOn = useNoteStore((s) => s.noteOn);
  const noteOff = useNoteStore((s) => s.noteOff);

  const [isSmall, setIsSmall] = useState(() => window.innerWidth < BREAKPOINT);

  useEffect(() => {
    const mq = window.matchMedia(`(max-width: ${BREAKPOINT - 1}px)`);
    const onChange = (e: MediaQueryListEvent) => setIsSmall(e.matches);
    mq.addEventListener('change', onChange);
    setIsSmall(mq.matches);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  const firstMidi = isSmall ? SMALL_FIRST : FULL_FIRST;
  const lastMidi = isSmall ? SMALL_LAST : FULL_LAST;

  const handleDown = useCallback(
    (midi: number) => { noteOn(midi, 80); },
    [noteOn],
  );

  const handleUp = useCallback(
    (midi: number) => { noteOff(midi); },
    [noteOff],
  );

  const midiToKeyLabel = useMemo(() => {
    const out: Record<number, string> = {};
    const baseMidi = (baseOctave + 1) * 12;
    for (const [code, offset] of Object.entries(ABLETON_KEY_MAP)) {
      out[baseMidi + offset] = getAbletonKeyLabel(code);
    }
    return out;
  }, [baseOctave]);

  const keys: JSX.Element[] = [];
  for (let midi = firstMidi; midi <= lastMidi; midi++) {
    const note = midiToNoteName(midi);
    const keyLabel = midiToKeyLabel[midi];
    const helpText = keyLabel
      ? `Play ${note}. Computer key: ${keyLabel}. Click and hold to sustain.`
      : `Play ${note}. Click and hold to sustain. (Out of computer-key range — press Z/X to shift octave.)`;
    keys.push(
      <PianoKey
        key={midi}
        midi={midi}
        isActive={activeNotes.has(midi)}
        helpText={helpText}
        onDown={handleDown}
        onUp={handleUp}
      />,
    );
  }

  return (
    <div
      className="piano-container"
      data-help-title="On-screen piano"
      data-help="Click any key to play. Use your MIDI keyboard or computer keyboard for faster input. Hover any key with Help mode on to see its computer-key binding."
    >
      {keys}
    </div>
  );
}
