import { useCallback } from 'react';
import { useNoteStore } from '../lib/noteStore';
import { isBlackKey } from '../lib/midiHelpers';

const FIRST_MIDI = 48; // C3
const LAST_MIDI = 84;  // C6

interface PianoKeyProps {
  midi: number;
  isActive: boolean;
  onDown: (midi: number) => void;
  onUp: (midi: number) => void;
}

function PianoKey({ midi, isActive, onDown, onUp }: PianoKeyProps) {
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
    />
  );
}

export default function Piano() {
  const activeNotes = useNoteStore((s) => s.activeNotes);
  const noteOn = useNoteStore((s) => s.noteOn);
  const noteOff = useNoteStore((s) => s.noteOff);

  const handleDown = useCallback(
    (midi: number) => {
      noteOn(midi, 80);
    },
    [noteOn],
  );

  const handleUp = useCallback(
    (midi: number) => {
      noteOff(midi);
    },
    [noteOff],
  );

  const keys: JSX.Element[] = [];
  for (let midi = FIRST_MIDI; midi <= LAST_MIDI; midi++) {
    keys.push(
      <PianoKey
        key={midi}
        midi={midi}
        isActive={activeNotes.has(midi)}
        onDown={handleDown}
        onUp={handleUp}
      />,
    );
  }

  return <div className="piano-container">{keys}</div>;
}
