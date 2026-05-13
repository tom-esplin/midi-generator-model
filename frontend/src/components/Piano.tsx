import { useCallback, useState, useEffect } from 'react';
import { useNoteStore } from '../lib/noteStore';
import { isBlackKey } from '../lib/midiHelpers';

const FULL_FIRST = 48;  // C3
const FULL_LAST = 84;   // C6
const SMALL_FIRST = 60; // C4
const SMALL_LAST = 72;  // C5
const BREAKPOINT = 640;

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

  const keys: JSX.Element[] = [];
  for (let midi = firstMidi; midi <= lastMidi; midi++) {
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
