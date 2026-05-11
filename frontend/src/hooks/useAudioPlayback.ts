import { useRef, useCallback } from 'react';
import * as Tone from 'tone';
import type { NoteEvent } from '../lib/types';
import { midiToNoteName } from '../lib/midiHelpers';

export function useAudioPlayback() {
  const synthRef = useRef<Tone.PolySynth | null>(null);

  const ensureSynth = useCallback(() => {
    if (!synthRef.current) {
      Tone.getContext().lookAhead = 0.01;
      synthRef.current = new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: 'triangle' },
        envelope: { attack: 0.005, decay: 0.1, sustain: 0.3, release: 0.4 },
      }).toDestination();
    }
    return synthRef.current;
  }, []);

  const playNote = useCallback(
    async (pitch: number, velocity: number, duration = 0.3) => {
      await Tone.start();
      const synth = ensureSynth();
      const name = midiToNoteName(pitch);
      const vol = Tone.gainToDb(velocity / 127);
      synth.triggerAttackRelease(name, duration, Tone.now(), Tone.dbToGain(vol));
    },
    [ensureSynth],
  );

  const playNotes = useCallback(
    async (notes: NoteEvent[]) => {
      await Tone.start();
      const synth = ensureSynth();
      const now = Tone.now();

      for (const n of notes) {
        const name = midiToNoteName(n.pitch);
        synth.triggerAttackRelease(name, n.duration, now + n.startTime);
      }
    },
    [ensureSynth],
  );

  const stopAll = useCallback(() => {
    synthRef.current?.releaseAll();
  }, []);

  return { playNote, playNotes, stopAll };
}
