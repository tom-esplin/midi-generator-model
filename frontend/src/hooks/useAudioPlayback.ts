import { useRef, useCallback } from 'react';
import * as Tone from 'tone';
import type { NoteEvent } from '../lib/types';
import { midiToNoteName } from '../lib/midiHelpers';
import { useNoteStore } from '../lib/noteStore';
import { buildSynthChain } from '../lib/synthEngine';
import type { SynthChain } from '../lib/synthEngine';

export function useAudioPlayback() {
  const chainRef = useRef<SynthChain | null>(null);

  const ensureChain = useCallback(() => {
    if (!chainRef.current) {
      const cfg = useNoteStore.getState().synthConfig;
      chainRef.current = buildSynthChain(cfg);
    }
    return chainRef.current;
  }, []);

  const playNote = useCallback(
    async (pitch: number, velocity: number, duration = 0.3) => {
      await Tone.start();
      const chain = ensureChain();
      const name = midiToNoteName(pitch);
      const vol = Tone.gainToDb(velocity / 127);
      chain.synth.triggerAttackRelease(name, duration, Tone.now(), Tone.dbToGain(vol));
    },
    [ensureChain],
  );

  const playNotes = useCallback(
    async (notes: NoteEvent[]) => {
      await Tone.start();
      const chain = ensureChain();
      const now = Tone.now();

      for (const n of notes) {
        const name = midiToNoteName(n.pitch);
        chain.synth.triggerAttackRelease(name, n.duration, now + n.startTime);
      }
    },
    [ensureChain],
  );

  const stopAll = useCallback(() => {
    if (chainRef.current) {
      chainRef.current.dispose();
      chainRef.current = null;
    }
  }, []);

  return { playNote, playNotes, stopAll };
}
