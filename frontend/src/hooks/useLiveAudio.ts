import { useEffect, useRef } from 'react';
import * as Tone from 'tone';
import { useNoteStore } from '../lib/noteStore';
import { midiToNoteName } from '../lib/midiHelpers';

export function useLiveAudio() {
  const synthRef = useRef<Tone.PolySynth | null>(null);
  const prevPitches = useRef(new Set<number>());
  const audioStarted = useRef(false);

  useEffect(() => {
    const unsubscribe = useNoteStore.subscribe((state) => {
      const currentPitches = new Set(state.activeNotes.keys());

      const added = [...currentPitches].filter((p) => !prevPitches.current.has(p));
      const removed = [...prevPitches.current].filter((p) => !currentPitches.has(p));

      if (added.length === 0 && removed.length === 0) return;

      if (!audioStarted.current) {
        Tone.start();
        Tone.getContext().lookAhead = 0.01;
        audioStarted.current = true;
      }

      if (!synthRef.current) {
        synthRef.current = new Tone.PolySynth(Tone.Synth, {
          maxPolyphony: 16,
          oscillator: { type: 'triangle' },
          envelope: { attack: 0.005, decay: 0.1, sustain: 0.3, release: 0.4 },
        }).toDestination();
      }

      const synth = synthRef.current;

      for (const pitch of added) {
        const info = state.activeNotes.get(pitch);
        const vel = info ? info.velocity / 127 : 0.6;
        synth.triggerAttack(midiToNoteName(pitch), Tone.now(), vel);
      }

      for (const pitch of removed) {
        synth.triggerRelease(midiToNoteName(pitch), Tone.now());
      }

      prevPitches.current = currentPitches;
    });

    return () => {
      unsubscribe();
      synthRef.current?.releaseAll();
    };
  }, []);
}
