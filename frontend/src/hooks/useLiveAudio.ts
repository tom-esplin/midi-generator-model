import { useEffect, useRef } from 'react';
import * as Tone from 'tone';
import { useNoteStore } from '../lib/noteStore';
import type { SynthConfig } from '../lib/noteStore';
import { midiToNoteName } from '../lib/midiHelpers';
import { buildSynthChain, applySynthConfig } from '../lib/synthEngine';
import type { SynthChain } from '../lib/synthEngine';

function configNeedsRebuild(a: SynthConfig, b: SynthConfig): boolean {
  return a.instrument !== b.instrument
    || a.attack !== b.attack || a.decay !== b.decay
    || a.sustain !== b.sustain || a.release !== b.release;
}

export function useLiveAudio() {
  const chainRef = useRef<SynthChain | null>(null);
  const prevPitches = useRef(new Set<number>());
  const audioStarted = useRef(false);
  const prevConfig = useRef<SynthConfig | null>(null);

  useEffect(() => {
    const unsubscribe = useNoteStore.subscribe((state) => {
      const cfg = state.synthConfig;
      const currentPitches = new Set(state.activeNotes.keys());

      // Rebuild synth if oscillator or envelope changed
      if (prevConfig.current && configNeedsRebuild(prevConfig.current, cfg) && chainRef.current) {
        chainRef.current.dispose();
        chainRef.current = null;
      }

      // Live-update effect parameters without rebuild
      if (chainRef.current && prevConfig.current) {
        applySynthConfig(chainRef.current, cfg);
      }

      prevConfig.current = cfg;

      const added = [...currentPitches].filter((p) => !prevPitches.current.has(p));
      const removed = [...prevPitches.current].filter((p) => !currentPitches.has(p));

      if (added.length === 0 && removed.length === 0) return;

      if (!audioStarted.current) {
        Tone.start();
        audioStarted.current = true;
      }

      if (!chainRef.current) {
        chainRef.current = buildSynthChain(cfg);
      }

      const synth = chainRef.current.synth;

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
      chainRef.current?.dispose();
      chainRef.current = null;
    };
  }, []);
}
