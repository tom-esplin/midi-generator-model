import { useEffect, useRef, useState, useCallback } from 'react';
import { useNoteStore } from '../lib/noteStore';
import {
  ABLETON_KEY_MAP,
  abletonKeyToMidi,
  DEFAULT_BASE_OCTAVE,
  MIN_OCTAVE,
  MAX_OCTAVE,
  VELOCITY_STEP,
  DEFAULT_VELOCITY,
  MIN_VELOCITY,
  MAX_VELOCITY,
} from '../lib/midiHelpers';

/**
 * Ableton-style computer keyboard input.
 * - Middle row = white keys, top row = black keys
 * - Z/X = octave down/up
 * - C/V = velocity down/up
 */
export function useKeyboardInput() {
  const [baseOctave, setBaseOctave] = useState(DEFAULT_BASE_OCTAVE);
  const [velocity, setVelocity] = useState(DEFAULT_VELOCITY);

  const heldKeys = useRef(new Map<string, number>());
  const storeRef = useRef(useNoteStore.getState());
  const octaveRef = useRef(baseOctave);
  const velocityRef = useRef(velocity);

  useEffect(() => {
    octaveRef.current = baseOctave;
  }, [baseOctave]);

  useEffect(() => {
    velocityRef.current = velocity;
  }, [velocity]);

  useEffect(() => {
    return useNoteStore.subscribe((s) => {
      storeRef.current = s;
    });
  }, []);

  const releaseAll = useCallback(() => {
    heldKeys.current.forEach((midi) => {
      storeRef.current.noteOff(midi);
    });
    heldKeys.current.clear();
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.repeat) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      // Z = octave down
      if (e.code === 'KeyZ') {
        e.preventDefault();
        setBaseOctave((prev) => Math.max(MIN_OCTAVE, prev - 1));
        releaseAll();
        return;
      }
      // X = octave up
      if (e.code === 'KeyX') {
        e.preventDefault();
        setBaseOctave((prev) => Math.min(MAX_OCTAVE, prev + 1));
        releaseAll();
        return;
      }
      // C = velocity down
      if (e.code === 'KeyC') {
        e.preventDefault();
        setVelocity((prev) => Math.max(MIN_VELOCITY, prev - VELOCITY_STEP));
        return;
      }
      // V = velocity up
      if (e.code === 'KeyV') {
        e.preventDefault();
        setVelocity((prev) => Math.min(MAX_VELOCITY, prev + VELOCITY_STEP));
        return;
      }

      if (!(e.code in ABLETON_KEY_MAP)) return;

      e.preventDefault();
      const midi = abletonKeyToMidi(e.code, octaveRef.current);
      if (midi == null || midi < 0 || midi > 127) return;

      heldKeys.current.set(e.code, midi);
      storeRef.current.noteOn(midi, velocityRef.current);
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (!(e.code in ABLETON_KEY_MAP)) return;

      e.preventDefault();
      const midi = heldKeys.current.get(e.code);
      if (midi != null) {
        storeRef.current.noteOff(midi);
        heldKeys.current.delete(e.code);
      }
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      releaseAll();
    };
  }, [releaseAll]);

  return { baseOctave, velocity };
}
