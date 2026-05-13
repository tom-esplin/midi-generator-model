import { useEffect, useRef } from 'react';
import * as Tone from 'tone';
import { useNoteStore } from '../lib/noteStore';

const COUNT_IN_BEATS = 4;

/**
 * Handles metronome click sounds and count-in timing.
 * Does NOT drive cursor position (that's useCursorAnimation).
 */
export function useMetronome() {
  const clickHi = useRef<Tone.Synth | null>(null);
  const clickLo = useRef<Tone.Synth | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const unsubscribe = useNoteStore.subscribe(
      (state, prev) => {
        const prevPhase = prev.recordingState;
        const curPhase = state.recordingState;
        const tempoChanged = prev.tempo !== state.tempo;
        const metronomeChanged = prev.metronomeOn !== state.metronomeOn;

        if (curPhase === 'counting_in' && prevPhase !== 'counting_in') {
          startCountIn(state.tempo);
        } else if (curPhase === 'recording' && (prevPhase !== 'recording' || tempoChanged || metronomeChanged)) {
          startMetronomeClicks(state.tempo, state.metronomeOn);
        } else if (curPhase !== 'recording' && curPhase !== 'counting_in' &&
                   (prevPhase === 'recording' || prevPhase === 'counting_in')) {
          stopTicking();
        }
      },
    );

    return () => {
      unsubscribe();
      stopTicking();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function ensureClicks() {
    if (!clickHi.current) {
      clickHi.current = new Tone.Synth({
        oscillator: { type: 'sine' },
        envelope: { attack: 0.001, decay: 0.05, sustain: 0, release: 0.05 },
      }).toDestination();
      clickHi.current.volume.value = -10;
    }
    if (!clickLo.current) {
      clickLo.current = new Tone.Synth({
        oscillator: { type: 'sine' },
        envelope: { attack: 0.001, decay: 0.05, sustain: 0, release: 0.05 },
      }).toDestination();
      clickLo.current.volume.value = -14;
    }
  }

  function startCountIn(tempo: number) {
    stopTicking();
    ensureClicks();
    Tone.start();

    const msPerBeat = (60 / tempo) * 1000;
    let beat = 0;

    const tick = () => {
      const store = useNoteStore.getState();
      if (store.recordingState !== 'counting_in') {
        stopTicking();
        return;
      }

      // After all 4 clicks have played, wait one full beat then start recording
      // so recording begins on the downbeat (beat 1) of the next measure.
      if (beat >= COUNT_IN_BEATS) {
        stopTicking();
        store.beginRecording();
        return;
      }

      store.setCountInBeat(beat);

      const synth = beat === 0 ? clickHi.current! : clickLo.current!;
      const clickNote = beat === 0 ? 'G5' : 'C5';
      synth.triggerAttackRelease(clickNote, '32n');

      beat++;
    };

    tick();
    intervalRef.current = setInterval(tick, msPerBeat);
  }

  function startMetronomeClicks(tempo: number, metronomeOn: boolean) {
    stopTicking();
    if (!metronomeOn) return;
    ensureClicks();
    Tone.start();

    const msPerBeat = (60 / tempo) * 1000;
    let beat = 0;

    const tick = () => {
      const store = useNoteStore.getState();
      if (store.recordingState !== 'recording') {
        stopTicking();
        return;
      }

      const synth = beat % 4 === 0 ? clickHi.current! : clickLo.current!;
      const clickNote = beat % 4 === 0 ? 'G5' : 'C5';
      synth.triggerAttackRelease(clickNote, '32n');
      beat++;
    };

    tick();
    intervalRef.current = setInterval(tick, msPerBeat);
  }

  function stopTicking() {
    if (intervalRef.current != null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }
}
