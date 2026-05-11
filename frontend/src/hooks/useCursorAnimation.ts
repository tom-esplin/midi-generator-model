import { useEffect, useRef } from 'react';
import { useNoteStore } from '../lib/noteStore';

/**
 * Module-level ref that PianoRoll reads directly in its rAF loop,
 * avoiding React re-renders for every cursor frame.
 */
export const cursorTimeRef = { current: 0 };

const STORE_UPDATE_INTERVAL = 66; // ~15 fps for UI components like beat dots

export function useCursorAnimation() {
  const rafRef = useRef<number | null>(null);
  const lastStoreUpdate = useRef(0);

  useEffect(() => {
    const animate = () => {
      const {
        recordingState, recordingStartTime,
        isPlaying, playbackStartedAt,
        setCursorTime, notes, generatedNotes, stopPlayback,
      } = useNoteStore.getState();

      const now = performance.now() / 1000;
      let t = 0;

      if (recordingState === 'recording' && recordingStartTime != null) {
        t = now - recordingStartTime;
      } else if (isPlaying && playbackStartedAt != null) {
        const elapsed = now - playbackStartedAt;
        const allNotes = [...notes, ...generatedNotes];
        const maxTime = allNotes.reduce(
          (max, n) => Math.max(max, n.startTime + n.duration), 0,
        );
        if (elapsed > maxTime + 0.5) {
          stopPlayback();
          cursorTimeRef.current = 0;
          setCursorTime(0);
          rafRef.current = requestAnimationFrame(animate);
          return;
        }
        t = elapsed;
      }

      cursorTimeRef.current = t;

      // Throttle zustand updates so RecordingControls doesn't re-render 60fps
      const nowMs = performance.now();
      if (nowMs - lastStoreUpdate.current > STORE_UPDATE_INTERVAL) {
        setCursorTime(t);
        lastStoreUpdate.current = nowMs;
      }

      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);
}
