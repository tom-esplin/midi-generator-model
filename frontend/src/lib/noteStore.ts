import { create } from 'zustand';
import type { NoteEvent, RecordingState } from './types';

export type OscillatorType =
  | 'sine' | 'square' | 'sawtooth' | 'triangle'
  | 'fmsine' | 'fmsquare' | 'fmsawtooth' | 'fmtriangle'
  | 'amsine' | 'amsquare' | 'amsawtooth' | 'amtriangle'
  | 'fatsine' | 'fatsquare' | 'fatsawtooth' | 'fattriangle';

export type FilterType = 'lowpass' | 'highpass' | 'bandpass';

export interface SynthConfig {
  instrument: OscillatorType;
  attack: number;
  decay: number;
  sustain: number;
  release: number;
  filterEnabled: boolean;
  filterType: FilterType;
  filterFreq: number;
  filterQ: number;
  reverbEnabled: boolean;
  reverbMix: number;
  reverbDecay: number;
  delayEnabled: boolean;
  delayMix: number;
  delayTime: number;
  delayFeedback: number;
  chorusEnabled: boolean;
  chorusMix: number;
  distortionEnabled: boolean;
  distortionAmount: number;
}

export const DEFAULT_SYNTH_CONFIG: SynthConfig = {
  instrument: 'triangle',
  attack: 0.005,
  decay: 0.1,
  sustain: 0.3,
  release: 0.4,
  filterEnabled: false,
  filterType: 'lowpass',
  filterFreq: 2000,
  filterQ: 1,
  reverbEnabled: false,
  reverbMix: 0.3,
  reverbDecay: 2,
  delayEnabled: false,
  delayMix: 0.25,
  delayTime: 0.3,
  delayFeedback: 0.3,
  chorusEnabled: false,
  chorusMix: 0.3,
  distortionEnabled: false,
  distortionAmount: 0.2,
};

interface NoteStore {
  notes: NoteEvent[];
  generatedNotes: NoteEvent[];
  recordingState: RecordingState;
  tempo: number;
  recordingStartTime: number | null;
  activeNotes: Map<number, { startTime: number; velocity: number }>;

  cursorTime: number;
  countInBeat: number;
  metronomeOn: boolean;
  isPlaying: boolean;
  playbackStartedAt: number | null;
  synthConfig: SynthConfig;

  startRecording: () => void;
  beginRecording: () => void;
  pauseRecording: () => void;
  stopRecording: () => void;
  resetRecording: () => void;

  startPlayback: () => void;
  stopPlayback: () => void;

  noteOn: (pitch: number, velocity: number) => void;
  noteOff: (pitch: number) => void;

  addNote: (note: NoteEvent) => void;
  updateNote: (index: number, note: Partial<NoteEvent>) => void;
  deleteNote: (index: number) => void;

  setGeneratedNotes: (notes: NoteEvent[]) => void;
  setTempo: (tempo: number) => void;
  setCursorTime: (t: number) => void;
  setCountInBeat: (beat: number) => void;
  setMetronomeOn: (on: boolean) => void;
  updateSynthConfig: (patch: Partial<SynthConfig>) => void;
}

function quantizeTo16th(timeInSeconds: number, tempo: number): number {
  const sixteenthDuration = 60 / tempo / 4;
  return Math.round(timeInSeconds / sixteenthDuration) * sixteenthDuration;
}

export const useNoteStore = create<NoteStore>((set, get) => ({
  notes: [],
  generatedNotes: [],
  recordingState: 'idle',
  tempo: 120,
  recordingStartTime: null,
  activeNotes: new Map(),
  cursorTime: 0,
  countInBeat: 0,
  metronomeOn: true,
  isPlaying: false,
  playbackStartedAt: null,
  synthConfig: { ...DEFAULT_SYNTH_CONFIG },

  startRecording: () => {
    const { isPlaying } = get();
    if (isPlaying) get().stopPlayback();
    set({
      recordingState: 'counting_in',
      countInBeat: 0,
      cursorTime: 0,
    });
  },

  beginRecording: () => {
    const now = performance.now() / 1000;
    set((s) => ({
      recordingState: 'recording',
      recordingStartTime: s.recordingStartTime ?? now,
      cursorTime: 0,
      countInBeat: 0,
    }));
  },

  pauseRecording: () => set({ recordingState: 'paused' }),

  stopRecording: () => {
    const { activeNotes: current } = get();
    current.forEach((_, pitch) => {
      get().noteOff(pitch);
    });
    set({ recordingState: 'stopped', activeNotes: new Map(), cursorTime: 0, countInBeat: 0 });
  },

  resetRecording: () => {
    const { isPlaying } = get();
    if (isPlaying) get().stopPlayback();
    set({
      notes: [],
      generatedNotes: [],
      recordingState: 'idle',
      recordingStartTime: null,
      activeNotes: new Map(),
      cursorTime: 0,
      countInBeat: 0,
    });
  },

  startPlayback: () => {
    const { recordingState } = get();
    if (recordingState === 'recording' || recordingState === 'counting_in') return;
    const now = performance.now() / 1000;
    set({ isPlaying: true, playbackStartedAt: now, cursorTime: 0 });
  },

  stopPlayback: () => {
    set({ isPlaying: false, playbackStartedAt: null, cursorTime: 0 });
  },

  noteOn: (pitch, velocity) => {
    const { recordingState, recordingStartTime } = get();
    const now = performance.now() / 1000;
    const next = new Map(get().activeNotes);
    const startTime = recordingState === 'recording' && recordingStartTime != null
      ? now - recordingStartTime
      : 0;
    next.set(pitch, { startTime, velocity });
    set({ activeNotes: next });
  },

  noteOff: (pitch) => {
    const { recordingState, recordingStartTime, tempo, activeNotes } = get();
    const noteInfo = activeNotes.get(pitch);
    const next = new Map(activeNotes);
    next.delete(pitch);
    set({ activeNotes: next });

    if (!noteInfo) return;
    if (recordingState !== 'recording' || recordingStartTime == null) return;

    const now = performance.now() / 1000;
    const rawEnd = now - recordingStartTime;
    const rawDuration = Math.max(0.05, rawEnd - noteInfo.startTime);

    const quantizedStart = quantizeTo16th(noteInfo.startTime, tempo);
    const sixteenthDuration = 60 / tempo / 4;
    const quantizedDuration = Math.max(
      sixteenthDuration,
      Math.round(rawDuration / sixteenthDuration) * sixteenthDuration,
    );

    const note: NoteEvent = {
      pitch,
      velocity: noteInfo.velocity,
      startTime: quantizedStart,
      duration: quantizedDuration,
    };

    set((s) => ({ notes: [...s.notes, note] }));
  },

  addNote: (note) => set((s) => ({ notes: [...s.notes, note] })),

  updateNote: (index, partial) =>
    set((s) => ({
      notes: s.notes.map((n, i) => (i === index ? { ...n, ...partial } : n)),
    })),

  deleteNote: (index) =>
    set((s) => ({ notes: s.notes.filter((_, i) => i !== index) })),

  setGeneratedNotes: (notes) => set({ generatedNotes: notes }),
  setTempo: (tempo) => set({ tempo }),
  setCursorTime: (t) => set({ cursorTime: t }),
  setCountInBeat: (beat) => set({ countInBeat: beat }),
  setMetronomeOn: (on) => set({ metronomeOn: on }),
  updateSynthConfig: (patch) =>
    set((s) => ({ synthConfig: { ...s.synthConfig, ...patch } })),
}));
