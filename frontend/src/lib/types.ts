export interface NoteEvent {
  pitch: number;
  velocity: number;
  startTime: number;
  duration: number;
}

export type RecordingState = 'idle' | 'counting_in' | 'recording' | 'paused' | 'stopped';

export interface GenerateRequest {
  notes: NoteEvent[];
  tempo: number;
  genre: string;
}

export interface GenerateResponse {
  notes: NoteEvent[];
  midiBase64?: string;
}
