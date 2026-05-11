import { useState } from 'react';
import { useNoteStore } from '../lib/noteStore';
import { useAudioPlayback } from '../hooks/useAudioPlayback';
import { generateMidi } from '../lib/midiApi';

export default function RecordingControls() {
  const recordingState = useNoteStore((s) => s.recordingState);
  const notes = useNoteStore((s) => s.notes);
  const generatedNotes = useNoteStore((s) => s.generatedNotes);
  const tempo = useNoteStore((s) => s.tempo);
  const cursorTime = useNoteStore((s) => s.cursorTime);
  const countInBeat = useNoteStore((s) => s.countInBeat);
  const metronomeOn = useNoteStore((s) => s.metronomeOn);
  const isPlaying = useNoteStore((s) => s.isPlaying);
  const startRecording = useNoteStore((s) => s.startRecording);
  const pauseRecording = useNoteStore((s) => s.pauseRecording);
  const stopRecording = useNoteStore((s) => s.stopRecording);
  const resetRecording = useNoteStore((s) => s.resetRecording);
  const startPlayback = useNoteStore((s) => s.startPlayback);
  const stopPlayback = useNoteStore((s) => s.stopPlayback);
  const setGeneratedNotes = useNoteStore((s) => s.setGeneratedNotes);
  const setTempo = useNoteStore((s) => s.setTempo);
  const setMetronomeOn = useNoteStore((s) => s.setMetronomeOn);

  const { playNotes, stopAll } = useAudioPlayback();

  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (notes.length === 0) return;
    setIsGenerating(true);
    setError(null);
    try {
      const res = await generateMidi({ notes, tempo, genre: 'jazz' });
      setGeneratedNotes(res.notes);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const handlePlay = () => {
    const allNotes = [...notes, ...generatedNotes];
    if (allNotes.length === 0) return;
    startPlayback();
    playNotes(allNotes);
  };

  const handleStop = () => {
    if (recordingState === 'recording' || recordingState === 'counting_in') {
      stopRecording();
    }
    if (isPlaying) {
      stopPlayback();
      stopAll();
    }
  };

  const isCountingIn = recordingState === 'counting_in';
  const isRecording = recordingState === 'recording';
  const isActive = isCountingIn || isRecording || isPlaying;

  const hasNotes = notes.length > 0 || generatedNotes.length > 0;
  const canRecord = !isPlaying;
  const canPlay = hasNotes && !isRecording && !isCountingIn && !isPlaying;

  const beatDuration = 60 / tempo;
  const currentBeat = cursorTime / beatDuration;
  const measure = Math.floor(currentBeat / 4) + 1;
  const beatInMeasure = Math.floor(currentBeat % 4) + 1;

  return (
    <div className="controls-container">
      <div className="controls-row">
        {/* Record / Pause / Resume / Count-in */}
        {recordingState === 'idle' || recordingState === 'stopped' ? (
          <button className="btn btn-record" onClick={startRecording} disabled={!canRecord}>
            <span className="btn-icon record-icon" />
            Record
          </button>
        ) : isRecording ? (
          <button className="btn btn-pause" onClick={pauseRecording}>
            <span className="btn-icon pause-icon" />
            Pause
          </button>
        ) : recordingState === 'paused' ? (
          <button className="btn btn-record" onClick={startRecording}>
            <span className="btn-icon record-icon" />
            Resume
          </button>
        ) : (
          <button className="btn btn-counting-in" disabled>
            <span className="spinner" />
            Count In...
          </button>
        )}

        {/* Play */}
        <button className="btn btn-play" onClick={handlePlay} disabled={!canPlay}>
          &#9654; Play
        </button>

        {/* Stop */}
        <button className="btn btn-stop" onClick={handleStop} disabled={!isActive}>
          <span className="btn-icon stop-icon" />
          Stop
        </button>

        <button className="btn btn-reset" onClick={() => { handleStop(); resetRecording(); }}>
          Reset
        </button>

        <button
          className={`btn btn-metronome ${metronomeOn ? 'active' : ''}`}
          onClick={() => setMetronomeOn(!metronomeOn)}
          title="Toggle metronome"
        >
          {metronomeOn ? '🔔' : '🔕'} Metronome
        </button>

        <div className="tempo-control">
          <label htmlFor="tempo">BPM</label>
          <input
            id="tempo"
            type="number"
            min={40}
            max={240}
            value={tempo}
            onChange={(e) => setTempo(Number(e.target.value))}
            disabled={isRecording || isCountingIn}
          />
        </div>
      </div>

      {isCountingIn && (
        <div className="beat-display count-in-display">
          <div className="beat-counter">
            <span className="beat-measure count-in-label">Count In</span>
            <div className="beat-dots">
              {[0, 1, 2, 3].map((b) => (
                <span
                  key={b}
                  className={`beat-dot ${b <= countInBeat ? 'beat-dot-active' : ''} ${b === 0 && b === countInBeat ? 'beat-dot-downbeat' : ''}`}
                >
                  {b <= countInBeat ? b + 1 : ''}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {(isRecording || isPlaying) && (
        <div className="beat-display">
          <div className="beat-counter">
            <span className="beat-measure">
              {isPlaying ? 'Playing' : 'Recording'} — Measure {measure}
            </span>
            <div className="beat-dots">
              {[1, 2, 3, 4].map((b) => (
                <span
                  key={b}
                  className={`beat-dot ${b === beatInMeasure ? 'beat-dot-active' : ''} ${b === 1 && b === beatInMeasure ? 'beat-dot-downbeat' : ''}`}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="controls-row">
        <button
          className="btn btn-generate"
          onClick={handleGenerate}
          disabled={notes.length === 0 || isGenerating || isActive}
        >
          {isGenerating ? (
            <span className="spinner" />
          ) : (
            <span className="btn-icon generate-icon">&#9733;</span>
          )}
          {isGenerating ? 'Generating...' : 'Generate'}
        </button>
      </div>

      {error && <p className="error-text">{error}</p>}

      <div className="status-bar">
        <span>Notes: <strong>{notes.length}</strong></span>
        {generatedNotes.length > 0 && (
          <span>Generated: <strong>{generatedNotes.length}</strong></span>
        )}
        <span className={`state-badge ${isPlaying ? 'state-playing' : `state-${recordingState}`}`}>
          {isPlaying ? 'PLAYING' : isCountingIn ? 'COUNT IN' : recordingState.toUpperCase()}
        </span>
      </div>
    </div>
  );
}
