import { useState } from 'react';
import { useNoteStore } from '../lib/noteStore';
import { useAudioPlayback } from '../hooks/useAudioPlayback';
import { generateMidi } from '../lib/midiApi';
import GenerateModal from './GenerateModal';

interface Props {
  isMobile: boolean;
}

export default function RecordingControls({ isMobile }: Props) {
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
  const editMode = useNoteStore((s) => s.editMode);
  const setEditMode = useNoteStore((s) => s.setEditMode);
  const helpMode = useNoteStore((s) => s.helpMode);
  const setHelpMode = useNoteStore((s) => s.setHelpMode);

  const { playNotes, stopAll } = useAudioPlayback();

  const [modalOpen, setModalOpen] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async (config: {
    genre: string;
    lengthMeasures: number;
    startMeasure: number;
  }) => {
    if (notes.length === 0) return;
    setIsGenerating(true);
    setError(null);
    try {
      const res = await generateMidi({
        notes,
        tempo,
        genre: config.genre,
        lengthMeasures: config.lengthMeasures,
        startMeasure: config.startMeasure,
      });
      setGeneratedNotes(res.notes);
      setModalOpen(false);
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
  const canEdit = !isActive && !isMobile;

  const beatDuration = 60 / tempo;
  const currentBeat = cursorTime / beatDuration;
  const measure = Math.floor(currentBeat / 4) + 1;
  const beatInMeasure = Math.floor(currentBeat % 4) + 1;

  return (
    <div className="controls-container">
      <div className="controls-row">
        {recordingState === 'idle' || recordingState === 'stopped' ? (
          <button
            className="btn btn-record"
            onClick={startRecording}
            disabled={!canRecord}
            data-help-title="Record"
            data-help="Start a 4-beat count-in, then capture every note you play (MIDI, computer keyboard, or on-screen piano) until you press Pause or Stop. Recorded notes snap to 16th notes."
          >
            <span className="btn-icon record-icon" />
            Record
          </button>
        ) : isRecording ? (
          <button
            className="btn btn-pause"
            onClick={pauseRecording}
            data-help-title="Pause"
            data-help="Stop capturing new notes but keep what you've recorded. Press Resume to continue."
          >
            <span className="btn-icon pause-icon" />
            Pause
          </button>
        ) : recordingState === 'paused' ? (
          <button
            className="btn btn-record"
            onClick={startRecording}
            data-help-title="Resume"
            data-help="Resume recording with another 4-beat count-in."
          >
            <span className="btn-icon record-icon" />
            Resume
          </button>
        ) : (
          <button className="btn btn-counting-in" disabled>
            <span className="spinner" />
            Count In...
          </button>
        )}

        <button className="btn btn-play" onClick={handlePlay} disabled={!canPlay}>
          &#9654; Play
        </button>

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
          data-help-title="Metronome"
          data-help="Toggle the audible click track during the count-in and recording. Helpful for staying on the beat."
        >
          {metronomeOn ? '🔔' : '🔕'} Metronome
        </button>

        <button
          className={`btn btn-edit ${editMode && !isMobile ? 'active' : ''}`}
          onClick={() => setEditMode(!editMode)}
          disabled={!canEdit}
          title={isMobile ? 'Edit mode is disabled on mobile' : 'Toggle edit mode'}
          data-help-title="Edit mode"
          data-help={
            isMobile
              ? 'Edit mode is disabled on mobile for better touch performance. Use a desktop pointer for precision note editing.'
              : 'Reshape your notes directly on the piano roll. Left-click an empty cell to add a quarter note (snapped to the beat). ' +
                'Drag a note body to move it (16th-note snap), drag its right edge to resize. ' +
                'Right-click a note to delete it, or right-drag across notes to sweep-delete. Disabled while recording or playing.'
          }
        >
          {editMode && !isMobile ? '\u270E Editing' : '\u270E Edit'}
        </button>

        {!isMobile && (
          <button
            className={`btn btn-help ${helpMode ? 'active' : ''}`}
            onClick={() => setHelpMode(!helpMode)}
            title={helpMode ? 'Exit help mode (Esc)' : 'Enter help mode'}
            aria-label="Toggle help mode"
            data-help-title="Help mode"
            data-help="You're in help mode. Move the pointer over any control or piano key to see what it does. Click this button again or press Esc to exit."
          >
            ?
          </button>
        )}

        <div
          className="tempo-control"
          data-help-title="Tempo"
          data-help="Beats per minute. Changes the metronome speed, recording quantization grid, and playback speed. Locked while recording or counting in."
        >
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
          onClick={() => setModalOpen(true)}
          disabled={notes.length === 0 || isGenerating || isActive}
          data-help-title="Generate"
          data-help="Open the generation dialog. Pick a genre, length in measures, and start measure, then the AI continues your recorded notes. Generated notes appear in purple on the piano roll."
        >
          <span className="btn-icon generate-icon">&#9733;</span>
          Generate
        </button>

        <button
          className="btn btn-reset"
          onClick={() => setGeneratedNotes([])}
          disabled={generatedNotes.length === 0 || isGenerating || isActive}
          title="Remove generated notes"
          data-help-title="Undo generation"
          data-help="Clear the purple AI-generated notes from the piano roll. Your recorded notes are kept."
        >
          &#8634; Undo Generation
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

      {modalOpen && (
        <GenerateModal
          open={modalOpen}
          onClose={() => setModalOpen(false)}
          onGenerate={handleGenerate}
          isGenerating={isGenerating}
        />
      )}
    </div>
  );
}
