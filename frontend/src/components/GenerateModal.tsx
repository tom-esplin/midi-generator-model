import { useState, useEffect, useRef } from 'react';
import { useNoteStore } from '../lib/noteStore';

const GENRES = [
  'soundtrack',
  'jazz',
  'classical',
] as const;

interface GenerateConfig {
  genre: string;
  lengthMeasures: number;
  startMeasure: number;
}

interface Props {
  open: boolean;
  onClose: () => void;
  onGenerate: (config: GenerateConfig) => void;
  isGenerating: boolean;
}

export default function GenerateModal({ open, onClose, onGenerate, isGenerating }: Props) {
  const notes = useNoteStore((s) => s.notes);
  const tempo = useNoteStore((s) => s.tempo);
  const backdropRef = useRef<HTMLDivElement>(null);

  const beatDuration = 60 / tempo;
  const maxEnd = notes.reduce((m, n) => Math.max(m, n.startTime + n.duration), 0);
  const totalInputMeasures = Math.max(1, Math.ceil(maxEnd / (beatDuration * 4)));

  const [genre, setGenre] = useState('soundtrack');
  const [lengthMeasures, setLengthMeasures] = useState(4);
  const [startMeasure, setStartMeasure] = useState(totalInputMeasures + 1);

  useEffect(() => {
    if (!open) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [open, onClose]);

  if (!open) return null;

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === backdropRef.current) onClose();
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onGenerate({ genre, lengthMeasures, startMeasure });
  };

  return (
    <div className="modal-backdrop" ref={backdropRef} onClick={handleBackdropClick}>
      <div className="modal-panel">
        <div className="modal-header">
          <h2 className="modal-title">Generation Settings</h2>
          <button className="modal-close" onClick={onClose} aria-label="Close">&times;</button>
        </div>

        <form onSubmit={handleSubmit} className="modal-body">
          <div className="modal-field">
            <label htmlFor="gen-genre">Genre</label>
            <select
              id="gen-genre"
              value={genre}
              onChange={(e) => setGenre(e.target.value)}
            >
              {GENRES.map((g) => (
                <option key={g} value={g}>
                  {g.charAt(0).toUpperCase() + g.slice(1)}
                </option>
              ))}
            </select>
          </div>

          <div className="modal-field">
            <label htmlFor="gen-length">Length (measures)</label>
            <input
              id="gen-length"
              type="number"
              min={1}
              max={64}
              value={lengthMeasures}
              onChange={(e) => setLengthMeasures(Math.max(1, Number(e.target.value)))}
            />
            <span className="modal-hint">
              {lengthMeasures * 4} beats &middot; {(lengthMeasures * 4 * beatDuration).toFixed(1)}s at {tempo} BPM
            </span>
          </div>

          <div className="modal-field">
            <label htmlFor="gen-start">Start at measure</label>
            <input
              id="gen-start"
              type="number"
              min={1}
              value={startMeasure}
              onChange={(e) => setStartMeasure(Math.max(1, Number(e.target.value)))}
            />
            <span className="modal-hint">
              Default: measure {totalInputMeasures + 1} (right after recorded content)
            </span>
          </div>

          <div className="modal-info">
            <span>Input: {notes.length} notes across {totalInputMeasures} measure{totalInputMeasures !== 1 ? 's' : ''}</span>
          </div>

          <div className="modal-actions">
            <button type="button" className="btn btn-reset" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-generate" disabled={isGenerating}>
              {isGenerating ? (
                <>
                  <span className="spinner" />
                  Generating...
                </>
              ) : (
                <>
                  <span className="btn-icon generate-icon">&#9733;</span>
                  Generate
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
