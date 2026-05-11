import { useState } from 'react';
import {
  ABLETON_KEY_MAP,
  getAbletonKeyLabel,
  midiToNoteName,
} from '../lib/midiHelpers';

interface KeyboardGuideProps {
  baseOctave: number;
  velocity: number;
}

export default function KeyboardGuide({ baseOctave, velocity }: KeyboardGuideProps) {
  const [open, setOpen] = useState(false);

  const baseMidi = (baseOctave + 1) * 12;

  const blackKeyCodes = ['KeyW', 'KeyE', 'KeyT', 'KeyY', 'KeyU', 'KeyO', 'KeyP'];
  const whiteKeyCodes = ['KeyA', 'KeyS', 'KeyD', 'KeyF', 'KeyG', 'KeyH', 'KeyJ', 'KeyK', 'KeyL', 'Semicolon'];

  return (
    <div className="keyboard-guide">
      <button className="btn btn-guide-toggle" onClick={() => setOpen(!open)}>
        {open ? 'Hide' : 'Show'} Keyboard Guide
      </button>
      {open && (
        <div className="guide-panel">
          <p className="guide-desc">
            Ableton-style keyboard layout. Hold keys to sustain notes.
          </p>

          <div className="guide-status">
            <span className="guide-status-item">
              Octave: <strong>C{baseOctave}</strong>
            </span>
            <span className="guide-status-item">
              Velocity: <strong>{velocity}</strong>
            </span>
          </div>

          <div className="guide-controls-row">
            <div className="guide-control-key">
              <span className="guide-key-label">Z</span>
              <span className="guide-key-note">Oct -</span>
            </div>
            <div className="guide-control-key">
              <span className="guide-key-label">X</span>
              <span className="guide-key-note">Oct +</span>
            </div>
            <div className="guide-control-key">
              <span className="guide-key-label">C</span>
              <span className="guide-key-note">Vel -</span>
            </div>
            <div className="guide-control-key">
              <span className="guide-key-label">V</span>
              <span className="guide-key-note">Vel +</span>
            </div>
          </div>

          <p className="guide-section-label">Black keys (sharps)</p>
          <div className="guide-row guide-row-black">
            {blackKeyCodes.map((code) => {
              const offset = ABLETON_KEY_MAP[code];
              const midi = baseMidi + offset;
              return (
                <div key={code} className="guide-key guide-key-black">
                  <span className="guide-key-label">{getAbletonKeyLabel(code)}</span>
                  <span className="guide-key-note">{midiToNoteName(midi)}</span>
                </div>
              );
            })}
          </div>

          <p className="guide-section-label">White keys</p>
          <div className="guide-row">
            {whiteKeyCodes.map((code) => {
              const offset = ABLETON_KEY_MAP[code];
              const midi = baseMidi + offset;
              return (
                <div key={code} className="guide-key guide-key-white">
                  <span className="guide-key-label">{getAbletonKeyLabel(code)}</span>
                  <span className="guide-key-note">{midiToNoteName(midi)}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
