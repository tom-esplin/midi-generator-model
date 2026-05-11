import Piano from './components/Piano';
import PianoRoll from './components/PianoRoll';
import RecordingControls from './components/RecordingControls';
import KeyboardGuide from './components/KeyboardGuide';
import { useMIDIInput } from './hooks/useMIDIInput';
import { useKeyboardInput } from './hooks/useKeyboardInput';
import { useLiveAudio } from './hooks/useLiveAudio';
import { useMetronome } from './hooks/useMetronome';
import { useCursorAnimation } from './hooks/useCursorAnimation';
import './App.css';

export default function App() {
  const { deviceName, error: midiError } = useMIDIInput();
  const { baseOctave, velocity } = useKeyboardInput();
  useLiveAudio();
  useMetronome();
  useCursorAnimation();

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">MIDI Generator</h1>
        <p className="app-subtitle">
          Record notes, view them on a staff, and generate continuations with AI
        </p>
        <div className="header-badges">
          {deviceName && (
            <span className="midi-badge">MIDI: {deviceName}</span>
          )}
          {midiError && (
            <span className="midi-badge midi-badge-warn">{midiError}</span>
          )}
          <span className="midi-badge">
            Oct: C{baseOctave} | Vel: {velocity}
          </span>
        </div>
      </header>

      <main className="app-main">
        <section className="section-piano-roll">
          <PianoRoll />
        </section>

        <section className="section-controls">
          <RecordingControls />
        </section>

        <section className="section-piano">
          <Piano />
        </section>

        <section className="section-guide">
          <KeyboardGuide baseOctave={baseOctave} velocity={velocity} />
        </section>
      </main>
    </div>
  );
}
