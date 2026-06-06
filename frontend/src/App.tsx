import { useEffect } from 'react';
import Piano from './components/Piano';
import PianoRoll from './components/PianoRoll';
import RecordingControls from './components/RecordingControls';
import SynthConfig from './components/SynthConfig';
import HelpOverlay from './components/HelpOverlay';
import KeyboardGuide from './components/KeyboardGuide';
import { useNoteStore } from './lib/noteStore';
import { useMIDIInput } from './hooks/useMIDIInput';
import { useKeyboardInput } from './hooks/useKeyboardInput';
import { useLiveAudio } from './hooks/useLiveAudio';
import { useMetronome } from './hooks/useMetronome';
import { useCursorAnimation } from './hooks/useCursorAnimation';
import { useIsMobile } from './hooks/useIsMobile';
import './App.css';

export default function App() {
  const { deviceName, error: midiError } = useMIDIInput();
  const { baseOctave, velocity } = useKeyboardInput();
  const isMobile = useIsMobile();
  const helpMode = useNoteStore((s) => s.helpMode);
  const setHelpMode = useNoteStore((s) => s.setHelpMode);
  useLiveAudio();
  useMetronome();
  useCursorAnimation();

  useEffect(() => {
    if (!helpMode) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setHelpMode(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [helpMode, setHelpMode]);

  return (
    <div className={`app ${isMobile ? 'app-mobile' : 'app-desktop'}`}>
      <header className="app-header">
        <h1 className="app-title">MIDI Generator</h1>
        <p className="app-subtitle">
          Record notes, view them on a staff, and generate continuations with AI
        </p>
        <div className="header-badges">
          {deviceName && (
            <span
              className="midi-badge"
              data-help-title="MIDI input"
              data-help={`Connected to ${deviceName}. Play your MIDI keyboard to record notes.`}
            >
              MIDI: {deviceName}
            </span>
          )}
          {midiError && (
            <span className="midi-badge midi-badge-warn">{midiError}</span>
          )}
          <span
            className="midi-badge"
            data-help-title="Computer keyboard"
            data-help={
              `Ableton-style layout. White keys: A S D F G H J K L ;  Black keys: W E T Y U O P. ` +
              `Z / X shift the octave (currently C${baseOctave}). C / V shift velocity (currently ${velocity}).`
            }
          >
            Oct: C{baseOctave} | Vel: {velocity}
          </span>
        </div>
      </header>

      <main className="app-main">
        <section className="section-controls">
          <RecordingControls isMobile={isMobile} />
        </section>

        <section className="section-piano">
          <SynthConfig />
          <Piano baseOctave={baseOctave} />
        </section>

        <section className="section-piano-roll">
          <PianoRoll isMobile={isMobile} />
        </section>

        <section className="section-guide">
          <KeyboardGuide baseOctave={baseOctave} velocity={velocity} />
        </section>
      </main>

      {!isMobile && helpMode && <HelpOverlay />}
    </div>
  );
}
