import { useState } from 'react';
import { useNoteStore } from '../lib/noteStore';
import type { OscillatorType, FilterType } from '../lib/noteStore';

const INSTRUMENTS: { value: OscillatorType; label: string }[] = [
  { value: 'triangle', label: 'Triangle' },
  { value: 'sine', label: 'Sine' },
  { value: 'square', label: 'Square' },
  { value: 'sawtooth', label: 'Sawtooth' },
  { value: 'fmtriangle', label: 'FM Triangle' },
  { value: 'fmsine', label: 'FM Sine' },
  { value: 'fmsquare', label: 'FM Square' },
  { value: 'fmsawtooth', label: 'FM Sawtooth' },
  { value: 'amtriangle', label: 'AM Triangle' },
  { value: 'amsine', label: 'AM Sine' },
  { value: 'amsquare', label: 'AM Square' },
  { value: 'amsawtooth', label: 'AM Sawtooth' },
  { value: 'fattriangle', label: 'Fat Triangle' },
  { value: 'fatsine', label: 'Fat Sine' },
  { value: 'fatsquare', label: 'Fat Square' },
  { value: 'fatsawtooth', label: 'Fat Sawtooth' },
];

const FILTER_TYPES: { value: FilterType; label: string }[] = [
  { value: 'lowpass', label: 'Low Pass' },
  { value: 'highpass', label: 'High Pass' },
  { value: 'bandpass', label: 'Band Pass' },
];

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (v: number) => void;
}

function Slider({ label, value, min, max, step, unit, onChange }: SliderProps) {
  return (
    <div className="sc-slider">
      <div className="sc-slider-header">
        <span className="sc-slider-label">{label}</span>
        <span className="sc-slider-value">{value.toFixed(step < 0.01 ? 3 : step < 1 ? 2 : 0)}{unit ?? ''}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export default function SynthConfig() {
  const [open, setOpen] = useState(false);
  const cfg = useNoteStore((s) => s.synthConfig);
  const update = useNoteStore((s) => s.updateSynthConfig);

  return (
    <div className="synth-config-wrapper">
      <button
        className="btn sc-toggle"
        onClick={() => setOpen(!open)}
        title="Synth Settings"
        data-help-title="Synth Config"
        data-help="Sculpt the sound used to play back notes: oscillator type, ADSR envelope, filter, reverb, delay, chorus, and distortion. Changes are applied in real time."
      >
        <span className="sc-gear">&#9881;</span> Synth Config
      </button>

      {open && (
        <div className="sc-panel">
          {/* Instrument */}
          <div className="sc-section">
            <h4 className="sc-section-title">Oscillator</h4>
            <select
              value={cfg.instrument}
              onChange={(e) => update({ instrument: e.target.value as OscillatorType })}
              className="sc-select"
            >
              {INSTRUMENTS.map((i) => (
                <option key={i.value} value={i.value}>{i.label}</option>
              ))}
            </select>
          </div>

          {/* ADSR Envelope */}
          <div className="sc-section">
            <h4 className="sc-section-title">Envelope (ADSR)</h4>
            <Slider label="Attack" value={cfg.attack} min={0.001} max={2} step={0.005} unit="s" onChange={(v) => update({ attack: v })} />
            <Slider label="Decay" value={cfg.decay} min={0.01} max={2} step={0.01} unit="s" onChange={(v) => update({ decay: v })} />
            <Slider label="Sustain" value={cfg.sustain} min={0} max={1} step={0.01} onChange={(v) => update({ sustain: v })} />
            <Slider label="Release" value={cfg.release} min={0.01} max={5} step={0.01} unit="s" onChange={(v) => update({ release: v })} />
          </div>

          {/* Filter */}
          <div className="sc-section">
            <div className="sc-section-header">
              <h4 className="sc-section-title">Filter</h4>
              <label className="sc-toggle-switch">
                <input type="checkbox" checked={cfg.filterEnabled} onChange={(e) => update({ filterEnabled: e.target.checked })} />
                <span className="sc-toggle-track" />
              </label>
            </div>
            {cfg.filterEnabled && (
              <>
                <div className="sc-row">
                  <span className="sc-slider-label">Type</span>
                  <select
                    value={cfg.filterType}
                    onChange={(e) => update({ filterType: e.target.value as FilterType })}
                    className="sc-select sc-select-sm"
                  >
                    {FILTER_TYPES.map((f) => (
                      <option key={f.value} value={f.value}>{f.label}</option>
                    ))}
                  </select>
                </div>
                <Slider label="Cutoff" value={cfg.filterFreq} min={50} max={15000} step={10} unit=" Hz" onChange={(v) => update({ filterFreq: v })} />
                <Slider label="Resonance" value={cfg.filterQ} min={0.1} max={20} step={0.1} onChange={(v) => update({ filterQ: v })} />
              </>
            )}
          </div>

          {/* Reverb */}
          <div className="sc-section">
            <div className="sc-section-header">
              <h4 className="sc-section-title">Reverb</h4>
              <label className="sc-toggle-switch">
                <input type="checkbox" checked={cfg.reverbEnabled} onChange={(e) => update({ reverbEnabled: e.target.checked })} />
                <span className="sc-toggle-track" />
              </label>
            </div>
            {cfg.reverbEnabled && (
              <>
                <Slider label="Mix" value={cfg.reverbMix} min={0} max={1} step={0.01} onChange={(v) => update({ reverbMix: v })} />
                <Slider label="Decay" value={cfg.reverbDecay} min={0.1} max={10} step={0.1} unit="s" onChange={(v) => update({ reverbDecay: v })} />
              </>
            )}
          </div>

          {/* Delay */}
          <div className="sc-section">
            <div className="sc-section-header">
              <h4 className="sc-section-title">Delay</h4>
              <label className="sc-toggle-switch">
                <input type="checkbox" checked={cfg.delayEnabled} onChange={(e) => update({ delayEnabled: e.target.checked })} />
                <span className="sc-toggle-track" />
              </label>
            </div>
            {cfg.delayEnabled && (
              <>
                <Slider label="Mix" value={cfg.delayMix} min={0} max={1} step={0.01} onChange={(v) => update({ delayMix: v })} />
                <Slider label="Time" value={cfg.delayTime} min={0.01} max={1} step={0.01} unit="s" onChange={(v) => update({ delayTime: v })} />
                <Slider label="Feedback" value={cfg.delayFeedback} min={0} max={0.9} step={0.01} onChange={(v) => update({ delayFeedback: v })} />
              </>
            )}
          </div>

          {/* Chorus */}
          <div className="sc-section">
            <div className="sc-section-header">
              <h4 className="sc-section-title">Chorus</h4>
              <label className="sc-toggle-switch">
                <input type="checkbox" checked={cfg.chorusEnabled} onChange={(e) => update({ chorusEnabled: e.target.checked })} />
                <span className="sc-toggle-track" />
              </label>
            </div>
            {cfg.chorusEnabled && (
              <Slider label="Mix" value={cfg.chorusMix} min={0} max={1} step={0.01} onChange={(v) => update({ chorusMix: v })} />
            )}
          </div>

          {/* Distortion */}
          <div className="sc-section">
            <div className="sc-section-header">
              <h4 className="sc-section-title">Distortion</h4>
              <label className="sc-toggle-switch">
                <input type="checkbox" checked={cfg.distortionEnabled} onChange={(e) => update({ distortionEnabled: e.target.checked })} />
                <span className="sc-toggle-track" />
              </label>
            </div>
            {cfg.distortionEnabled && (
              <Slider label="Amount" value={cfg.distortionAmount} min={0} max={1} step={0.01} onChange={(v) => update({ distortionAmount: v })} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
