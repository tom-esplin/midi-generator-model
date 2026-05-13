import * as Tone from 'tone';
import type { SynthConfig } from './noteStore';

export interface SynthChain {
  synth: Tone.PolySynth;
  filter: Tone.Filter;
  reverb: Tone.Reverb;
  delay: Tone.FeedbackDelay;
  chorus: Tone.Chorus;
  distortion: Tone.Distortion;
  dispose: () => void;
}

export function buildSynthChain(cfg: SynthConfig): SynthChain {
  Tone.getContext().lookAhead = 0.01;

  const synth = new Tone.PolySynth(Tone.Synth, {
    maxPolyphony: 16,
    oscillator: { type: cfg.instrument },
    envelope: {
      attack: cfg.attack,
      decay: cfg.decay,
      sustain: cfg.sustain,
      release: cfg.release,
    },
  });

  const filter = new Tone.Filter({
    frequency: cfg.filterEnabled ? cfg.filterFreq : 20000,
    type: cfg.filterType,
    Q: cfg.filterQ,
  });

  const reverb = new Tone.Reverb({ decay: cfg.reverbDecay, wet: cfg.reverbEnabled ? cfg.reverbMix : 0 });
  const delay = new Tone.FeedbackDelay({
    delayTime: cfg.delayTime,
    feedback: cfg.delayFeedback,
    wet: cfg.delayEnabled ? cfg.delayMix : 0,
  });
  const chorus = new Tone.Chorus({ wet: cfg.chorusEnabled ? cfg.chorusMix : 0 }).start();
  const distortion = new Tone.Distortion({
    distortion: cfg.distortionAmount,
    wet: cfg.distortionEnabled ? 1 : 0,
  });

  synth.chain(filter, distortion, chorus, delay, reverb, Tone.getDestination());

  return {
    synth,
    filter,
    reverb,
    delay,
    chorus,
    distortion,
    dispose: () => {
      synth.releaseAll();
      synth.dispose();
      filter.dispose();
      reverb.dispose();
      delay.dispose();
      chorus.dispose();
      distortion.dispose();
    },
  };
}

export function applySynthConfig(chain: SynthChain, cfg: SynthConfig) {
  chain.filter.frequency.value = cfg.filterEnabled ? cfg.filterFreq : 20000;
  chain.filter.type = cfg.filterType;
  chain.filter.Q.value = cfg.filterQ;

  chain.reverb.decay = cfg.reverbDecay;
  chain.reverb.wet.value = cfg.reverbEnabled ? cfg.reverbMix : 0;

  chain.delay.delayTime.value = cfg.delayTime;
  chain.delay.feedback.value = cfg.delayFeedback;
  chain.delay.wet.value = cfg.delayEnabled ? cfg.delayMix : 0;

  chain.chorus.wet.value = cfg.chorusEnabled ? cfg.chorusMix : 0;

  chain.distortion.distortion = cfg.distortionAmount;
  chain.distortion.wet.value = cfg.distortionEnabled ? 1 : 0;
}
