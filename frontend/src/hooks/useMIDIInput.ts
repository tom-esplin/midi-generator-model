import { useEffect, useRef, useState } from 'react';
import { useNoteStore } from '../lib/noteStore';

export function useMIDIInput() {
  const [midiAccess, setMidiAccess] = useState<MIDIAccess | null>(null);
  const [deviceName, setDeviceName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const storeRef = useRef(useNoteStore.getState());

  useEffect(() => {
    return useNoteStore.subscribe((s) => {
      storeRef.current = s;
    });
  }, []);

  useEffect(() => {
    if (!navigator.requestMIDIAccess) {
      setError('Web MIDI API not supported in this browser');
      return;
    }

    let cancelled = false;

    navigator.requestMIDIAccess().then(
      (access) => {
        if (cancelled) return;
        setMidiAccess(access);

        const handleMessage = (e: MIDIMessageEvent) => {
          const [status, note, velocity] = e.data!;
          const command = status & 0xf0;

          if (command === 0x90 && velocity > 0) {
            storeRef.current.noteOn(note, velocity);
          } else if (command === 0x80 || (command === 0x90 && velocity === 0)) {
            storeRef.current.noteOff(note);
          }
        };

        const connectInputs = () => {
          const inputs = Array.from(access.inputs.values());
          if (inputs.length > 0) {
            setDeviceName(inputs[0].name ?? 'MIDI Device');
          } else {
            setDeviceName(null);
          }
          inputs.forEach((input) => {
            input.onmidimessage = handleMessage;
          });
        };

        connectInputs();
        access.onstatechange = connectInputs;
      },
      (err) => {
        if (!cancelled) setError(`MIDI access denied: ${err.message}`);
      },
    );

    return () => {
      cancelled = true;
      if (midiAccess) {
        midiAccess.inputs.forEach((input) => {
          input.onmidimessage = null;
        });
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { deviceName, error };
}
