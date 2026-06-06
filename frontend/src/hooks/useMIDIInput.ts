import { useEffect, useRef, useState } from 'react';
import { useNoteStore } from '../lib/noteStore';

export function useMIDIInput() {
  const [deviceName, setDeviceName] = useState<string | null>(null);
  const midiSupported = 'requestMIDIAccess' in navigator;
  const [error, setError] = useState<string | null>(
    midiSupported ? null : 'Web MIDI API not supported in this browser',
  );
  const storeRef = useRef(useNoteStore.getState());

  useEffect(() => {
    return useNoteStore.subscribe((s) => {
      storeRef.current = s;
    });
  }, []);

  useEffect(() => {
    if (!midiSupported) return;

    let cancelled = false;
    let accessRef: MIDIAccess | null = null;

    navigator.requestMIDIAccess().then(
      (access) => {
        if (cancelled) return;
        accessRef = access;

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
      if (accessRef) {
        accessRef.onstatechange = null;
        accessRef.inputs.forEach((input) => {
          input.onmidimessage = null;
        });
      }
    };
  }, [midiSupported]);

  return { deviceName, error };
}
