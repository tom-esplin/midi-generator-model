import type { GenerateRequest, GenerateResponse } from './types';

const API_BASE = import.meta.env.VITE_API_URL ?? '';

export async function generateMidi(request: GenerateRequest): Promise<GenerateResponse> {
  const res = await fetch(`${API_BASE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Generation failed (${res.status}): ${text}`);
  }

  return res.json();
}
