import type { GenerateRequest, GenerateResponse } from './types';

const API_BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '');

function apiUrl(path: string) {
  if (!API_BASE) return path;
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

export async function generateMidi(request: GenerateRequest): Promise<GenerateResponse> {
  const res = await fetch(apiUrl('/api/generate'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const contentType = res.headers.get('content-type') ?? '';
    const detail = contentType.includes('application/json')
      ? await res.json().then((body) => body.error ?? JSON.stringify(body))
      : await res.text();
    throw new Error(`Generation failed (${res.status}): ${detail || res.statusText}`);
  }

  return res.json();
}
