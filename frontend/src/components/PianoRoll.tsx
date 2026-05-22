import { useEffect, useRef, useCallback, useMemo } from 'react';
import { useNoteStore } from '../lib/noteStore';
import { midiToNoteName } from '../lib/midiHelpers';
import { cursorTimeRef } from '../hooks/useCursorAnimation';
import type { NoteEvent } from '../lib/types';

const ROW_HEIGHT = 14;
const PIXELS_PER_BEAT = 50;
const MIN_PITCH = 21;
const MAX_PITCH = 108;
const TOTAL_ROWS = MAX_PITCH - MIN_PITCH;
const CANVAS_HEIGHT = TOTAL_ROWS * ROW_HEIGHT;
const BEATS_PER_MEASURE = 4;
const WIDTH_CHUNK = 800;
const MIN_WIDTH = BEATS_PER_MEASURE * 16 * PIXELS_PER_BEAT;
const EDGE_HIT_PX = 6;
const DEFAULT_VELOCITY = 100;

const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);

const BG_WHITE = '#1e1e30';
const BG_BLACK = '#171728';
const LINE_FAINT = '#252542';
const LINE_BEAT = '#2e2e4c';
const LINE_MEASURE = '#42425e';
const LINE_OCTAVE = '#42425e';
const LINE_16TH = '#ffffff06';
const NOTE_FILL = '#38bdf8';
const NOTE_STROKE = '#0ea5e9';
const ACTIVE_FILL = '#38bdf870';
const ACTIVE_STROKE = '#0ea5e9a0';
const GEN_FILL = '#a78bfa';
const GEN_STROKE = '#7c3aed';
const CURSOR_CLR = '#ef4444d0';

type NoteSource = 'notes' | 'generatedNotes';

type HitTarget =
  | { kind: 'note'; source: NoteSource; index: number; zone: 'body' | 'right-edge' }
  | { kind: 'empty' };

type DragState = {
  kind: 'move' | 'resize';
  source: NoteSource;
  index: number;
  startX: number;
  startY: number;
  origNote: NoteEvent;
};

function snapTime(t: number, sixteenth: number): number {
  return Math.max(0, Math.round(t / sixteenth) * sixteenth);
}

function screenToGrid(
  clientX: number,
  clientY: number,
  grid: HTMLDivElement,
  pxPerSec: number,
) {
  const rect = grid.getBoundingClientRect();
  const x = clientX - rect.left + grid.scrollLeft;
  const y = clientY - rect.top + grid.scrollTop;
  const time = x / pxPerSec;
  const pitch = MAX_PITCH - 1 - Math.floor(y / ROW_HEIGHT);
  return { x, y, time, pitch };
}

function hitTest(
  x: number,
  y: number,
  notes: NoteEvent[],
  generatedNotes: NoteEvent[],
  pxPerSec: number,
): HitTarget {
  const checkList = (
    list: NoteEvent[],
    source: NoteSource,
  ): HitTarget | null => {
    for (let i = list.length - 1; i >= 0; i--) {
      const n = list[i];
      const row = MAX_PITCH - 1 - n.pitch;
      if (row < 0 || row >= TOTAL_ROWS) continue;
      const nx = n.startTime * pxPerSec;
      const nw = Math.max(n.duration * pxPerSec, 4);
      const ny = row * ROW_HEIGHT;
      const nh = ROW_HEIGHT;
      if (x < nx || x > nx + nw || y < ny || y > ny + nh) continue;
      const zone = x >= nx + nw - EDGE_HIT_PX ? 'right-edge' : 'body';
      return { kind: 'note', source, index: i, zone };
    }
    return null;
  };

  const recorded = checkList(notes, 'notes');
  if (recorded) return recorded;
  const generated = checkList(generatedNotes, 'generatedNotes');
  if (generated) return generated;
  return { kind: 'empty' };
}

function cursorForHit(hit: HitTarget): string {
  // For 'empty' we return '' so the CSS pencil cursor (set on
  // .piano-roll.edit-mode .pr-grid) shows through. Inline styles win otherwise.
  if (hit.kind === 'empty') return '';
  if (hit.zone === 'right-edge') return 'ew-resize';
  return 'grab';
}

// ── Static grid rendered to an offscreen canvas ──────────────

function buildGrid(width: number, _tempo: number, dpr: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = width * dpr;
  c.height = CANVAS_HEIGHT * dpr;
  const ctx = c.getContext('2d')!;
  ctx.scale(dpr, dpr);

  for (let i = 0; i < TOTAL_ROWS; i++) {
    const pitch = MAX_PITCH - 1 - i;
    ctx.fillStyle = BLACK_KEYS.has(pitch % 12) ? BG_BLACK : BG_WHITE;
    ctx.fillRect(0, i * ROW_HEIGHT, width, ROW_HEIGHT);
  }

  for (let i = 0; i <= TOTAL_ROWS; i++) {
    const pitch = MAX_PITCH - i;
    if (pitch % 12 === 0) {
      ctx.strokeStyle = LINE_OCTAVE;
      ctx.lineWidth = 1;
    } else {
      ctx.strokeStyle = LINE_FAINT;
      ctx.lineWidth = 0.5;
    }
    ctx.beginPath();
    ctx.moveTo(0, i * ROW_HEIGHT);
    ctx.lineTo(width, i * ROW_HEIGHT);
    ctx.stroke();
  }

  const totalBeats = Math.ceil(width / PIXELS_PER_BEAT);
  for (let b = 0; b <= totalBeats; b++) {
    const x = b * PIXELS_PER_BEAT;
    if (b % BEATS_PER_MEASURE === 0) {
      ctx.strokeStyle = LINE_MEASURE;
      ctx.lineWidth = 1.5;
    } else {
      ctx.strokeStyle = LINE_BEAT;
      ctx.lineWidth = 0.5;
    }
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, CANVAS_HEIGHT);
    ctx.stroke();
  }

  const px16 = PIXELS_PER_BEAT / 4;
  const total16 = Math.ceil(width / px16);
  ctx.strokeStyle = LINE_16TH;
  ctx.lineWidth = 0.5;
  for (let s = 0; s <= total16; s++) {
    if (s % 4 === 0) continue;
    ctx.beginPath();
    ctx.moveTo(s * px16, 0);
    ctx.lineTo(s * px16, CANVAS_HEIGHT);
    ctx.stroke();
  }

  ctx.font = '10px Inter, system-ui, sans-serif';
  ctx.fillStyle = '#64748b';
  ctx.textBaseline = 'top';
  for (let b = 0; b <= totalBeats; b++) {
    if (b % BEATS_PER_MEASURE === 0) {
      ctx.fillText(String(b / BEATS_PER_MEASURE + 1), b * PIXELS_PER_BEAT + 4, 3);
    }
  }

  return c;
}

// ── Component ────────────────────────────────────────────────

export default function PianoRoll() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const labelRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const didInitScroll = useRef(false);
  const lastScrollX = useRef(0);
  const rafRef = useRef<number | null>(null);
  const gridCache = useRef<{ canvas: HTMLCanvasElement; key: string } | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const rightDragRef = useRef(false);

  const editMode = useNoteStore((s) => s.editMode);

  const showPlaceholder = useNoteStore(
    (s) => s.notes.length === 0 && s.generatedNotes.length === 0
           && s.recordingState !== 'recording' && !s.isPlaying,
  );

  const canEdit = useCallback(() => {
    const s = useNoteStore.getState();
    return (
      s.editMode
      && s.recordingState !== 'recording'
      && s.recordingState !== 'counting_in'
      && !s.isPlaying
    );
  }, []);

  const applyNoteUpdate = useCallback(
    (source: NoteSource, index: number, partial: Partial<NoteEvent>) => {
      const store = useNoteStore.getState();
      if (source === 'notes') {
        store.updateNote(index, partial);
      } else {
        store.updateGeneratedNote(index, partial);
      }
    },
    [],
  );

  const deleteNoteAt = useCallback((source: NoteSource, index: number) => {
    const store = useNoteStore.getState();
    if (source === 'notes') {
      store.deleteNote(index);
    } else {
      store.deleteGeneratedNote(index);
    }
  }, []);

  // rAF draw loop — reads store directly, no React re-renders
  useEffect(() => {
    const dpr = window.devicePixelRatio || 1;

    const drawBlock = (
      ctx: CanvasRenderingContext2D,
      n: NoteEvent,
      fill: string,
      stroke: string,
      pxPerSec: number,
    ) => {
      const x = n.startTime * pxPerSec;
      const w = Math.max(n.duration * pxPerSec, 4);
      const row = MAX_PITCH - 1 - n.pitch;
      if (row < 0 || row >= TOTAL_ROWS) return;
      const y = row * ROW_HEIGHT + 1;
      const h = ROW_HEIGHT - 2;
      ctx.fillStyle = fill;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(x + 0.5, y, w - 1, h, 2);
      ctx.fill();
      ctx.stroke();
    };

    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);

      const canvas = canvasRef.current;
      if (!canvas) return;

      const state = useNoteStore.getState();
      const ct = cursorTimeRef.current;
      const { notes, generatedNotes, tempo, recordingState, isPlaying, activeNotes, editMode: em } = state;
      const showCursor = recordingState === 'recording' || isPlaying;
      const pxPerSec = PIXELS_PER_BEAT * (tempo / 60);

      const maxEnd = [...notes, ...generatedNotes].reduce(
        (m, n) => Math.max(m, n.startTime + n.duration), 0,
      );
      const cursorPx = showCursor ? ct * pxPerSec : 0;
      const rawWidth = Math.max(maxEnd * pxPerSec + 400, cursorPx + 400, MIN_WIDTH);
      const contentWidth = Math.ceil(rawWidth / WIDTH_CHUNK) * WIDTH_CHUNK;

      const targetW = Math.round(contentWidth * dpr);
      const targetH = Math.round(CANVAS_HEIGHT * dpr);
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
        canvas.style.width = `${contentWidth}px`;
        canvas.style.height = `${CANVAS_HEIGHT}px`;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const gridKey = `${contentWidth}:${tempo}`;
      let gc = gridCache.current;
      if (!gc || gc.key !== gridKey) {
        gc = { canvas: buildGrid(contentWidth, tempo, dpr), key: gridKey };
        gridCache.current = gc;
      }
      ctx.drawImage(gc.canvas, 0, 0, contentWidth, CANVAS_HEIGHT);

      for (const n of notes) drawBlock(ctx, n, NOTE_FILL, NOTE_STROKE, pxPerSec);
      for (const n of generatedNotes) drawBlock(ctx, n, GEN_FILL, GEN_STROKE, pxPerSec);

      if (recordingState === 'recording') {
        activeNotes.forEach((info, pitch) => {
          const row = MAX_PITCH - 1 - pitch;
          if (row < 0 || row >= TOTAL_ROWS) return;
          const x = info.startTime * pxPerSec;
          const dur = Math.max(ct - info.startTime, 0);
          const w = Math.max(dur * pxPerSec, 4);
          const y = row * ROW_HEIGHT + 1;
          const h = ROW_HEIGHT - 2;
          ctx.fillStyle = ACTIVE_FILL;
          ctx.strokeStyle = ACTIVE_STROKE;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.roundRect(x + 0.5, y, w - 1, h, 2);
          ctx.fill();
          ctx.stroke();
        });
      } else if (activeNotes.size > 0) {
        const sc = gridRef.current;
        const viewLeft = sc ? sc.scrollLeft : 0;
        const previewX = viewLeft + 8;
        const previewW = 30;
        activeNotes.forEach((_, pitch) => {
          const row = MAX_PITCH - 1 - pitch;
          if (row < 0 || row >= TOTAL_ROWS) return;
          const y = row * ROW_HEIGHT + 1;
          const h = ROW_HEIGHT - 2;
          ctx.fillStyle = ACTIVE_FILL;
          ctx.strokeStyle = ACTIVE_STROKE;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.roundRect(previewX, y, previewW, h, 2);
          ctx.fill();
          ctx.stroke();
        });
      }

      if (showCursor) {
        const cx = ct * pxPerSec;
        ctx.fillStyle = CURSOR_CLR;
        ctx.fillRect(cx - 1, 0, 2.5, CANVAS_HEIGHT);
      }

      if (em && !showCursor) {
        ctx.font = '10px Inter, system-ui, sans-serif';
        ctx.fillStyle = '#a78bfa90';
        ctx.textBaseline = 'top';
        ctx.fillText('EDIT', contentWidth - 36, 4);
      }

      if (showCursor && gridRef.current) {
        const sc = gridRef.current;
        const cx = ct * pxPerSec;
        const target = cx - sc.clientWidth * 0.6;
        const clamped = Math.max(0, target);
        if (Math.abs(clamped - lastScrollX.current) > 10) {
          sc.scrollLeft = clamped;
          lastScrollX.current = clamped;
        }
      }
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  useEffect(() => {
    if (didInitScroll.current) return;
    const frame = requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (didInitScroll.current || !gridRef.current) return;
        didInitScroll.current = true;
        const sc = gridRef.current;
        const c4Row = MAX_PITCH - 1 - 60;
        const targetY = c4Row * ROW_HEIGHT - sc.clientHeight / 2;
        sc.scrollTop = Math.max(0, targetY);
        if (labelRef.current) labelRef.current.scrollTop = sc.scrollTop;
      });
    });
    return () => cancelAnimationFrame(frame);
  }, []);

  const handleGridScroll = useCallback(() => {
    if (gridRef.current && labelRef.current) {
      labelRef.current.scrollTop = gridRef.current.scrollTop;
    }
  }, []);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!canEdit() || !gridRef.current) return;
      const grid = gridRef.current;
      const { tempo, notes, generatedNotes } = useNoteStore.getState();
      const pxPerSec = PIXELS_PER_BEAT * (tempo / 60);
      const beatDuration = 60 / tempo;
      const { x, y, time, pitch } = screenToGrid(e.clientX, e.clientY, grid, pxPerSec);
      const hit = hitTest(x, y, notes, generatedNotes, pxPerSec);

      if (e.button === 2) {
        e.preventDefault();
        rightDragRef.current = true;
        grid.style.cursor = 'not-allowed';
        if (hit.kind === 'note') {
          deleteNoteAt(hit.source, hit.index);
        }
        return;
      }

      if (e.button !== 0) return;

      if (hit.kind === 'empty') {
        const clampedPitch = Math.max(MIN_PITCH, Math.min(MAX_PITCH - 1, pitch));
        const beatStart = Math.max(0, Math.floor(time / beatDuration) * beatDuration);
        const note: NoteEvent = {
          pitch: clampedPitch,
          velocity: DEFAULT_VELOCITY,
          startTime: beatStart,
          duration: beatDuration,
        };
        useNoteStore.getState().addNote(note);
        return;
      }

      const list = hit.source === 'notes' ? notes : generatedNotes;
      const origNote = { ...list[hit.index] };
      dragRef.current = {
        kind: hit.zone === 'right-edge' ? 'resize' : 'move',
        source: hit.source,
        index: hit.index,
        startX: x,
        startY: y,
        origNote,
      };
      grid.style.cursor = hit.zone === 'right-edge' ? 'ew-resize' : 'grabbing';
      e.preventDefault();
    },
    [canEdit, deleteNoteAt],
  );

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      const grid = gridRef.current;
      if (!grid) return;

      if (rightDragRef.current) {
        if (!canEdit()) {
          rightDragRef.current = false;
          grid.style.cursor = '';
          return;
        }
        const { tempo, notes, generatedNotes } = useNoteStore.getState();
        const pxPerSec = PIXELS_PER_BEAT * (tempo / 60);
        const { x, y } = screenToGrid(e.clientX, e.clientY, grid, pxPerSec);
        const hit = hitTest(x, y, notes, generatedNotes, pxPerSec);
        if (hit.kind === 'note') {
          deleteNoteAt(hit.source, hit.index);
        }
        return;
      }

      const drag = dragRef.current;
      if (drag) {
        const { tempo } = useNoteStore.getState();
        const pxPerSec = PIXELS_PER_BEAT * (tempo / 60);
        const sixteenth = 60 / tempo / 4;
        const { x, y } = screenToGrid(e.clientX, e.clientY, grid, pxPerSec);
        const dx = x - drag.startX;
        const dy = y - drag.startY;
        const orig = drag.origNote;

        if (drag.kind === 'move') {
          const dt = dx / pxPerSec;
          const rowDelta = Math.round(dy / ROW_HEIGHT);
          const newPitch = Math.max(
            MIN_PITCH,
            Math.min(MAX_PITCH - 1, orig.pitch - rowDelta),
          );
          const newStart = snapTime(orig.startTime + dt, sixteenth);
          applyNoteUpdate(drag.source, drag.index, {
            pitch: newPitch,
            startTime: newStart,
          });
        } else {
          const newDurPx = Math.max(
            orig.duration * pxPerSec + dx,
            sixteenth * pxPerSec,
          );
          const newDuration = snapTime(newDurPx / pxPerSec, sixteenth);
          applyNoteUpdate(drag.source, drag.index, {
            duration: Math.max(sixteenth, newDuration),
          });
        }
        return;
      }

      if (!canEdit()) {
        grid.style.cursor = '';
        return;
      }

      const { tempo, notes, generatedNotes } = useNoteStore.getState();
      const pxPerSec = PIXELS_PER_BEAT * (tempo / 60);
      const { x, y } = screenToGrid(e.clientX, e.clientY, grid, pxPerSec);
      const hit = hitTest(x, y, notes, generatedNotes, pxPerSec);
      grid.style.cursor = cursorForHit(hit);
    };

    const onMouseUp = () => {
      const wasDragging = dragRef.current != null || rightDragRef.current;
      dragRef.current = null;
      rightDragRef.current = false;
      if (wasDragging && gridRef.current) {
        gridRef.current.style.cursor = '';
      }
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [canEdit, applyNoteUpdate, deleteNoteAt]);

  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      if (canEdit()) e.preventDefault();
    },
    [canEdit],
  );

  const handleGridMouseLeave = useCallback(() => {
    if (!dragRef.current && gridRef.current) {
      gridRef.current.style.cursor = '';
    }
  }, []);

  const labels = useMemo(() => {
    const result: JSX.Element[] = [];
    for (let i = 0; i < TOTAL_ROWS; i++) {
      const pitch = MAX_PITCH - 1 - i;
      const isC = pitch % 12 === 0;
      const isBlack = BLACK_KEYS.has(pitch % 12);
      result.push(
        <div
          key={pitch}
          className={`pr-label ${isBlack ? 'pr-label-black' : ''} ${isC ? 'pr-label-c' : ''}`}
          style={{ height: ROW_HEIGHT }}
        >
          {isC ? midiToNoteName(pitch) : ''}
        </div>,
      );
    }
    return result;
  }, []);

  return (
    <div
      className={`piano-roll ${editMode ? 'edit-mode' : ''}`}
      data-help-title="Piano roll"
      data-help={
        editMode
          ? 'Edit mode is active. Left-click to add a quarter note, drag a note to move it, drag its right edge to resize, right-click (or right-drag) to delete.'
          : 'Scrolling note grid showing recorded notes (blue) and AI-generated notes (purple). Enable Edit mode to modify notes directly.'
      }
    >
      <div className="pr-labels" ref={labelRef}>
        {labels}
      </div>
      <div
        className="pr-grid"
        ref={gridRef}
        onScroll={handleGridScroll}
        onMouseDown={handleMouseDown}
        onContextMenu={handleContextMenu}
        onMouseLeave={handleGridMouseLeave}
      >
        <canvas ref={canvasRef} />
      </div>
      {showPlaceholder && (
        <div className="pr-placeholder">
          Start recording to see notes on the grid...
        </div>
      )}
    </div>
  );
}
