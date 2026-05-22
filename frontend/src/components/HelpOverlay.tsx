import { useEffect, useState } from 'react';
import { useNoteStore } from '../lib/noteStore';

interface Tip {
  x: number;
  y: number;
  title: string;
  body: string;
}

const TOOLTIP_OFFSET_X = 16;
const TOOLTIP_OFFSET_Y = 18;
const TOOLTIP_MAX_WIDTH = 280;
const TOOLTIP_FALLBACK_HEIGHT = 80;

export default function HelpOverlay() {
  const helpMode = useNoteStore((s) => s.helpMode);
  const [tip, setTip] = useState<Tip | null>(null);

  useEffect(() => {
    document.body.classList.toggle('help-mode-active', helpMode);
    return () => document.body.classList.remove('help-mode-active');
  }, [helpMode]);

  useEffect(() => {
    if (!helpMode) {
      setTip(null);
      return;
    }

    const onMove = (e: MouseEvent) => {
      const target = (e.target as HTMLElement | null)?.closest<HTMLElement>('[data-help]');
      if (!target) {
        setTip(null);
        return;
      }

      const body = target.getAttribute('data-help') ?? '';
      const title = target.getAttribute('data-help-title') ?? '';
      if (!body) {
        setTip(null);
        return;
      }

      let x = e.clientX + TOOLTIP_OFFSET_X;
      let y = e.clientY + TOOLTIP_OFFSET_Y;
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      if (x + TOOLTIP_MAX_WIDTH > vw) x = Math.max(8, e.clientX - TOOLTIP_MAX_WIDTH - 8);
      if (y + TOOLTIP_FALLBACK_HEIGHT > vh) y = Math.max(8, e.clientY - TOOLTIP_FALLBACK_HEIGHT);

      setTip({ x, y, title, body });
    };

    const onLeave = () => setTip(null);

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseleave', onLeave);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseleave', onLeave);
    };
  }, [helpMode]);

  if (!helpMode) return null;

  return (
    <>
      <div className="help-mode-banner" role="status">
        Help mode: hover any control to see what it does. Press Esc or click the
        <span className="help-mode-banner-pill">?</span>
        button again to exit.
      </div>
      {tip && (
        <div
          className="help-tooltip"
          style={{ left: tip.x, top: tip.y, maxWidth: TOOLTIP_MAX_WIDTH }}
          role="tooltip"
        >
          {tip.title && <div className="help-tooltip-title">{tip.title}</div>}
          <div className="help-tooltip-body">{tip.body}</div>
        </div>
      )}
    </>
  );
}
