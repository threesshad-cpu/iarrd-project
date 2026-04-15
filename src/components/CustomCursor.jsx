import React, { useRef, useEffect } from 'react';

/**
 * Dual-layer custom cursor:
 *  - Small dot  → follows mouse at native speed
 *  - Larger ring → lags behind (spring physics via requestAnimationFrame)
 */
export function CustomCursor() {
  const dotRef  = useRef(null);
  const ringRef = useRef(null);
  const pos     = useRef({ x: 0, y: 0 });
  const ring    = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const onMove = (e) => {
      pos.current = { x: e.clientX, y: e.clientY };
      // Dot snaps immediately
      if (dotRef.current) {
        dotRef.current.style.left = `${e.clientX}px`;
        dotRef.current.style.top  = `${e.clientY}px`;
      }

      // Expand ring on interactive elements
      const tag = e.target.tagName.toLowerCase();
      const interactive = ['a','button','label','input','select','textarea'].includes(tag)
        || e.target.closest('a, button, label');
      if (ringRef.current) {
        ringRef.current.classList.toggle('hover', !!interactive);
      }
    };

    // RAF loop for ring spring
    let raf;
    const loop = () => {
      ring.current.x += (pos.current.x - ring.current.x) * 0.1;
      ring.current.y += (pos.current.y - ring.current.y) * 0.1;
      if (ringRef.current) {
        ringRef.current.style.left = `${ring.current.x}px`;
        ringRef.current.style.top  = `${ring.current.y}px`;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    window.addEventListener('mousemove', onMove);
    return () => {
      window.removeEventListener('mousemove', onMove);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <>
      <div ref={dotRef}  className="cursor-dot" />
      <div ref={ringRef} className="cursor-ring" />
    </>
  );
}
