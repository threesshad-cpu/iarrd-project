import { useState, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';

/**
 * ImageComparisonSlider
 * @param {string}  originalSrc      — left-side image URL
 * @param {string}  reconstructedSrc — right-side image URL
 * @param {string}  leftLabel        — label on the left half  (default "RAW INPUT")
 * @param {string}  rightLabel       — label on the right half (default "RECONSTRUCTED")
 * @param {string}  aspectRatio      — CSS aspect-ratio (default "auto" — natural image height)
 * @param {string}  minHeight        — CSS min-height for the wrapper (default "260px")
 */
const ImageComparisonSlider = ({
  originalSrc,
  reconstructedSrc,
  leftLabel  = 'RAW INPUT',
  rightLabel = 'RECONSTRUCTED',
  aspectRatio = 'auto',
  minHeight   = '260px',
}) => {
  const [sliderPosition, setSliderPosition] = useState(50);
  const isDragging   = useRef(false);
  const containerRef = useRef(null);

  const handleMove = useCallback((clientX) => {
    if (!isDragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pct  = ((clientX - rect.left) / rect.width) * 100;
    setSliderPosition(Math.max(1, Math.min(99, pct)));
  }, []);

  return (
    <div className="img-cmp-root">
      <div
        ref={containerRef}
        className="img-cmp-wrapper"
        style={{ aspectRatio, minHeight }}
        onMouseMove={(e) => handleMove(e.clientX)}
        onMouseDown={() => { isDragging.current = true; }}
        onMouseUp={() => { isDragging.current = false; }}
        onMouseLeave={() => { isDragging.current = false; }}
        onTouchMove={(e) => handleMove(e.touches[0].clientX)}
        onTouchStart={() => { isDragging.current = true; }}
        onTouchEnd={() => { isDragging.current = false; }}
      >
        {/* Background — RIGHT side (reconstructed / enhanced) */}
        <img
          src={reconstructedSrc}
          alt={rightLabel}
          className="img-cmp-base"
          style={{ imageRendering: 'high-quality' }}
        />

        {/* Foreground — LEFT side (original, revealed by clip) */}
        <div
          className="img-cmp-clip"
          style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
        >
          <img
            src={originalSrc}
            alt={leftLabel}
            className="img-cmp-base"
            style={{ imageRendering: 'high-quality' }}
          />
        </div>

        {/* Divider */}
        <div className="img-cmp-line" style={{ left: `${sliderPosition}%` }} />

        {/* Handle */}
        <motion.div
          className="img-cmp-handle"
          style={{ left: `${sliderPosition}%` }}
          whileHover={{ scale: 1.12 }}
          whileTap={{ scale: 0.92 }}
          onMouseDown={(e) => { e.stopPropagation(); isDragging.current = true; }}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z"/>
            <path d="M8.59 16.59L10 18l6-6-6-6-1.41 1.41L13.17 12z"/>
          </svg>
        </motion.div>

        {/* Labels */}
        <div className="img-cmp-label img-cmp-label-left">{leftLabel}</div>
        <div className="img-cmp-label img-cmp-label-right">{rightLabel}</div>
      </div>
      <div className="img-cmp-caption">← Drag handle to compare →</div>
    </div>
  );
};

export default ImageComparisonSlider;
