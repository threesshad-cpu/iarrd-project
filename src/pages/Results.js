import React, { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { Link } from 'react-router-dom';
import {
  Target, BarChart3, Zap, MapPin,
  DownloadCloud, RotateCcw, Database,
  Eye, EyeOff, Activity, Layers, ScanSearch, FileJson, Share2,
  X, ZoomIn, FileText, Brain, Globe, CheckCircle,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { PageTransition, FadeUp } from '../components/Animations';
import ImageComparisonSlider from '../components/ImageComparisonSlider';
import { getLastResult } from '../ResultsStore';

/* ── Zoom Modal ──────────────────────────────────────────────────── */
function ZoomModal({ src, label, onClose }) {
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        style={{
          position: 'fixed', inset: 0, zIndex: 9999,
          background: 'rgba(3,3,18,0.92)',
          backdropFilter: 'blur(12px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '2rem',
        }}
      >
        <motion.div
          initial={{ scale: 0.88, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.88, opacity: 0 }}
          transition={{ ease: [0.16, 1, 0.3, 1], duration: 0.35 }}
          onClick={e => e.stopPropagation()}
          style={{ position: 'relative', maxWidth: '90vw', maxHeight: '85vh' }}
        >
          <button
            onClick={onClose}
            style={{
              position: 'absolute', top: -40, right: 0,
              background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: '50%', width: 32, height: 32,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: '#fff', cursor: 'pointer',
            }}
          >
            <X size={15} />
          </button>
          <img
            src={src}
            alt={label}
            style={{ display: 'block', maxWidth: '100%', maxHeight: '80vh', objectFit: 'contain', borderRadius: 12, border: '1px solid rgba(255,255,255,0.12)' }}
          />
          <div style={{ textAlign: 'center', marginTop: 10, fontFamily: 'var(--font-mono)', fontSize: '0.62rem', color: 'var(--text-lo)', letterSpacing: '0.1em' }}>
            {label} — ESC to close
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

/* ── Browser-mode classifier ─────────────────────────────────── */
function classify(obj) {
  if (obj.area > 20000)     return { label: 'Large Structure',  color: '#7c3aed' };
  if (obj.brightness > 220) return { label: 'Bright Source',   color: '#38bdf8' };
  if (obj.area > 8000)      return { label: 'Diffuse Region',  color: '#1d4ed8' };
  return                           { label: 'Point Object',    color: '#475569' };
}

/* ── Label colours ───────────────────────────────────────────── */
const LABEL_COLORS = {
  'Galaxy':            '#7c3aed',
  'Star Cluster':      '#38bdf8',
  'Nebula':            '#f59e0b',
  'Quasar':            '#ef4444',
  'Supernova Remnant': '#10b981',
  'Unknown Object':    '#475569',
};
const getColor = (label) => LABEL_COLORS[label] ?? '#38bdf8';

/* ── Animated number — 0.8s (was 1.2s) ──────────────────────── */
function AnimNum({ value, format = v => v, duration = 0.8 }) {
  const [displayed, setDisplayed] = useState(0);
  useEffect(() => {
    const start = performance.now();
    const to = Number(value) || 0;
    const tick = (now) => {
      const t = Math.min((now - start) / (duration * 1000), 1);
      const ease = 1 - Math.pow(1 - t, 3);
      setDisplayed(to * ease);
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [value, duration]);
  return <>{format(displayed)}</>;
}

/* ── Shared card entrance — no blur (GPU cost removed) ──────── */
const cardVariants = {
  hidden: { opacity: 0, y: 18 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] } },
};


/* ═══════════════════════════════════════════════════════════════ */
/*  ENHANCEMENT PANEL (user-facing: "Enhanced Image")             */
/* ═══════════════════════════════════════════════════════════════ */
function EnhancementPanel({ enhancement, originalImage }) {
  const { image_b64, meta, premium_chain: pc } = enhancement;
  const [zoom, setZoom] = useState(false);
  const [zoomSrc, setZoomSrc] = useState(null);
  const [zoomLabel, setZoomLabel] = useState('');
  // detect JPEG vs PNG from backend
  const mimeType = image_b64?.startsWith('/9j/') ? 'image/jpeg' : 'image/jpeg';
  const src = `data:${mimeType};base64,${image_b64}`;

  const pcFinalSrc = pc?.enhanced_b64 ? `data:image/png;base64,${pc.enhanced_b64}` : null;

  const download = () => {
    const a = document.createElement('a');
    a.href = src;
    a.download = 'iarrd-enhanced.jpg';
    a.click();
  };

  /* Quality badges — human-readable labels only */
  const qualityBadges = pc ? [
    { label: 'Clarity Score',    value: pc.psnr != null ? `${pc.psnr} dB` : '—',          color: '#10b981', desc: 'Higher = cleaner image' },
    { label: 'Noise Reduced',    value: pc.noise_reduction_pct != null ? `${pc.noise_reduction_pct}%` : '—', color: '#a78bfa', desc: 'Sensor noise removed' },
    { label: 'Contrast Boost',   value: pc.contrast_boost_pct != null ? `${pc.contrast_boost_pct > 0 ? '+' : ''}${pc.contrast_boost_pct}%` : '—', color: '#f59e0b', desc: 'Dynamic range improved' },
  ] : [];

  return (
    <motion.div
      className="glass enhancement-panel"
      variants={cardVariants} initial="hidden" animate="show"
    >
      {/* Header */}
      <div className="enhancement-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div className="enhancement-icon-wrap">
            <ZoomIn size={18} style={{ color: '#10b981' }} />
          </div>
          <div>
            <div className="enhancement-title">ENHANCED IMAGE</div>
            <div className="enhancement-sub">
              {meta?.native_resolution} → {meta?.output_resolution}
              &nbsp;·&nbsp; clarity optimised
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <motion.button
            className="btn-enhancement-view"
            onClick={() => setZoom(true)}
            whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
          >
            <ZoomIn size={12} /> Full View
          </motion.button>
          <motion.button
            className="btn-enhancement-dl"
            onClick={download}
            whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
          >
            <DownloadCloud size={13} /> Download
          </motion.button>
        </div>
      </div>

      {/* Enhanced image */}
      <div
        style={{ position: 'relative', borderRadius: 12, overflow: 'hidden', border: '1.5px solid rgba(16,185,129,0.35)', boxShadow: '0 0 40px rgba(16,185,129,0.12)', cursor: 'zoom-in', marginBottom: '1.2rem' }}
        onClick={() => setZoom(true)}
        title="Click to view full resolution"
      >
        <img
          src={src}
          alt="Enhanced"
          style={{ width: '100%', display: 'block', objectFit: 'contain', maxHeight: 420 }}
        />
        <div style={{ position: 'absolute', bottom: 8, right: 12, fontFamily: 'var(--font-mono)', fontSize: '0.52rem', color: '#10b981', background: 'rgba(0,0,0,0.75)', padding: '3px 8px', borderRadius: 5, display: 'flex', alignItems: 'center', gap: 5, pointerEvents: 'none' }}>
          <ZoomIn size={10} /> Click to enlarge
        </div>
      </div>

      {/* Quality badges */}
      {qualityBadges.length > 0 && (
        <div style={{ display: 'flex', gap: '0.45rem', flexWrap: 'wrap', marginBottom: '1.2rem' }}>
          {qualityBadges.map(({ label, value, color, desc }) => (
            <div key={label} style={{ flex: '1 1 90px', padding: '0.55rem 0.6rem', borderRadius: 10, background: `${color}0a`, border: `1px solid ${color}28`, textAlign: 'center' }}>
              <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.46rem', letterSpacing: '0.12em', color: 'var(--text-lo)', marginBottom: 3 }}>{label}</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.88rem', color, fontWeight: 700, textShadow: `0 0 12px ${color}66` }}>{value}</div>
              <div style={{ fontSize: '0.5rem', color: 'var(--text-lo)', marginTop: 2 }}>{desc}</div>
            </div>
          ))}
        </div>
      )}

      {/* Zoom Modal */}
      {zoomSrc && <ZoomModal src={zoomSrc} label={zoomLabel} onClose={() => setZoomSrc(null)} />}
      {zoom && <ZoomModal src={src} label="Enhanced Image — Full Resolution" onClose={() => setZoom(false)} />}
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
/*  CLASSIFICATION PANEL                                           */
/* ═══════════════════════════════════════════════════════════════ */
function ClassificationPanel({ classification, minConf = 0 }) {
  const { label, confidence, all_scores, confidence_flag } = classification;
  const color = getColor(label);

  const flagColor = confidence_flag === 'high'   ? '#10b981' :
                    confidence_flag === 'medium' ? '#f59e0b' : '#ef4444';
  const flagLabel = confidence_flag === 'high'   ? 'Strong Match' :
                    confidence_flag === 'medium' ? 'Probable Match' : 'Low Certainty';

  return (
    <motion.div
      className="glass cnn-panel result-card"
      style={{ borderColor: `${color}33` }}
      variants={cardVariants} initial="hidden" animate="show"
    >
      <div className="panel-header">
        <div className="panel-title">
          <Target size={13} strokeWidth={1.5} style={{ color: 'var(--sky)' }} />
          OBJECT CLASSIFICATION
        </div>
        {confidence_flag && (
          <span style={{ fontSize: '0.52rem', fontFamily: 'var(--font-mono)', color: flagColor, padding: '2px 8px', borderRadius: 999, border: `1px solid ${flagColor}55`, background: `${flagColor}12`, letterSpacing: '0.08em' }}>
            {flagLabel}
          </span>
        )}
      </div>

      {/* Top prediction */}
      <div className="cnn-label-row">
        <div className="cnn-label-dot" style={{ background: color, boxShadow: `0 0 18px ${color}` }} />
        <div>
          <div className="cnn-label-name gradient-label" style={{
            background: `linear-gradient(135deg, ${color} 0%, #8b5cf6 100%)`,
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
            fontSize: '1.15rem', fontWeight: 700, letterSpacing: '-0.01em',
          }}>{label}</div>
          <div className="cnn-label-conf">
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', color: '#10b981', fontWeight: 700, textShadow: '0 0 12px rgba(16,185,129,0.6)' }}>
              <AnimNum value={confidence} format={v => `${v.toFixed(1)}%`} />
            </span>{' '}confidence
          </div>
        </div>
      </div>

      {/* All scores */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
        {Object.entries(all_scores)
          .sort(([,a],[,b]) => b - a)
          .filter(([, score]) => score >= minConf)
          .map(([lbl, score], idx) => {
            const c = getColor(lbl);
            const isTop = lbl === label;
            return (
              <div key={lbl} className="score-row">
                <div className="score-row-header">
                  <span className={`score-name ${isTop ? 'top' : ''}`}>{lbl}</span>
                  <span className="score-pct mono" style={{ color: isTop ? c : 'var(--text-lo)' }}>
                    {score.toFixed(1)}%
                  </span>
                </div>
                <div className="score-track">
                  <motion.div
                    className={`score-fill ${isTop ? 'shimmer-bar' : ''}`}
                    style={{ background: isTop ? `linear-gradient(90deg, ${c}cc, ${c})` : c, boxShadow: isTop ? `0 0 12px ${c}88` : 'none' }}
                    initial={{ width: 0 }}
                    animate={{ width: `${score}%` }}
                    transition={{ duration: 0.8, ease: [0.16,1,0.3,1], delay: 0.05 + idx * 0.04 }}
                  />
                </div>
              </div>
            );
          })}
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
/*  IMAGE QUALITY PANEL  (was "Autoencoder")                       */
/* ═══════════════════════════════════════════════════════════════ */
function AnomalyPanel({ reconstruction, originalImage }) {
  const {
    anomaly_score, quality, image_b64,
    noise_reduction_pct, gaussian_preview_b64,
  } = reconstruction;
  const [zoomSrc, setZoomSrc] = useState(null);
  const [zoomLabel, setZoomLabel] = useState('');
  const [useSlider, setUseSlider] = useState(true);
  const scoreColor =
    quality === 'Normal'   ? '#10b981' :
    quality === 'Elevated' ? '#f59e0b' : '#ef4444';

  // Human-readable quality description
  const qualityDesc = quality === 'Normal' ? 'Clean image — minimal interference' :
                      quality === 'Elevated' ? 'Moderate noise detected' :
                      'High noise / image anomaly';

  const reconSrc    = image_b64            ? `data:image/png;base64,${image_b64}` : null;
  const gaussianSrc = gaussian_preview_b64 ? `data:image/png;base64,${gaussian_preview_b64}` : null;

  return (
    <motion.div
      className="glass cnn-panel result-card"
      variants={cardVariants} initial="hidden" animate="show"
    >
      <div className="panel-header">
        <div className="panel-title">
          <Activity size={13} strokeWidth={1.5} style={{ color: 'var(--sky)' }} />
          IMAGE QUALITY REPORT
        </div>
      </div>

      {/* Noise gauge */}
      <div className="anomaly-gauge-wrap">
        <div className="anomaly-ring" style={{ borderColor: scoreColor, boxShadow: `0 0 28px ${scoreColor}44, inset 0 0 18px ${scoreColor}18` }}>
          <div className="anomaly-score-val" style={{ color: scoreColor }}>
            <AnimNum value={anomaly_score} format={v => v.toFixed(1)} />
          </div>
          <div className="anomaly-max">/100</div>
        </div>
        <div>
          <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.48rem', letterSpacing: '0.12em', color: 'var(--text-lo)', marginBottom: 3 }}>NOISE LEVEL</div>
          <div className="anomaly-quality" style={{ color: scoreColor, textShadow: `0 0 18px ${scoreColor}88` }}>{quality}</div>
          <div className="anomaly-desc">{qualityDesc}</div>
        </div>
      </div>

      {/* 3-way: Original → Filtered → Enhanced */}
      {originalImage && gaussianSrc && reconSrc && (
        <div style={{ marginBottom: '1rem' }}>
          <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.54rem', letterSpacing: '0.14em', color: 'var(--text-lo)', marginBottom: '0.6rem' }}>
            IMAGE PROCESSING STEPS
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '0.45rem' }}>
            {[
              { src: originalImage, label: 'ORIGINAL',      borderColor: 'rgba(255,255,255,0.18)', textColor: 'rgba(255,255,255,0.6)',  zLbl: 'Original Image' },
              { src: gaussianSrc,  label: 'NOISE REMOVED',  borderColor: '#38bdf8',               textColor: '#7dd3fc',                zLbl: 'Noise Removed' },
              { src: reconSrc,     label: 'ENHANCED',       borderColor: '#10b981',               textColor: '#6ee7b7',                zLbl: 'Enhanced Output' },
            ].map(({ src, label, borderColor, textColor, zLbl }) => (
              <div
                key={label}
                style={{ position: 'relative', cursor: 'zoom-in', borderRadius: 8, overflow: 'hidden', border: `1px solid ${borderColor}` }}
                onClick={() => { setZoomSrc(src); setZoomLabel(zLbl); }}
              >
                <img src={src} alt={label} style={{ width: '100%', display: 'block', aspectRatio: '1 / 1', objectFit: 'cover' }} />
                <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '4px 5px', background: 'rgba(3,3,18,0.78)', fontFamily: 'var(--font-mono)', fontSize: '0.48rem', color: textColor }}>
                  {label}
                </div>
                <div style={{ position: 'absolute', top: 4, right: 4 }}>
                  <ZoomIn size={9} style={{ color: 'rgba(255,255,255,0.5)' }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Before / After slider */}
      {image_b64 && originalImage && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div className="recon-label" style={{ marginBottom: 0 }}>BEFORE vs AFTER</div>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <button
                onClick={() => setUseSlider(s => !s)}
                style={{ background: useSlider ? 'rgba(56,189,248,0.12)' : 'transparent', border: '1px solid rgba(56,189,248,0.3)', borderRadius: 6, padding: '3px 8px', cursor: 'pointer', color: 'var(--sky)', fontSize: '0.58rem', fontFamily: 'var(--font-mono)', transition: 'background 0.2s' }}
              >
                {useSlider ? '⇔ Slider' : '⊞ Split'}
              </button>
            </div>
          </div>
          {useSlider ? (
            <ImageComparisonSlider originalSrc={originalImage} reconstructedSrc={reconSrc} />
          ) : (
            <div className="recon-images">
              <div className="recon-img-wrap" style={{ cursor: 'zoom-in', position: 'relative' }} onClick={() => { setZoomSrc(originalImage); setZoomLabel('Original'); }}>
                <img src={originalImage} alt="Original" style={{ width: '100%', display: 'block' }} />
                <div className="recon-img-tag">ORIGINAL</div>
              </div>
              <div className="recon-img-wrap" style={{ cursor: 'zoom-in', position: 'relative' }} onClick={() => { setZoomSrc(reconSrc); setZoomLabel('Enhanced'); }}>
                <img src={reconSrc} alt="Enhanced" style={{ width: '100%', display: 'block' }} />
                <div className="recon-img-tag" style={{ background: 'rgba(16,185,129,0.15)', borderColor: 'rgba(16,185,129,0.4)', color: '#6ee7b7' }}>ENHANCED</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Noise bar */}
      <div style={{ marginTop: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
          <span style={{ fontFamily: 'var(--font-head)', fontSize: '0.58rem', letterSpacing: '0.14em', color: 'var(--text-lo)' }}>NOISE LEVEL</span>
          <span className="mono" style={{ fontSize: '0.65rem', color: scoreColor }}>{anomaly_score}/100</span>
        </div>
        <div style={{ height: 7, borderRadius: 7, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
          <motion.div
            className="shimmer-bar"
            style={{ height: '100%', borderRadius: 7, background: `linear-gradient(90deg, ${scoreColor}cc, ${scoreColor})`, boxShadow: `0 0 14px ${scoreColor}88` }}
            initial={{ width: 0 }}
            animate={{ width: `${anomaly_score}%` }}
            transition={{ duration: 0.9, ease: [0.16,1,0.3,1], delay: 0.15 }}
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: '0.58rem', color: 'var(--text-lo)', fontFamily: 'var(--font-mono)' }}>
          <span>0 — Clean</span><span>50 — Moderate</span><span>100 — Noisy</span>
        </div>
      </div>

      {zoomSrc && <ZoomModal src={zoomSrc} label={zoomLabel} onClose={() => setZoomSrc(null)} />}
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
/*  REGION MAP PANEL  (was "U-Net Segmentation")                   */
/* ═══════════════════════════════════════════════════════════════ */
function SegmentationPanel({ segmentation, originalImage }) {
  const { overlay_b64, coverage } = segmentation;
  const [showOverlay, setShowOverlay] = useState(true);
  const [zoomSrc, setZoomSrc] = useState(null);
  const [overlayOpacity, setOverlayOpacity] = useState(
    () => parseFloat(localStorage.getItem('iarrd_segOpacity') || '0.85')
  );

  const SEG_COLORS = {
    'Galaxy':             '#7c3aed',
    'Star Cluster':       '#38bdf8',
    'Nebula':             '#f59e0b',
    'Quasar':             '#ef4444',
    'Supernova Remnant':  '#10b981',
    'Unknown Object':     '#475569',
  };

  const allClasses = Object.keys(SEG_COLORS);
  const [visibleClasses, setVisibleClasses] = useState(() => {
    const init = {};
    allClasses.forEach(k => { init[k] = true; });
    return init;
  });

  const handleOpacity = (v) => {
    setOverlayOpacity(v);
    localStorage.setItem('iarrd_segOpacity', String(v));
  };

  const downloadOverlay = () => {
    if (!overlay_b64) return;
    const a = document.createElement('a');
    a.href = `data:image/png;base64,${overlay_b64}`;
    a.download = 'iarrd-region-map.png'; a.click();
  };

  const sorted = useMemo(() =>
    Object.entries(coverage).filter(([,v]) => v > 0.1).sort(([,a],[,b]) => b - a),
    [coverage]
  );

  return (
    <motion.div
      className="glass cnn-panel result-card"
      variants={cardVariants} initial="hidden" animate="show"
    >
      <div className="panel-header">
        <div className="panel-title">
          <Layers size={13} strokeWidth={1.5} style={{ color: 'var(--sky)' }} />
          REGION MAP
        </div>
        <motion.button
          className={`seg-toggle-btn ${showOverlay ? 'on' : ''}`}
          onClick={() => setShowOverlay(o => !o)}
          whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
        >
          {showOverlay ? <Eye size={11} /> : <EyeOff size={11} />}
          {showOverlay ? 'Overlay ON' : 'Overlay OFF'}
        </motion.button>
      </div>

      {overlay_b64 && originalImage && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.58rem', letterSpacing: '0.14em', color: 'var(--text-lo)' }}>DETECTED REGIONS</div>
            <motion.button
              onClick={downloadOverlay}
              whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
              style={{ background: 'rgba(52,211,153,0.1)', border: '1px solid rgba(52,211,153,0.3)', borderRadius: 6, padding: '3px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4, color: '#34d399', fontSize: '0.6rem', fontFamily: 'var(--font-mono)' }}
            >
              <DownloadCloud size={11} /> Save Map
            </motion.button>
          </div>

          <div className="seg-img-wrap" style={{ cursor: 'zoom-in', position: 'relative' }} onClick={() => setZoomSrc(showOverlay ? `data:image/png;base64,${overlay_b64}` : originalImage)}>
            <img src={originalImage} alt="Original" style={{ width: '100%', display: 'block' }} />
            <AnimatePresence>
              {showOverlay && (
                <motion.img
                  key="overlay"
                  src={`data:image/png;base64,${overlay_b64}`}
                  alt="Region overlay"
                  style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'fill', opacity: overlayOpacity }}
                  initial={{ opacity: 0 }} animate={{ opacity: overlayOpacity }} exit={{ opacity: 0 }}
                />
              )}
            </AnimatePresence>
            <div style={{ position: 'absolute', top: 6, right: 6, background: 'rgba(0,0,0,0.55)', borderRadius: 5, padding: '3px 5px', display: 'flex', alignItems: 'center', gap: 3 }}>
              <ZoomIn size={11} style={{ color: '#34d399' }} />
            </div>
          </div>

          {showOverlay && (
            <div style={{ marginTop: '0.8rem' }}>
              <div className="ctrl-label" style={{ marginBottom: 6 }}>
                Overlay Opacity
                <em style={{ fontStyle: 'normal', fontFamily: 'var(--font-mono)', fontSize: '0.62rem', color: 'var(--sky)' }}>
                  {Math.round(overlayOpacity * 100)}%
                </em>
              </div>
              <input type="range" className="slider" min="0.05" max="1" step="0.05" value={overlayOpacity} onChange={e => handleOpacity(Number(e.target.value))} />
            </div>
          )}

          <div className="seg-class-toggles">
            <span style={{ fontFamily: 'var(--font-head)', fontSize: '0.52rem', letterSpacing: '0.16em', color: 'var(--text-lo)', flexShrink: 0 }}>FILTER</span>
            {allClasses.map(cls => {
              const c = SEG_COLORS[cls];
              const active = visibleClasses[cls];
              return (
                <motion.button
                  key={cls}
                  whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                  onClick={() => setVisibleClasses(prev => ({ ...prev, [cls]: !prev[cls] }))}
                  style={{ padding: '3px 10px', background: active ? `${c}22` : 'rgba(255,255,255,0.03)', border: `1px solid ${active ? c : 'rgba(255,255,255,0.08)'}`, borderRadius: 999, color: active ? c : 'var(--text-lo)', fontSize: '0.6rem', cursor: 'pointer', fontFamily: 'var(--font-body)', transition: 'all 0.25s', boxShadow: active ? `0 0 10px ${c}44` : 'none', whiteSpace: 'nowrap' }}
                >
                  {cls}
                </motion.button>
              );
            })}
          </div>
        </div>
      )}

      {zoomSrc && <ZoomModal src={zoomSrc} label="Region Map" onClose={() => setZoomSrc(null)} />}

      {/* Coverage bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: '0.75rem' }}>
        {sorted
          .filter(([lbl]) => visibleClasses[lbl] !== false)
          .map(([lbl, pct], idx) => {
            const c = SEG_COLORS[lbl] ?? '#38bdf8';
            return (
              <div key={lbl} className="coverage-row">
                <div className="coverage-header">
                  <span className="coverage-name">{lbl}</span>
                  <span className="coverage-pct mono" style={{ color: c }}>{pct.toFixed(1)}%</span>
                </div>
                <div className="coverage-track">
                  <motion.div
                    className="shimmer-bar"
                    style={{ height: '100%', borderRadius: 5, background: `linear-gradient(90deg, ${c}aa, ${c})`, boxShadow: `0 0 8px ${c}66` }}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(pct * 3, 100)}%` }}
                    transition={{ duration: 0.8, ease: [0.16,1,0.3,1], delay: 0.05 + idx * 0.05 }}
                  />
                </div>
              </div>
            );
          })}
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
/*  IMAGE PROPERTIES PANEL  (was "Feature Extraction")            */
/* ═══════════════════════════════════════════════════════════════ */
function FeaturesPanel({ features }) {
  const { brightness, shape, size, color } = features;

  const statTiles = useMemo(() => [
    {
      icon: '☀', label: 'Peak Brightness',
      value: brightness.peak_normalized != null ? brightness.peak_normalized.toFixed(1) : (brightness.peak * 100).toFixed(1),
      unit: '/ 100', color: '#f59e0b',
      desc: 'Brightest point in the image',
    },
    {
      icon: '💡', label: 'Average Brightness',
      value: brightness.mean_luminosity != null ? brightness.mean_luminosity.toFixed(1) : (brightness.mean * 100).toFixed(1),
      unit: '/ 100', color: '#38bdf8',
      desc: 'Overall image brightness',
    },
    {
      icon: '⊕', label: 'Object Size (est.)',
      value: size.largest_region_px2 != null ? size.largest_region_px2.toLocaleString() : size.bright_pixels.toLocaleString(),
      unit: 'px²', color: '#34d399',
      desc: 'Largest detected bright region',
    },
    {
      icon: '◈', label: 'Shape Profile',
      value: shape.eccentricity != null ? shape.eccentricity.toFixed(3) : '—',
      unit: '', color: '#a78bfa',
      desc: '0 = circular  ·  1 = elongated',
    },
  ], [brightness, shape, size]);

  const detailRows = useMemo(() => [
    { label: 'Brightness Range', value: brightness.dynamic_range?.toFixed(3), c: '#f59e0b' },
    { label: 'Edge Sharpness',   value: shape.edge_energy?.toFixed(4),        c: '#a78bfa' },
    { label: 'Coverage',         value: size.fill_ratio?.toFixed(3),          c: '#34d399' },
    { label: 'Objects Detected', value: size.object_count_est,                c: '#38bdf8' },
    { label: 'Complexity',       value: shape.complexity,   c: '#a78bfa', isText: true },
    { label: 'Dominant Color',   value: color.dominant_channel, c: '#38bdf8', isText: true },
  ], [brightness, shape, size, color]);

  return (
    <motion.div
      className="glass cnn-panel result-card"
      variants={cardVariants} initial="hidden" animate="show"
      style={{ borderColor: 'rgba(245,158,11,0.25)' }}
    >
      <div className="panel-header">
        <div className="panel-title">
          <span style={{ fontSize: '0.9rem', lineHeight: 1 }}>📡</span>
          IMAGE PROPERTIES
        </div>
        <span style={{ fontSize: '0.52rem', fontFamily: 'var(--font-mono)', color: '#f59e0b', padding: '2px 8px', borderRadius: 999, border: '1px solid rgba(245,158,11,0.3)', background: 'rgba(245,158,11,0.08)', letterSpacing: '0.08em', whiteSpace: 'nowrap' }}>
          {color.spectral_hint?.split('(')[0]?.trim() ?? 'Balanced Spectrum'}
        </span>
      </div>

      {/* 4 stat tiles */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: '0.6rem', marginBottom: '0.9rem' }}>
        {statTiles.map(({ icon, label, value, unit, color: c, desc }) => (
          <div key={label} style={{ padding: '0.85rem 0.9rem', borderRadius: 12, background: `${c}09`, border: `1px solid ${c}28`, position: 'relative', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', top: 8, right: 10, fontSize: '1.15rem', opacity: 0.14 }}>{icon}</div>
            <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.5rem', letterSpacing: '0.14em', color: 'var(--text-lo)', marginBottom: 5 }}>{label}</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1.1rem', color: c, fontWeight: 700, lineHeight: 1 }}>{value}</span>
              {unit && <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.58rem', color: `${c}88` }}>{unit}</span>}
            </div>
            <div style={{ fontSize: '0.57rem', color: 'var(--text-lo)', marginTop: 4, lineHeight: 1.4 }}>{desc}</div>
          </div>
        ))}
      </div>

      {/* Detail rows */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(160px,1fr))', gap: '0.4rem' }}>
        {detailRows.map(({ label, value, c, isText }) => (
          <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '5px 8px', borderRadius: 7, background: 'rgba(255,255,255,0.025)' }}>
            <span style={{ fontSize: '0.6rem', color: 'var(--text-lo)' }}>{label}</span>
            <span style={{ fontFamily: isText ? 'inherit' : 'var(--font-mono)', fontSize: '0.64rem', color: c, fontWeight: 600 }}>{value}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
/*  ANALYZED IMAGE PANEL  (was "Stage 08 Labeled Output")          */
/* ═══════════════════════════════════════════════════════════════ */
function LabeledOutputPanel({ labeledOutput, classification }) {
  const [zoom, setZoom] = useState(false);
  // Support JPEG or PNG from backend
  const mimeType = labeledOutput.labeled_image_b64?.startsWith('/9j/') ? 'image/jpeg' : 'image/jpeg';
  const src   = `data:${mimeType};base64,${labeledOutput.labeled_image_b64}`;
  const label = classification?.label ?? 'Unknown';
  const color = LABEL_COLORS[label] ?? '#38bdf8';

  const download = () => {
    const a = document.createElement('a');
    a.href = src;
    a.download = `iarrd-analyzed-${label.toLowerCase().replace(/ /g, '-')}.jpg`;
    a.click();
  };

  return (
    <motion.div className="glass" variants={cardVariants} initial="hidden" animate="show"
      style={{ borderRadius: 18, padding: '1.4rem 1.6rem', borderColor: `${color}33` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 34, height: 34, borderRadius: 10, background: `${color}12`, border: `1.5px solid ${color}55`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1rem' }}>🏷️</div>
          <div>
            <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.62rem', letterSpacing: '0.18em', color }}>ANALYZED IMAGE</div>
            <div style={{ fontSize: '0.67rem', color: 'var(--text-lo)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>
              {labeledOutput.composite_size} · annotated output
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <motion.button onClick={() => setZoom(true)} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
            style={{ background: `${color}12`, border: `1px solid ${color}44`, borderRadius: 8, padding: '6px 14px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5, color, fontSize: '0.66rem', fontFamily: 'var(--font-mono)' }}>
            <ZoomIn size={12} /> Full View
          </motion.button>
          <motion.button onClick={download} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
            style={{ background: `${color}12`, border: `1px solid ${color}44`, borderRadius: 8, padding: '6px 14px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5, color, fontSize: '0.66rem', fontFamily: 'var(--font-mono)' }}>
            <DownloadCloud size={13} /> Download
          </motion.button>
        </div>
      </div>
      <div style={{ position: 'relative', cursor: 'zoom-in', borderRadius: 12, overflow: 'hidden', border: `1px solid ${color}22` }} onClick={() => setZoom(true)}>
        <img src={src} alt="Analyzed Output" style={{ width: '100%', display: 'block' }} />
        <div style={{ position: 'absolute', bottom: 8, right: 8, fontFamily: 'var(--font-mono)', fontSize: '0.55rem', color, background: 'rgba(0,0,0,0.7)', padding: '3px 8px', borderRadius: 5, display: 'flex', alignItems: 'center', gap: 5 }}>
          <ZoomIn size={10} /> Click to enlarge
        </div>
      </div>
      {zoom && <ZoomModal src={src} label={`${label} — Annotated`} onClose={() => setZoom(false)} />}
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
/*  EXPERT VERIFICATION PANEL  (was "Stage 09 Second Opinion")     */
/* ═══════════════════════════════════════════════════════════════ */
function SecondOpinionPanel({ secondOpinion }) {
  const { cnn_label, cnn_confidence, gemini, catalog_xref } = secondOpinion;
  const geminiOk = gemini?.status === 'ok';

  return (
    <motion.div className="glass result-card" variants={cardVariants} initial="hidden" animate="show"
      style={{ borderColor: 'rgba(124,58,237,0.3)', marginTop: '1.5rem' }}>
      <div className="panel-header">
        <div className="panel-title">
          <Brain size={14} style={{ color: '#7c3aed' }} />
          EXPERT VERIFICATION
        </div>
        <span style={{ fontSize: '0.52rem', fontFamily: 'var(--font-mono)', color: '#7c3aed', padding: '2px 8px', borderRadius: 999, border: '1px solid rgba(124,58,237,0.3)', background: 'rgba(124,58,237,0.08)' }}>
          {geminiOk ? 'CROSS-CHECK COMPLETE' : 'CATALOG CHECK'}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
        {/* AI verification */}
        <div style={{ background: 'rgba(124,58,237,0.05)', border: '1px solid rgba(124,58,237,0.2)', borderRadius: 12, padding: '1rem' }}>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-lo)', marginBottom: '0.5rem', lineHeight: 1.5 }}>
            Confidence was below our threshold ({cnn_confidence?.toFixed(0)}%). An additional review was performed.
          </div>
          {geminiOk && gemini.label && (
            <div style={{ fontSize: '0.7rem', color: '#e0f4ff', fontWeight: 600, marginBottom: 4 }}>
              Expert suggests: <span style={{ color: getColor(gemini.label) }}>{gemini.label}</span>
              {gemini.confidence && <span style={{ color: '#10b981', marginLeft: 6, fontFamily: 'var(--font-mono)' }}>{gemini.confidence.toFixed(0)}%</span>}
            </div>
          )}
          {gemini?.reasoning && (
            <div style={{ fontSize: '0.62rem', color: 'var(--text-lo)', lineHeight: 1.5, fontStyle: 'italic' }}>
              "{gemini.reasoning}"
            </div>
          )}
          {gemini?.agrees_with_cnn !== undefined && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: '0.7rem', padding: '0.45rem', background: 'rgba(0,0,0,0.2)', borderRadius: 8 }}>
              {gemini.agrees_with_cnn ? <CheckCircle size={14} color="#10b981" /> : <X size={14} color="#f59e0b" />}
              <span style={{ fontSize: '0.62rem', color: gemini.agrees_with_cnn ? '#10b981' : '#f59e0b' }}>
                {gemini.agrees_with_cnn ? 'Confirms initial finding' : 'Suggests a different classification'}
              </span>
            </div>
          )}
        </div>

        {/* Catalog cross-checks */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {[
            { name: 'SIMBAD Database', icon: <Globe size={12} />, data: catalog_xref?.simbad, color: '#34d399' },
            { name: 'NASA Catalog',    icon: <Brain size={12} />, data: catalog_xref?.ned,    color: '#38bdf8' },
          ].map(({ name, icon, data, color }) => (
            <div key={name} style={{ background: `${color}08`, border: `1px solid ${color}22`, borderRadius: 12, padding: '0.8rem', flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: '0.4rem' }}>
                <span style={{ color }}>{icon}</span>
                <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-head)', letterSpacing: '0.1em', color }}>{name}</span>
              </div>
              <div style={{ fontSize: '0.63rem', color: 'var(--text-lo)' }}>
                {data?.status === 'ok'
                  ? data.result_count > 0
                    ? `${data.result_count} known object(s) found matching this type.`
                    : 'No catalog matches for this field.'
                  : 'Catalog check unavailable.'}
              </div>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
/*  RESULTS PAGE                                                   */
/* ═══════════════════════════════════════════════════════════════ */
export default function Results() {
  const [result,   setResult]   = useState(null);
  const [loading,  setLoading]  = useState(true);
  const [selected, setSelected] = useState(null);
  const [copied,   setCopied]   = useState(false);
  const [history,  setHistory]  = useState([]);
  const canvasRef = useRef(null);

  useEffect(() => {
    try {
      const memResult = getLastResult();
      if (memResult) {
        setResult(memResult);
      } else {
        const raw = localStorage.getItem('lastResult');
        if (raw) setResult(JSON.parse(raw));
      }
      const hist = localStorage.getItem('iarrd_history');
      if (hist) setHistory(JSON.parse(hist));
    } catch (_) {}
    setLoading(false);
  }, []);

  /* Draw image + markers for browser-mode */
  useEffect(() => {
    if (!result || result.mode === 'ai' || !canvasRef.current) return;
    const img = new window.Image();
    img.onload = () => {
      const c = canvasRef.current;
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      const ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0);
      (result.objects || []).forEach((o, i) => {
        const { color } = classify(o);
        const isSel = selected === i;
        const r = isSel ? 30 : 21;
        ctx.beginPath(); ctx.arc(o.x, o.y, r, 0, 2 * Math.PI);
        ctx.lineWidth = isSel ? 3 : 2; ctx.strokeStyle = color;
        ctx.shadowColor = color; ctx.shadowBlur = isSel ? 22 : 10;
        ctx.stroke();
        ctx.beginPath(); ctx.arc(o.x, o.y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = color; ctx.shadowBlur = 12; ctx.fill();
        ctx.shadowBlur = 0;
        ctx.font = '600 13px "Inter", sans-serif';
        ctx.fillStyle = '#edf4ff';
        ctx.fillText(`T-${i + 1}`, o.x + r + 5, o.y + 5);
      });
    };
    img.src = result.image;
  }, [result, selected]);

  const exportJSON = useCallback(() => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'iarrd-report.json'; a.click();
    URL.revokeObjectURL(url);
  }, [result]);

  const copyShare = useCallback(() => {
    const payload = {
      object:     result?.classification?.label,
      confidence: result?.classification?.confidence,
      noiseLevel: result?.reconstruction?.anomaly_score,
      quality:    result?.reconstruction?.quality,
      analysisMs: result?.elapsed_ms,
    };
    navigator.clipboard.writeText(JSON.stringify(payload, null, 2)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [result]);

  const isAI = result?.mode === 'ai';

  return (
    <PageTransition>
      <section className="scene" style={{ alignItems: 'flex-start', paddingTop: '6.5rem' }}>
        <div className="page-inner">

          {/* Header */}
          <FadeUp>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <div className="eyebrow">
                  <Target size={13} strokeWidth={1.5} />
                  SCENE 03 — ANALYSIS RESULTS
                </div>
                <h1 className="section-title">Analysis Results</h1>
              </div>
              {result && (
                <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
                  <motion.button className="btn-ghost" onClick={copyShare} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }} style={{ fontSize: '0.68rem' }}>
                    <Share2 size={13} /> {copied ? 'Copied!' : 'Share'}
                  </motion.button>
                  <motion.button className="btn-ghost" onClick={exportJSON} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }} style={{ fontSize: '0.68rem' }}>
                    <FileJson size={13} /> Export Report
                  </motion.button>
                  <Link to="/upload" className="btn-primary" style={{ fontSize: '0.68rem', padding: '10px 20px' }}>
                    <RotateCcw size={13} /> New Analysis
                  </Link>
                </div>
              )}
            </div>
          </FadeUp>

          {/* Skeleton */}
          {loading && (
            <FadeUp delay={0.05}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '0.75rem' }}>
                  {[1,2,3,4].map(i => (
                    <div key={i} className="glass stat-cell" style={{ height: 80 }}>
                      <div style={{ height: 10, borderRadius: 5, background: 'rgba(255,255,255,0.06)', width: '50%', marginBottom: 12, animation: 'pulse 1.5s ease infinite' }} />
                      <div style={{ height: 22, borderRadius: 5, background: 'rgba(255,255,255,0.06)', width: '75%', animation: 'pulse 1.5s ease infinite' }} />
                    </div>
                  ))}
                </div>
              </div>
            </FadeUp>
          )}

          {/* Recent scans */}
          {!loading && history.length > 0 && (
            <FadeUp delay={0.06}>
              <div style={{ marginBottom: '0.5rem' }}>
                <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.58rem', letterSpacing: '0.18em', color: 'var(--text-lo)', marginBottom: '0.7rem', display: 'flex', alignItems: 'center', gap: 7 }}>
                  <Database size={11} strokeWidth={1.5} />
                  RECENT SCANS — LAST {history.length}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))', gap: '0.6rem' }}>
                  {history.map((h, i) => {
                    const qual = h.quality === 'Normal' ? '#10b981' : h.quality === 'Elevated' ? '#f59e0b' : '#ef4444';
                    const lbl  = getColor(h.label);
                    return (
                      <div key={i} className="glass" style={{ padding: '0.9rem 1rem', borderRadius: 12, borderColor: `${lbl}22` }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
                          <div style={{ fontSize: '0.72rem', color: lbl, fontWeight: 700 }}>{h.label ?? '—'}</div>
                          <span style={{ fontSize: '0.55rem', fontFamily: 'var(--font-mono)', color: 'var(--text-lo)' }}>#{i + 1}</span>
                        </div>
                        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                          {[
                            { k: 'Conf',    v: `${(h.confidence ?? 0).toFixed(1)}%`, c: lbl },
                            { k: 'Noise',   v: `${(h.anomaly_score ?? 0).toFixed(0)}`, c: qual },
                            { k: 'Time',    v: `${h.elapsed_ms ?? '—'}ms`, c: 'var(--text-lo)' },
                          ].map(({ k, v, c }) => (
                            <div key={k} style={{ fontSize: '0.62rem', fontFamily: 'var(--font-mono)' }}>
                              <span style={{ color: 'var(--text-lo)' }}>{k} </span>
                              <span style={{ color: c }}>{v}</span>
                            </div>
                          ))}
                        </div>
                        <div style={{ fontSize: '0.58rem', color: 'var(--text-lo)', marginTop: 5, fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {h.filename}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </FadeUp>
          )}

          {/* Empty state */}
          {!loading && !result && (
            <FadeUp delay={0.1}>
              <div className="glass" style={{ padding: '4rem', textAlign: 'center', borderRadius: 18, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem' }}>
                <Database size={54} strokeWidth={1} style={{ color: 'rgba(255,255,255,0.07)' }} />
                <p style={{ color: 'var(--text-lo)', maxWidth: 380, lineHeight: 1.7 }}>
                  No results yet. Upload an astronomical image to get started.
                </p>
                <Link to="/upload" className="btn-primary">
                  <Target size={14} /> Start Analysis
                </Link>
              </div>
            </FadeUp>
          )}

          {/* ── AI mode results ── */}
          {result && isAI && (
            <>
              {/* Summary strip */}
              <FadeUp delay={0.05}>
                <div className="stats-strip">
                  {[
                    {
                      label: 'DETECTED OBJECT',
                      value: result.classification?.label ?? '—',
                      style: { fontSize: '0.9rem', color: getColor(result.classification?.label) },
                    },
                    {
                      label: 'CONFIDENCE',
                      value: <AnimNum value={result.classification?.confidence ?? 0} format={v => `${v.toFixed(1)}%`} />,
                    },
                    {
                      label: 'NOISE LEVEL',
                      value: <AnimNum value={result.reconstruction?.anomaly_score ?? 0} format={v => v.toFixed(1)} />,
                    },
                    {
                      label: 'ANALYSIS TIME',
                      value: <AnimNum value={result.elapsed_ms ?? 0} format={v => `${v.toFixed(0)} ms`} />,
                    },
                  ].map(({ label, value, style }) => (
                    <div key={label} className="stat-cell glass">
                      <div className="stat-label">{label}</div>
                      <div className="stat-value" style={style}>{value}</div>
                    </div>
                  ))}
                </div>
              </FadeUp>

              {/* Compact pipeline progress — no internal detail */}
              <FadeUp delay={0.07}>
                <div className="glass" style={{ padding: '1rem 1.5rem', borderRadius: 14 }}>
                  <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.6rem', letterSpacing: '0.18em', color: '#38bdf8', marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Activity size={12} strokeWidth={1.5} />
                    ANALYSIS COMPLETE
                    {result.pipeline?.total_ms && (
                      <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', color: '#10b981', fontSize: '0.58rem' }}>
                        {result.pipeline.total_ms} ms
                      </span>
                    )}
                  </div>
                  <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                    {[
                      { label: 'Image received',   color: '#38bdf8' },
                      { label: 'Quality optimised', color: '#a78bfa' },
                      { label: 'Features detected', color: '#34d399' },
                      { label: 'Object classified', color: '#38bdf8' },
                      { label: 'Noise scored',      color: '#f59e0b' },
                      { label: 'Regions mapped',    color: '#10b981' },
                      { label: 'Output annotated',  color: '#34d399' },
                    ].map(({ label, color }, i) => (
                      <motion.div
                        key={label}
                        initial={{ opacity: 0, scale: 0.88 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.06 + i * 0.04, ease: [0.16,1,0.3,1] }}
                        style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '4px 10px', borderRadius: 8, background: `${color}08`, border: `1px solid ${color}22`, fontSize: '0.62rem', color: '#e0f4ff' }}
                      >
                        <span style={{ width: 6, height: 6, borderRadius: '50%', background: color, boxShadow: `0 0 8px ${color}`, flexShrink: 0 }} />
                        {label}
                      </motion.div>
                    ))}
                  </div>
                </div>
              </FadeUp>

              {/* Enhancement panel */}
              {result.enhancement?.image_b64 && (
                <FadeUp delay={0.08}>
                  <div style={{ marginBottom: '1.5rem' }}>
                    <EnhancementPanel enhancement={result.enhancement} originalImage={result.image} />
                  </div>
                </FadeUp>
              )}

              {/* Main panels grid */}
              <FadeUp delay={0.1}>
                <div className="telemetry-grid">
                  {result.classification && (
                    <ClassificationPanel classification={result.classification} minConf={result.minConf ?? 0} />
                  )}
                  {result.reconstruction && (
                    <AnomalyPanel reconstruction={result.reconstruction} originalImage={result.image} />
                  )}
                  {result.features && (
                    <FeaturesPanel features={result.features} />
                  )}
                  {result.segmentation && (
                    <SegmentationPanel segmentation={result.segmentation} originalImage={result.image} />
                  )}
                </div>
              </FadeUp>

              {/* Annotated output */}
              {result.labeled_output?.labeled_image_b64 && (
                <FadeUp delay={0.12}>
                  <LabeledOutputPanel labeledOutput={result.labeled_output} classification={result.classification} />
                </FadeUp>
              )}

              {/* Expert verification */}
              {result.second_opinion && (
                <FadeUp delay={0.14}>
                  <SecondOpinionPanel secondOpinion={result.second_opinion} />
                </FadeUp>
              )}

              {/* Action bar */}
              <FadeUp delay={0.16}>
                <div className="action-bar">
                  <span className="action-bar-label">SAVE RESULTS</span>
                  <motion.button className="btn-ghost" onClick={exportJSON} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }} style={{ fontSize: '0.68rem', padding: '8px 16px' }}>
                    <DownloadCloud size={13} /> Download Report
                  </motion.button>
                  <motion.button className="btn-ghost" onClick={copyShare} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }} style={{ fontSize: '0.68rem', padding: '8px 16px' }}>
                    <Share2 size={13} /> {copied ? 'Copied!' : 'Share Summary'}
                  </motion.button>
                  <motion.button
                    className="btn-ghost"
                    whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
                    style={{ fontSize: '0.68rem', padding: '8px 16px' }}
                    onClick={() => {
                      const rows = [
                        ['Field', 'Value'],
                        ['Detected Object', result.classification?.label ?? '—'],
                        ['Confidence', `${(result.classification?.confidence ?? 0).toFixed(1)}%`],
                        ['Noise Level', result.reconstruction?.anomaly_score ?? '—'],
                        ['Image Quality', result.reconstruction?.quality ?? '—'],
                        ['Analysis Time (ms)', result.elapsed_ms ?? '—'],
                      ];
                      const csv = rows.map(r => r.join(',')).join('\n');
                      const a = document.createElement('a');
                      a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
                      a.download = 'iarrd-summary.csv'; a.click();
                    }}
                  >
                    <FileText size={13} /> Export CSV
                  </motion.button>
                  <Link to="/upload" className="btn-primary" style={{ fontSize: '0.68rem', padding: '8px 16px', marginLeft: 'auto' }}>
                    <RotateCcw size={13} /> New Analysis
                  </Link>
                </div>
              </FadeUp>
            </>
          )}

          {/* ── Browser mode ── */}
          {result && !isAI && (
            <>
              <FadeUp delay={0.05}>
                <div className="stats-strip">
                  {[
                    { label: 'OBJECTS FOUND',  value: result.objects?.length ?? 0 },
                    { label: 'DETAIL LEVEL',   value: result.meta?.std?.toFixed(1) ?? '—' },
                    { label: 'FILTER STRENGTH', value: result.meta?.sigma ?? '—' },
                  ].map(({ label, value }) => (
                    <div key={label} className="stat-cell glass">
                      <div className="stat-label">{label}</div>
                      <div className="stat-value">{value}</div>
                    </div>
                  ))}
                </div>
              </FadeUp>

              <FadeUp delay={0.1}>
                <div className="results-layout">
                  <div className="results-screen glass">
                    <div className="scanline" />
                    <canvas ref={canvasRef} />
                  </div>

                  <div className="results-sidebar">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '0.4rem' }}>
                      <ScanSearch size={15} style={{ color: 'var(--sky)' }} strokeWidth={1.5} />
                      <span style={{ fontFamily: 'var(--font-head)', fontSize: '0.62rem', letterSpacing: '0.16em' }}>
                        DETECTED OBJECTS
                      </span>
                    </div>

                    <AnimatePresence>
                      {(result.objects || []).map((o, i) => {
                        const { label, color } = classify(o);
                        const isSel = selected === i;
                        return (
                          <motion.div
                            key={i}
                            className={`obj-card ${isSel ? 'selected' : ''}`}
                            style={isSel ? { borderColor: color, background: `${color}14` } : {}}
                            onClick={() => setSelected(isSel ? null : i)}
                            initial={{ opacity: 0, x: 16 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.05, ease: [0.16,1,0.3,1] }}
                          >
                            <div className="obj-header">
                              <span className="obj-dot" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />
                              T-{i+1} — {label}
                              <span className="obj-badge" style={{ color, borderColor:`${color}66` }}>#{i+1}</span>
                            </div>
                            <div className="obj-metrics">
                              <div className="metric"><Zap size={11} className="metric-icon" />{o.brightness.toFixed(1)} lum</div>
                              <div className="metric"><MapPin size={11} className="metric-icon" />{o.x.toFixed(0)}, {o.y.toFixed(0)}</div>
                              <div className="metric"><BarChart3 size={11} className="metric-icon" />{o.area} px²</div>
                            </div>
                          </motion.div>
                        );
                      })}
                    </AnimatePresence>
                  </div>
                </div>
              </FadeUp>
            </>
          )}

        </div>
      </section>
    </PageTransition>
  );
}
