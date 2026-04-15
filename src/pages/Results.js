import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  Target, BarChart3, Zap, MapPin,
  DownloadCloud, RotateCcw, Database, Cpu,
  Eye, EyeOff, Activity, Layers, ScanSearch, FileJson, Share2,
  X, ZoomIn, FileText, Brain, BookOpen, Globe, AlertTriangle,
  ExternalLink, CheckCircle,
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
  if (obj.area > 20000)     return { label: 'Class-A Galaxy',  color: '#7c3aed' };
  if (obj.brightness > 220) return { label: 'Alpha Star',      color: '#38bdf8' };
  if (obj.area > 8000)      return { label: 'Nebular Cluster', color: '#1d4ed8' };
  return                           { label: 'Bright Anomaly',  color: '#475569' };
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

/* ── Animated number ─────────────────────────────────────────── */
function AnimNum({ value, format = v => v, duration = 1.2 }) {
  const [displayed, setDisplayed] = useState(0);
  useEffect(() => {
    const start = performance.now();
    const from = 0;
    const to = Number(value) || 0;
    const tick = (now) => {
      const t = Math.min((now - start) / (duration * 1000), 1);
      const ease = 1 - Math.pow(1 - t, 3);
      setDisplayed(from + (to - from) * ease);
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [value, duration]);
  return <>{format(displayed)}</>;
}

/* ═══════════════════════════════════════════════════════════════ */
/*  ULTRA 8K ENHANCEMENT PANEL                                     */
/* ═══════════════════════════════════════════════════════════════ */
function EnhancementPanel({ enhancement, originalImage }) {
  const { image_b64, meta, multi, premium_chain: pc } = enhancement;
  const [zoom, setZoom] = useState(false);
  const [zoomSrc, setZoomSrc] = useState(null);
  const [zoomLabel, setZoomLabel] = useState('');
  const [showPremium, setShowPremium] = useState(true);
  const src = `data:image/png;base64,${image_b64}`;
  const multiSrc = (key) => multi?.[key] ? `data:image/png;base64,${multi[key]}` : null;

  // premium chain helpers
  const pcSrc = (key) => pc?.[key] ? `data:image/png;base64,${pc[key]}` : null;

  const download = () => {
    const a = document.createElement('a');
    a.href = src;
    a.download = 'iarrd-ultra-8k-enhanced.png';
    a.click();
  };

  const STAGE_COLORS = ['#38bdf8','#a78bfa','#f59e0b','#34d399','#10b981','#38bdf8','#a78bfa','#10b981'];

  return (
    <motion.div
      className="glass enhancement-panel"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.25 }}
    >
      {/* ── Header ── */}
      <div className="enhancement-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div className="enhancement-icon-wrap">
            <ZoomIn size={18} style={{ color: '#10b981' }} />
          </div>
          <div>
            <div className="enhancement-title">ULTRA 8K IMAGE ENHANCEMENT</div>
            <div className="enhancement-sub">
              {meta?.native_resolution} → {meta?.output_resolution}
              &nbsp;&middot;&nbsp; {meta?.upscale_factor}× LANCZOS
              &nbsp;&middot;&nbsp; 8-stage processing pipeline
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <span className="enhancement-badge">4096 × 4096 &middot; 16 MP</span>
          <motion.button
            className="btn-enhancement-view"
            onClick={() => setZoom(true)}
            whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
          >
            <ZoomIn size={12} /> View Native 4K
          </motion.button>
          <motion.button
            className="btn-enhancement-dl"
            onClick={download}
            whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
          >
            <DownloadCloud size={13} /> Download 8K PNG
          </motion.button>
        </div>
      </div>

      {/* ── Enhanced image — single clean display (slider removed) ── */}
      <div
        style={{ position: 'relative', borderRadius: 12, overflow: 'hidden', border: '1.5px solid rgba(16,185,129,0.35)', boxShadow: '0 0 40px rgba(16,185,129,0.12)', cursor: 'zoom-in', marginBottom: '1.2rem' }}
        onClick={() => setZoom(true)}
        title="Click to view at native 4096×4096 resolution"
      >
        <img
          src={src}
          alt="Ultra 8K Enhanced"
          style={{ width: '100%', display: 'block', objectFit: 'contain', maxHeight: 420, imageRendering: 'high-quality' }}
        />
        <div style={{ position: 'absolute', bottom: 8, right: 12, fontFamily: 'var(--font-mono)', fontSize: '0.52rem', color: '#10b981', background: 'rgba(0,0,0,0.75)', padding: '3px 8px', borderRadius: 5, display: 'flex', alignItems: 'center', gap: 5, pointerEvents: 'none' }}>
          <ZoomIn size={10} /> Click to view 4096 × 4096
        </div>
      </div>

      {/* ── PSNR / SSIM metric badges ── */}
      {multi && (
        <div style={{ display: 'flex', gap: '0.45rem', flexWrap: 'wrap', marginBottom: '1.2rem' }}>

          {[
            { label: 'PSNR',        value: `${multi.psnr ?? '—'} dB`,  color: '#10b981', desc: 'Signal quality' },
            { label: 'SSIM',        value: multi.ssim?.toFixed(4) ?? '—', color: '#38bdf8', desc: 'Structural similarity' },
            { label: 'NOISE ↓',     value: `${multi.noise_reduction_pct ?? '—'}%`,  color: '#a78bfa', desc: 'Noise suppressed' },
            { label: 'CONTRAST ↑',  value: `${(multi.contrast_improvement_pct ?? 0) > 0 ? '+' : ''}${multi.contrast_improvement_pct ?? '—'}%`, color: '#f59e0b', desc: 'Dynamic range boost' },
          ].map(({ label, value, color, desc }) => (
            <div key={label} style={{ flex: '1 1 90px', padding: '0.55rem 0.6rem', borderRadius: 10, background: `${color}0a`, border: `1px solid ${color}28`, textAlign: 'center' }}>
              <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.46rem', letterSpacing: '0.12em', color: 'var(--text-lo)', marginBottom: 3 }}>{label}</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.88rem', color, fontWeight: 700, textShadow: `0 0 12px ${color}66` }}>{value}</div>
              <div style={{ fontSize: '0.5rem', color: 'var(--text-lo)', marginTop: 2 }}>{desc}</div>
            </div>
          ))}
        </div>
      )}



      {/* ── Original pipeline stage chips ── */}
      <div style={{ marginTop: '1.2rem' }}>
        <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.56rem', letterSpacing: '0.18em', color: 'var(--text-lo)', marginBottom: '0.7rem' }}>
          PROCESSING PIPELINE — {(meta?.stages_applied ?? []).length} STAGES APPLIED
        </div>
        <div className="enhancement-stages-grid">
          {(meta?.stages_applied ?? []).map((stage, i) => (
            <motion.div
              key={i}
              className="enhancement-stage-item"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + i * 0.06 }}
            >
              <div className="enhancement-stage-num" style={{ color: STAGE_COLORS[i], borderColor: STAGE_COLORS[i], background: `${STAGE_COLORS[i]}14` }}>
                {String(i + 1).padStart(2, '0')}
              </div>
              <div className="enhancement-stage-text">{stage}</div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* ── 2025 PREMIUM CHAIN — CLAHE → Wavelet → TDR ── */}
      {pc && pcSrc('enhanced_b64') && (
        <div style={{ marginTop: '1.4rem', borderTop: '1px solid rgba(56,189,248,0.12)', paddingTop: '1.2rem' }}>
          {/* Header row */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.8rem' }}>
            <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.56rem', letterSpacing: '0.18em', color: '#38bdf8', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#38bdf8', boxShadow: '0 0 10px #38bdf8', display: 'inline-block' }} />
              2025 PREMIUM CHAIN — 3-TECHNIQUE PIPELINE
            </div>
            <button
              onClick={() => setShowPremium(s => !s)}
              style={{ background: 'none', border: '1px solid rgba(56,189,248,0.2)', borderRadius: 6, padding: '2px 10px', cursor: 'pointer', color: 'var(--text-lo)', fontFamily: 'var(--font-mono)', fontSize: '0.55rem' }}
            >
              {showPremium ? '▲ HIDE' : '▼ SHOW'}
            </button>
          </div>

          {showPremium && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} style={{ overflow: 'hidden' }}>
              {/* PSNR / SSIM / Noise badges */}
              <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginBottom: '0.9rem' }}>
                {[
                  { label: 'PSNR',        value: `${pc.psnr ?? '—'} dB`,  color: '#10b981', desc: 'vs raw input' },
                  { label: 'SSIM',        value: pc.ssim?.toFixed(4) ?? '—',  color: '#38bdf8', desc: 'Structural similarity' },
                  { label: 'NOISE ↓',     value: `${pc.noise_reduction_pct ?? '—'}%`, color: '#a78bfa', desc: 'Noise suppressed' },
                  { label: 'CONTRAST ↑',  value: `${(pc.contrast_boost_pct ?? 0) > 0 ? '+' : ''}${pc.contrast_boost_pct ?? '—'}%`, color: '#f59e0b', desc: 'Contrast boost' },
                ].map(({ label, value, color, desc }) => (
                  <div key={label} style={{ flex: '1 1 80px', padding: '0.5rem 0.55rem', borderRadius: 9, background: `${color}0a`, border: `1px solid ${color}28`, textAlign: 'center' }}>
                    <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.44rem', letterSpacing: '0.12em', color: 'var(--text-lo)', marginBottom: 3 }}>{label}</div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color, fontWeight: 700, textShadow: `0 0 10px ${color}55` }}>{value}</div>
                    <div style={{ fontSize: '0.48rem', color: 'var(--text-lo)', marginTop: 2 }}>{desc}</div>
                  </div>
                ))}
              </div>

              {/* 3-stage thumbnail progression + final enlarged */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '0.4rem', marginBottom: '0.8rem' }}>
                {[
                  { key: 'clahe_b64',   label: 'CLAHE',   badge: '2-pass',  bc: '#38bdf8', ms: pc.stage_ms?.clahe },
                  { key: 'wavelet_b64', label: 'WAVELET',  badge: '3-level', bc: '#a78bfa', ms: pc.stage_ms?.wavelet },
                  { key: 'tdr_b64',     label: 'TDR',      badge: '2025',    bc: '#10b981', ms: pc.stage_ms?.tdr },
                ].map(({ key, label, badge, bc, ms }, idx) => {
                  const imgSrc = pcSrc(key);
                  return imgSrc ? (
                    <div
                      key={key}
                      style={{ position: 'relative', borderRadius: 8, overflow: 'hidden', border: `1px solid ${bc}40`, cursor: 'zoom-in' }}
                      onClick={() => { setZoomSrc(imgSrc); setZoomLabel(`${label} — 2025 Premium Chain`); }}
                    >
                      <img src={imgSrc} alt={label} style={{ width: '100%', display: 'block', aspectRatio: '1', objectFit: 'cover' }} />
                      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '3px 5px', background: 'rgba(3,3,18,0.82)', fontFamily: 'var(--font-mono)', fontSize: '0.43rem', color: bc, display: 'flex', justifyContent: 'space-between' }}>
                        <span>{label}</span>
                        <span style={{ color: 'var(--text-lo)' }}>{ms != null ? `${ms}ms` : badge}</span>
                      </div>
                      {idx < 2 && (
                        <div style={{ position: 'absolute', top: '50%', right: -9, transform: 'translateY(-50%)', color: 'rgba(255,255,255,0.3)', fontSize: '0.75rem', zIndex: 2, pointerEvents: 'none' }}>→</div>
                      )}
                    </div>
                  ) : null;
                })}
              </div>

              {/* Final TDR output enlarged */}
              <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.5rem', letterSpacing: '0.14em', color: '#10b981', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: 7 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#10b981', boxShadow: '0 0 12px #10b981', display: 'inline-block', flexShrink: 0 }} />
                TDR DENOISED FINAL — TOTAL {pc.stage_ms?.total ?? '—'} ms
              </div>
              <div
                style={{ borderRadius: 10, overflow: 'hidden', border: '2px solid rgba(16,185,129,0.4)', boxShadow: '0 0 24px rgba(16,185,129,0.14)', cursor: 'zoom-in' }}
                onClick={() => { setZoomSrc(pcSrc('enhanced_b64')); setZoomLabel('TDR DENOISED — 2025 Premium Chain Final'); }}
              >
                <img src={pcSrc('enhanced_b64')} alt="TDR Final" style={{ width: '100%', display: 'block', objectFit: 'contain', maxHeight: 340 }} />
              </div>

              {/* Technique list */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginTop: '0.8rem' }}>
                {(pc.techniques ?? []).map((t, i) => {
                  const colors = ['#38bdf8', '#a78bfa', '#10b981'];
                  return (
                    <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, fontSize: '0.62rem', color: 'var(--text-lo)', lineHeight: 1.5 }}>
                      <span style={{ fontFamily: 'var(--font-mono)', color: colors[i], flexShrink: 0 }}>{String(i + 1).padStart(2, '0')}</span>
                      {t}
                    </div>
                  );
                })}
              </div>
            </motion.div>
          )}
        </div>
      )}

      {/* ── Zoom Modal ── */}
      {zoomSrc && <ZoomModal src={zoomSrc} label={zoomLabel} onClose={() => setZoomSrc(null)} />}
      {zoom && <ZoomModal src={src} label="ULTRA 8K — 4096×4096 (16 MP) — NATIVE RESOLUTION" onClose={() => setZoom(false)} />}

    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
/*  CLASSIFICATION PANEL                                           */
/* ═══════════════════════════════════════════════════════════════ */
function ClassificationPanel({ classification, minConf = 0 }) {
  const { label, confidence, all_scores, confidence_flag } = classification;
  const color = getColor(label);

  const flagColor  = confidence_flag === 'high'   ? '#10b981' :
                     confidence_flag === 'medium' ? '#f59e0b' : '#ef4444';
  const flagLabel  = confidence_flag === 'high'   ? 'HIGH CONFIDENCE' :
                     confidence_flag === 'medium' ? 'MEDIUM CONFIDENCE' : 'LOW — UNCERTAIN';

  return (
    <motion.div
      className="glass cnn-panel result-card"
      style={{ borderColor: `${color}33` }}
      initial={{ opacity: 0, y: 24, scale: 0.97, filter: 'blur(8px)' }}
      animate={{ opacity: 1, y: 0, scale: 1, filter: 'blur(0px)' }}
      transition={{ duration: 0.6, ease: [0.43, 0.13, 0.23, 0.96] }}
    >
      <div className="panel-header">
        <div className="panel-title">
          <Cpu size={13} strokeWidth={1.5} style={{ color: 'var(--sky)' }} />
          CNN CLASSIFICATION
        </div>
        {confidence_flag && (
          <span style={{ fontSize: '0.52rem', fontFamily: 'var(--font-mono)', color: flagColor, padding: '2px 8px', borderRadius: 999, border: `1px solid ${flagColor}55`, background: `${flagColor}12`, letterSpacing: '0.08em' }}>
            {flagLabel}
          </span>
        )}
      </div>

      {/* Top prediction — gradient label */}
      <div className="cnn-label-row">
        <div className="cnn-label-dot" style={{ background: color, boxShadow: `0 0 18px ${color}` }} />
        <div>
          <div className="cnn-label-name gradient-label" style={{
            background: `linear-gradient(135deg, ${color} 0%, #8b5cf6 100%)`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            fontSize: '1.15rem',
            fontWeight: 700,
            letterSpacing: '-0.01em',
          }}>{label}</div>
          <div className="cnn-label-conf">
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1rem', color: '#10b981', fontWeight: 700, textShadow: '0 0 12px rgba(16,185,129,0.6)' }}>
              <AnimNum value={confidence} format={v => `${v.toFixed(1)}%`} />
            </span>{' '}confidence
          </div>
        </div>
      </div>

      {/* All scores — shimmer bars */}
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
                    style={{
                      background: isTop ? `linear-gradient(90deg, ${c}cc, ${c})` : c,
                      boxShadow: isTop ? `0 0 12px ${c}88` : 'none'
                    }}
                    initial={{ width: 0 }}
                    animate={{ width: `${score}%` }}
                    transition={{ duration: 1.0, ease: [0.16,1,0.3,1], delay: 0.1 + idx * 0.07 }}
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
/*  IMAGE ENHANCEMENT REPORT PANEL  (Autoencoder)                  */
/* ═══════════════════════════════════════════════════════════════ */
function AnomalyPanel({ reconstruction, originalImage }) {
  const {
    anomaly_score, quality, image_b64,
    psnr_db, snr_improvement_pct, noise_reduction_pct,
    gaussian_preview_b64,
  } = reconstruction;
  const [zoomSrc, setZoomSrc] = useState(null);
  const [zoomLabel, setZoomLabel] = useState('');
  const [useSlider, setUseSlider] = useState(true);
  const scoreColor =
    quality === 'Normal'   ? '#10b981' :
    quality === 'Elevated' ? '#f59e0b' : '#ef4444';

  const downloadEnhanced = () => {
    if (!image_b64) return;
    const a = document.createElement('a');
    a.href = `data:image/png;base64,${image_b64}`;
    a.download = 'iarrd-enhanced.png'; a.click();
  };

  const reconSrc = image_b64 ? `data:image/png;base64,${image_b64}` : null;

  const gaussianSrc = gaussian_preview_b64 ? `data:image/png;base64,${gaussian_preview_b64}` : null;

  return (
    <motion.div
      className="glass cnn-panel result-card"
      initial={{ opacity: 0, y: 24, scale: 0.97, filter: 'blur(8px)' }}
      animate={{ opacity: 1, y: 0, scale: 1, filter: 'blur(0px)' }}
      transition={{ duration: 0.6, delay: 0.1, ease: [0.43, 0.13, 0.23, 0.96] }}
    >
      <div className="panel-header">
        <div className="panel-title">
          <Activity size={13} strokeWidth={1.5} style={{ color: 'var(--sky)' }} />
          AUTOENCODER — IMAGE ENHANCEMENT REPORT
        </div>
      </div>

      {/* Enhancement metrics row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '0.55rem', marginBottom: '1rem' }}>
        {[
          { label: 'PSNR',           value: psnr_db           != null ? `${psnr_db} dB`                                        : '—', color: '#10b981', desc: 'Signal quality'   },
          { label: 'SNR IMPROVE.',   value: snr_improvement_pct != null ? `${snr_improvement_pct > 0 ? '+' : ''}${snr_improvement_pct}%` : '—', color: '#38bdf8', desc: 'Signal-to-noise'  },
          { label: 'NOISE REDUCED',  value: noise_reduction_pct != null ? `${noise_reduction_pct}%`                              : '—', color: '#a78bfa', desc: 'Noise suppressed'  },
        ].map(({ label, value, color, desc }) => (
          <div key={label} style={{ padding: '0.65rem 0.5rem', borderRadius: 10, background: `${color}0a`, border: `1px solid ${color}22`, textAlign: 'center' }}>
            <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.48rem', letterSpacing: '0.12em', color: 'var(--text-lo)', marginBottom: 4 }}>{label}</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.84rem', color, fontWeight: 700 }}>{value}</div>
            <div style={{ fontSize: '0.54rem', color: 'var(--text-lo)', marginTop: 2 }}>{desc}</div>
          </div>
        ))}
      </div>

      {/* Noise Level gauge row */}
      <div className="anomaly-gauge-wrap">
        <div className="anomaly-ring" style={{
          borderColor: scoreColor,
          boxShadow: `0 0 28px ${scoreColor}44, inset 0 0 18px ${scoreColor}18`
        }}>
          <div className="anomaly-score-val" style={{ color: scoreColor }}>
            <AnimNum value={anomaly_score} format={v => v.toFixed(1)} />
          </div>
          <div className="anomaly-max">/100</div>
        </div>
        <div>
          <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.48rem', letterSpacing: '0.12em', color: 'var(--text-lo)', marginBottom: 3 }}>NOISE LEVEL</div>
          <div className="anomaly-quality" style={{
            color: scoreColor,
            textShadow: `0 0 18px ${scoreColor}88`
          }}>{quality}</div>
          <div className="anomaly-desc">
            Low score = clean image.<br />High = elevated noise / anomaly.
          </div>
        </div>
      </div>

      {/* 3-way thumbnail: Raw → Gaussian Filtered → Denoised Output */}
      {originalImage && gaussianSrc && reconSrc && (
        <div style={{ marginBottom: '1rem' }}>
          <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.54rem', letterSpacing: '0.14em', color: 'var(--text-lo)', marginBottom: '0.6rem' }}>
            PREPROCESSING VISUAL — RAW → GAUSSIAN → DENOISED
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '0.45rem' }}>
            {[
              { src: originalImage, label: 'RAW INPUT',        badge: null,      borderColor: 'rgba(255,255,255,0.18)', textColor: 'rgba(255,255,255,0.6)', zLbl: 'RAW INPUT — Original Telescope Image' },
              { src: gaussianSrc,  label: 'GAUSSIAN FILTERED', badge: 'σ = 1.5', borderColor: '#38bdf8',               textColor: '#7dd3fc',               zLbl: 'GAUSSIAN FILTERED — σ=1.5 Preprocessing' },
              { src: reconSrc,     label: 'DENOISED OUTPUT',   badge: null,      borderColor: '#10b981',               textColor: '#6ee7b7',               zLbl: 'DENOISED OUTPUT — Autoencoder Reconstruction' },
            ].map(({ src, label, badge, borderColor, textColor, zLbl }) => (
              <div
                key={label}
                style={{ position: 'relative', cursor: 'zoom-in', borderRadius: 8, overflow: 'hidden', border: `1px solid ${borderColor}` }}
                onClick={() => { setZoomSrc(src); setZoomLabel(zLbl); }}
              >
                <img src={src} alt={label} style={{ width: '100%', display: 'block', aspectRatio: '1 / 1', objectFit: 'cover' }} />
                <div style={{
                  position: 'absolute', bottom: 0, left: 0, right: 0,
                  padding: '4px 5px', background: 'rgba(3,3,18,0.78)',
                  fontFamily: 'var(--font-mono)', fontSize: '0.48rem',
                  color: textColor, letterSpacing: '0.05em',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                }}>
                  {label}
                  {badge && (
                    <span style={{ background: `rgba(56,189,248,0.18)`, border: '1px solid rgba(56,189,248,0.4)', borderRadius: 4, padding: '1px 5px', fontSize: '0.48rem', color: '#38bdf8' }}>
                      {badge}
                    </span>
                  )}
                </div>
                <div style={{ position: 'absolute', top: 4, right: 4 }}>
                  <ZoomIn size={9} style={{ color: 'rgba(255,255,255,0.5)' }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Slider comparison */}
      {image_b64 && originalImage && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div className="recon-label" style={{ marginBottom: 0 }}>ORIGINAL vs DENOISED — SLIDER</div>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              {/* Slider / side-by-side toggle */}
              <button
                onClick={() => setUseSlider(s => !s)}
                style={{ background: useSlider ? 'rgba(56,189,248,0.12)' : 'transparent', border: '1px solid rgba(56,189,248,0.3)', borderRadius: 6, padding: '3px 8px', cursor: 'pointer', color: 'var(--sky)', fontSize: '0.58rem', fontFamily: 'var(--font-mono)', transition: 'background 0.2s' }}
              >
                {useSlider ? '⇔ Slider' : '⊞ Split'}
              </button>
              <motion.button
                onClick={downloadEnhanced}
                whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                style={{ background: 'rgba(56,189,248,0.12)', border: '1px solid rgba(56,189,248,0.35)', borderRadius: 6, padding: '3px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4, color: 'var(--sky)', fontSize: '0.6rem', fontFamily: 'var(--font-mono)' }}
              >
                <DownloadCloud size={11} /> Recon
              </motion.button>
            </div>
          </div>

          {useSlider ? (
            <ImageComparisonSlider originalSrc={originalImage} reconstructedSrc={reconSrc} />
          ) : (
            <div className="recon-images">
              <div
                className="recon-img-wrap"
                style={{ cursor: 'zoom-in', position: 'relative' }}
                onClick={() => { setZoomSrc(originalImage); setZoomLabel('RAW INPUT — Telescope Image'); }}
              >
                <img src={originalImage} alt="Original" style={{ width: '100%', display: 'block' }} />
                <div className="recon-img-tag">RAW INPUT</div>
                <div style={{ position: 'absolute', top: 6, right: 6, background: 'rgba(0,0,0,0.5)', borderRadius: 5, padding: '3px 5px', display: 'flex', alignItems: 'center' }}>
                  <ZoomIn size={11} style={{ color: '#fff' }} />
                </div>
              </div>
              <div
                className="recon-img-wrap"
                style={{ cursor: 'zoom-in', position: 'relative' }}
                onClick={() => { setZoomSrc(reconSrc); setZoomLabel('DENOISED OUTPUT — Autoencoder Reconstruction (128×128)'); }}
              >
                <img src={reconSrc} alt="Denoised Output" style={{ width: '100%', display: 'block' }} />
                <div className="recon-img-tag" style={{ background: 'rgba(16,185,129,0.15)', borderColor: 'rgba(16,185,129,0.4)', color: '#6ee7b7' }}>DENOISED OUTPUT</div>
                <div style={{ position: 'absolute', top: 6, right: 6, background: 'rgba(0,0,0,0.5)', borderRadius: 5, padding: '3px 5px', display: 'flex', alignItems: 'center' }}>
                  <ZoomIn size={11} style={{ color: '#38bdf8' }} />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Zoom modal */}
      {zoomSrc && <ZoomModal src={zoomSrc} label={zoomLabel} onClose={() => setZoomSrc(null)} />}

      {/* Noise level bar */}
      <div style={{ marginTop: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
          <span style={{ fontFamily: 'var(--font-head)', fontSize: '0.58rem', letterSpacing: '0.14em', color: 'var(--text-lo)' }}>NOISE LEVEL</span>
          <span className="mono" style={{ fontSize: '0.65rem', color: scoreColor }}>{anomaly_score}/100</span>
        </div>
        <div style={{ height: 7, borderRadius: 7, background: 'rgba(255,255,255,0.06)', overflow: 'hidden', position: 'relative' }}>
          <motion.div
            className="shimmer-bar"
            style={{ height: '100%', borderRadius: 7, background: `linear-gradient(90deg, ${scoreColor}cc, ${scoreColor})`, boxShadow: `0 0 14px ${scoreColor}88` }}
            initial={{ width: 0 }}
            animate={{ width: `${anomaly_score}%` }}
            transition={{ duration: 1.2, ease: [0.16,1,0.3,1], delay: 0.2 }}
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: '0.58rem', color: 'var(--text-lo)', fontFamily: 'var(--font-mono)' }}>
          <span>0 — Clean</span><span>50 — Elevated</span><span>100 — Anomalous</span>
        </div>
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
/*  SEGMENTATION PANEL                                             */
/* ═══════════════════════════════════════════════════════════════ */
function SegmentationPanel({ segmentation, originalImage }) {
  const { overlay_b64, coverage } = segmentation;
  const [showOverlay, setShowOverlay] = useState(true);
  const [zoomSrc, setZoomSrc]         = useState(null);
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
    a.download = 'iarrd-segmentation.png'; a.click();
  };

  const sorted = Object.entries(coverage)
    .filter(([,v]) => v > 0.1)
    .sort(([,a],[,b]) => b - a);

  return (
    <motion.div
      className="glass cnn-panel result-card"
      initial={{ opacity: 0, y: 24, scale: 0.97, filter: 'blur(8px)' }}
      animate={{ opacity: 1, y: 0, scale: 1, filter: 'blur(0px)' }}
      transition={{ duration: 0.6, delay: 0.2, ease: [0.43, 0.13, 0.23, 0.96] }}
    >
      <div className="panel-header">
        <div className="panel-title">
          <Layers size={13} strokeWidth={1.5} style={{ color: 'var(--sky)' }} />
          U-NET SEGMENTATION
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

      {/* Composite image */}
      {overlay_b64 && originalImage && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.58rem', letterSpacing: '0.14em', color: 'var(--text-lo)' }}>PIXEL SEGMENTATION MAP</div>
            <motion.button
              onClick={downloadOverlay}
              whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
              style={{ background: 'rgba(52,211,153,0.1)', border: '1px solid rgba(52,211,153,0.3)', borderRadius: 6, padding: '3px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4, color: '#34d399', fontSize: '0.6rem', fontFamily: 'var(--font-mono)' }}
            >
              <DownloadCloud size={11} /> Overlay
            </motion.button>
          </div>

          <div
            className="seg-img-wrap"
            style={{ cursor: 'zoom-in', position: 'relative' }}
            onClick={() => setZoomSrc(showOverlay ? `data:image/png;base64,${overlay_b64}` : originalImage)}
          >
            <img src={originalImage} alt="Original" style={{ width: '100%', display: 'block' }} />
            <AnimatePresence>
              {showOverlay && (
                <motion.img
                  key="overlay"
                  src={`data:image/png;base64,${overlay_b64}`}
                  alt="Segmentation overlay"
                  style={{
                    position: 'absolute', top: 0, left: 0,
                    width: '100%', height: '100%',
                    objectFit: 'fill', opacity: overlayOpacity
                  }}
                  initial={{ opacity: 0 }} animate={{ opacity: overlayOpacity }} exit={{ opacity: 0 }}
                />
              )}
            </AnimatePresence>
            <div style={{ position: 'absolute', top: 6, right: 6, background: 'rgba(0,0,0,0.55)', borderRadius: 5, padding: '3px 5px', display: 'flex', alignItems: 'center', gap: 3 }}>
              <ZoomIn size={11} style={{ color: '#34d399' }} />
            </div>
          </div>

          {/* Opacity slider */}
          {showOverlay && (
            <div style={{ marginTop: '0.8rem' }}>
              <div className="ctrl-label" style={{ marginBottom: 6 }}>
                Overlay Opacity
                <em style={{ fontStyle: 'normal', fontFamily: 'var(--font-mono)', fontSize: '0.62rem', color: 'var(--sky)' }}>
                  {Math.round(overlayOpacity * 100)}%
                </em>
              </div>
              <input
                type="range" className="slider" min="0.05" max="1" step="0.05"
                value={overlayOpacity}
                onChange={e => handleOpacity(Number(e.target.value))}
              />
            </div>
          )}

          {/* Class toggle pills */}
          <div className="seg-class-toggles">
            <span style={{ fontFamily: 'var(--font-head)', fontSize: '0.52rem', letterSpacing: '0.16em', color: 'var(--text-lo)', flexShrink: 0 }}>CLASSES</span>
            {allClasses.map(cls => {
              const c = SEG_COLORS[cls];
              const active = visibleClasses[cls];
              return (
                <motion.button
                  key={cls}
                  whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                  onClick={() => setVisibleClasses(prev => ({ ...prev, [cls]: !prev[cls] }))}
                  style={{
                    padding: '3px 10px',
                    background: active ? `${c}22` : 'rgba(255,255,255,0.03)',
                    border: `1px solid ${active ? c : 'rgba(255,255,255,0.08)'}`,
                    borderRadius: 999,
                    color: active ? c : 'var(--text-lo)',
                    fontSize: '0.6rem',
                    cursor: 'pointer',
                    fontFamily: 'var(--font-body)',
                    transition: 'all 0.25s',
                    boxShadow: active ? `0 0 10px ${c}44` : 'none',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {cls}
                </motion.button>
              );
            })}
          </div>
        </div>
      )}

      {zoomSrc && <ZoomModal src={zoomSrc} label="SEGMENTATION OVERLAY" onClose={() => setZoomSrc(null)} />}

      {/* Coverage bars — filtered by toggle */}
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
                    transition={{ duration: 0.9, ease: [0.16,1,0.3,1], delay: 0.1 + idx * 0.06 }}
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
/*  STAGE 04 — FEATURE EXTRACTION PANEL                           */
/* ═══════════════════════════════════════════════════════════════ */
function FeaturesPanel({ features }) {
  const { brightness, shape, size, color } = features;

  const STAT_TILES = [
    {
      icon: '☀', label: 'Peak Brightness',
      value: brightness.peak_normalized != null
        ? brightness.peak_normalized.toFixed(1)
        : (brightness.peak * 100).toFixed(1),
      unit: '/ 100', color: '#f59e0b',
      desc: 'Max pixel intensity (normalized 0–100)',
    },
    {
      icon: '💡', label: 'Mean Luminosity',
      value: brightness.mean_luminosity != null
        ? brightness.mean_luminosity.toFixed(1)
        : (brightness.mean * 100).toFixed(1),
      unit: '/ 100', color: '#38bdf8',
      desc: 'Average frame luminance (0–100)',
    },
    {
      icon: '⊕', label: 'Est. Object Size',
      value: size.largest_region_px2 != null
        ? size.largest_region_px2.toLocaleString()
        : size.bright_pixels.toLocaleString(),
      unit: 'px²', color: '#34d399',
      desc: 'Largest connected bright region',
    },
    {
      icon: '◈', label: 'Shape Eccentricity',
      value: shape.eccentricity != null ? shape.eccentricity.toFixed(4) : '—',
      unit: '', color: '#a78bfa',
      desc: '0 = circle  ·  1 = linear streak',
    },
  ];

  return (
    <motion.div
      className="glass cnn-panel result-card"
      initial={{ opacity: 0, y: 24, scale: 0.97, filter: 'blur(8px)' }}
      animate={{ opacity: 1, y: 0, scale: 1, filter: 'blur(0px)' }}
      transition={{ duration: 0.6, delay: 0.15, ease: [0.43, 0.13, 0.23, 0.96] }}
      style={{ borderColor: 'rgba(245,158,11,0.25)' }}
    >
      <div className="panel-header">
        <div className="panel-title">
          <span style={{ fontSize: '0.9rem', lineHeight: 1 }}>📡</span>
          FEATURE EXTRACTION — STAGE 04
        </div>
        <span style={{
          fontSize: '0.52rem', fontFamily: 'var(--font-mono)', color: '#f59e0b',
          padding: '2px 8px', borderRadius: 999,
          border: '1px solid rgba(245,158,11,0.3)', background: 'rgba(245,158,11,0.08)',
          letterSpacing: '0.08em', whiteSpace: 'nowrap',
        }}>
          {color.spectral_hint?.split(' ')[0]?.toUpperCase() ?? 'BALANCED'}
        </span>
      </div>

      {/* 4 primary stat tiles — 2×2 grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: '0.6rem', marginBottom: '0.9rem' }}>
        {STAT_TILES.map(({ icon, label, value, unit, color: c, desc }) => (
          <motion.div
            key={label}
            initial={{ opacity: 0, scale: 0.94 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.18 }}
            style={{
              padding: '0.85rem 0.9rem', borderRadius: 12,
              background: `${c}09`, border: `1px solid ${c}28`,
              position: 'relative', overflow: 'hidden',
            }}
          >
            <div style={{ position: 'absolute', top: 8, right: 10, fontSize: '1.15rem', opacity: 0.14 }}>{icon}</div>
            <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.5rem', letterSpacing: '0.14em', color: 'var(--text-lo)', marginBottom: 5 }}>{label}</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1.1rem', color: c, fontWeight: 700, lineHeight: 1 }}>{value}</span>
              {unit && <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.58rem', color: `${c}88` }}>{unit}</span>}
            </div>
            <div style={{ fontSize: '0.57rem', color: 'var(--text-lo)', marginTop: 4, lineHeight: 1.4 }}>{desc}</div>
          </motion.div>
        ))}
      </div>

      {/* Secondary detail rows */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(160px,1fr))', gap: '0.4rem' }}>
        {[
          { label: 'Dynamic Range', value: brightness.dynamic_range?.toFixed(4), c: '#f59e0b' },
          { label: 'Edge Energy',   value: shape.edge_energy?.toFixed(6),          c: '#a78bfa' },
          { label: 'Fill Ratio',    value: size.fill_ratio?.toFixed(4),             c: '#34d399' },
          { label: 'Est. Objects',  value: size.object_count_est,                   c: '#38bdf8' },
          { label: 'Complexity',    value: shape.complexity,   c: '#a78bfa', isText: true },
          { label: 'Dominant Ch.',  value: color.dominant_channel, c: '#38bdf8', isText: true },
        ].map(({ label, value, c, isText }) => (
          <div key={label} style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '5px 8px', borderRadius: 7, background: 'rgba(255,255,255,0.025)',
          }}>
            <span style={{ fontSize: '0.6rem', color: 'var(--text-lo)' }}>{label}</span>
            <span style={{ fontFamily: isText ? 'inherit' : 'var(--font-mono)', fontSize: '0.64rem', color: c, fontWeight: 600 }}>{value}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
/*  STAGE 08 — LABELED OUTPUT PANEL                               */
/* ═══════════════════════════════════════════════════════════════ */
function LabeledOutputPanel({ labeledOutput, classification }) {
  const [zoom, setZoom] = useState(false);
  const src   = `data:image/png;base64,${labeledOutput.labeled_image_b64}`;
  const label = classification?.label ?? 'Unknown';
  const color = LABEL_COLORS[label] ?? '#38bdf8';

  const download = () => {
    const a = document.createElement('a');
    a.href     = src;
    a.download = `iarrd-labeled-${label.toLowerCase().replace(/ /g, '-')}.png`;
    a.click();
  };

  return (
    <motion.div className="glass" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
      style={{ borderRadius: 18, padding: '1.4rem 1.6rem', borderColor: `${color}33` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 34, height: 34, borderRadius: 10, background: `${color}12`, border: `1.5px solid ${color}55`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1rem' }}>🏷️</div>
          <div>
            <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.62rem', letterSpacing: '0.18em', color }}>STAGE 08 — ENHANCED IMAGE + OBJECT LABELS</div>
            <div style={{ fontSize: '0.67rem', color: 'var(--text-lo)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>
              {labeledOutput.composite_size} &nbsp;·&nbsp; {labeledOutput.layers_composited?.length ?? 5} layers composited
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
        <img src={src} alt="Labeled Output" style={{ width: '100%', display: 'block', imageRendering: 'high-quality' }} />
        <div style={{ position: 'absolute', bottom: 8, right: 8, fontFamily: 'var(--font-mono)', fontSize: '0.55rem', color, background: 'rgba(0,0,0,0.7)', padding: '3px 8px', borderRadius: 5, display: 'flex', alignItems: 'center', gap: 5 }}>
          <ZoomIn size={10} /> Click to enlarge
        </div>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: '0.9rem' }}>
        {(labeledOutput.layers_composited ?? []).map((layer, i) => (
          <motion.span key={i} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 + i * 0.05 }}
            style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color, padding: '2px 8px', borderRadius: 999, border: `1px solid ${color}44`, background: `${color}10` }}>
            ✓ {layer}
          </motion.span>
        ))}
      </div>
      {zoom && <ZoomModal src={src} label={`STAGE 08 — ${label.toUpperCase()}`} onClose={() => setZoom(false)} />}
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════════════ */
/*  STAGE 09 — SECOND OPINION (GEMINI + SIMBAD + NED)              */
/* ═══════════════════════════════════════════════════════════════ */
function SecondOpinionPanel({ secondOpinion }) {
  const { cnn_label, cnn_confidence, gemini, catalog_xref } = secondOpinion;
  
  return (
    <motion.div className="glass result-card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} style={{ borderColor: 'rgba(124, 58, 237, 0.3)', marginTop: '1.5rem' }}>
      <div className="panel-header">
        <div className="panel-title">
          <Brain size={14} style={{ color: '#7c3aed' }} />
          STAGE 09 — SECOND OPINION CROSS-VALIDATION
        </div>
        <span style={{ fontSize: '0.52rem', fontFamily: 'var(--font-mono)', color: '#7c3aed', padding: '2px 8px', borderRadius: 999, border: '1px solid rgba(124, 58, 237, 0.3)', background: 'rgba(124, 58, 237, 0.08)' }}>
          {gemini?.status === 'ok' ? 'GEMINI 1.5 ACTIVE' : 'GEMINI OFFLINE'}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
        {/* Gemini Analysis */}
        <div style={{ background: 'rgba(124,58,237,0.05)', border: '1px solid rgba(124,58,237,0.2)', borderRadius: 12, padding: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: '0.8rem' }}>
            <AlertTriangle size={13} color="#f59e0b" />
            <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-head)', letterSpacing: '0.1em', color: '#f59e0b' }}>
              CNN CONFIDENCE LOW ({cnn_confidence?.toFixed(1)}%)
            </span>
          </div>
          
          <div style={{ fontSize: '0.65rem', color: 'var(--text-lo)', lineHeight: 1.6, marginBottom: '0.5rem' }}>
            {gemini?.analysis || "Gemini cross-validation unavailable or processing failed."}
          </div>
          
          {gemini?.agrees_with_cnn !== undefined && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: '0.8rem', padding: '0.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: 8 }}>
              {gemini.agrees_with_cnn ? <CheckCircle size={14} color="#10b981" /> : <X size={14} color="#ef4444" />}
              <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: gemini.agrees_with_cnn ? '#10b981' : '#ef4444' }}>
                GEMINI {gemini.agrees_with_cnn ? 'AGREES' : 'DISAGREES'} WITH {cnn_label?.toUpperCase()}
              </span>
            </div>
          )}
        </div>

        {/* Database Catalogs */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {/* SIMBAD */}
          <div style={{ background: 'rgba(52,211,153,0.05)', border: '1px solid rgba(52,211,153,0.2)', borderRadius: 12, padding: '0.8rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Globe size={12} color="#34d399" />
                <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-head)', letterSpacing: '0.1em', color: '#34d399' }}>SIMBAD CATALOG (CDS)</span>
              </div>
              <span style={{ fontSize: '0.5rem', fontFamily: 'var(--font-mono)', color: 'var(--text-lo)' }}>{catalog_xref?.simbad?.elapsed_ms ?? 0}ms</span>
            </div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-lo)' }}>
              {catalog_xref?.simbad?.matches ? (
                <span>Found {catalog_xref.simbad.matches} match(es) for {catalog_xref.predicted_label}.</span>
              ) : (
                <span>No definitive matches found in region.</span>
              )}
            </div>
          </div>

          {/* NED */}
          <div style={{ background: 'rgba(56,189,248,0.05)', border: '1px solid rgba(56,189,248,0.2)', borderRadius: 12, padding: '0.8rem', flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <BookOpen size={12} color="#38bdf8" />
                <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-head)', letterSpacing: '0.1em', color: '#38bdf8' }}>NED CATALOG (NASA/IPAC)</span>
              </div>
              <span style={{ fontSize: '0.5rem', fontFamily: 'var(--font-mono)', color: 'var(--text-lo)' }}>{catalog_xref?.ned?.elapsed_ms ?? 0}ms</span>
            </div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-lo)' }}>
              {catalog_xref?.ned?.matches ? (
                <span>Found {catalog_xref.ned.matches} extragalactic object(s).</span>
              ) : (
                <span>Query failed or no objects in coordinate field.</span>
              )}
            </div>
          </div>
        </div>
      </div>
      
      <div style={{ borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '0.8rem', display: 'flex', alignItems: 'center', gap: 6 }}>
         <ExternalLink size={12} color="var(--text-lo)" />
         <span style={{ fontSize: '0.55rem', fontFamily: 'var(--font-mono)', color: 'var(--text-lo)' }}>
           CROSS-REFERENCE COMPLETE. AI DETERMINATION SUPERSEDES CATALOG MISSES.
         </span>
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
      // In-memory store has full result with all b64 images (set by Upload.js, no quota limit)
      const memResult = getLastResult();
      if (memResult) {
        setResult(memResult);
      } else {
        // Fallback: localStorage slim copy (survives hard refresh, but images are null)
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
      c.width  = img.naturalWidth;
      c.height = img.naturalHeight;
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
    a.href = url; a.download = 'iarrd-telemetry.json'; a.click();
    URL.revokeObjectURL(url);
  }, [result]);

  const copyShare = useCallback(() => {
    const payload = {
      label: result?.classification?.label,
      confidence: result?.classification?.confidence,
      anomaly: result?.reconstruction?.anomaly_score,
      quality: result?.reconstruction?.quality,
      elapsed: result?.elapsed_ms,
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
                  SCENE 03 — ASTRONOMICAL ANALYSIS DECK
                </div>
                <h1 className="section-title">Mission Telemetry</h1>
              </div>
              {result && (
                <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
                  <motion.button
                    className="btn-ghost"
                    onClick={copyShare}
                    whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
                    style={{ fontSize: '0.68rem' }}
                  >
                    <Share2 size={13} />
                    {copied ? 'Copied!' : 'Share'}
                  </motion.button>
                  <motion.button
                    className="btn-ghost"
                    onClick={exportJSON}
                    whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
                    style={{ fontSize: '0.68rem' }}
                  >
                    <FileJson size={13} />
                    Export JSON
                  </motion.button>
                  <Link to="/upload" className="btn-primary" style={{ fontSize: '0.68rem', padding: '10px 20px' }}>
                    <RotateCcw size={13} />
                    New Scan
                  </Link>
                </div>
              )}
            </div>
          </FadeUp>

          {/* ── Skeleton while loading ── */}
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
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '0.75rem' }}>
                  {[1,2,3].map(i => (
                    <div key={i} className="glass" style={{ height: 240, borderRadius: 14, padding: '1.2rem' }}>
                      <div style={{ height: 10, borderRadius: 5, background: 'rgba(255,255,255,0.06)', width: '40%', marginBottom: 16, animation: 'pulse 1.5s ease infinite' }} />
                      <div style={{ height: 10, borderRadius: 5, background: 'rgba(255,255,255,0.06)', width: '90%', marginBottom: 8, animation: 'pulse 1.5s ease infinite' }} />
                      <div style={{ height: 10, borderRadius: 5, background: 'rgba(255,255,255,0.06)', width: '70%', animation: 'pulse 1.5s ease infinite' }} />
                    </div>
                  ))}
                </div>
              </div>
            </FadeUp>
          )}

          {/* ── Last 3 scans history ── */}
          {!loading && history.length > 0 && (
            <FadeUp delay={0.08}>
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
                          <span style={{ fontSize: '0.55rem', fontFamily: 'var(--font-mono)', color: 'var(--text-lo)' }}>
                            #{i + 1}
                          </span>
                        </div>
                        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                          {[
                            { k: 'Conf',    v: `${(h.confidence ?? 0).toFixed(1)}%`, c: lbl },
                            { k: 'Anomaly', v: `${(h.anomaly_score ?? 0).toFixed(0)}`, c: qual },
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
                  No telemetry recorded. Navigate to the Uplink Station to acquire and process an astronomical image.
                </p>
                <Link to="/upload" className="btn-primary">
                  <Target size={14} />
                  Begin Acquisition
                </Link>
              </div>
            </FadeUp>
          )}


          {/* ── AI mode results ────────────────────────────────── */}
          {result && isAI && (
            <>
              {/* Stats strip */}
              <FadeUp delay={0.05}>
                <div className="stats-strip">
                  {[
                    {
                      label: 'AI CLASSIFICATION',
                      value: result.classification?.label ?? '—',
                      style: { fontSize: '0.9rem', color: getColor(result.classification?.label) },
                    },
                    {
                      label: 'CONFIDENCE',
                      value: <AnimNum value={result.classification?.confidence ?? 0} format={v => `${v.toFixed(1)}%`} />,
                    },
                    {
                      label: 'ANOMALY SCORE',
                      value: <AnimNum value={result.reconstruction?.anomaly_score ?? 0} format={v => v.toFixed(1)} />,
                    },
                    {
                      label: 'INFERENCE TIME',
                      value: <AnimNum value={result.elapsed_ms ?? 0} format={v => `${v.toFixed(0)} ms`} />,
                    },
                    {
                      label: 'PIPELINE STAGES',
                      value: result.pipeline?.stages?.length ?? 8,
                      style: { color: '#10b981' },
                    },
                  ].map(({ label, value, style }) => (
                    <div key={label} className="stat-cell glass">
                      <div className="stat-label">{label}</div>
                      <div className="stat-value" style={style}>{value}</div>
                    </div>
                  ))}
                </div>
              </FadeUp>

              {/* ── 8-Stage Pipeline Execution Tracker ── */}
              <FadeUp delay={0.07}>
                <div className="glass" style={{ padding: '1.2rem 1.5rem', borderRadius: 14, marginTop: 0 }}>
                  <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.6rem', letterSpacing: '0.18em', color: '#38bdf8', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Activity size={12} strokeWidth={1.5} />
                    PIPELINE EXECUTION — ALL {result.pipeline?.stages?.length ?? 8} STAGES COMPLETED
                    {result.pipeline?.total_ms && (
                      <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', color: '#10b981', fontSize: '0.58rem' }}>
                        TOTAL: {result.pipeline.total_ms} ms
                      </span>
                    )}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '0.55rem' }}>
                    {(result.pipeline?.stages ?? [
                      { num:'01', name:'Astronomical Image Acquisition',                   detail:'File validated · metadata extracted',           ms: null, color:'#38bdf8' },
                      { num:'02', name:'Preprocessing  (Noise Removal & Normalization)',    detail:'Gaussian σ=1.2 · normalized to [0,1]',          ms: null, color:'#38bdf8' },
                      { num:'03', name:'Image Enhancement  (Contrast · Deblurring · SR)', detail:'8-pass pipeline · 4096×4096 LANCZOS upscale',    ms: null, color:'#a78bfa' },
                      { num:'04', name:'Feature Extraction  (Brightness · Shape · Size)',  detail:'Luminosity · edge energy · object count',        ms: null, color:'#34d399' },
                      { num:'05', name:'AI Model Processing  (CNN)',                        detail:'6-class softmax · ResNet-style backbone',        ms: null, color:'#38bdf8' },
                      { num:'06', name:'Celestial Object Detection & Classification',       detail:'Autoencoder anomaly · MSE reconstruction',      ms: null, color:'#f59e0b' },
                      { num:'07', name:'U-Net Pixel Segmentation',                          detail:'Encoder–decoder · 6-class pixel mask',           ms: null, color:'#10b981' },
                      { num:'08', name:'Enhanced Image + Object Labels',                    detail:'4K composite · overlay · label banner drawn',   ms: null, color:'#34d399' },
                    ]).map((stage, i) => {
                      const STAGE_COLORS = ['#38bdf8','#38bdf8','#a78bfa','#34d399','#38bdf8','#f59e0b','#10b981','#34d399'];
                      const color = stage.color ?? STAGE_COLORS[i] ?? '#38bdf8';
                      return (
                        <motion.div
                          key={stage.num ?? i}
                          initial={{ opacity: 0, x: -12 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.08 + i * 0.06, ease: [0.16,1,0.3,1] }}
                          style={{ display: 'flex', gap: 10, alignItems: 'flex-start', padding: '8px 10px', borderRadius: 10, background: `${color}07`, border: `1px solid ${color}22` }}
                        >
                          <div style={{
                            width: 26, height: 26, borderRadius: '50%', flexShrink: 0,
                            background: `${color}18`, border: `1.5px solid ${color}`,
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: '0.58rem', fontFamily: 'var(--font-mono)', color,
                          }}>✓</div>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: '0.7rem', color: '#e0f4ff', fontWeight: 600, lineHeight: 1.3 }}>
                              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.55rem', color: 'var(--text-lo)', marginRight: 5 }}>{stage.num}</span>
                              {stage.name}
                            </div>
                            <div style={{ fontSize: '0.6rem', color: 'var(--text-lo)', marginTop: 2, lineHeight: 1.4 }}>{stage.detail}</div>
                            {stage.ms != null && (
                              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.55rem', color, marginTop: 3 }}>{stage.ms} ms</div>
                            )}
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                </div>
              </FadeUp>

              {/* ── 8K Enhancement panel — full width ── */}
              {result.enhancement?.image_b64 && (
                <FadeUp delay={0.08}>
                  <div style={{ marginBottom: '1.5rem' }}>
                    <EnhancementPanel
                      enhancement={result.enhancement}
                      originalImage={result.image}
                    />
                  </div>
                </FadeUp>
              )}

              {/* ── CNN + Enhancement Report + Feature Extraction + U-Net ── */}
              <FadeUp delay={0.1}>
                <div className="telemetry-grid">
                  {result.classification && (
                    <ClassificationPanel
                      classification={result.classification}
                      minConf={result.minConf ?? 0}
                    />
                  )}
                  {result.reconstruction && (
                    <AnomalyPanel
                      reconstruction={result.reconstruction}
                      originalImage={result.image}
                    />
                  )}
                  {result.features && (
                    <FeaturesPanel features={result.features} />
                  )}
                  {result.segmentation && (
                    <SegmentationPanel
                      segmentation={result.segmentation}
                      originalImage={result.image}
                    />
                  )}
                </div>
              </FadeUp>

              {/* ── Labeled Output Panel (Stage 08) ── */}
              {result.labeled_output?.labeled_image_b64 && (
                <FadeUp delay={0.14}>
                  <LabeledOutputPanel labeledOutput={result.labeled_output} classification={result.classification} />
                </FadeUp>
              )}

              {/* ── Stage 09 — Second Opinion (Gemini + SIMBAD + NED) ── */}
              {result.second_opinion && (
                <FadeUp delay={0.16}>
                  <SecondOpinionPanel secondOpinion={result.second_opinion} />
                </FadeUp>
              )}

              {/* Action bar */}
              <FadeUp delay={0.2}>
                <div className="action-bar">
                  <span className="action-bar-label">EXPORT RESULTS</span>
                  <motion.button className="btn-ghost" onClick={exportJSON}
                    whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
                    style={{ fontSize: '0.68rem', padding: '8px 16px' }}>
                    <DownloadCloud size={13} /> Download JSON
                  </motion.button>
                  <motion.button className="btn-ghost" onClick={copyShare}
                    whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
                    style={{ fontSize: '0.68rem', padding: '8px 16px' }}>
                    <Share2 size={13} /> {copied ? 'Copied!' : 'Copy Summary'}
                  </motion.button>
                  <motion.button
                    className="btn-ghost"
                    whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
                    style={{ fontSize: '0.68rem', padding: '8px 16px' }}
                    onClick={() => {
                      const rows = [
                        ['Field', 'Value'],
                        ['Classification', result.classification?.label ?? '—'],
                        ['Confidence', `${(result.classification?.confidence ?? 0).toFixed(1)}%`],
                        ['Anomaly Score', result.reconstruction?.anomaly_score ?? '—'],
                        ['Quality', result.reconstruction?.quality ?? '—'],
                        ['Inference Time (ms)', result.elapsed_ms ?? '—'],
                      ];
                      const csv = rows.map(r => r.join(',')).join('\n');
                      const a = document.createElement('a');
                      a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
                      a.download = 'iarrd-report.csv'; a.click();
                    }}
                  >
                    <FileText size={13} /> Download Report
                  </motion.button>
                  <Link to="/upload" className="btn-primary"
                    style={{ fontSize: '0.68rem', padding: '8px 16px', marginLeft: 'auto' }}>
                    <RotateCcw size={13} /> New Analysis
                  </Link>
                </div>
              </FadeUp>
            </>
          )}

          {/* ── Browser-mode results ───────────────────────────── */}
          {result && !isAI && (
            <>
              <FadeUp delay={0.05}>
                <div className="stats-strip">
                  {[
                    { label: 'TARGETS FOUND', value: result.objects?.length ?? 0 },
                    { label: 'NOISE FLOOR σ', value: result.meta?.std?.toFixed(1) ?? '—' },
                    { label: 'SIGMA USED',    value: result.meta?.sigma ?? '—' },
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
                        DETECTED TARGETS
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
                            transition={{ delay: i * 0.06, ease: [0.16,1,0.3,1] }}
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
