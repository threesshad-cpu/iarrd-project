import React, { useRef, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  UploadCloud, FileImage, RotateCcw,
  Zap, SlidersHorizontal, Info, AlertCircle, Cpu, Globe,
  CheckCircle2, Clock, Shield, ChevronDown, ChevronUp, Layers
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { PageTransition, FadeUp } from '../components/Animations';
import { setLastResult } from '../ResultsStore';

const API_BASE = (
  process.env.REACT_APP_API_BASE ||
  process.env.REACT_APP_API_URL ||
  ''
).replace(/\/$/, '');

/* ═══════════════════════════════════════════════ */
/*  BROWSER-SIDE PROCESSING (fallback)             */
/* ═══════════════════════════════════════════════ */
function gaussianKernel(sigma) {
  const radius = Math.ceil(sigma * 3);
  const size   = radius * 2 + 1;
  const kernel = new Float32Array(size * size);
  const two    = 2 * sigma * sigma;
  let sum = 0;
  for (let y = -radius; y <= radius; y++) {
    for (let x = -radius; x <= radius; x++) {
      const v = Math.exp(-(x*x + y*y) / two) / (Math.PI * two);
      kernel[(y+radius)*size + (x+radius)] = v;
      sum += v;
    }
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  return { kernel, size, radius };
}

function convolveGray(src, width, height, { kernel, size, radius }) {
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let ky = -radius; ky <= radius; ky++) {
        for (let kx = -radius; kx <= radius; kx++) {
          const ix = Math.min(width-1,  Math.max(0, x+kx));
          const iy = Math.min(height-1, Math.max(0, y+ky));
          acc += src[iy*width+ix] * kernel[(ky+radius)*size + (kx+radius)];
        }
      }
      out[y*width+x] = acc;
    }
  }
  return out;
}

function toGray(imgData) {
  const d = imgData.data, n = d.length / 4;
  const g = new Float32Array(n);
  for (let i = 0; i < n; i++)
    g[i] = 0.2989*d[i*4] + 0.5870*d[i*4+1] + 0.1140*d[i*4+2];
  return g;
}

function normalize(arr) {
  let mn = Infinity, mx = -Infinity;
  for (const v of arr) { if (v < mn) mn = v; if (v > mx) mx = v; }
  const out = new Uint8ClampedArray(arr.length), rng = mx - mn || 1;
  for (let i = 0; i < arr.length; i++) out[i] = ((arr[i] - mn) / rng) * 255;
  return out;
}

function detectObjects(gray, width, height, kThreshold) {
  const n = gray.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += gray[i];
  const mean = sum / n;
  let sq = 0;
  for (let i = 0; i < n; i++) sq += (gray[i] - mean) ** 2;
  const std    = Math.sqrt(sq / n);
  const thresh = mean + kThreshold * std;
  const visited = new Uint8Array(n);
  const objects = [];
  for (let i = 0; i < n; i++) {
    if (visited[i] || gray[i] <= thresh) continue;
    const stack = [i];
    visited[i] = 1;
    let sx = 0, sy = 0, sv = 0, cnt = 0;
    while (stack.length) {
      const idx = stack.pop();
      const py = Math.floor(idx / width), px = idx % width;
      const val = gray[idx];
      sx += px*val; sy += py*val; sv += val; cnt++;
      for (let ny = Math.max(0,py-1); ny <= Math.min(height-1,py+1); ny++)
        for (let nx = Math.max(0,px-1); nx <= Math.min(width-1,px+1); nx++) {
          const ni = ny*width+nx;
          if (!visited[ni] && gray[ni] > thresh) { visited[ni]=1; stack.push(ni); }
        }
    }
    if (cnt >= 3) objects.push({ x: sx/sv, y: sy/sv, area: cnt, brightness: sv/cnt });
  }
  return { objects, thresh, mean, std };
}

/* ═══════════════════════════════════════════════ */
/*  AI PIPELINE STEPS for visual feedback          */
/* ═══════════════════════════════════════════════ */
const AI_STEPS = [
  'Preparing your image…',
  'Optimizing image quality…',
  'Enhancing clarity…',
  'Identifying features…',
  'Analyzing objects…',
  'Finalizing results…',
];

/* ═══════════════════════════════════════════════ */
/*  MODE TOGGLE                                    */
/* ═══════════════════════════════════════════════ */
function ModeToggle({ useAI, setUseAI }) {
  return (
    <div className="mode-toggle">
      {[
        { val: true,  icon: <Cpu size={13} />,   label: 'AI Models' },
        { val: false, icon: <Globe size={13} />,  label: 'Browser' },
      ].map(({ val, icon, label }) => (
        <motion.button
          key={String(val)}
          className={`mode-tab ${useAI === val ? 'active' : ''}`}
          onClick={() => setUseAI(val)}
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
        >
          {icon}{label}
        </motion.button>
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════════ */
/*  DROPZONE                                       */
/* ═══════════════════════════════════════════════ */
function Dropzone({ onFile, dragOver, setDragOver }) {
  return (
    <div className="dropzone-shell">
      <motion.div
        className={`dropzone-body ${dragOver ? 'active' : ''}`}
        animate={{ y: [0, -10, 0] }}
        transition={{ y: { repeat: Infinity, duration: 4.5, ease: 'easeInOut' } }}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => { e.preventDefault(); setDragOver(false); e.dataTransfer.files[0] && onFile(e.dataTransfer.files[0]); }}
      >
        <div className="reticle reticle-tl" />
        <div className="reticle reticle-tr" />
        <div className="reticle reticle-bl" />
        <div className="reticle reticle-br" />

        <motion.div
          animate={{ scale: dragOver ? 1.2 : 1, rotate: dragOver ? 10 : 0 }}
          transition={{ duration: 0.3 }}
        >
          <UploadCloud size={56} strokeWidth={1.2} className="dz-icon" />
        </motion.div>

        <div>
          <div className="dz-title">
            {dragOver ? 'Release to deploy' : 'Data Intake Chamber'}
          </div>
          <div className="dz-hint" style={{ marginTop: 6 }}>
            {dragOver ? 'Drop your astronomical image here' : 'Drag & drop or click to select'}
          </div>
          <div className="dz-formats" style={{ marginTop: 10 }}>
            {['PNG', 'JPG', 'TIFF', 'WEBP'].map(f => (
              <span key={f} className="dz-tag">{f}</span>
            ))}
          </div>
          <div style={{ marginTop: 8, fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-lo)', textAlign: 'center' }}>
            MAX 10 MB
          </div>
        </div>

        <input
          type="file"
          accept="image/*"
          className="dz-input"
          onChange={e => e.target.files[0] && onFile(e.target.files[0])}
        />
      </motion.div>
    </div>
  );
}

/* ═══════════════════════════════════════════════ */
/*  PROCESSING OVERLAY                             */
/* ═══════════════════════════════════════════════ */
function ProcessingOverlay({ stepIndex, statusMsg }) {
  return (
    <motion.div
      className="proc-overlay"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="scan-orbit" />

      <div className="proc-step-label">
        {statusMsg || 'COMPUTING TELEMETRY'}
      </div>

      <div className="proc-steps">
        {AI_STEPS.map((_, i) => (
          <div
            key={i}
            className={`proc-step-dot ${i < stepIndex ? 'done' : i === stepIndex ? 'active' : ''}`}
          />
        ))}
      </div>

      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-lo)', textAlign: 'center', maxWidth: 180, lineHeight: 1.5 }}>
        {AI_STEPS[Math.min(stepIndex, AI_STEPS.length - 1)]}
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════ */
/*  UPLOAD PAGE                                    */
/* ═══════════════════════════════════════════════ */
export default function Upload() {
  const navigate   = useNavigate();
  const canvasRef  = useRef(null);
  const fileRef    = useRef(null);

  const [preview,    setPreview]    = useState(null);
  const [fileName,   setFileName]   = useState('');
  const [dims,       setDims]       = useState(null);
  const [fileSize,   setFileSize]   = useState(0);
  const [dragOver,   setDragOver]   = useState(false);
  const [processing, setProcessing] = useState(false);
  const [useAI,      setUseAI]      = useState(true);
  const [sigma,      setSigma]      = useState(1.4);
  const [kThr,       setKThr]       = useState(1.4);
  const [error,      setError]      = useState('');
  const [statusMsg,  setStatusMsg]  = useState('');
  const [stepIndex,  setStepIndex]  = useState(0);
  const [advOpen,    setAdvOpen]    = useState(false);
  const [minConf,    setMinConf]    = useState(
    () => parseFloat(localStorage.getItem('iarrd_minConf') || '0')
  );

  const handleFile = useCallback((file) => {
    setError('');
    if (file.size > 10 * 1024 * 1024) { setError('File exceeds 10 MB limit.'); return; }
    fileRef.current = file;
    setFileSize(file.size);
    const url = URL.createObjectURL(file);
    setPreview(url);
    setFileName(file.name);
    const img = new Image();
    img.onload = () => {
      const c = canvasRef.current;
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext('2d').drawImage(img, 0, 0);
      setDims({ w: img.naturalWidth, h: img.naturalHeight });
    };
    img.src = url;
  }, []);

  const advanceStep = (i, msg) => {
    setStepIndex(i);
    setStatusMsg(msg);
  };

  /* ── AI backend pipeline ──────────────────────────────────── */
  const runAIPipeline = useCallback(async () => {
    if (!fileRef.current) return;
    setProcessing(true);
    setError('');
    setStepIndex(0);

    try {
      advanceStep(0, 'Connecting to analysis engine…');
      const formData = new FormData();
      formData.append('file', fileRef.current);

      advanceStep(1, 'Optimizing image quality…');
      await new Promise(r => setTimeout(r, 200));

      advanceStep(2, 'Analyzing your image…');
      let res;
      try {
        res = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: formData });
      } catch (netErr) {
        throw Object.assign(new Error(
          'Something went wrong. Please check your connection and try again.'
        ), { kind: 'connection' });
      }

      advanceStep(3, 'Mapping object regions…');
      await new Promise(r => setTimeout(r, 100));

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        const detail = errData.detail || '';
        if (res.status === 422 || detail.toLowerCase().includes('image') || detail.toLowerCase().includes('format')) {
          throw Object.assign(new Error(
            'This file format is not supported. Please upload a PNG, JPG, TIFF or WEBP image.'
          ), { kind: 'format' });
        }
        throw Object.assign(new Error(
          'Something went wrong. Please try again.'
        ), { kind: 'model' });
      }

      advanceStep(4, 'Building segmentation map…');
      const data = await res.json();

      advanceStep(5, 'Preparing your results…');
      await new Promise(r => setTimeout(r, 200));

      const canvas = canvasRef.current;
      const resultPayload = {
        mode: 'ai',
        // In AI mode, we don't strictly need to carry the massive original image in localStorage
        // downsample it if we do to avoid QuotaExceeded on 4k/8k images
        image: (() => {
          if (!canvas) return null;
          const maxDim = 800; // max width/height
          const w = canvas.width, h = canvas.height;
          if (w <= maxDim && h <= maxDim) return canvas.toDataURL('image/jpeg', 0.85);
          const scale = maxDim / Math.max(w, h);
          const c2 = document.createElement('canvas');
          c2.width = w * scale; c2.height = h * scale;
          c2.getContext('2d').drawImage(canvas, 0, 0, c2.width, c2.height);
          return c2.toDataURL('image/jpeg', 0.85);
        })(),
        filename: data.filename,
        elapsed_ms: data.elapsed_ms,
        // ── Core analysis panels ──────────────────────────────────
        classification: data.classification,
        reconstruction: data.reconstruction,
        segmentation:   data.segmentation,
        // ── Extended pipeline panels (previously dropped) ─────────
        enhancement:    data.enhancement,
        features:       data.features,
        labeled_output: data.labeled_output,
        pipeline:       data.pipeline,
        acquisition:    data.acquisition,
        preprocessing:  data.preprocessing,
        second_opinion: data.second_opinion ?? null,
        // ── Meta ─────────────────────────────────────────────────
        objects: [],
        meta: { width: dims?.w, height: dims?.h },
        minConf,
        timestamp: Date.now(),
      };
      // Full payload (all base64 images) → in-memory module store (no quota, survives navigation)
      setLastResult(resultPayload);
      // Slim payload (no b64 blobs) → localStorage for cross-session history only
      const slim = {
        ...resultPayload,
        image: null,
        enhancement: resultPayload.enhancement
          ? { ...resultPayload.enhancement, image_b64: null, multi: null, premium_chain: null }
          : null,
        segmentation:   resultPayload.segmentation
          ? { ...resultPayload.segmentation, overlay_b64: null } : null,
        reconstruction: resultPayload.reconstruction
          ? { ...resultPayload.reconstruction, image_b64: null, gaussian_preview_b64: null } : null,
        labeled_output: resultPayload.labeled_output
          ? { ...resultPayload.labeled_output, labeled_image_b64: null } : null,
      };
      try { localStorage.setItem('lastResult', JSON.stringify(slim)); } catch (_) {}

      // Save to history (max 3 scans)
      try {
        const prevHistory = JSON.parse(localStorage.getItem('iarrd_history') || '[]');
        const histEntry = {
          timestamp: Date.now(),
          label: data.classification?.label,
          confidence: data.classification?.confidence,
          anomaly_score: data.reconstruction?.anomaly_score,
          elapsed_ms: data.elapsed_ms,
          quality: data.reconstruction?.quality,
          filename: data.filename || fileRef.current?.name || '—',
        };
        const newHistory = [histEntry, ...prevHistory].slice(0, 3);
        localStorage.setItem('iarrd_history', JSON.stringify(newHistory));
      } catch (_) {}

      navigate('/results');
    } catch (e) {
      setError('Something went wrong. Please try again.');
    } finally {
      setProcessing(false);
      setStatusMsg('');
      setStepIndex(0);
    }
  }, [navigate, dims, minConf]);

  /* ── Browser-side pipeline ────────────────────────────────── */
  const runBrowserPipeline = useCallback(() => {
    const c = canvasRef.current;
    if (!c?.width) return;
    setProcessing(true);
    setStatusMsg('Running Gaussian blur + blob detection …');
    setTimeout(() => {
      const ctx = c.getContext('2d');
      const w = c.width, h = c.height;
      const img = ctx.getImageData(0, 0, w, h);
      const gray = toGray(img);
      const blurred = convolveGray(gray, w, h, gaussianKernel(sigma));
      const normed  = normalize(blurred);
      const out = ctx.createImageData(w, h);
      for (let i = 0; i < w * h; i++) {
        const v = normed[i];
        out.data[i*4]=v; out.data[i*4+1]=v; out.data[i*4+2]=v; out.data[i*4+3]=255;
      }
      ctx.putImageData(out, 0, 0);
      const { objects, thresh, mean, std } = detectObjects(blurred, w, h, kThr);
      const imgScale = (() => {
        const maxD = 800;
        if (w <= maxD && h <= maxD) return c.toDataURL('image/jpeg', 0.85);
        const s = maxD / Math.max(w, h);
        const c2 = document.createElement('canvas');
        c2.width = w * s; c2.height = h * s;
        c2.getContext('2d').drawImage(c, 0, 0, c2.width, c2.height);
        return c2.toDataURL('image/jpeg', 0.85);
      })();

      localStorage.setItem('lastResult', JSON.stringify({
        mode: 'browser',
        image: imgScale,
        objects,
        meta: { thresh, mean, std, sigma, kThreshold: kThr, width: w, height: h }
      }));
      setProcessing(false);
      setStatusMsg('');
      navigate('/results');
    }, 80);
  }, [sigma, kThr, navigate]);

  const runPipeline = useCallback(() => {
    if (useAI) runAIPipeline();
    else        runBrowserPipeline();
  }, [useAI, runAIPipeline, runBrowserPipeline]);

  const reset = () => {
    setPreview(null); setFileName(''); setDims(null);
    setError(''); setStatusMsg(''); fileRef.current = null; setFileSize(0);
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024*1024) return `${(bytes/1024).toFixed(1)} KB`;
    return `${(bytes/1024/1024).toFixed(2)} MB`;
  };

  return (
    <PageTransition>
      <section className="scene" style={{ alignItems: 'flex-start', paddingTop: '6.5rem' }}>
        <div className="page-inner">

          {/* Header */}
          <FadeUp>
            <div className="eyebrow">
              <UploadCloud size={13} strokeWidth={1.5} />
              SCENE 02 — DATA INTAKE CHAMBER
            </div>
            <h1 className="section-title">Uplink Station</h1>
            <p className="section-desc">
              Deploy an astronomical image into the analysis pipeline.
              Choose <strong style={{ color: 'var(--sky)' }}>AI Models</strong> for
              deep-learning inference (CNN · Autoencoder · U-Net) or{' '}
              <strong style={{ color: 'var(--text-hi)' }}>Browser</strong> for
              in-browser Gaussian + blob detection.
            </p>
          </FadeUp>

          {/* Status + mode row */}
          <FadeUp delay={0.05}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <ModeToggle useAI={useAI} setUseAI={setUseAI} />

              <div className="upload-status-bar" style={{ flex: 1, minWidth: 200 }}>
                <div className={`status-indicator ${processing ? '' : error ? 'error' : 'idle'}`} />
                <span>
                  {processing
                    ? statusMsg || 'Processing…'
                    : error ? 'Error — check details below'
                    : preview ? `Ready — ${fileName}`
                    : 'Awaiting image upload'}
                </span>
              </div>
            </div>
          </FadeUp>

          {/* Main layout */}
          <FadeUp delay={0.1}>
            <div className="upload-grid">

              {/* LEFT — drop or preview */}
              <div>
                {!preview ? (
                  <Dropzone onFile={handleFile} dragOver={dragOver} setDragOver={setDragOver} />
                ) : (
                  <div className="preview-wrap" style={{ position: 'relative' }}>
                    <canvas ref={canvasRef} className="preview-canvas" />

                    {/* Metadata badges */}
                    {dims && !processing && (
                      <div className="preview-meta">
                        <span className="preview-badge">{dims.w} x {dims.h}px</span>
                        <span className="preview-badge">{formatSize(fileSize)}</span>
                      </div>
                    )}

                    <AnimatePresence>
                      {processing && (
                        <ProcessingOverlay stepIndex={stepIndex} statusMsg={statusMsg} />
                      )}
                    </AnimatePresence>
                  </div>
                )}

                {!preview && <canvas ref={canvasRef} style={{ display: 'none' }} />}

                {/* Error */}
                <AnimatePresence>
                  {error && (
                    <motion.div
                      className="error-note"
                      style={{ marginTop: '1rem' }}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                    >
                      <AlertCircle size={15} style={{ flexShrink: 0 }} />
                      {error}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* RIGHT — controls */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

                {/* File badge */}
                <AnimatePresence>
                  {preview && (
                    <motion.div
                      className="file-badge"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <FileImage size={20} className="file-badge-icon" strokeWidth={1.5} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div className="file-badge-name">{fileName}</div>
                        {dims && <div className="file-badge-dim">{dims.w} x {dims.h}px · {formatSize(fileSize)}</div>}
                      </div>
                      <CheckCircle2 size={16} style={{ color: 'var(--emerald)', flexShrink: 0 }} />
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* AI / Browser info card */}
                <AnimatePresence mode="wait">
                  {useAI ? (
                    <motion.div
                      key="ai"
                      className="glass controls-card"
                      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <Cpu size={14} strokeWidth={1.5} style={{ color: 'var(--sky)' }} />
                        <span style={{ fontFamily: 'var(--font-head)', fontSize: '0.64rem', letterSpacing: '0.16em', color: 'var(--text-hi)' }}>
                          AI PIPELINE — 3 MODELS
                        </span>
                      </div>

                      {[
                        { name: 'CNN Classifier',   desc: 'Preprocessing → Feature extraction → 6-class classification',  color: '#38bdf8', icon: <Cpu size={12} />, tag: 'CLASSIFY' },
                        { name: 'Autoencoder',       desc: 'Image enhancement · reconstruction · anomaly scoring (replaces super-resolution step)', color: '#a78bfa', icon: <Clock size={12} />, tag: 'ENHANCE' },
                        { name: 'U-Net Segmentor',   desc: 'Pixel-level semantic segmentation — labels every object region', color: '#34d399', icon: <Layers size={12} strokeWidth={1.5} />, tag: 'SEGMENT' },
                      ].map(m => (
                        <div key={m.name} style={{
                          display: 'flex', gap: 10, alignItems: 'flex-start',
                          padding: '10px 0', borderBottom: '1px solid rgba(255,255,255,0.04)'
                        }}>
                          <div style={{
                            width: 28, height: 28, borderRadius: 8, flexShrink: 0,
                            background: `${m.color}14`, border: `1px solid ${m.color}30`,
                            display: 'flex', alignItems: 'center', justifyContent: 'center', color: m.color
                          }}>{m.icon}</div>
                          <div style={{ flex: 1 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <div style={{ fontSize: '0.78rem', color: 'var(--text-hi)', fontWeight: 600 }}>{m.name}</div>
                              <span style={{ padding: '1px 6px', borderRadius: 4, background: `${m.color}22`, border: `1px solid ${m.color}44`, fontSize: '0.55rem', color: m.color, fontFamily: 'var(--font-mono)', letterSpacing: '0.08em' }}>{m.tag}</span>
                            </div>
                            <div style={{ fontSize: '0.68rem', color: 'var(--text-lo)', marginTop: 3, lineHeight: 1.5 }}>{m.desc}</div>
                          </div>
                        </div>
                      ))}

                      {/* Super-resolution note */}
                      <div style={{
                        display: 'flex', gap: 8, alignItems: 'flex-start',
                        padding: '10px 12px', borderRadius: 8,
                        background: 'rgba(167,139,250,0.06)',
                        border: '1px solid rgba(167,139,250,0.2)',
                        marginTop: 4
                      }}>
                        <Info size={13} style={{ color: '#a78bfa', flexShrink: 0, marginTop: 1 }} />
                        <div style={{ fontSize: '0.68rem', color: '#c4b5fd', lineHeight: 1.55 }}>
                          <strong style={{ color: '#a78bfa' }}>Image Enhancement note:</strong> The Autoencoder reconstructs the input at 128×128 px — this reconstructed output IS the enhanced image referenced in the pipeline flow.
                        </div>
                      </div>

                      {/* Advanced options toggle */}
                      <button
                        onClick={() => setAdvOpen(o => !o)}
                        style={{
                          background: 'none', border: 'none', color: 'var(--text-lo)',
                          display: 'flex', alignItems: 'center', gap: 6,
                          fontFamily: 'var(--font-head)', fontSize: '0.6rem', letterSpacing: '0.12em',
                          cursor: 'pointer', padding: 0
                        }}
                      >
                        {advOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        ADVANCED OPTIONS
                      </button>

                      <AnimatePresence>
                        {advOpen && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            style={{ overflow: 'hidden' }}
                          >
                            <div className="ctrl-note" style={{ marginTop: 0 }}>
                              <Info size={14} style={{ flexShrink: 0, color: 'var(--sky)', marginTop: 2 }} />
                              <span>
                                Connect to a running analysis server to use AI mode.
                              </span>
                            </div>

                            {/* Min confidence slider */}
                            <div className="ctrl-row" style={{ marginTop: 10 }}>
                              <div className="ctrl-label">
                                Min. Confidence Filter
                                <em>{minConf.toFixed(0)}%</em>
                              </div>
                              <input
                                type="range" className="slider" min="0" max="50" step="1"
                                value={minConf}
                                onChange={e => {
                                  const v = Number(e.target.value);
                                  setMinConf(v);
                                  localStorage.setItem('iarrd_minConf', String(v));
                                }}
                              />
                              <div className="ctrl-bounds">
                                <span>0% — Show all</span>
                                <span>50% — Top only</span>
                              </div>
                            </div>

                            <div style={{ marginTop: 10, display: 'flex', gap: 6 }}>
                              {[
                                { icon: <Shield size={11}/>, label: 'Max 10MB' },
                                { icon: <Clock size={11}/>,  label: '~500ms avg' },
                              ].map(({ icon, label }) => (
                                <div key={label} style={{
                                  display: 'flex', alignItems: 'center', gap: 5,
                                  padding: '4px 10px', borderRadius: 'var(--r-full)',
                                  border: '1px solid rgba(255,255,255,0.06)',
                                  fontSize: '0.62rem', color: 'var(--text-lo)',
                                  fontFamily: 'var(--font-mono)'
                                }}>
                                  {icon}{label}
                                </div>
                              ))}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="browser"
                      className="glass controls-card"
                      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <SlidersHorizontal size={14} strokeWidth={1.5} style={{ color: 'var(--sky)' }} />
                        <span style={{ fontFamily: 'var(--font-head)', fontSize: '0.64rem', letterSpacing: '0.16em', color: 'var(--text-hi)' }}>
                          PIPELINE PARAMETERS
                        </span>
                      </div>

                      <div className="ctrl-row">
                        <div className="ctrl-label">
                          Gaussian Sigma
                          <em>{sigma.toFixed(1)}</em>
                        </div>
                        <input type="range" className="slider" min="0.5" max="6" step="0.1"
                          value={sigma} onChange={e => setSigma(Number(e.target.value))} />
                        <div className="ctrl-bounds"><span>0.5 — Fine</span><span>6.0 — Coarse</span></div>
                      </div>

                      <div className="ctrl-row">
                        <div className="ctrl-label">
                          Detection Threshold k
                          <em>{kThr.toFixed(1)}</em>
                        </div>
                        <input type="range" className="slider" min="0.5" max="3" step="0.1"
                          value={kThr} onChange={e => setKThr(Number(e.target.value))} />
                        <div className="ctrl-bounds"><span>0.5 — Sensitive</span><span>3.0 — Strict</span></div>
                      </div>

                      <div className="ctrl-divider" />

                      <div className="ctrl-note">
                        <Info size={14} style={{ flexShrink: 0, color: 'var(--sky)', marginTop: 2 }} />
                        Higher sigma = stronger blur. Higher k = fewer, brighter targets only.
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Actions */}
                <div className="action-row">
                  <motion.button
                    className="btn-primary"
                    style={{ flex: 1 }}
                    onClick={runPipeline}
                    disabled={!preview || processing}
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.96 }}
                  >
                    <Zap size={15} strokeWidth={2} />
                    {processing
                      ? (statusMsg || 'Processing…')
                      : useAI ? 'Run AI Analysis' : 'Initiate Scan'}
                  </motion.button>

                  {preview && (
                    <motion.button
                      className="btn-icon"
                      onClick={reset}
                      whileHover={{ scale: 1.08 }}
                      whileTap={{ scale: 0.92 }}
                      title="Reset"
                    >
                      <RotateCcw size={16} />
                    </motion.button>
                  )}
                </div>

                {/* Capability chips */}
                {!preview && (
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {['CNN · 6-class', 'Autoencoder', 'U-Net Seg'].map(t => (
                      <span key={t} style={{
                        padding: '3px 10px', borderRadius: 'var(--r-full)',
                        border: '1px solid var(--sky-border)',
                        fontSize: '0.62rem', fontFamily: 'var(--font-mono)',
                        color: 'var(--text-lo)'
                      }}>{t}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </FadeUp>

        </div>
      </section>
    </PageTransition>
  );
}
