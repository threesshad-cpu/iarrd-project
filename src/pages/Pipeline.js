import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Satellite, Zap, Database, Cpu,
  Target, Layers, ChevronRight, Rocket,
  AlertCircle, CheckCircle, ArrowDown, Activity,
  Filter, Brain, Image,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { FadeUp, PageTransition } from '../components/Animations';

/* ─────────────────────────────────────────────────────────────────
   PROBLEM STATEMENT DATA
───────────────────────────────────────────────────────────────── */
const PROBLEM = {
  title: 'Problem Statement',
  body: 'Astronomical images captured by telescopes often contain noise and distortions due to atmospheric and instrumental limitations. Manual analysis of these images is inefficient. An automated system using AI and image processing techniques is required to enhance image quality and accurately identify celestial objects.',
};

const SOLUTION = {
  title: 'Solution',
  body: 'An AI-powered image processing system that enhances astronomical images and automatically identifies celestial objects with high accuracy.',
  algorithms: ['Gaussian Filter', 'Convolutional Neural Network (CNN)'],
};

/* ─────────────────────────────────────────────────────────────────
   FLOW STEPS — exact flow diagram from problem statement
───────────────────────────────────────────────────────────────── */
const FLOW = [
  {
    num: '01',
    isTerminal: false,
    color: '#38bdf8',
    Icon: Satellite,
    badge: 'INPUT',
    title: 'Astronomical Image Acquisition',
    subtitle: 'Telescope Image Intake',
    algo: 'File API — PNG / JPG / TIFF / WEBP  ≤ 10 MB',
    desc: 'A raw telescope image is ingested via the upload interface. The system validates format and file size, decodes pixel data, and extracts native resolution metadata (width × height, file size, mean luminosity). This constitutes the raw input before any processing.',
    detail: 'native resolution · file_size_kb · mean luminosity · std luminosity',
    backend: 'stage_01_acquisition(raw, filename)',
    tags: ['File Validation', 'Metadata Extraction', 'Resolution Check'],
  },
  {
    num: '02',
    color: '#38bdf8',
    Icon: Filter,
    badge: 'FILTER',
    title: 'Preprocessing',
    subtitle: 'Noise Removal & Normalization',
    algo: 'Gaussian Filter (σ = 1.2)  +  [0, 1] Min-Max Normalization',
    desc: 'A Gaussian blur kernel (σ = 1.2) is convolved over the image to suppress high-frequency sensor noise and atmospheric distortion — the primary classical algorithm stated in the problem. Pixel values are then normalized to the [0, 1] floating-point range for stable neural network input. The image is resized to 128 × 128 px for model inference.',
    detail: 'K(x,y,σ) = (1/2πσ²) · e^(−(x²+y²)/2σ²)  →  normalize [0, 1]  →  resize 128×128',
    backend: 'stage_02_preprocessing(pil_img)',
    tags: ['Gaussian Filter', 'Normalization', 'Resize 128×128', 'SNR Estimation'],
  },
  {
    num: '03',
    color: '#a78bfa',
    Icon: Image,
    badge: 'ENHANCE',
    title: 'Image Enhancement',
    subtitle: 'Contrast · Deblurring · Super-Resolution',
    algo: '8-Pass Pipeline: CLAHE + Laplacian Blend + Triple USM + Gamma + LANCZOS 4096×4096',
    desc: 'The original image (not just the 128×128 model input) is enhanced through 8 deterministic passes: per-channel CLAHE histogram stretching, high-frequency detail layer boost via Laplacian blending, triple unsharp masking at fine/mid/large scales, gamma lift γ=0.80, saturation/contrast/brightness tuning, edge-preserving denoising, and dual deconvolution sharpening. The result is upscaled to 4096 × 4096 via LANCZOS for ultra-HD display.',
    detail: 'CLAHE → detail boost → USM(r=1.2/4/10) → γ lift → colour +35% → denoise → LANCZOS 4096×4096',
    backend: 'stage_03_enhancement(raw_bytes, native_w, native_h)',
    tags: ['CLAHE', 'Unsharp Masking', 'Gamma Correction', 'LANCZOS 4×SR', '16 MP Output'],
  },
  {
    num: '04',
    color: '#34d399',
    Icon: Database,
    badge: 'EXTRACT',
    title: 'Feature Extraction',
    subtitle: 'Brightness · Shape · Size',
    algo: 'Luminosity Statistics  +  Gradient Energy  +  Threshold Blob Detection',
    desc: 'Radiometric and morphological features are extracted from the 128×128 normalized array. Brightness: mean, peak, dynamic range, standard deviation. Shape: horizontal and vertical gradient energy (Sobel-style), edge complexity. Size: adaptive threshold (mean + 1.5σ) to isolate bright structures, pixel count, fill ratio, estimated object count. Color: per-channel means and dominant spectral classification.',
    detail: 'gray = 0.299R+0.587G+0.114B  →  ∇I  →  thresh = μ+1.5σ  →  {brightness, shape, size, color}',
    backend: 'stage_04_feature_extraction(arr)',
    tags: ['Brightness Analysis', 'Edge Energy', 'Blob Count', 'Spectral Hint'],
  },
  {
    num: '05',
    color: '#818cf8',
    Icon: Brain,
    badge: 'CNN',
    title: 'AI Model Processing (CNN)',
    subtitle: 'Deep Neural Classification',
    algo: 'Convolutional Neural Network — ResNet-style backbone — 6-class Softmax',
    desc: 'The preprocessed 128×128×3 float32 array is passed to a fine-tuned Convolutional Neural Network — the primary AI algorithm from the problem statement. The ResNet-style backbone extracts hierarchical spatial features through stacked convolutional blocks, and a fully-connected softmax head outputs a 6-class probability distribution. Predictions below 30% confidence are automatically remapped to "Unknown Object" to prevent false positives.',
    detail: 'Input[128×128×3] → Conv blocks → GlobalAvgPool → FC(256) → Dropout → Softmax[6]',
    backend: 'stage_05_cnn(arr)',
    tags: ['CNN', 'Softmax[6]', 'Confidence Threshold 30%', 'Unknown Fallback'],
  },
  {
    num: '06',
    color: '#f59e0b',
    Icon: Target,
    badge: 'DETECT',
    title: 'Celestial Object Detection & Classification',
    subtitle: '6-Class Taxonomy + Anomaly Scoring',
    algo: 'Convolutional Autoencoder → MSE Reconstruction Error → Anomaly Score [0–100]',
    desc: 'A convolutional autoencoder compresses the image to a latent vector and reconstructs it. The mean-squared error between input and reconstruction quantifies how "anomalous" the object is relative to the training set — objects unlike any known class produce high reconstruction error. Results are fused with the CNN classification into a structured telemetry packet. Quadrant-level hotspot analysis identifies the most anomalous spatial region.',
    detail: 'Encoder→latent z→Decoder → MSE = mean((x−x̂)²) → anomaly = min(MSE×1000, 100)',
    backend: 'stage_06_detection(arr)',
    tags: ['Autoencoder', 'Anomaly Score', 'Quadrant Analysis', '6-Class Output'],
  },
  {
    num: '07',
    color: '#10b981',
    Icon: Layers,
    badge: 'SEGMENT',
    title: 'Enhanced Image + Object Labels',
    subtitle: 'U-Net Pixel Segmentation + Labeled Composite',
    algo: 'U-Net Encoder–Decoder + Skip Connections → 6-class mask → 4K Composite PNG',
    desc: 'A U-Net with encoder–decoder architecture and skip connections maps every pixel to one of 6 semantic classes, producing a full-resolution RGBA color overlay. Coverage percentages are computed per class. The enhanced 4096×4096 image is then composited with the segmentation overlay, CNN classification banner, anomaly badge, and coverage legend — producing the final labeled analysis image.',
    detail: 'Input→Encoder(4 blocks)→Bridge→Decoder(4 blocks)+skip→6-class mask → composite PNG',
    backend: 'stage_07_segmentation(arr)  +  stage_08_labeled_output(...)',
    tags: ['U-Net', '6-class Mask', 'Overlay', '4K Composite', 'Downloadable PNG'],
  },
];

/* ─────────────────────────────────────────────────────────────────
   ALGORITHM COMPARISON TABLE
───────────────────────────────────────────────────────────────── */
const ALGOS = [
  { name: 'Gaussian Filter',            type: 'Classical DSP',  use: 'Noise removal — primary preprocessing algorithm', complexity: 'O(n·k²)',  color: '#38bdf8', primary: true },
  { name: 'CNN Classifier',             type: 'Deep Learning',  use: 'Primary AI algorithm — 6-class classification',   complexity: '~487 ms',  color: '#818cf8', primary: true },
  { name: 'Convolutional Autoencoder',  type: 'Deep Learning',  use: 'Image enhancement + anomaly detection',           complexity: '~90 ms',   color: '#a78bfa', primary: false },
  { name: 'U-Net Segmentor',            type: 'Deep Learning',  use: 'Pixel-level semantic labeling (6 classes)',        complexity: '~120 ms',  color: '#34d399', primary: false },
  { name: 'CLAHE + USM',               type: 'Classical CV',   use: 'Contrast enhancement + super-resolution output',  complexity: 'O(n)',     color: '#f59e0b', primary: false },
];

/* ─────────────────────────────────────────────────────────────────
   STEP CARD COMPONENT
───────────────────────────────────────────────────────────────── */
function StepCard({ step, index, active, onClick, isLast }) {
  const { num, color, Icon, title, subtitle, algo, desc, detail, backend, badge, tags } = step;

  return (
    <div style={{ display: 'flex', gap: '1.2rem', alignItems: 'flex-start' }}>
      {/* Timeline spine */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0, paddingTop: 4 }}>
        <motion.div
          onClick={onClick}
          whileHover={{ scale: 1.12 }}
          whileTap={{ scale: 0.93 }}
          style={{
            width: 50, height: 50, borderRadius: '50%', cursor: 'pointer',
            background: active ? `${color}22` : 'rgba(255,255,255,0.04)',
            border: `2px solid ${active ? color : 'rgba(255,255,255,0.09)'}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: active ? color : 'rgba(255,255,255,0.25)',
            boxShadow: active ? `0 0 24px ${color}44` : 'none',
            transition: 'all 0.3s ease',
          }}
        >
          <Icon size={20} strokeWidth={1.4} />
        </motion.div>
        {!isLast && (
          <motion.div
            initial={{ scaleY: 0 }}
            animate={{ scaleY: 1 }}
            transition={{ delay: index * 0.08 + 0.3, duration: 0.5 }}
            style={{
              width: 2, flex: 1, minHeight: 36, marginTop: 4, transformOrigin: 'top',
              background: `linear-gradient(180deg, ${color}66 0%, transparent 100%)`,
            }}
          />
        )}
      </div>

      {/* Card */}
      <div style={{ flex: 1, paddingBottom: isLast ? 0 : '1.6rem' }}>
        <motion.div
          onClick={onClick}
          initial={{ opacity: 0, x: -18 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.07, ease: [0.16, 1, 0.3, 1] }}
          style={{
            padding: '1.1rem 1.3rem', borderRadius: 14, cursor: 'pointer',
            background: active ? `${color}0a` : 'rgba(255,255,255,0.025)',
            border: `1px solid ${active ? color + '44' : 'rgba(255,255,255,0.07)'}`,
            transition: 'all 0.3s ease',
          }}
        >
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, marginBottom: 8 }}>
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 5 }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.58rem', color, padding: '2px 7px', borderRadius: 4, background: `${color}18`, border: `1px solid ${color}44`, letterSpacing: '0.08em' }}>{badge}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.56rem', color: 'var(--text-lo)' }}>STEP {num}</span>
              </div>
              <h3 style={{ fontSize: '0.97rem', color: '#e0f4ff', fontWeight: 700, margin: 0, lineHeight: 1.3 }}>{title}</h3>
              <p style={{ fontSize: '0.7rem', color, margin: '3px 0 0', fontWeight: 600 }}>{subtitle}</p>
            </div>
            <motion.div animate={{ rotate: active ? 90 : 0 }} transition={{ duration: 0.22 }}>
              <ChevronRight size={16} style={{ color: active ? color : 'var(--text-lo)', flexShrink: 0, marginTop: 6 }} />
            </motion.div>
          </div>

          {/* Algorithm badge */}
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.61rem', color: 'var(--text-lo)', padding: '4px 10px', borderRadius: 6, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', display: 'inline-block', lineHeight: 1.5 }}>
            {algo}
          </div>

          {/* Expanded detail */}
          <AnimatePresence>
            {active && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
                style={{ overflow: 'hidden' }}
              >
                <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                  <p style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.8)', lineHeight: 1.7, margin: 0 }}>{desc}</p>

                  {detail && (
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.66rem', color, padding: '8px 14px', borderRadius: 8, background: `${color}0a`, border: `1px solid ${color}33`, lineHeight: 1.7 }}>
                      {detail}
                    </div>
                  )}

                  {/* Tags */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                    {tags.map(t => (
                      <span key={t} style={{ fontSize: '0.58rem', fontFamily: 'var(--font-mono)', color, padding: '2px 8px', borderRadius: 999, border: `1px solid ${color}44`, background: `${color}0e` }}>
                        {t}
                      </span>
                    ))}
                  </div>

                  {/* Backend function */}
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: 7 }}>
                    <Cpu size={11} style={{ color: '#10b981', flexShrink: 0, marginTop: 2 }} />
                    <span style={{ fontSize: '0.64rem', color: 'var(--text-lo)' }}>
                      <strong style={{ color: '#10b981' }}>Backend:</strong>&nbsp;
                      <code style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>{backend}</code>
                    </span>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   TERMINAL NODE (START / END)
───────────────────────────────────────────────────────────────── */
function TerminalNode({ label, isEnd = false }) {
  const color = isEnd ? '#10b981' : '#38bdf8';
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ ease: [0.16, 1, 0.3, 1] }}
      style={{
        display: 'flex', alignItems: 'center', gap: '1.2rem', paddingLeft: 0,
      }}
    >
      <div style={{
        width: 50, height: 50, borderRadius: '50%', flexShrink: 0,
        background: `${color}18`, border: `2px solid ${color}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        boxShadow: `0 0 20px ${color}44`,
      }}>
        {isEnd
          ? <CheckCircle size={20} style={{ color }} strokeWidth={1.5} />
          : <AlertCircle  size={20} style={{ color }} strokeWidth={1.5} />
        }
      </div>
      <div style={{
        padding: '8px 22px', borderRadius: 999,
        background: `${color}12`, border: `1.5px solid ${color}55`,
        fontFamily: 'var(--font-head)', fontSize: '0.75rem', letterSpacing: '0.22em',
        color, fontWeight: 700,
      }}>
        {label}
      </div>
    </motion.div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   ARROW CONNECTOR
───────────────────────────────────────────────────────────────── */
function FlowArrow({ color = 'rgba(255,255,255,0.15)' }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '1.2rem', paddingLeft: 0 }}>
      <div style={{ width: 50, display: 'flex', justifyContent: 'center' }}>
        <ArrowDown size={16} style={{ color }} strokeWidth={1.5} />
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────
   MAIN PAGE
───────────────────────────────────────────────────────────────── */
export default function Pipeline() {
  const [activeStep, setActiveStep] = useState(0);
  const toggle = (i) => setActiveStep(prev => (prev === i ? -1 : i));

  return (
    <PageTransition>
      <section className="scene" style={{ alignItems: 'flex-start', paddingTop: '6.5rem', paddingBottom: '6rem' }}>
        <div className="page-inner">

          {/* ── Page header ── */}
          <FadeUp>
            <div className="eyebrow">
              <Layers size={13} strokeWidth={1.5} />
              SCENE 04 — SYSTEM ARCHITECTURE
            </div>
            <h1 className="section-title">Analysis Pipeline</h1>
            <p className="section-desc" style={{ maxWidth: 640 }}>
              From raw telescope pixels to structured astronomical telemetry — an AI-powered
              8-stage pipeline using <strong style={{ color: '#38bdf8' }}>Gaussian filtering</strong> and
              &nbsp;<strong style={{ color: '#818cf8' }}>Convolutional Neural Networks</strong>.
            </p>
          </FadeUp>

          {/* ── Problem Statement + Solution cards ── */}
          <FadeUp delay={0.06}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', margin: '1.5rem 0 2rem' }}>
              {/* Problem */}
              <div style={{ padding: '1.3rem 1.5rem', borderRadius: 16, background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.25)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '0.8rem' }}>
                  <AlertCircle size={15} style={{ color: '#ef4444' }} strokeWidth={1.5} />
                  <span style={{ fontFamily: 'var(--font-head)', fontSize: '0.6rem', letterSpacing: '0.18em', color: '#ef4444' }}>PROBLEM STATEMENT</span>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'rgba(255,255,255,0.75)', lineHeight: 1.7, margin: 0 }}>
                  {PROBLEM.body}
                </p>
              </div>

              {/* Solution */}
              <div style={{ padding: '1.3rem 1.5rem', borderRadius: 16, background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.25)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '0.8rem' }}>
                  <CheckCircle size={15} style={{ color: '#10b981' }} strokeWidth={1.5} />
                  <span style={{ fontFamily: 'var(--font-head)', fontSize: '0.6rem', letterSpacing: '0.18em', color: '#10b981' }}>SOLUTION</span>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'rgba(255,255,255,0.75)', lineHeight: 1.7, margin: '0 0 0.8rem' }}>
                  {SOLUTION.body}
                </p>
                <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
                  {SOLUTION.algorithms.map((a, i) => {
                    const c = i === 0 ? '#38bdf8' : '#818cf8';
                    return (
                      <span key={a} style={{ fontSize: '0.62rem', fontFamily: 'var(--font-mono)', color: c, padding: '3px 10px', borderRadius: 999, border: `1px solid ${c}55`, background: `${c}12` }}>
                        ✓ {a}
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>
          </FadeUp>

          {/* ── Quick stats ── */}
          <FadeUp delay={0.08}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '0.7rem', marginBottom: '2.5rem' }}>
              {[
                { icon: <Brain size={14} />,      label: 'AI Models',       value: '3',        color: '#818cf8' },
                { icon: <Target size={14} />,      label: 'Object Classes',  value: '6',        color: '#a78bfa' },
                { icon: <Zap size={14} />,         label: 'Avg Inference',   value: '~500 ms',  color: '#34d399' },
                { icon: <Activity size={14} />,    label: 'Pipeline Stages', value: '8',        color: '#f59e0b' },
                { icon: <Filter size={14} />,      label: 'Main Algorithms', value: '2',        color: '#38bdf8' },
                { icon: <Image size={14} />,       label: 'Output (8K)',     value: '4096×4096',color: '#10b981' },
              ].map(({ icon, label, value, color }) => (
                <div key={label} className="glass stat-cell" style={{ borderColor: `${color}22` }}>
                  <div className="stat-label" style={{ display: 'flex', alignItems: 'center', gap: 5, color }}>{icon} {label}</div>
                  <div className="stat-value" style={{ color, fontSize: value.length > 6 ? '0.85rem' : undefined }}>{value}</div>
                </div>
              ))}
            </div>
          </FadeUp>

          {/* ── Two-column: flow + sidebar ── */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '2rem', alignItems: 'flex-start' }}>

            {/* Left: animated flow diagram */}
            <FadeUp delay={0.1}>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.58rem', letterSpacing: '0.18em', color: 'var(--text-lo)', marginBottom: '1.4rem', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Activity size={11} />  CLICK ANY STEP TO EXPAND DETAILS
                </div>

                {/* START terminal */}
                <TerminalNode label="START" />
                <FlowArrow color="#38bdf844" />

                {/* Flow steps */}
                {FLOW.map((step, i) => (
                  <React.Fragment key={step.num}>
                    <StepCard
                      step={step}
                      index={i}
                      active={activeStep === i}
                      onClick={() => toggle(i)}
                      isLast={false}
                    />
                    {i < FLOW.length - 1 && <FlowArrow />}
                  </React.Fragment>
                ))}

                {/* END terminal */}
                <FlowArrow color="#10b98144" />
                <TerminalNode label="END" isEnd />
              </div>
            </FadeUp>

            {/* Right: sticky sidebar */}
            <FadeUp delay={0.14}>
              <div style={{ position: 'sticky', top: '7rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>

                {/* Active step detail */}
                <div className="glass" style={{ padding: '1.3rem', borderRadius: 16, borderColor: activeStep >= 0 ? `${FLOW[activeStep]?.color ?? '#38bdf8'}33` : undefined }}>
                  <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.56rem', letterSpacing: '0.18em', color: 'var(--text-lo)', marginBottom: '0.8rem' }}>SELECTED STEP</div>
                  {activeStep >= 0 ? (
                    <motion.div key={activeStep} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 10 }}>
                        <div style={{ width: 34, height: 34, borderRadius: '50%', background: `${FLOW[activeStep].color}18`, border: `1.5px solid ${FLOW[activeStep].color}`, display: 'flex', alignItems: 'center', justifyContent: 'center', color: FLOW[activeStep].color }}>
                          {React.createElement(FLOW[activeStep].Icon, { size: 15, strokeWidth: 1.5 })}
                        </div>
                        <div>
                          <div style={{ fontSize: '0.85rem', color: '#e0f4ff', fontWeight: 700, lineHeight: 1.3 }}>{FLOW[activeStep].title}</div>
                          <div style={{ fontSize: '0.65rem', color: FLOW[activeStep].color, marginTop: 2 }}>{FLOW[activeStep].subtitle}</div>
                        </div>
                      </div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.62rem', color: FLOW[activeStep].color, lineHeight: 1.6, padding: '8px 10px', background: `${FLOW[activeStep].color}0a`, borderRadius: 8, border: `1px solid ${FLOW[activeStep].color}33` }}>
                        {FLOW[activeStep].algo}
                      </div>
                    </motion.div>
                  ) : (
                    <p style={{ color: 'var(--text-lo)', fontSize: '0.73rem' }}>Select a step to view details →</p>
                  )}
                </div>

                {/* Algorithm table */}
                <div className="glass" style={{ padding: '1.2rem', borderRadius: 16 }}>
                  <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.56rem', letterSpacing: '0.18em', color: 'var(--text-lo)', marginBottom: '0.8rem' }}>
                    ALGORITHMS USED
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {ALGOS.map(({ name, type, use, complexity, color, primary }) => (
                      <div key={name} style={{ padding: '8px 10px', borderRadius: 9, background: `${color}08`, border: `1px solid ${primary ? color + '44' : color + '1a'}`, position: 'relative', overflow: 'hidden' }}>
                        {primary && (
                          <span style={{ position: 'absolute', top: 6, right: 6, fontSize: '0.48rem', fontFamily: 'var(--font-mono)', color, letterSpacing: '0.08em', background: `${color}18`, padding: '1px 5px', borderRadius: 3 }}>
                            PRIMARY
                          </span>
                        )}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 6, marginBottom: 3 }}>
                          <div style={{ fontSize: '0.72rem', color: '#dbeafe', fontWeight: 600, paddingRight: primary ? 52 : 0 }}>{name}</div>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.55rem', color, padding: '1px 5px', borderRadius: 3, background: `${color}18`, flexShrink: 0 }}>{complexity}</span>
                        </div>
                        <div style={{ fontSize: '0.6rem', color: 'var(--text-lo)' }}>{type} &middot; {use}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* CTA */}
                <Link to="/upload" className="btn-primary" style={{ justifyContent: 'center', padding: '12px' }}>
                  <Rocket size={14} /> Try the Pipeline
                </Link>
              </div>
            </FadeUp>
          </div>

          {/* ── Horizontal flow summary bar ── */}
          <FadeUp delay={0.18}>
            <div style={{ marginTop: '3.5rem', padding: '1.3rem 1.5rem', borderRadius: 16, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ fontFamily: 'var(--font-head)', fontSize: '0.56rem', letterSpacing: '0.18em', color: 'var(--text-lo)', marginBottom: '1rem' }}>
                COMPLETE FLOW DIAGRAM
              </div>

              {/* Terminal START */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 0, overflowX: 'auto', paddingBottom: 6 }}>
                <div style={{ flexShrink: 0, padding: '6px 14px', borderRadius: 999, background: 'rgba(56,189,248,0.12)', border: '1.5px solid rgba(56,189,248,0.4)', fontFamily: 'var(--font-head)', fontSize: '0.6rem', color: '#38bdf8', letterSpacing: '0.18em' }}>
                  START
                </div>
                <div style={{ width: 12, height: 1.5, background: 'rgba(56,189,248,0.3)', flexShrink: 0 }} />

                {FLOW.map((step, i) => (
                  <React.Fragment key={step.num}>
                    <motion.div
                      whileHover={{ scale: 1.06 }}
                      onClick={() => { toggle(i); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
                      style={{
                        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
                        padding: '7px 10px', borderRadius: 9, cursor: 'pointer', flexShrink: 0,
                        background: activeStep === i ? `${step.color}18` : 'transparent',
                        border: `1px solid ${activeStep === i ? step.color + '55' : 'transparent'}`,
                        transition: 'all 0.2s',
                      }}
                    >
                      <div style={{ width: 26, height: 26, borderRadius: '50%', background: `${step.color}18`, border: `1.5px solid ${step.color}55`, display: 'flex', alignItems: 'center', justifyContent: 'center', color: step.color }}>
                        {React.createElement(step.Icon, { size: 11, strokeWidth: 1.5 })}
                      </div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.48rem', color: step.color }}>{step.num}</div>
                      <div style={{ fontSize: '0.5rem', color: 'rgba(255,255,255,0.55)', whiteSpace: 'nowrap', maxWidth: 64, textAlign: 'center', lineHeight: 1.3 }}>{step.title}</div>
                    </motion.div>
                    {i < FLOW.length - 1 && (
                      <div style={{ width: 14, height: 1.5, background: 'rgba(255,255,255,0.1)', flexShrink: 0 }} />
                    )}
                  </React.Fragment>
                ))}

                <div style={{ width: 12, height: 1.5, background: 'rgba(16,185,129,0.3)', flexShrink: 0 }} />
                <div style={{ flexShrink: 0, padding: '6px 14px', borderRadius: 999, background: 'rgba(16,185,129,0.12)', border: '1.5px solid rgba(16,185,129,0.4)', fontFamily: 'var(--font-head)', fontSize: '0.6rem', color: '#10b981', letterSpacing: '0.18em' }}>
                  END
                </div>
              </div>
            </div>
          </FadeUp>

        </div>
      </section>
    </PageTransition>
  );
}
