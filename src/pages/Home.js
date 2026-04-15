import React from 'react';
import { Link } from 'react-router-dom';
import {
  Rocket, ChevronRight, Cpu, Activity,
  Layers, Satellite, Telescope, Scan, Orbit, ChevronDown
} from 'lucide-react';
import { motion } from 'framer-motion';
import { FadeUp, PageTransition } from '../components/Animations';

/* ── Animated counter ────────────────────────────────────────── */
function Counter({ to, suffix = '', duration = 2 }) {
  const [val, setVal] = React.useState(0);
  const hasRun = React.useRef(false);
  const ref = React.useRef(null);

  React.useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !hasRun.current) {
        hasRun.current = true;
        const start = performance.now();
        const tick = (now) => {
          const t = Math.min((now - start) / (duration * 1000), 1);
          const ease = 1 - Math.pow(1 - t, 4);
          setVal(Math.round(ease * to));
          if (t < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }
    }, { threshold: 0.4 });
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [to, duration]);

  return <span ref={ref}>{val.toLocaleString()}{suffix}</span>;
}

/* ── Pipeline steps data ─────────────────────────────────────── */
const PIPELINE = [
  { Icon: Satellite, name: 'INGEST',          desc: 'Upload & acquire' },
  { Icon: Scan,      name: 'GAUSSIAN FILTER', desc: 'σ=1.5 noise removal' },
  { Icon: Activity,  name: 'ENHANCE',         desc: 'Contrast · deblur' },
  { Icon: Cpu,       name: 'EXTRACT',         desc: 'Brightness · shape · size' },
  { Icon: Layers,    name: 'CLASSIFY',        desc: 'CNN + detection' },
];

/* ── Capability cards ────────────────────────────────────────── */
const FEATURES = [
  {
    Icon: Cpu,
    color: '#38bdf8',
    title: 'Neural Classification',
    desc: 'A fine-tuned CNN classifies each image into 6 astronomical categories with per-class probability scores.',
  },
  {
    Icon: Activity,
    color: '#a78bfa',
    title: 'Anomaly Detection',
    desc: 'Deep autoencoder reconstructs the image and flags pixel-level divergence as an anomaly score from 0–100.',
  },
  {
    Icon: Layers,
    color: '#34d399',
    title: 'Semantic Segmentation',
    desc: 'U-Net architecture maps every pixel to one of 6 semantic classes, rendered as a live color overlay.',
  },
  {
    Icon: Telescope,
    color: '#f59e0b',
    title: 'Browser Fallback',
    desc: 'Gaussian blur convolving with flood-fill blob detection runs entirely client-side — zero backend needed.',
  },
];

export default function Home() {
  return (
    <PageTransition>
      {/* ── HERO ──────────────────────────────────────────── */}
      <section className="scene scene--hero">
        <div className="hero-inner">

          <FadeUp delay={0.05}>
            <div className="hero-pill">
              <span className="hero-pill-dot" />
              CELESTIAL IMAGING PIPELINE — IARRD V9
            </div>
          </FadeUp>

          <FadeUp delay={0.15}>
            <h1 className="hero-title">
              ASTRONOMICAL<br />TELEMETRY<br />SYSTEM
            </h1>
          </FadeUp>

          <FadeUp delay={0.25}>
            <p className="hero-sub">
              Decoding the cosmos with neural intelligence.
              Upload any astronomical image — CNN, Autoencoder, and U-Net
              analyze it in under 600&nbsp;ms.
            </p>
          </FadeUp>

          <FadeUp delay={0.35}>
            <div className="hero-actions">
              <Link to="/upload" className="btn-primary">
                <Rocket size={15} strokeWidth={2} />
                Launch Uplink
              </Link>
              <Link to="/pipeline" className="btn-ghost">
                View Pipeline
                <ChevronRight size={15} />
              </Link>
            </div>
          </FadeUp>

          {/* Stats */}
          <FadeUp delay={0.45}>
            <div className="hero-stats">
              {[
                { to: 3,    suffix: '',    label: 'NEURAL MODELS' },
                { to: 6,    suffix: '',    label: 'OBJECT CLASSES' },
                { to: 128,  suffix: 'px', label: 'INPUT RESOLUTION' },
                { to: 600,  suffix: 'ms', label: 'AVG INFERENCE' },
              ].map(({ to, suffix, label }) => (
                <div key={label} className="hero-stat">
                  <div className="hero-stat-value">
                    <Counter to={to} suffix={suffix} />
                  </div>
                  <div className="hero-stat-label">{label}</div>
                </div>
              ))}
            </div>
          </FadeUp>

          {/* Scroll hint */}
          <FadeUp delay={0.6}>
            <motion.div
              style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, marginTop: '3rem', color: 'var(--text-lo)' }}
              animate={{ y: [0, 6, 0] }}
              transition={{ repeat: Infinity, duration: 2.2, ease: 'easeInOut' }}
            >
              <span style={{ fontFamily: 'var(--font-head)', fontSize: '0.56rem', letterSpacing: '0.22em' }}>SCROLL</span>
              <ChevronDown size={14} />
            </motion.div>
          </FadeUp>
        </div>
      </section>

      {/* ── PIPELINE STRIP ──────────────────────────────────── */}
      <section className="scene" style={{ paddingTop: 0, paddingBottom: '3rem', minHeight: 'auto' }}>
        <div className="page-inner">
          <FadeUp>
            <div className="eyebrow">
              <Orbit size={13} strokeWidth={1.5} />
              Analysis Pipeline
            </div>
            <h2 className="section-title">Five Stages. One Decision.</h2>
          </FadeUp>

          <FadeUp delay={0.1}>
            <div className="hero-pipeline">
              {PIPELINE.map(({ Icon, name, desc }, i) => (
                <React.Fragment key={name}>
                  <motion.div
                    className="pipeline-step"
                    whileHover={{ scale: 1.04 }}
                    transition={{ duration: 0.2 }}
                  >
                    <div className="pipeline-step-icon">
                      <Icon size={18} strokeWidth={1.4} />
                    </div>
                    <div className="pipeline-step-name">{name}</div>
                    <div className="pipeline-step-desc">{desc}</div>
                  </motion.div>
                  {i < PIPELINE.length - 1 && <div className="pipeline-arrow" />}
                </React.Fragment>
              ))}
            </div>
          </FadeUp>
        </div>
      </section>

      {/* ── HOW IT WORKS ─────────────────────────────────────── */}
      <section className="scene" style={{ paddingTop: '2rem', paddingBottom: '2rem', minHeight: 'auto' }}>
        <div className="page-inner">
          <FadeUp>
            <div className="eyebrow">
              <Telescope size={13} strokeWidth={1.5} />
              How It Works
            </div>
            <h2 className="section-title">From Pixels to Telemetry</h2>
            <p className="section-desc" style={{ marginTop: '0.4rem' }}>
              Each step of the problem statement flow maps directly to a working UI screen and algorithm.
            </p>
          </FadeUp>

          <FadeUp delay={0.1}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '0.85rem', marginTop: '1.8rem' }}>
              {[
                { step: '01', label: 'Image Acquisition',         screen: 'Upload Page',   color: '#38bdf8', desc: 'Drag-and-drop any telescope image up to 10 MB — PNG, JPG, TIFF, WEBP.' },
                { step: '02', label: 'Preprocessing',             screen: 'Auto (FastAPI / Browser)', color: '#38bdf8', desc: 'Gaussian noise removal + 0–1 pixel normalization before inference.' },
                { step: '03', label: 'Image Enhancement',         screen: 'Results — Autoencoder Panel', color: '#a78bfa', desc: 'Autoencoder reconstructs a clean 128×128 enhanced output.' },
                { step: '04', label: 'Feature Extraction',        screen: 'Results — Browser Mode targets', color: '#34d399', desc: 'Brightness, shape and centroid extracted per detected blob.' },
                { step: '05', label: 'CNN Model Processing',      screen: 'Backend FastAPI', color: '#38bdf8', desc: 'CNN classifies object type across 6 classes in ~500 ms.' },
                { step: '06', label: 'Object Detection & Classification', screen: 'Results — Classification Panel', color: '#f59e0b', desc: 'Galaxy, Nebula, Quasar, Star Cluster, Supernova Remnant, Unknown.' },
                { step: '07', label: 'Enhanced Image + Labels',   screen: 'Results — Segmentation Panel', color: '#34d399', desc: 'U-Net colorizes every pixel by semantic class with live overlay.' },
              ].map(({ step, label, screen, color, desc }, i) => (
                <motion.div
                  key={step}
                  className="glass"
                  initial={{ opacity: 0, y: 16 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.06, ease: [0.16, 1, 0.3, 1] }}
                  whileHover={{ y: -4, boxShadow: `0 0 30px ${color}18` }}
                  style={{ padding: '1.1rem 1.2rem', borderRadius: 14, borderColor: `${color}22` }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <div style={{
                      width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
                      background: `${color}18`, border: `1.5px solid ${color}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontFamily: 'var(--font-mono)', fontSize: '0.58rem', color, fontWeight: 700,
                    }}>{step}</div>
                    <div style={{ fontSize: '0.8rem', color: '#e0f4ff', fontWeight: 700 }}>{label}</div>
                  </div>
                  <p style={{ fontSize: '0.74rem', color: 'rgba(255,255,255,0.65)', lineHeight: 1.55, margin: '0 0 8px' }}>{desc}</p>
                  <div style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color, padding: '3px 8px', borderRadius: 5, background: `${color}12`, border: `1px solid ${color}30`, display: 'inline-block' }}>
                    {screen}
                  </div>
                </motion.div>
              ))}
            </div>
          </FadeUp>

          <FadeUp delay={0.2}>
            <div style={{ display: 'flex', justifyContent: 'center', marginTop: '1.5rem' }}>
              <Link to="/pipeline" className="btn-ghost" style={{ fontSize: '0.75rem' }}>
                View Full Pipeline Architecture
                <ChevronRight size={14} />
              </Link>
            </div>
          </FadeUp>
        </div>
      </section>

      {/* ── FEATURES ────────────────────────────────────────── */}
      <section className="scene" style={{ paddingTop: '2rem', paddingBottom: '8rem' }}>
        <div className="page-inner">
          <FadeUp>
            <div className="eyebrow">
              <Cpu size={13} strokeWidth={1.5} />
              Core Capabilities
            </div>
            <h2 className="section-title">Built for Deep Space</h2>
            <p className="section-desc" style={{ marginTop: '0.5rem' }}>
              Three production Keras models working in concert.
              Every computation runs in your local Python runtime.
            </p>
          </FadeUp>

          <div className="cards-grid" style={{ marginTop: '2rem' }}>
            {FEATURES.map(({ Icon, color, title, desc }, i) => (
              <FadeUp key={title} delay={0.08 * i}>
                <motion.div
                  className="info-card glass"
                  whileHover={{ y: -6, boxShadow: `0 0 40px ${color}18, 0 20px 40px rgba(0,0,0,0.5)` }}
                  transition={{ duration: 0.3 }}
                >
                  <div style={{
                    width: 44, height: 44,
                    borderRadius: 12,
                    background: `${color}14`,
                    border: `1px solid ${color}33`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color,
                  }}>
                    <Icon size={20} strokeWidth={1.4} />
                  </div>
                  <h3 style={{ fontSize: '0.82rem' }}>{title}</h3>
                  <p style={{ fontSize: '0.85rem' }}>{desc}</p>
                </motion.div>
              </FadeUp>
            ))}
          </div>

          {/* CTA */}
          <FadeUp delay={0.2}>
            <div style={{ display: 'flex', justifyContent: 'center', marginTop: '1rem' }}>
              <Link to="/upload" className="btn-primary" style={{ padding: '15px 40px', fontSize: '0.8rem' }}>
                <Rocket size={16} strokeWidth={2} />
                Begin Mission
              </Link>
            </div>
          </FadeUp>
        </div>
      </section>
    </PageTransition>
  );
}
