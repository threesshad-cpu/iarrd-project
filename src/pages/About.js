import React from 'react';
import { Telescope, Cpu, Globe, ArrowUpRight } from 'lucide-react';
import { FadeUp, PageTransition } from '../components/Animations';

export default function About() {
  return (
    <PageTransition>
      <section className="scene" style={{ alignItems: 'flex-start', paddingTop: '7rem' }}>
        <div className="page-inner">
          <FadeUp>
            <div className="section-eyebrow"><Globe size={14} /> ABOUT THE PROJECT</div>
            <h1 className="section-title">IARRD System</h1>
            <p className="section-desc">
              Image Analysis & Rapid Recognition of Deep-space data. A research prototype for
              browser-native astronomical preprocessing, designed to feed server-side CNN models.
            </p>
          </FadeUp>
          <div className="info-grid mt-4">
            {[
              { icon: Telescope, title: 'Mission Context', desc: 'Ground-based telescopes generate terabytes of raw imagery. Manual inspection is impossible at scale — automated pipelines are essential.' },
              { icon: Cpu,       title: 'Technology',      desc: 'Built on React 19, Three.js, Framer Motion. Processing uses pure JavaScript typed arrays — zero native dependencies.' },
              { icon: Globe,     title: 'Next Steps',      desc: 'FITS format support, PyTorch CNN backend integration, multi-user cloud storage, and morphological deblending algorithms.' },
            ].map(({ icon: Icon, title, desc }) => (
              <FadeUp key={title}>
                <div className="info-card glass">
                  <Icon size={26} strokeWidth={1.5} style={{ color: 'var(--cyan)' }} />
                  <h3 style={{ fontSize:'1rem' }}>{title}</h3>
                  <p>{desc}</p>
                </div>
              </FadeUp>
            ))}
          </div>
        </div>
      </section>
    </PageTransition>
  );
}
