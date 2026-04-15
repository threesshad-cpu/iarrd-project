import React from 'react';
import { Users, Code, Telescope } from 'lucide-react';
import { FadeUp, PageTransition } from '../components/Animations';

const TEAM = [
  { icon: Telescope, name: 'Research Division',  role: 'Photometric analysis, detection algorithm design' },
  { icon: Code,      name: 'Engineering',         role: 'Browser pipeline, React Three Fiber integration' },
  { icon: Users,     name: 'Data Science',        role: 'CNN model training, feature validation, benchmarking' },
];

export default function Team() {
  return (
    <PageTransition>
      <section className="scene" style={{ alignItems: 'flex-start', paddingTop: '7rem' }}>
        <div className="page-inner">
          <FadeUp>
            <div className="section-eyebrow"><Users size={14} /> THE CREW</div>
            <h1 className="section-title">Mission Team</h1>
          </FadeUp>
          <div className="info-grid mt-4">
            {TEAM.map(({ icon: Icon, name, role }, i) => (
              <FadeUp key={name} delay={i * 0.08}>
                <div className="info-card glass" style={{ alignItems: 'center', textAlign: 'center' }}>
                  <div style={{ width: 64, height: 64, borderRadius: '50%', background: 'var(--cyan-dim)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent:'center' }}>
                    <Icon size={28} strokeWidth={1.5} style={{ color: 'var(--cyan)' }} />
                  </div>
                  <h3 style={{ fontSize: '1rem' }}>{name}</h3>
                  <p style={{ fontSize: '0.85rem' }}>{role}</p>
                </div>
              </FadeUp>
            ))}
          </div>
        </div>
      </section>
    </PageTransition>
  );
}
