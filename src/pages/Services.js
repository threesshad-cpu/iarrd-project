import React from 'react';
import { ScanSearch, ImagePlus, Database, Share2 } from 'lucide-react';
import { FadeUp, PageTransition } from '../components/Animations';

const SERVICES = [
  { icon: ImagePlus,  title: 'Image Preprocessing', desc: 'Gaussian blur, grayscale conversion, and adaptive normalization prepare raw telescope output for detection algorithms.' },
  { icon: ScanSearch, title: 'Object Detection',     desc: 'Statistical bright-spot clustering locates candidate stars, galaxies, and nebular regions with sub-pixel centroid accuracy.' },
  { icon: Database,   title: 'Feature Extraction',   desc: 'Each detected object yields structured metadata: centroid XY, area, photometric brightness, and classification label.' },
  { icon: Share2,     title: 'Data Export',           desc: 'Results serialized to JSON for export or server-side ingestion into CNN classification pipelines.' },
];

export default function Services() {
  return (
    <PageTransition>
      <section className="scene" style={{ alignItems: 'flex-start', paddingTop: '7rem' }}>
        <div className="page-inner">
          <FadeUp>
            <div className="section-eyebrow"><ScanSearch size={14} /> CAPABILITIES</div>
            <h1 className="section-title">Services</h1>
            <p className="section-desc">What IARRD does for you — every step of the pipeline.</p>
          </FadeUp>
          <div className="info-grid mt-4">
            {SERVICES.map(({ icon: Icon, title, desc }, i) => (
              <FadeUp key={title} delay={i * 0.08}>
                <div className="info-card glass">
                  <Icon size={26} strokeWidth={1.5} style={{ color: 'var(--cyan)' }} />
                  <h3 style={{ fontSize: '1rem' }}>{title}</h3>
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
