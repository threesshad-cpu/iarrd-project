import React, { useState } from 'react';
import { Send, Mail, MessageSquare } from 'lucide-react';
import { motion } from 'framer-motion';
import { FadeUp, PageTransition } from '../components/Animations';

export default function Contact() {
  const [sent, setSent] = useState(false);

  return (
    <PageTransition>
      <section className="scene" style={{ alignItems: 'flex-start', paddingTop: '7rem' }}>
        <div className="page-inner" style={{ maxWidth: 640 }}>
          <FadeUp>
            <div className="section-eyebrow"><Send size={14} /> CONTACT</div>
            <h1 className="section-title">Open Channel</h1>
            <p className="section-desc">Questions, research partnerships, or integration requests — transmit below.</p>
          </FadeUp>

          <FadeUp delay={0.1}>
            {!sent ? (
              <form
                className="glass"
                style={{ padding: '2rem', borderRadius: 20, marginTop: '2rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}
                onSubmit={e => { e.preventDefault(); setSent(true); }}
              >
                <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <span style={{ fontFamily: 'var(--font-head)', fontSize: '0.7rem', letterSpacing: '0.12em', color: 'var(--text-lo)' }}>
                    <Mail size={12} style={{ marginRight: 5 }} />
                    EMAIL ADDRESS
                  </span>
                  <input
                    type="email" required placeholder="you@observatory.space"
                    style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border)', borderRadius: 10, padding: '12px 16px', color: 'var(--text-hi)', fontFamily: 'var(--font-body)', outline: 'none', fontSize: '0.95rem' }}
                  />
                </label>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <span style={{ fontFamily: 'var(--font-head)', fontSize: '0.7rem', letterSpacing: '0.12em', color: 'var(--text-lo)' }}>
                    <MessageSquare size={12} style={{ marginRight: 5 }} />
                    MESSAGE
                  </span>
                  <textarea
                    required rows={5} placeholder="Transmission content..."
                    style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border)', borderRadius: 10, padding: '12px 16px', color: 'var(--text-hi)', fontFamily: 'var(--font-body)', outline: 'none', resize: 'vertical', fontSize: '0.95rem' }}
                  />
                </label>
                <motion.button type="submit" className="btn-primary" whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>
                  <Send size={15} /> Transmit
                </motion.button>
              </form>
            ) : (
              <div className="glass" style={{ padding: '3rem 2rem', borderRadius: 20, marginTop: '2rem', textAlign: 'center' }}>
                <div style={{ color: 'var(--cyan)', fontFamily: 'var(--font-head)', fontSize: '1rem', letterSpacing: '0.15em' }}>SIGNAL RECEIVED</div>
                <p style={{ color: 'var(--text-lo)', marginTop: '1rem' }}>We will respond on a secure channel shortly.</p>
              </div>
            )}
          </FadeUp>
        </div>
      </section>
    </PageTransition>
  );
}
