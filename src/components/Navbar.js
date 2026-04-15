import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Globe, Layers, ImagePlus, Database,
  Info, Menu, X, Sun, Moon
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const LINKS = [
  { to: '/',         label: 'Home',     Icon: Globe     },
  { to: '/pipeline', label: 'Pipeline', Icon: Layers    },
  { to: '/upload',   label: 'Uplink',   Icon: ImagePlus },
  { to: '/results',  label: 'Databank', Icon: Database  },
  { to: '/about',    label: 'About',    Icon: Info      },
];

export default function Navbar() {
  const loc  = useLocation();
  const [open,  setOpen]  = useState(false);
  const [light, setLight] = useState(
    () => localStorage.getItem('iarrd_theme') === 'light'
  );

  useEffect(() => setOpen(false), [loc]);

  useEffect(() => {
    if (light) {
      document.body.classList.add('light-mode');
      localStorage.setItem('iarrd_theme', 'light');
    } else {
      document.body.classList.remove('light-mode');
      localStorage.setItem('iarrd_theme', 'dark');
    }
  }, [light]);

  return (
    <motion.nav
      className="glass-nav"
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.75, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Logo */}
      <Link to="/" className="nav-brand" aria-label="IARRD Home">
        <img src="/logo.svg" alt="IARRD" />
      </Link>

      {/* Desktop links */}
      <div className="nav-links">
        {LINKS.map(({ to, label, Icon }) => (
          <Link
            key={to}
            to={to}
            className={`nav-item ${loc.pathname === to ? 'active' : ''}`}
          >
            <Icon size={12} strokeWidth={1.8} />
            {label}
          </Link>
        ))}
      </div>

      {/* Right controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginLeft: 'auto', flexShrink: 0 }}>
        {/* Dark / Light toggle */}
        <motion.button
          onClick={() => setLight(l => !l)}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          title={light ? 'Switch to Dark Mode' : 'Switch to Light Mode'}
          style={{
            background: light ? 'rgba(251,191,36,0.15)' : 'rgba(56,189,248,0.1)',
            border: `1px solid ${light ? 'rgba(251,191,36,0.35)' : 'rgba(56,189,248,0.25)'}`,
            borderRadius: 8, width: 32, height: 32,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'pointer', color: light ? '#fbbf24' : '#38bdf8',
          }}
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={light ? 'sun' : 'moon'}
              initial={{ rotate: -30, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 30, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              {light ? <Sun size={14} /> : <Moon size={14} />}
            </motion.div>
          </AnimatePresence>
        </motion.button>

        {/* Mobile burger */}
        <button className="nav-mobile-toggle" onClick={() => setOpen(s => !s)} aria-label="Toggle navigation">
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {/* Mobile drawer */}
      <AnimatePresence>
        {open && (
          <motion.div
            className="nav-links open"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.28 }}
          >
            {LINKS.map(({ to, label, Icon }) => (
              <Link key={to} to={to} className={`nav-item ${loc.pathname === to ? 'active' : ''}`}>
                <Icon size={14} strokeWidth={1.8} /> {label}
              </Link>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
}
