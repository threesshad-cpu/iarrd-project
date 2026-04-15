import React from 'react';
import { Link } from 'react-router-dom';
import { Orbit } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="site-footer">
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <Orbit size={16} strokeWidth={1.5} style={{ color: 'var(--cyan)' }} />
        <span>IARRD &copy; {new Date().getFullYear()}</span>
      </div>
      <nav className="footer-links">
        <Link to="/about">About</Link>
        <Link to="/services">Services</Link>
        <Link to="/pipeline">Pipeline</Link>
        <Link to="/contact">Contact</Link>
      </nav>
    </footer>
  );
}
