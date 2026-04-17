import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import './App.css';

// 3D Scene (already suspended at mount)
import { Scene } from './3d/Scene';

// UI Shell — always loaded (tiny, needed on every route)
import Navbar    from './components/Navbar';
import Footer    from './components/Footer';
import { CustomCursor } from './components/CustomCursor';

// Pages — code-split; each chunk fetched only when first visited
const Home     = lazy(() => import('./pages/Home'));
const Upload   = lazy(() => import('./pages/Upload'));
const Results  = lazy(() => import('./pages/Results'));
const About    = lazy(() => import('./pages/About'));
const Services = lazy(() => import('./pages/Services'));
const Pipeline = lazy(() => import('./pages/Pipeline'));
const Team     = lazy(() => import('./pages/Team'));
const Contact  = lazy(() => import('./pages/Contact'));



// Minimal fallback — matches app chrome, no layout shift
function PageLoadingFallback() {
  return (
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'transparent',
    }}>
      <div style={{
        width: 28, height: 28, borderRadius: '50%',
        border: '2px solid rgba(56,189,248,0.15)',
        borderTopColor: '#38bdf8',
        animation: 'spin 0.7s linear infinite',
      }} />
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

/* AnimatePresence needs location access — extract routes */
function AppRoutes() {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/"         element={<Home     />} />
        <Route path="/upload"   element={<Upload   />} />
        <Route path="/results"  element={<Results  />} />
        <Route path="/about"    element={<About    />} />
        <Route path="/services" element={<Services />} />
        <Route path="/pipeline" element={<Pipeline />} />
        <Route path="/team"     element={<Team     />} />
        <Route path="/contact"  element={<Contact  />} />
      </Routes>
    </AnimatePresence>
  );
}

export default function App() {
  return (
    <>
      {/* Custom spring cursor */}
      <CustomCursor />

      {/* Full-screen persistent 3D environment */}
      <Suspense fallback={null}>
        <Scene />
      </Suspense>

      {/* HTML overlay */}
      <div className="overlay-root">
        <Router>
          <Navbar />
          <main>
          <Suspense fallback={<PageLoadingFallback />}>
            <AppRoutes />
          </Suspense>
          </main>
          <Footer />
        </Router>
      </div>
    </>
  );
}
