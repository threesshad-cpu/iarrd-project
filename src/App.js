import React, { Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';

// 3D Scene
import { Scene } from './3d/Scene';

// UI Shell
import Navbar    from './components/Navbar';
import Footer    from './components/Footer';
import { CustomCursor } from './components/CustomCursor';

// Pages
import Home     from './pages/Home';
import Upload   from './pages/Upload';
import Results  from './pages/Results';
import About    from './pages/About';
import Services from './pages/Services';
import Pipeline from './pages/Pipeline';
import Team     from './pages/Team';
import Contact  from './pages/Contact';

import './App.css';

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
            <AppRoutes />
          </main>
          <Footer />
        </Router>
      </div>
    </>
  );
}
