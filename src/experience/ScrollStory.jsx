import React, { useState } from 'react';
import { ScrollControls, Scroll } from '@react-three/drei';
import { Scene } from '../3d/Scene';
import { UploadScene } from './UploadScene';
import { ResultsScene } from './ResultsScene';
import { NavigationOrbit } from './NavigationOrbit';

export function ScrollStory() {
  const [file, setFile] = useState(null);

  return (
    <ScrollControls pages={3} damping={0.25} distance={2}>
      {/* 1. Global Scene Background */}
      <Scene />

      {/* 2. Scrollable 3D Objects */}
      <Scroll>
        <UploadScene onFileSelect={setFile} />
        {file && <ResultsScene file={file} />}
      </Scroll>

      {/* Fixed UI Orbit */}
      <NavigationOrbit />
      
      {/* 3. HTML Overlays attached to scroll */}
      <Scroll html style={{ width: '100vw' }}>
        <h1 style={{ position: 'absolute', top: '30vh', left: '10vw', fontSize: '4rem', textShadow: '0 0 20px #00f3ff', color: '#fff' }}>
          COSMIC<br />DATA<br />LINKS
        </h1>
        
        <p style={{ position: 'absolute', top: '120vh', left: '10vw', fontSize: '1.5rem', maxWidth: '30%', color: '#ccc' }}>
          Drop orbital telemetry files into the intake chamber to begin preprocessing.
        </p>

        <p style={{ position: 'absolute', top: '220vh', right: '10vw', fontSize: '1.5rem', maxWidth: '30%', color: '#ccc', textAlign: 'right' }}>
          Analyze clustered anomalies extracted from deep space images.
        </p>
      </Scroll>
    </ScrollControls>
  );
}
