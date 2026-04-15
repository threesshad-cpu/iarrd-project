import React, { Suspense } from 'react';
import { Canvas }          from '@react-three/fiber';
import { Starfield }       from './Starfield';
import { NebulaShader }    from './NebulaShader';
import { FloatingCamera, GalaxyBackground, DustLayer } from './FloatingObjects';
import { PostProcessing }  from './PostProcessing';

/**
 * Persistent full-screen WebGL backdrop.
 * Three depth layers:
 *   Background (z ~−200): NebulaShader sphere
 *   Midground  (z ~−35):  GalaxyBackground spiral
 *   Foreground (z ~ 0):   Starfield + DustLayer
 * Camera floats via FloatingCamera rig.
 */
export function Scene() {
  return (
    <div className="canvas-root">
      <Canvas
        camera={{ position: [0, 0, 12], fov: 72, near: 0.1, far: 600 }}
        gl={{ antialias: false, powerPreference: 'high-performance', alpha: false }}
        dpr={[1, 1.5]}
      >
        <color attach="background" args={['#050510']} />
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 8]}  intensity={2.5} color="#38bdf8" />
        <pointLight position={[-8, -8, -8]} intensity={1.8} color="#7c3aed" />

        <Suspense fallback={null}>
          {/* Layer 1: Background nebula */}
          <NebulaShader />
          {/* Layer 2: Mid galaxy */}
          <GalaxyBackground />
          {/* Layer 3: Foreground stars + dust */}
          <Starfield />
          <DustLayer />
          {/* Camera rig */}
          <FloatingCamera />
        </Suspense>

        {/* Postprocessing */}
        <PostProcessing />
      </Canvas>
    </div>
  );
}
