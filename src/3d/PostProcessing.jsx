import React from 'react';
import { EffectComposer, Bloom, ChromaticAberration } from '@react-three/postprocessing';
import { BlendFunction } from 'postprocessing';
import * as THREE from 'three';

/**
 * Postprocessing stack:
 *  1. Bloom  — glow on self-luminous geometry
 *  2. ChromaticAberration — subtle lens dispersion adds cinematic realism
 */
export function PostProcessing() {
  return (
    <EffectComposer disableNormalPass multisampling={0}>
      <Bloom
        luminanceThreshold={0.15}
        luminanceSmoothing={0.9}
        intensity={2.2}
        mipmapBlur
      />
      <ChromaticAberration
        blendFunction={BlendFunction.NORMAL}
        offset={new THREE.Vector2(0.0008, 0.0008)}
        radialModulation={false}
        modulationOffset={0.0}
      />
    </EffectComposer>
  );
}
