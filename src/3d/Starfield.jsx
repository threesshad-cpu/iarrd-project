import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * TRUE GLSL twinkling starfield.
 * Each star has a unique phase baked into vertexColor.a
 * Fragment shader reads time + phase → per-star flicker.
 */

const VERT = /* glsl */`
  attribute float aPhase;
  attribute float aSize;
  varying float vPhase;
  varying float vBrightness;

  void main() {
    vPhase = aPhase;
    vBrightness = aSize;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * (300.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const FRAG = /* glsl */`
  uniform float uTime;
  varying float vPhase;
  varying float vBrightness;

  // Smooth noise-based twinkle
  float twinkle(float phase, float t) {
    return 0.65 + 0.35 * sin(t * 2.1 + phase * 6.28318);
  }

  void main() {
    // Circular soft star shape
    vec2 uv = gl_PointCoord - 0.5;
    float d  = length(uv);
    if (d > 0.5) discard;

    float alpha = smoothstep(0.5, 0.0, d);
    float flicker = twinkle(vPhase, uTime);
    alpha *= flicker;

    // Colour: cool white → faint blue tint
    vec3 col = mix(vec3(0.75, 0.87, 1.0), vec3(1.0, 1.0, 1.0), vBrightness);
    gl_FragColor = vec4(col, alpha * 0.88);
  }
`;

export function Starfield() {
  const ref  = useRef();
  const mat  = useRef();
  const COUNT = 7000;

  const [positions, phases, sizes] = useMemo(() => {
    const pos = new Float32Array(COUNT * 3);
    const ph  = new Float32Array(COUNT);
    const sz  = new Float32Array(COUNT);

    for (let i = 0; i < COUNT; i++) {
      // Spherical shell distribution
      const theta = Math.random() * Math.PI * 2;
      const phi   = Math.acos(2 * Math.random() - 1);
      const r     = 70 + Math.random() * 130;

      pos[i * 3 + 0] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);

      ph[i] = Math.random();                     // twinkle phase
      sz[i] = 0.35 + Math.random() * 1.1;        // star size
    }
    return [pos, ph, sz];
  }, []);

  useFrame(({ clock }) => {
    if (mat.current) mat.current.uniforms.uTime.value = clock.elapsedTime;
    if (ref.current)  ref.current.rotation.y += 0.00015;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={COUNT} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-aPhase"   count={COUNT} array={phases}    itemSize={1} />
        <bufferAttribute attach="attributes-aSize"    count={COUNT} array={sizes}     itemSize={1} />
      </bufferGeometry>
      <shaderMaterial
        ref={mat}
        vertexShader={VERT}
        fragmentShader={FRAG}
        uniforms={{ uTime: { value: 0 } }}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}
