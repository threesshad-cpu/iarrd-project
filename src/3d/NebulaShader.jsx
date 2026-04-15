import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * Full-screen GLSL nebula shader.
 * Rendered on a large sphere (inside-out) so it wraps the entire scene.
 * Uses layered FBM (fractional Brownian motion) noise to create
 * volumetric-looking cloud structures — no texture files required.
 */

const VERT = /* glsl */`
  varying vec3 vWorldPos;
  void main() {
    vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const FRAG = /* glsl */`
  uniform float uTime;
  varying vec3 vWorldPos;

  // ── Hash & noise ──────────────────────────────
  vec3 hash3(vec3 p) {
    p = vec3(dot(p,vec3(127.1,311.7,74.7)),
             dot(p,vec3(269.5,183.3,246.1)),
             dot(p,vec3(113.5,271.9,124.6)));
    return fract(sin(p) * 43758.5453123);
  }

  float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f*f*(3.0-2.0*f);
    return mix(
      mix(mix(dot(hash3(i+vec3(0,0,0)),f-vec3(0,0,0)),
              dot(hash3(i+vec3(1,0,0)),f-vec3(1,0,0)),f.x),
          mix(dot(hash3(i+vec3(0,1,0)),f-vec3(0,1,0)),
              dot(hash3(i+vec3(1,1,0)),f-vec3(1,1,0)),f.x),f.y),
      mix(mix(dot(hash3(i+vec3(0,0,1)),f-vec3(0,0,1)),
              dot(hash3(i+vec3(1,0,1)),f-vec3(1,0,1)),f.x),
          mix(dot(hash3(i+vec3(0,1,1)),f-vec3(0,1,1)),
              dot(hash3(i+vec3(1,1,1)),f-vec3(1,1,1)),f.x),f.y),f.z);
  }

  // Fractional Brownian Motion — 5 octaves
  float fbm(vec3 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) {
      v += a * noise(p);
      p  = p * 2.1 + vec3(1.7, 9.2, 0.9);
      a *= 0.5;
    }
    return v;
  }

  void main() {
    vec3 dir = normalize(vWorldPos);
    float t  = uTime * 0.04;

    // Two warped fbm layers
    float n1 = fbm(dir * 2.5 + vec3(t, t * 0.7, t * 0.4));
    float n2 = fbm(dir * 3.8 + vec3(-t * 0.5, t * 0.3, t * 0.6) + n1 * 0.6);

    // Cloud density
    float density = smoothstep(0.35, 0.72, n2);

    // Sky-blue / violet nebula colors (matching logo palette)
    vec3 colBlue   = vec3(0.04, 0.28, 0.55);   // deep blue
    vec3 colCyan   = vec3(0.08, 0.48, 0.72);   // sky-blue mid
    vec3 colViolet = vec3(0.22, 0.08, 0.40);   // violet accent

    vec3 col = mix(colBlue, colCyan, n1);
    col      = mix(col, colViolet, smoothstep(0.5, 1.0, n2) * 0.55);

    // Very subtle horizontal banding
    float band = smoothstep(0.0, 0.4, sin(dir.y * 3.0 + t));
    col += vec3(0.01, 0.04, 0.06) * band;

    gl_FragColor = vec4(col * density, density * 0.38);
  }
`;

export function NebulaShader() {
  const matRef = useRef();

  useFrame(({ clock }) => {
    if (matRef.current) matRef.current.uniforms.uTime.value = clock.elapsedTime;
  });

  return (
    <mesh scale={[-1, 1, 1]}>  {/* invert normals for inside-out view */}
      <sphereGeometry args={[200, 32, 32]} />
      <shaderMaterial
        ref={matRef}
        vertexShader={VERT}
        fragmentShader={FRAG}
        uniforms={{ uTime: { value: 0 } }}
        transparent
        depthWrite={false}
        side={THREE.BackSide}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
}
