import React, { useRef, useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * FloatingCamera — mouse-based parallax tilt + continuous anti-gravity drift.
 * Runs entirely in useFrame, zero state updates.
 */
export function FloatingCamera() {
  const { camera } = useThree();
  const mouse = useRef({ x: 0, y: 0 });
  const origin = useRef(new THREE.Vector3(0, 0, 0));

  React.useEffect(() => {
    const onMove = (e) => {
      mouse.current.x = (e.clientX / window.innerWidth)  * 2 - 1;
      mouse.current.y = (e.clientY / window.innerHeight) * 2 - 1;
    };
    window.addEventListener('mousemove', onMove, { passive: true });
    return () => window.removeEventListener('mousemove', onMove);
  }, []);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    // Slow breathing bob
    const bobX = Math.sin(t * 0.13) * 0.5;
    const bobY = Math.cos(t * 0.10) * 0.3;
    // Mouse tilt (max ±1.8 units)
    const tx = mouse.current.x * 1.8;
    const ty = -mouse.current.y * 1.0;
    // Spring interpolation
    camera.position.x += (bobX + tx - camera.position.x) * 0.022;
    camera.position.y += (bobY + ty - camera.position.y) * 0.022;
    camera.lookAt(origin.current);
  });

  return null;
}

/**
 * Galaxy spiral — 14,000 particles, deep background layer.
 * Sky-blue inner → deep blue outer, matching logo palette.
 */
export function GalaxyBackground() {
  const ref = useRef();
  const COUNT = 14000;

  const [positions, colors] = useMemo(() => {
    const pos = new Float32Array(COUNT * 3);
    const col = new Float32Array(COUNT * 3);
    // Logo sky-blue → deep navy
    const c1 = new THREE.Color('#38bdf8');
    const c2 = new THREE.Color('#0f2060');

    for (let i = 0; i < COUNT; i++) {
      const radius      = Math.random() * 20;
      const branches    = 3;
      const spin        = radius * 0.85;
      const branchAngle = ((i % branches) / branches) * Math.PI * 2;

      const pow = 3;
      const rnd = (v) => Math.pow(Math.random(), pow) * (Math.random() < 0.5 ? 1 : -1) * v * 0.45;

      pos[i*3+0] = Math.cos(branchAngle + spin) * radius + rnd(radius);
      pos[i*3+1] = rnd(1.2);
      pos[i*3+2] = Math.sin(branchAngle + spin) * radius + rnd(radius);

      const mixed = c1.clone().lerp(c2, radius / 20);
      col[i*3+0] = mixed.r;
      col[i*3+1] = mixed.g;
      col[i*3+2] = mixed.b;
    }
    return [pos, col];
  }, []);

  useFrame((_, delta) => {
    if (ref.current) ref.current.rotation.y += delta * 0.035;
  });

  return (
    <points ref={ref} position={[0, -16, -38]} rotation={[0.22, 0, -0.12]}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={COUNT} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-color"    count={COUNT} array={colors}    itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.09}
        sizeAttenuation
        vertexColors
        transparent
        opacity={0.9}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

/**
 * Foreground dust — tiny slow-drifting particles adding depth parallax.
 */
export function DustLayer() {
  const ref = useRef();
  const COUNT = 800;

  const positions = useMemo(() => {
    const arr = new Float32Array(COUNT * 3);
    for (let i = 0; i < COUNT; i++) {
      arr[i*3+0] = (Math.random() - 0.5) * 40;
      arr[i*3+1] = (Math.random() - 0.5) * 40;
      arr[i*3+2] = (Math.random() - 0.5) * 15;
    }
    return arr;
  }, []);

  useFrame(({ clock }) => {
    if (!ref.current) return;
    ref.current.rotation.y = clock.elapsedTime * 0.003;
    ref.current.rotation.x = Math.sin(clock.elapsedTime * 0.04) * 0.02;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={COUNT} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.06}
        sizeAttenuation
        color="#7dd3fc"
        transparent opacity={0.4}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}
