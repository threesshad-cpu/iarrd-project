import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * Procedural nebula using two large transparent planes with
 * vertex-displaced noise — no texture file needed.
 * Creates depth-layered color clouds.
 */
function NebulaPlane({ color, position, rotation, opacity, speed }) {
  const meshRef  = useRef();
  const matRef   = useRef();

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const t = clock.elapsedTime * speed;
    meshRef.current.rotation.z = t * 0.05;
    if (matRef.current) {
      matRef.current.opacity = opacity * (0.8 + Math.sin(t * 0.8) * 0.2);
    }
  });

  return (
    <mesh ref={meshRef} position={position} rotation={rotation}>
      <planeGeometry args={[220, 220, 1, 1]} />
      <meshBasicMaterial
        ref={matRef}
        color={color}
        transparent
        opacity={opacity}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/**
 * Floating dust particles in the midground for depth.
 */
function DustParticles() {
  const ref   = useRef();
  const COUNT = 1200;

  const positions = useMemo(() => {
    const arr = new Float32Array(COUNT * 3);
    for (let i = 0; i < COUNT; i++) {
      arr[i * 3 + 0] = (Math.random() - 0.5) * 60;
      arr[i * 3 + 1] = (Math.random() - 0.5) * 60;
      arr[i * 3 + 2] = (Math.random() - 0.5) * 30;
    }
    return arr;
  }, []);

  useFrame(({ clock }) => {
    if (!ref.current) return;
    ref.current.rotation.y += 0.0008;
    ref.current.rotation.x = Math.sin(clock.elapsedTime * 0.1) * 0.03;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={COUNT}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.08}
        sizeAttenuation
        color="#5ec3e8"
        transparent
        opacity={0.5}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

export function Nebula() {
  return (
    <group>
      {/* Far background nebula layers */}
      <NebulaPlane
        color="#0d1f6e"
        position={[0, 0, -80]}
        rotation={[0, 0, 0.3]}
        opacity={0.45}
        speed={0.3}
      />
      <NebulaPlane
        color="#1a0a3d"
        position={[20, -10, -100]}
        rotation={[0.1, 0.2, -0.2]}
        opacity={0.35}
        speed={0.2}
      />
      {/* Accent: subtle violet bloom near center */}
      <NebulaPlane
        color="#3b1278"
        position={[-15, 8, -60]}
        rotation={[0, -0.1, 0.5]}
        opacity={0.2}
        speed={0.5}
      />
      {/* Cyan teal haze */}
      <NebulaPlane
        color="#002e3d"
        position={[5, -5, -70]}
        rotation={[0.05, 0.05, 0]}
        opacity={0.25}
        speed={0.4}
      />
      {/* Midground dust */}
      <DustParticles />
    </group>
  );
}
