import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { useScroll, Html } from '@react-three/drei';
import * as THREE from 'three';

export function NavigationOrbit() {
  const groupRef = useRef();
  const scroll = useScroll();

  useFrame(({ clock }) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = clock.elapsedTime * 0.1;

      // Keep orbit fixed to the camera by offsetting against scroll
      groupRef.current.position.y = -scroll.offset * 10; // Match negative scroll distance
    }
  });

  const handleNav = (offset) => {
     scroll.el.scrollTo({ top: offset * scroll.el.scrollHeight, behavior: 'smooth' });
  };

  return (
    <group ref={groupRef} position={[4, 2, 0]}>
      {/* Central Star */}
      <mesh>
         <sphereGeometry args={[0.3, 32, 32]} />
         <meshStandardMaterial color="#00f3ff" emissive="#00f3ff" emissiveIntensity={2} />
      </mesh>

      {/* Orbit Rings */}
      <mesh rotation-x={Math.PI / 2}>
         <ringGeometry args={[1.5, 1.55, 64]} />
         <meshBasicMaterial color="rgba(157, 78, 221, 0.5)" side={THREE.DoubleSide} />
      </mesh>
      
      {/* Planet 1 - Uplink */}
      <mesh position={[1.5, 0, 0]} onClick={() => handleNav(0.3)}>
         <sphereGeometry args={[0.15, 16, 16]} />
         <meshStandardMaterial color="#9d4edd" />
         <Html center position={[0, -0.3, 0]}>
            <div style={{ color: '#aaa', fontSize: '0.6rem', fontFamily: 'Orbitron', pointerEvents: 'none' }}>
               UPLINK
            </div>
         </Html>
      </mesh>

      {/* Planet 2 - Databanks */}
      <mesh position={[-1.5, 0, 0]} onClick={() => handleNav(0.8)}>
         <sphereGeometry args={[0.15, 16, 16]} />
         <meshStandardMaterial color="#f72585" />
         <Html center position={[0, -0.3, 0]}>
            <div style={{ color: '#aaa', fontSize: '0.6rem', fontFamily: 'Orbitron', pointerEvents: 'none' }}>
               DATABANKS
            </div>
         </Html>
      </mesh>

    </group>
  );
}
