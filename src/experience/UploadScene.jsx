import React, { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import { UploadCloud } from 'lucide-react';

export function UploadScene({ onFileSelect }) {
  const groupRef = useRef();
  const cubeRef = useRef();
  const [hovered, setHovered] = useState(false);

  useFrame((state, delta) => {
    if (cubeRef.current) {
      cubeRef.current.rotation.x += delta * 0.5;
      cubeRef.current.rotation.y += delta * 0.5;
    }
    if (groupRef.current) {
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime) * 0.2;
    }
  });

  return (
    <group ref={groupRef} position={[0, -5, 0]}>
      <mesh 
        ref={cubeRef} 
        onPointerOver={() => setHovered(true)} 
        onPointerOut={() => setHovered(false)}
      >
        <boxGeometry args={[2, 2, 2]} />
        <meshStandardMaterial 
          color={hovered ? "#00f3ff" : "#050510"} 
          emissive={hovered ? "#00f3ff" : "#9d4edd"}
          emissiveIntensity={hovered ? 0.8 : 0.2}
          wireframe={!hovered}
        />
      </mesh>
      
      <Html position={[0, 0, 0]} center transform style={{ pointerEvents: 'none' }}>
        <div style={{ textAlign: 'center', color: '#fff', textShadow: '0 0 10px #00f3ff', fontFamily: 'Orbitron', pointerEvents: 'auto' }}>
           <UploadCloud size={48} color="#00f3ff" style={{ marginBottom: 10 }} />
           <h3>DATA INTAKE</h3>
           <input 
             type="file" 
             style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer' }}
             onChange={(e) => {
               if(e.target.files[0]) onFileSelect(e.target.files[0]);
             }}
           />
        </div>
      </Html>
    </group>
  );
}
