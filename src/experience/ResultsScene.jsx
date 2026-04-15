import React, { useRef, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html, useTexture } from '@react-three/drei';
import * as THREE from 'three';
import { Database, Zap, MapPin } from 'lucide-react';

export function ResultsScene({ file }) {
  const meshRef = useRef();
  const groupRef = useRef();
  const [texture, setTexture] = useState(null);
  const [aspect, setAspect] = useState(1);
  const [data, setData] = useState(null);

  useEffect(() => {
    if(!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.src = url;
    img.onload = () => {
      const tex = new THREE.Texture(img);
      tex.needsUpdate = true;
      setTexture(tex);
      setAspect(img.width / img.height);
      
      // Simulate detection delay
      setTimeout(() => {
        setData({
          objects: [
            { id: 1, x: Math.random() * 2 - 1, y: Math.random() * 2 - 1, type: "Alpha Star", energy: "450 kW" },
            { id: 2, x: Math.random() * 2 - 1, y: Math.random() * 2 - 1, type: "Nebular Cluster", energy: "120 kW" }
          ]
        });
      }, 1000);
    };
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useFrame((state, delta) => {
    if (groupRef.current) {
      groupRef.current.position.y = -10; // offset down for Scene 3
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
      groupRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.2) * 0.05;
    }
  });

  return (
    <group ref={groupRef}>
      {texture && (
        <mesh ref={meshRef} position={[0, 0, 0]}>
          <planeGeometry args={[6 * aspect, 6]} />
          <meshBasicMaterial map={texture} transparent opacity={0.8} />
          
          {/* Scanning Beam */}
          <mesh position={[0, 0, 0.1]}>
             <planeGeometry args={[6 * aspect, 0.2]} />
             <meshStandardMaterial color="#00f3ff" emissive="#00f3ff" emissiveIntensity={2} transparent opacity={0.5} />
          </mesh>
        </mesh>
      )}

      {/* Detected 3D Anomaly Nodes */}
      {data && data.objects.map((obj, i) => (
        <group key={obj.id} position={[obj.x * 3 * aspect, obj.y * 3, 0.5]}>
          <mesh>
            <sphereGeometry args={[0.1, 16, 16]} />
            <meshStandardMaterial color="#9d4edd" emissive="#9d4edd" emissiveIntensity={2} />
          </mesh>
          <mesh>
             <ringGeometry args={[0.2, 0.25, 32]} />
             <meshBasicMaterial color="#00f3ff" side={THREE.DoubleSide} />
          </mesh>

          {/* Floating Data Cards */}
          <Html position={[0.3, 0.3, 0]} transform>
            <div style={{
              background: 'rgba(10, 10, 25, 0.6)',
              border: '1px solid rgba(0, 243, 255, 0.3)',
              padding: '10px 15px',
              borderRadius: '8px',
              backdropFilter: 'blur(10px)',
              color: '#fff',
              fontFamily: 'Orbitron',
              fontSize: '0.8rem',
              whiteSpace: 'nowrap',
              textShadow: '0 0 5px #00f3ff',
              pointerEvents: 'none'
            }}>
               <div style={{ color: '#00f3ff', marginBottom: 5 }}>T-{obj.id} // {obj.type}</div>
               <div style={{ display: 'flex', gap: 10, fontSize: '0.7rem' }}>
                  <span><Zap size={10} /> {obj.energy}</span>
                  <span><MapPin size={10} /> Lock</span>
               </div>
            </div>
          </Html>
        </group>
      ))}

      {/* Side Holographic Controls / Satellite Panels */}
      <Html position={[-4, 0, 1]} transform rotation-y={Math.PI / 8}>
         <div style={{
            width: 200, padding: 20, 
            background: 'rgba(0, 0, 0, 0.5)',
            borderLeft: '4px solid #9d4edd',
            color: '#fff', fontFamily: 'Orbitron',
            boxShadow: '0 0 20px rgba(157, 78, 221, 0.3)'
         }}>
            <Database size={24} color="#9d4edd" style={{ marginBottom: 10 }} />
            <h4 style={{ margin: '0 0 10px 0' }}>ANALYSIS LOGS</h4>
            <div style={{ fontSize: '0.7rem', color: '#aaa', lineHeight: 1.5 }}>
              Reading deep space emissions.<br/>
              Spectrometry: ACTIVE<br/>
              Noise Floor: 1.2 dB<br/>
              Resolution: 4K<br/>
            </div>
         </div>
      </Html>
    </group>
  );
}
