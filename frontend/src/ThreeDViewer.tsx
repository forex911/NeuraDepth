import { useRef, useState, useEffect, useMemo, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { Grid3x3, RotateCcw, ZoomIn, Move, LocateFixed, Box, Grip } from "lucide-react";

// ── OBJ Parser — extracts positions, indices, and computes UVs ──────────────
function parseOBJWithUVs(text: string): THREE.BufferGeometry {
  const positions: number[] = [];
  const indices: number[] = [];

  const lines = text.split("\n");
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("v ")) {
      const parts = trimmed.split(/\s+/);
      positions.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
    } else if (trimmed.startsWith("f ")) {
      const parts = trimmed.split(/\s+/);
      const faceIndices = parts.slice(1).map((p) => parseInt(p.split("/")[0]) - 1);
      if (faceIndices.length === 3) {
        indices.push(faceIndices[0], faceIndices[1], faceIndices[2]);
      } else if (faceIndices.length === 4) {
        indices.push(faceIndices[0], faceIndices[1], faceIndices[2]);
        indices.push(faceIndices[0], faceIndices[2], faceIndices[3]);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  if (indices.length > 0) {
    geometry.setIndex(indices);
  }

  // Compute UVs from XY positions (the mesh is a height-field grid)
  const posAttr = geometry.getAttribute("position");
  const count = posAttr.count;

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < count; i++) {
    const x = posAttr.getX(i);
    const y = posAttr.getY(i);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const uvs = new Float32Array(count * 2);

  for (let i = 0; i < count; i++) {
    // U = normalized X, V = flipped normalized Y (image coords are top-down)
    uvs[i * 2] = (posAttr.getX(i) - minX) / rangeX;
    uvs[i * 2 + 1] = (posAttr.getY(i) - minY) / rangeY;
  }

  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.computeVertexNormals();
  geometry.center();

  return geometry;
}

// ── Dense Point Cloud Generator ─────────────────────────────────────────────
function createDensePoints(geometry: THREE.BufferGeometry, multiplier: number = 10): THREE.BufferGeometry {
  const posAttr = geometry.getAttribute("position");
  const uvAttr = geometry.getAttribute("uv");
  const indices = geometry.getIndex();
  
  if (!posAttr || !indices || !uvAttr) return geometry;

  const triCount = indices.count / 3;
  const newCount = triCount * multiplier; 
  
  const newPositions = new Float32Array(newCount * 3);
  const newUvs = new Float32Array(newCount * 2);

  const posA = new THREE.Vector3();
  const posB = new THREE.Vector3();
  const posC = new THREE.Vector3();
  const uvA = new THREE.Vector2();
  const uvB = new THREE.Vector2();
  const uvC = new THREE.Vector2();

  for (let i = 0; i < newCount; i++) {
    const triIdx = Math.floor(Math.random() * triCount) * 3;
    const a = indices.getX(triIdx);
    const b = indices.getX(triIdx + 1);
    const c = indices.getX(triIdx + 2);

    posA.fromBufferAttribute(posAttr as THREE.BufferAttribute, a);
    posB.fromBufferAttribute(posAttr as THREE.BufferAttribute, b);
    posC.fromBufferAttribute(posAttr as THREE.BufferAttribute, c);
    
    uvA.fromBufferAttribute(uvAttr as THREE.BufferAttribute, a);
    uvB.fromBufferAttribute(uvAttr as THREE.BufferAttribute, b);
    uvC.fromBufferAttribute(uvAttr as THREE.BufferAttribute, c);

    let r1 = Math.random();
    let r2 = Math.random();
    if (r1 + r2 > 1) {
      r1 = 1 - r1;
      r2 = 1 - r2;
    }
    const r3 = 1 - r1 - r2;

    newPositions[i * 3] = r1 * posA.x + r2 * posB.x + r3 * posC.x;
    newPositions[i * 3 + 1] = r1 * posA.y + r2 * posB.y + r3 * posC.y;
    newPositions[i * 3 + 2] = r1 * posA.z + r2 * posB.z + r3 * posC.z;

    newUvs[i * 2] = r1 * uvA.x + r2 * uvB.x + r3 * uvC.x;
    newUvs[i * 2 + 1] = r1 * uvA.y + r2 * uvB.y + r3 * uvC.y;
  }

  const denseGeo = new THREE.BufferGeometry();
  denseGeo.setAttribute("position", new THREE.Float32BufferAttribute(newPositions, 3));
  denseGeo.setAttribute("uv", new THREE.Float32BufferAttribute(newUvs, 2));
  return denseGeo;
}

// ── Textured 3D Mesh ────────────────────────────────────────────────────────
function TexturedMesh({
  geometry,
  textureUrl,
  displayMode,
}: {
  geometry: THREE.BufferGeometry;
  textureUrl: string;
  displayMode: 'solid' | 'wireframe' | 'points';
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const pointsRef = useRef<THREE.Points>(null);

  // Load the original image as a texture
  const texture = useMemo(() => {
    const tex = new THREE.TextureLoader().load(textureUrl);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true;
    return tex;
  }, [textureUrl]);

  const maxDim = useMemo(() => {
    geometry.computeBoundingBox();
    const size = new THREE.Vector3();
    geometry.boundingBox?.getSize(size);
    return Math.max(size.x, size.y, size.z) || 100;
  }, [geometry]);

  // Subtle idle rotation
  useFrame((state, delta) => {
    const rotSpeed = delta * 0.05;
    if (meshRef.current) meshRef.current.rotation.y += rotSpeed;
    if (pointsRef.current) pointsRef.current.rotation.y += rotSpeed;
  });

  const customShader = useMemo(() => ({
    uniforms: {
      tex: { value: texture },
      pointSize: { value: 0.8 }, // Extremely small points for ultra-dense look
      uMaxDim: { value: maxDim }
    },
    vertexShader: `
      varying vec2 vUv;
      uniform float pointSize;

      void main() {
        vUv = uv;
        
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        
        // Dynamic point size scaling based on depth
        gl_PointSize = pointSize * (200.0 / -mvPosition.z);
      }
    `,
    fragmentShader: `
      uniform sampler2D tex;
      varying vec2 vUv;

      void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (dot(coord, coord) > 0.25) discard;
        
        vec4 color = texture2D(tex, vUv);
        // Original true color matching upload exactly
        gl_FragColor = color;
      }
    `
  }), [texture, maxDim]);

  const densePointsGeometry = useMemo(() => {
    return createDensePoints(geometry, 75); // 75x multiplier for ultra-dense look
  }, [geometry]);

  return (
    <group>
      {displayMode === 'solid' && (
        <mesh ref={meshRef} geometry={geometry}>
          <meshBasicMaterial map={texture} side={THREE.DoubleSide} />
        </mesh>
      )}

      {displayMode === 'wireframe' && (
        <mesh ref={meshRef} geometry={geometry}>
          <meshBasicMaterial map={texture} side={THREE.DoubleSide} wireframe={true} />
        </mesh>
      )}

      {displayMode === 'points' && (
        <points ref={pointsRef} geometry={densePointsGeometry}>
          <shaderMaterial 
            attach="material" 
            args={[customShader]} 
            transparent={true}
            blending={THREE.NormalBlending}
            depthWrite={false}
          />
        </points>
      )}
    </group>
  );
}

// ── Scene Controller (Auto-fit & Orbit Controls) ────────────────────────────
function SceneController({ geometry, resetToggle }: { geometry: THREE.BufferGeometry; resetToggle: boolean }) {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);

  useEffect(() => {
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    if (!box) return;

    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim * 1.6;

    if (camera instanceof THREE.PerspectiveCamera) {
      camera.position.set(dist * 0.6, dist * 0.5, dist * 0.8);
      camera.lookAt(0, 0, 0);
      camera.near = 0.1;
      camera.far = dist * 10;
      camera.updateProjectionMatrix();
    }

    if (controlsRef.current) {
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();
    }
  }, [geometry, camera, resetToggle]);

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      minDistance={5}
      maxDistance={500}
      enablePan
      panSpeed={0.8}
      rotateSpeed={0.6}
      zoomSpeed={0.8}
      zoomToCursor
    />
  );
}

// ── Main Component ──────────────────────────────────────────────────────────
interface ThreeDViewerProps {
  file: File | null;
  controls: Record<string, number>;
  apiUrl: string;
  sourceUrl: string;
}

export default function ThreeDViewer({ file, controls, apiUrl, sourceUrl }: ThreeDViewerProps) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [displayMode, setDisplayMode] = useState<'solid' | 'wireframe' | 'points'>('solid');
  const [resetToggle, setResetToggle] = useState(false);
  const [vertexCount, setVertexCount] = useState(0);
  const [faceCount, setFaceCount] = useState(0);

  const fetchModel = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const form = new FormData();
    form.append("image", file);
    form.append("mode", "export_obj");
    form.append("scan_density", String(controls.scanDensity));
    form.append("noise_level", String(controls.noiseLevel));
    form.append("edge_sensitivity", String(controls.edgeSensitivity));
    form.append("depth_contrast", String(controls.depthContrast));
    form.append("smoothing", String(controls.smoothing));
    form.append("point_density", String(controls.pointDensity));

    try {
      const response = await fetch(`${apiUrl}/scan`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        throw new Error("Failed to generate 3D mesh");
      }

      const text = await response.text();
      const geo = parseOBJWithUVs(text);
      setGeometry(geo);
      setVertexCount(geo.getAttribute("position").count);
      setFaceCount(geo.index ? geo.index.count / 3 : 0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load 3D model");
    } finally {
      setLoading(false);
    }
  }, [file, controls, apiUrl]);

  // Fetch on mount and when controls change (debounced)
  useEffect(() => {
    if (!file) return;
    const timeout = setTimeout(() => fetchModel(), 800);
    return () => clearTimeout(timeout);
  }, [fetchModel]);

  return (
    <div className="three-viewer-container">
      {/* Toolbar */}
      <div className="three-viewer-toolbar">
        <button
          onClick={() => setDisplayMode('solid')}
          className={`neu-btn three-viewer-tool-btn ${displayMode === 'solid' ? "neu-inset active" : "neu-raised"}`}
          title="Solid Mesh"
        >
          <Box size={15} />
          <span>Solid</span>
        </button>
        <button
          onClick={() => setDisplayMode('points')}
          className={`neu-btn three-viewer-tool-btn ${displayMode === 'points' ? "neu-inset active" : "neu-raised"}`}
          title="Vertex Dots"
        >
          <Grip size={15} />
          <span>Points</span>
        </button>
        <button
          onClick={() => setDisplayMode('wireframe')}
          className={`neu-btn three-viewer-tool-btn ${displayMode === 'wireframe' ? "neu-inset active" : "neu-raised"}`}
          title="Wireframe"
        >
          <Grid3x3 size={15} />
          <span>Wire</span>
        </button>
        <button
          onClick={() => setResetToggle((prev) => !prev)}
          className="neu-btn neu-raised three-viewer-tool-btn"
          title="Center View"
          disabled={!geometry}
        >
          <LocateFixed size={15} />
          <span>Center</span>
        </button>
        <button
          onClick={fetchModel}
          className="neu-btn neu-raised three-viewer-tool-btn"
          title="Refresh Model"
          disabled={loading}
        >
          <RotateCcw size={15} className={loading ? "animate-spin" : ""} />
          <span>Refresh</span>
        </button>

        {/* Stats */}
        {geometry && (
          <div className="three-viewer-stats">
            <span>{vertexCount.toLocaleString()} verts</span>
            <span className="three-viewer-stats-dot">•</span>
            <span>{faceCount.toLocaleString()} faces</span>
          </div>
        )}
      </div>

      {/* Canvas */}
      <div className="three-viewer-canvas-wrap">
        {loading && (
          <div className="three-viewer-loading">
            <div className="three-viewer-spinner" />
            <span>Generating 3D mesh…</span>
          </div>
        )}

        {error && !loading && (
          <div className="three-viewer-error">
            <p>{error}</p>
            <button onClick={fetchModel} className="neu-btn neu-raised three-viewer-tool-btn" style={{ marginTop: 12 }}>
              <RotateCcw size={14} />
              <span>Retry</span>
            </button>
          </div>
        )}

        {!error && (
          <Canvas
            flat
            camera={{ fov: 45, near: 0.1, far: 2000, position: [100, 80, 120] }}
            style={{ background: "transparent" }}
            gl={{ antialias: true, alpha: true }}
          >
            {geometry && sourceUrl && (
              <>
                <SceneController geometry={geometry} resetToggle={resetToggle} />
                <TexturedMesh geometry={geometry} textureUrl={sourceUrl} displayMode={displayMode} />
              </>
            )}
          </Canvas>
        )}
      </div>

      {/* Controls hint */}
      <div className="three-viewer-hints">
        <span><Move size={12} /> Drag to orbit</span>
        <span><ZoomIn size={12} /> Scroll to zoom</span>
        <span>Right-click to pan</span>
      </div>
    </div>
  );
}
