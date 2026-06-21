import { useRef, useState, useEffect, useMemo, useCallback } from "react";
import { Canvas, useFrame, useThree, useLoader } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { Grid3x3, RotateCcw, ZoomIn, Move } from "lucide-react";

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

// ── Textured 3D Mesh ────────────────────────────────────────────────────────
function TexturedMesh({
  geometry,
  textureUrl,
  wireframe,
}: {
  geometry: THREE.BufferGeometry;
  textureUrl: string;
  wireframe: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const wireRef = useRef<THREE.LineSegments>(null);

  // Load the original image as a texture
  const texture = useMemo(() => {
    const tex = new THREE.TextureLoader().load(textureUrl);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true;
    return tex;
  }, [textureUrl]);

  // Wireframe edges
  const wireGeo = useMemo(() => {
    return new THREE.WireframeGeometry(geometry);
  }, [geometry]);

  // Subtle idle rotation
  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.05;
    }
    if (wireRef.current) {
      wireRef.current.rotation.y = meshRef.current?.rotation.y ?? 0;
    }
  });

  return (
    <group>
      {/* Solid textured mesh */}
      <mesh ref={meshRef} geometry={geometry}>
        <meshBasicMaterial
          map={texture}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Optional wireframe overlay */}
      {wireframe && (
        <lineSegments ref={wireRef} geometry={wireGeo}>
          <lineBasicMaterial color="#32C5FF" transparent opacity={0.25} />
        </lineSegments>
      )}
    </group>
  );
}

// ── Auto-fit camera to bounding box ─────────────────────────────────────────
function AutoFit({ geometry }: { geometry: THREE.BufferGeometry }) {
  const { camera } = useThree();

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
  }, [geometry, camera]);

  return null;
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
  const [wireframe, setWireframe] = useState(false);
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
          onClick={() => setWireframe(!wireframe)}
          className={`neu-btn three-viewer-tool-btn ${wireframe ? "neu-inset active" : "neu-raised"}`}
          title="Toggle Wireframe Overlay"
        >
          <Grid3x3 size={15} />
          <span>{wireframe ? "Solid" : "Wire"}</span>
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
            camera={{ fov: 45, near: 0.1, far: 2000, position: [100, 80, 120] }}
            style={{ background: "transparent" }}
            gl={{ antialias: true, alpha: true }}
          >
            {geometry && sourceUrl && (
              <>
                <AutoFit geometry={geometry} />
                <TexturedMesh geometry={geometry} textureUrl={sourceUrl} wireframe={wireframe} />
              </>
            )}

            <OrbitControls
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
