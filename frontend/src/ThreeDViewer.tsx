import { useRef, useState, useEffect, useMemo, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { RotateCcw, ZoomIn, Move, LocateFixed, Box, Grip, Grid3x3 } from "lucide-react";

type ScanMode = "depth" | "lidar" | "wireframe" | "mesh" | "scanner" | "photogrammetry" | "topographic";
type DisplayMode = "solid" | "wireframe" | "points" | "effect";

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

  const posAttr = geometry.getAttribute("position");
  const count = posAttr.count;

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < count; i++) {
    const x = posAttr.getX(i);
    const y = posAttr.getY(i);
    const z = posAttr.getZ(i);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const uvs = new Float32Array(count * 2);
  const normalizedZ = new Float32Array(count);
  const rangeZ = maxZ - minZ || 1;

  for (let i = 0; i < count; i++) {
    uvs[i * 2] = (posAttr.getX(i) - minX) / rangeX;
    uvs[i * 2 + 1] = (posAttr.getY(i) - minY) / rangeY;
    normalizedZ[i] = (posAttr.getZ(i) - minZ) / rangeZ;
  }

  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setAttribute("normalizedHeight", new THREE.Float32BufferAttribute(normalizedZ, 1));
  geometry.computeVertexNormals();
  geometry.center();

  return geometry;
}

// ── Dense Point Cloud Generator ─────────────────────────────────────────────
function createDensePoints(geometry: THREE.BufferGeometry, multiplier: number = 10): THREE.BufferGeometry {
  const posAttr = geometry.getAttribute("position");
  const uvAttr = geometry.getAttribute("uv");
  const heightAttr = geometry.getAttribute("normalizedHeight");
  const indices = geometry.getIndex();

  if (!posAttr || !indices || !uvAttr) return geometry;

  const triCount = indices.count / 3;
  const newCount = triCount * multiplier;

  const newPositions = new Float32Array(newCount * 3);
  const newUvs = new Float32Array(newCount * 2);
  const newHeights = new Float32Array(newCount);

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

    const hA = heightAttr ? (heightAttr as THREE.BufferAttribute).getX(a) : 0;
    const hB = heightAttr ? (heightAttr as THREE.BufferAttribute).getX(b) : 0;
    const hC = heightAttr ? (heightAttr as THREE.BufferAttribute).getX(c) : 0;

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

    newHeights[i] = r1 * hA + r2 * hB + r3 * hC;
  }

  const denseGeo = new THREE.BufferGeometry();
  denseGeo.setAttribute("position", new THREE.Float32BufferAttribute(newPositions, 3));
  denseGeo.setAttribute("uv", new THREE.Float32BufferAttribute(newUvs, 2));
  denseGeo.setAttribute("normalizedHeight", new THREE.Float32BufferAttribute(newHeights, 1));
  return denseGeo;
}

// ══════════════════════════════════════════════════════════════════════════════
// SHADER DEFINITIONS PER SCAN MODE
// ══════════════════════════════════════════════════════════════════════════════

// ── Depth Map: Grayscale height ──────────────────────────────────────────────
const depthMapShader = {
  vertexShader: `
    attribute float normalizedHeight;
    varying float vHeight;
    varying vec3 vNormal;
    void main() {
      vHeight = normalizedHeight;
      vNormal = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      #ifdef IS_POINTS
        gl_PointSize = 2.0 * (150.0 / -mvPos.z);
      #endif
    }
  `,
  fragmentShader: `
    varying float vHeight;
    varying vec3 vNormal;
    void main() {
      #ifdef IS_POINTS
        vec2 ptCoord = gl_PointCoord - vec2(0.5);
        if (dot(ptCoord, ptCoord) > 0.25) discard;
      #endif
      float h = clamp(vHeight, 0.0, 1.0);
      // Add subtle lighting
      vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));
      float diffuse = max(dot(vNormal, lightDir), 0.0) * 0.3 + 0.7;
      vec3 color = vec3(h * diffuse);
      gl_FragColor = vec4(color, 1.0);
    }
  `
};

// ── LiDAR: Turbo colormap points ─────────────────────────────────────────────
const lidarShader = {
  vertexShader: `
    attribute float normalizedHeight;
    varying float vHeight;
    varying float vScanline;
    uniform float uTime;
    void main() {
      vHeight = normalizedHeight;
      vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
      // Scanline Y position
      vScanline = position.y;
      gl_Position = projectionMatrix * mvPos;
      gl_PointSize = 2.0 * (150.0 / -mvPos.z);
    }
  `,
  fragmentShader: `
    varying float vHeight;
    varying float vScanline;
    uniform float uTime;

    // Approximation of Turbo colormap
    vec3 turbo(float t) {
      t = clamp(t, 0.0, 1.0);
      vec3 r = vec3(0.13572138, 4.61539260, -42.66032258);
      vec3 g = vec3(0.09140261, 2.19418839, 4.84296658);
      vec3 b = vec3(0.10667330, 12.64194608, -60.58204836);
      vec3 ro = vec3(11.60249308, -82.66631913, 132.13108234);
      vec3 go = vec3(-3.26450440, 13.24825890, -24.97509825);
      vec3 bo = vec3(28.40740723, -163.72821945, 278.68679559);
      return clamp(r + ro * t + (g + go * t) * t + (b + bo * t) * t * t, 0.0, 1.0);
    }

    void main() {
      vec2 coord = gl_PointCoord - vec2(0.5);
      if (dot(coord, coord) > 0.25) discard;

      float h = clamp(vHeight, 0.0, 1.0);
      vec3 color = turbo(h);

      // Animated scanline pulse
      float scanPulse = sin(vScanline * 0.5 + uTime * 2.0) * 0.5 + 0.5;
      float scanBand = smoothstep(0.92, 1.0, scanPulse);
      color = mix(color, vec3(0.27, 0.82, 1.0), scanBand * 0.4);

      gl_FragColor = vec4(color, 0.92);
    }
  `
};

// ── Wireframe: Cyan contour glow ─────────────────────────────────────────────
const wireframeShader = {
  vertexShader: `
    attribute float normalizedHeight;
    varying float vHeight;
    varying vec3 vNormal;
    varying vec3 vViewPos;
    void main() {
      vHeight = normalizedHeight;
      vNormal = normalize(normalMatrix * normal);
      vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
      vViewPos = mvPos.xyz;
      gl_Position = projectionMatrix * mvPos;
      #ifdef IS_POINTS
        gl_PointSize = 2.0 * (150.0 / -mvPos.z);
      #endif
    }
  `,
  fragmentShader: `
    varying float vHeight;
    varying vec3 vNormal;
    varying vec3 vViewPos;
    uniform float uTime;

    void main() {
      #ifdef IS_POINTS
        vec2 ptCoord = gl_PointCoord - vec2(0.5);
        if (dot(ptCoord, ptCoord) > 0.25) discard;
      #endif
      float h = clamp(vHeight, 0.0, 1.0);

      // Contour bands
      float bands = fract(h * 12.0);
      float contourLine = smoothstep(0.02, 0.0, abs(bands - 0.5) - 0.46);

      // Edge detection via fresnel
      vec3 viewDir = normalize(-vViewPos);
      float fresnel = 1.0 - abs(dot(viewDir, vNormal));
      float edgeGlow = pow(fresnel, 2.5);

      // Base dark with cyan edges and bands
      vec3 baseColor = vec3(0.02, 0.06, 0.1);
      vec3 cyanEdge = vec3(0.16, 0.86, 1.0);
      vec3 dimCyan = vec3(0.1, 0.42, 0.58);

      vec3 color = baseColor;
      color = mix(color, dimCyan, h * 0.35);
      color = mix(color, cyanEdge, contourLine * 0.8);
      color = mix(color, cyanEdge, edgeGlow * 0.7);

      // Pulsing glow
      float pulse = sin(uTime * 1.5) * 0.5 + 0.5;
      color += cyanEdge * edgeGlow * pulse * 0.15;

      gl_FragColor = vec4(color, 1.0);
    }
  `
};

// ── Mesh: Viridis + triangle grid overlay ────────────────────────────────────
const meshShader = {
  vertexShader: `
    attribute float normalizedHeight;
    varying float vHeight;
    varying vec3 vNormal;
    varying vec3 vWorldPos;
    void main() {
      vHeight = normalizedHeight;
      vNormal = normalize(normalMatrix * normal);
      vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      #ifdef IS_POINTS
        gl_PointSize = 2.0 * (150.0 / -mvPos.z);
      #endif
    }
  `,
  fragmentShader: `
    varying float vHeight;
    varying vec3 vNormal;
    varying vec3 vWorldPos;

    // Viridis colormap approximation
    vec3 viridis(float t) {
      t = clamp(t, 0.0, 1.0);
      vec3 c0 = vec3(0.267, 0.004, 0.329);
      vec3 c1 = vec3(0.282, 0.140, 0.457);
      vec3 c2 = vec3(0.253, 0.265, 0.529);
      vec3 c3 = vec3(0.190, 0.407, 0.556);
      vec3 c4 = vec3(0.127, 0.566, 0.550);
      vec3 c5 = vec3(0.153, 0.733, 0.498);
      vec3 c6 = vec3(0.360, 0.868, 0.390);
      vec3 c7 = vec3(0.667, 0.966, 0.224);
      vec3 c8 = vec3(0.993, 0.906, 0.144);
      float s = t * 8.0;
      int i = int(floor(s));
      float f = fract(s);
      if (i == 0) return mix(c0, c1, f);
      if (i == 1) return mix(c1, c2, f);
      if (i == 2) return mix(c2, c3, f);
      if (i == 3) return mix(c3, c4, f);
      if (i == 4) return mix(c4, c5, f);
      if (i == 5) return mix(c5, c6, f);
      if (i == 6) return mix(c6, c7, f);
      return mix(c7, c8, f);
    }

    void main() {
      #ifdef IS_POINTS
        vec2 ptCoord = gl_PointCoord - vec2(0.5);
        if (dot(ptCoord, ptCoord) > 0.25) discard;
      #endif
      float h = clamp(vHeight, 0.0, 1.0);
      vec3 color = viridis(h);

      // Grid overlay
      float gridScale = 4.0;
      vec2 grid = abs(fract(vWorldPos.xy * gridScale) - 0.5);
      float gridLine = 1.0 - smoothstep(0.0, 0.04, min(grid.x, grid.y));

      // Lighting
      vec3 lightDir = normalize(vec3(0.5, 1.0, 0.8));
      float diffuse = max(dot(vNormal, lightDir), 0.0) * 0.4 + 0.6;
      color *= diffuse;

      // Grid lines in bright cyan-green
      vec3 gridColor = vec3(0.12, 0.8, 0.9);
      color = mix(color, gridColor, gridLine * 0.65);

      gl_FragColor = vec4(color, 1.0);
    }
  `
};

// ── Scanner: Ocean colormap + pulse + HUD ────────────────────────────────────
const scannerShader = {
  vertexShader: `
    attribute float normalizedHeight;
    varying float vHeight;
    varying vec3 vNormal;
    varying vec3 vWorldPos;
    varying vec3 vViewPos;
    void main() {
      vHeight = normalizedHeight;
      vNormal = normalize(normalMatrix * normal);
      vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
      vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
      vViewPos = mvPos.xyz;
      gl_Position = projectionMatrix * mvPos;
      #ifdef IS_POINTS
        gl_PointSize = 2.0 * (150.0 / -mvPos.z);
      #endif
    }
  `,
  fragmentShader: `
    varying float vHeight;
    varying vec3 vNormal;
    varying vec3 vWorldPos;
    varying vec3 vViewPos;
    uniform float uTime;

    void main() {
      #ifdef IS_POINTS
        vec2 ptCoord = gl_PointCoord - vec2(0.5);
        if (dot(ptCoord, ptCoord) > 0.25) discard;
      #endif
      float h = clamp(vHeight, 0.0, 1.0);

      // Ocean colormap
      vec3 deepBlue = vec3(0.0, 0.05, 0.2);
      vec3 midBlue = vec3(0.0, 0.3, 0.6);
      vec3 lightCyan = vec3(0.3, 0.85, 1.0);
      vec3 baseColor = h < 0.5
        ? mix(deepBlue, midBlue, h * 2.0)
        : mix(midBlue, lightCyan, (h - 0.5) * 2.0);

      // Lighting
      vec3 lightDir = normalize(vec3(0.3, 1.0, 0.5));
      float diff = max(dot(vNormal, lightDir), 0.0) * 0.35 + 0.65;
      baseColor *= diff;

      // Pulsing scan bands
      float scanFreq = 18.0 + sin(uTime * 0.3) * 4.0;
      float scanBand = sin(h * scanFreq + uTime * 3.0);
      float scanLine = smoothstep(0.88, 1.0, scanBand);
      baseColor = mix(baseColor, vec3(0.33, 0.9, 1.0), scanLine * 0.7);

      // Edge glow (fresnel)
      vec3 viewDir = normalize(-vViewPos);
      float fresnel = 1.0 - abs(dot(viewDir, vNormal));
      float edgeGlow = pow(fresnel, 3.0);
      baseColor += vec3(0.25, 0.88, 1.0) * edgeGlow * 0.45;

      // Crosshair shimmer
      float cross = smoothstep(0.01, 0.0, abs(vWorldPos.x)) + smoothstep(0.01, 0.0, abs(vWorldPos.y));
      baseColor += vec3(0.25, 0.9, 1.0) * cross * 0.2;

      gl_FragColor = vec4(baseColor, 1.0);
    }
  `
};

// ── Photogrammetry: Textured + noise + holes ─────────────────────────────────
const photogrammetryShader = {
  vertexShader: `
    attribute float normalizedHeight;
    varying vec2 vUv;
    varying float vHeight;
    varying vec3 vNormal;
    varying vec3 vWorldPos;
    uniform float uTime;
    void main() {
      vUv = uv;
      vHeight = normalizedHeight;
      vNormal = normalize(normalMatrix * normal);
      vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      #ifdef IS_POINTS
        gl_PointSize = 2.0 * (150.0 / -mvPos.z);
      #endif
    }
  `,
  fragmentShader: `
    uniform sampler2D tex;
    uniform float uTime;
    varying vec2 vUv;
    varying float vHeight;
    varying vec3 vNormal;
    varying vec3 vWorldPos;

    // Simple hash for noise
    float hash(vec2 p) {
      return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }

    float noise2D(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      float a = hash(i);
      float b = hash(i + vec2(1.0, 0.0));
      float c = hash(i + vec2(0.0, 1.0));
      float d = hash(i + vec2(1.0, 1.0));
      return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }

    void main() {
      #ifdef IS_POINTS
        vec2 ptCoord = gl_PointCoord - vec2(0.5);
        if (dot(ptCoord, ptCoord) > 0.25) discard;
      #endif
      vec4 texColor = texture2D(tex, vUv);

      // Add film grain noise
      float grain = (hash(vUv * 500.0 + uTime * 0.1) - 0.5) * 0.12;
      texColor.rgb += grain;

      // Reconstruction artifact: occlusion gaps
      float gapNoise = noise2D(vWorldPos.xy * 8.0);
      float gradientStrength = length(vNormal.xy);
      if (gapNoise > 0.82 && gradientStrength > 0.6) {
        texColor.rgb = vec3(0.06, 0.06, 0.06);
      }

      // Subtle color shift to simulate multi-view blending
      float chromatic = sin(vWorldPos.x * 20.0 + vWorldPos.y * 15.0) * 0.02;
      texColor.r += chromatic;
      texColor.b -= chromatic;

      // Lighting
      vec3 lightDir = normalize(vec3(0.5, 1.0, 0.7));
      float diff = max(dot(vNormal, lightDir), 0.0) * 0.25 + 0.75;
      texColor.rgb *= diff;

      gl_FragColor = vec4(clamp(texColor.rgb, 0.0, 1.0), 1.0);
    }
  `
};

// ── Topographic: Terrain gradient + contour lines ────────────────────────────
const topographicShader = {
  vertexShader: `
    attribute float normalizedHeight;
    varying float vHeight;
    varying vec3 vNormal;
    void main() {
      vHeight = normalizedHeight;
      vNormal = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      #ifdef IS_POINTS
        gl_PointSize = 2.0 * (150.0 / -mvPos.z);
      #endif
    }
  `,
  fragmentShader: `
    varying float vHeight;
    varying vec3 vNormal;

    vec3 terrainColor(float h) {
      // Deep water -> green -> brown -> snow
      vec3 deepWater = vec3(0.08, 0.24, 0.47);
      vec3 shallowWater = vec3(0.15, 0.52, 0.55);
      vec3 lowland = vec3(0.22, 0.60, 0.25);
      vec3 midland = vec3(0.45, 0.72, 0.22);
      vec3 highland = vec3(0.65, 0.55, 0.30);
      vec3 mountain = vec3(0.55, 0.40, 0.28);
      vec3 snow = vec3(0.92, 0.94, 0.96);

      if (h < 0.12) return mix(deepWater, shallowWater, h / 0.12);
      if (h < 0.25) return mix(shallowWater, lowland, (h - 0.12) / 0.13);
      if (h < 0.45) return mix(lowland, midland, (h - 0.25) / 0.20);
      if (h < 0.65) return mix(midland, highland, (h - 0.45) / 0.20);
      if (h < 0.82) return mix(highland, mountain, (h - 0.65) / 0.17);
      return mix(mountain, snow, (h - 0.82) / 0.18);
    }

    void main() {
      #ifdef IS_POINTS
        vec2 ptCoord = gl_PointCoord - vec2(0.5);
        if (dot(ptCoord, ptCoord) > 0.25) discard;
      #endif
      float h = clamp(vHeight, 0.0, 1.0);
      vec3 color = terrainColor(h);

      // Contour lines
      float contourSpacing = 20.0;
      float contour = fract(h * contourSpacing);
      float contourLine = smoothstep(0.02, 0.0, abs(contour - 0.5) - 0.47);

      // Every 5th contour is thicker (index contours)
      float indexContour = fract(h * contourSpacing / 5.0);
      float indexLine = smoothstep(0.04, 0.0, abs(indexContour - 0.5) - 0.44);

      // Lighting
      vec3 lightDir = normalize(vec3(0.5, 1.0, 0.6));
      float diff = max(dot(vNormal, lightDir), 0.0) * 0.35 + 0.65;
      color *= diff;

      // Draw contour lines
      color = mix(color, vec3(1.0), contourLine * 0.5);
      color = mix(color, vec3(1.0), indexLine * 0.8);

      gl_FragColor = vec4(color, 1.0);
    }
  `
};

// ══════════════════════════════════════════════════════════════════════════════
// 3D MESH COMPONENT — renders the appropriate effect per scan mode
// ══════════════════════════════════════════════════════════════════════════════
function ScanModeMesh({
  geometry,
  textureUrl,
  scanMode,
  displayMode,
}: {
  geometry: THREE.BufferGeometry;
  textureUrl: string;
  scanMode: ScanMode;
  displayMode: DisplayMode;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const pointsRef = useRef<THREE.Points>(null);
  const timeRef = useRef(0);

  // Load texture for photogrammetry mode
  const texture = useMemo(() => {
    const tex = new THREE.TextureLoader().load(textureUrl);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true;
    return tex;
  }, [textureUrl]);

  // Dense points for LiDAR
  const densePointsGeometry = useMemo(() => {
    return createDensePoints(geometry, 75);
  }, [geometry]);

  // Animate time uniform
  useFrame((_, delta) => {
    timeRef.current += delta;
    const rotSpeed = delta * 0.05;

    if (meshRef.current) {
      meshRef.current.rotation.y += rotSpeed;
      const mat = meshRef.current.material as THREE.ShaderMaterial;
      if (mat.uniforms?.uTime) mat.uniforms.uTime.value = timeRef.current;
    }
    if (pointsRef.current) {
      pointsRef.current.rotation.y += rotSpeed;
      const mat = pointsRef.current.material as THREE.ShaderMaterial;
      if (mat.uniforms?.uTime) mat.uniforms.uTime.value = timeRef.current;
    }
  });


  // Get shader based on scan mode
  let shaderDef: any;
  if (scanMode === "depth") shaderDef = depthMapShader;
  else if (scanMode === "lidar") shaderDef = lidarShader;
  else if (scanMode === "wireframe") shaderDef = wireframeShader;
  else if (scanMode === "mesh") shaderDef = meshShader;
  else if (scanMode === "scanner") shaderDef = scannerShader;
  else if (scanMode === "photogrammetry") shaderDef = photogrammetryShader;
  else if (scanMode === "topographic") shaderDef = topographicShader;


  if (displayMode !== "effect") {
    if (displayMode === "points") {
      return (
        <points ref={pointsRef} geometry={densePointsGeometry}>
          <shaderMaterial
            vertexShader={`
              varying vec2 vUv;
              void main() {
                vUv = uv;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_Position = projectionMatrix * mvPosition;
                gl_PointSize = 2.0 * (150.0 / -mvPosition.z);
              }
            `}
            fragmentShader={`
              uniform sampler2D tex;
              varying vec2 vUv;
              void main() {
                vec2 coord = gl_PointCoord - vec2(0.5);
                if (dot(coord, coord) > 0.25) discard;
                gl_FragColor = texture2D(tex, vUv);
              }
            `}
            uniforms={{ tex: { value: texture } }}
            transparent
            depthWrite={false}
            blending={THREE.NormalBlending}
          />
        </points>
      );
    }

    return (
      <mesh ref={meshRef} geometry={geometry}>
        <meshBasicMaterial 
          map={texture} 
          side={THREE.DoubleSide} 
          wireframe={displayMode === "wireframe"} 
        />
      </mesh>
    );
  }

  // --- Effect Mode Rendering ---
  const isPointsEffect = scanMode === "lidar";
  const isWireframeEffect = scanMode === "wireframe";
  
  if (isPointsEffect) {
    return (
      <points ref={pointsRef} geometry={densePointsGeometry}>
        <shaderMaterial
          vertexShader={shaderDef.vertexShader}
          fragmentShader={shaderDef.fragmentShader}
          uniforms={{ uTime: { value: 0 }, tex: { value: texture } }}
          defines={{ IS_POINTS: "" }}
          transparent
          depthWrite={false}
          blending={THREE.NormalBlending}
        />
      </points>
    );
  }

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <shaderMaterial
        vertexShader={shaderDef.vertexShader}
        fragmentShader={shaderDef.fragmentShader}
        uniforms={{ uTime: { value: 0 }, tex: { value: texture } }}
        side={THREE.DoubleSide}
        wireframe={isWireframeEffect}
        transparent={scanMode === "scanner"}
      />
    </mesh>
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

// ── Scan Mode Label Map ──────────────────────────────────────────────────────
const scanModeLabels: Record<ScanMode, string> = {
  depth: "Depth Map",
  lidar: "LiDAR",
  wireframe: "Wireframe",
  mesh: "Mesh",
  scanner: "Scanner",
  photogrammetry: "Photogrammetry",
  topographic: "Topographic",
};

// ── Main Component ──────────────────────────────────────────────────────────
interface ThreeDViewerProps {
  file: File | null;
  controls: Record<string, number>;
  apiUrl: string;
  sourceUrl: string;
  scanMode: ScanMode;
}

export default function ThreeDViewer({ file, controls, apiUrl, sourceUrl, scanMode }: ThreeDViewerProps) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [displayMode, setDisplayMode] = useState<DisplayMode>("effect");
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
        
        <div className="w-px h-6 bg-black/10 mx-2"></div>

        {/* Active scan mode effect button */}
        <button
          onClick={() => setDisplayMode('effect')}
          className={`neu-btn three-viewer-tool-btn ${displayMode === 'effect' ? "neu-inset active" : "neu-raised"}`}
          title="Apply Scan Mode Effect"
        >
          <span className={displayMode === 'effect' ? "text-[#F62440]" : ""}>{scanModeLabels[scanMode]} Effect</span>
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
                <ScanModeMesh geometry={geometry} textureUrl={sourceUrl} scanMode={scanMode} displayMode={displayMode} />
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
