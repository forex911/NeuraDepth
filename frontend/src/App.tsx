import { ChangeEvent, DragEvent, useEffect, useState, useCallback } from "react";
import { Download, Scan, Upload, Settings2, Image as ImageIcon, Layers, Activity, Github } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

type ScanMode = "depth" | "lidar" | "wireframe" | "mesh" | "scanner" | "photogrammetry" | "topographic";

type ControlKey =
  | "scanDensity"
  | "noiseLevel"
  | "edgeSensitivity"
  | "depthContrast"
  | "smoothing"
  | "pointDensity";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

const modes: Array<{ id: ScanMode; label: string }> = [
  { id: "depth", label: "Depth Map" },
  { id: "lidar", label: "LiDAR" },
  { id: "wireframe", label: "Wireframe" },
  { id: "mesh", label: "Mesh" },
  { id: "scanner", label: "Scanner" },
  { id: "photogrammetry", label: "Photogrammetry" },
  { id: "topographic", label: "Topographic" },
];

const controlMeta: Array<{ key: ControlKey; label: string }> = [
  { key: "scanDensity", label: "Scan Density" },
  { key: "noiseLevel", label: "Noise Amount" },
  { key: "edgeSensitivity", label: "Edge Sensitivity" },
  { key: "depthContrast", label: "Depth Contrast" },
  { key: "smoothing", label: "Smoothing" },
  { key: "pointDensity", label: "Point Density" },
];

const defaults: Record<ControlKey, number> = {
  scanDensity: 62,
  noiseLevel: 18,
  edgeSensitivity: 52,
  depthContrast: 68,
  smoothing: 46,
  pointDensity: 58,
};

function App() {
  const [error, setError] = useState<string | null>(null);
  const [progressMsg, setProgressMsg] = useState<string | null>(null);

  // Initialize WebSocket connection for live progress
  useEffect(() => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws/progress");
    ws.onmessage = (event) => {
      setProgressMsg(event.data);
      if (event.data.includes("successfully") || event.data.includes("Error")) {
        setTimeout(() => setProgressMsg(null), 4000);
      }
    };
    return () => ws.close();
  }, []);
  const [file, setFile] = useState<File | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string>("");
  const [resultUrl, setResultUrl] = useState<string>("");
  const [resultMime, setResultMime] = useState<string>("image/png");
  const [mode, setMode] = useState<ScanMode>("scanner");
  const [controls, setControls] = useState(defaults);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState<"checking" | "ready" | "offline">("checking");
  const [sliderPos, setSliderPos] = useState(50);
  const [isSliding, setIsSliding] = useState(false);
  const [imageAspect, setImageAspect] = useState(1);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((response) => setHealth(response.ok ? "ready" : "offline"))
      .catch(() => setHealth("offline"));
  }, []);

  const runScan = useCallback(async () => {
    if (!file || health !== "ready") return;

    setIsProcessing(true);
    setError("");

    const form = new FormData();
    form.append("image", file);
    form.append("mode", mode);
    form.append("scan_density", String(controls.scanDensity));
    form.append("noise_level", String(controls.noiseLevel));
    form.append("edge_sensitivity", String(controls.edgeSensitivity));
    form.append("depth_contrast", String(controls.depthContrast));
    form.append("smoothing", String(controls.smoothing));
    form.append("point_density", String(controls.pointDensity));

    try {
      const response = await fetch(`${API_URL}/scan`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({ detail: "Processing failed." }));
        throw new Error(body.detail ?? "Processing failed.");
      }

      const blob = await response.blob();
      setResultMime(blob.type);
      setResultUrl((oldUrl) => {
        if (oldUrl) URL.revokeObjectURL(oldUrl);
        return URL.createObjectURL(blob);
      });
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Processing failed.");
    } finally {
      setIsProcessing(false);
    }
  }, [file, mode, controls, health]);

  const downloadExport = useCallback(async (exportMode: string, extension: string) => {
    if (!file || health !== "ready") return;
    setIsProcessing(true);
    setError("");
    const form = new FormData();
    form.append("image", file);
    form.append("mode", exportMode);
    form.append("scan_density", String(controls.scanDensity));
    form.append("noise_level", String(controls.noiseLevel));
    form.append("edge_sensitivity", String(controls.edgeSensitivity));
    form.append("depth_contrast", String(controls.depthContrast));
    form.append("smoothing", String(controls.smoothing));
    form.append("point_density", String(controls.pointDensity));

    try {
      const response = await fetch(`${API_URL}/scan`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) throw new Error("Export failed.");
      
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `depthforge_${exportMode}.${extension}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Export failed.");
    } finally {
      setIsProcessing(false);
    }
  }, [file, controls, health]);

  // Auto-scan on changes with a 500ms debounce
  useEffect(() => {
    if (!file || health !== "ready") return;
    
    const timeoutId = setTimeout(() => {
      runScan();
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [runScan, file, health]);

  function acceptFile(nextFile: File) {
    if (!["image/jpeg", "image/png", "image/webp"].includes(nextFile.type)) {
      setError("Upload a JPG, PNG, or WEBP image.");
      return;
    }
    setFile(nextFile);
    setSourceUrl((oldUrl) => {
      if (oldUrl) URL.revokeObjectURL(oldUrl);
      return URL.createObjectURL(nextFile);
    });
    setError("");
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    const nextFile = event.dataTransfer.files[0];
    if (nextFile) {
      acceptFile(nextFile);
    }
  }

  function handleInput(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0];
    if (nextFile) {
      acceptFile(nextFile);
    }
  }

  function updateControl(key: ControlKey, value: string) {
    setControls((current) => ({ ...current, [key]: Number(value) }));
  }

  function saveResult() {
    if (!resultUrl) return;
    const a = document.createElement("a");
    a.href = resultUrl;
    a.download = `depth_scan_${mode}.${resultMime.split("/")[1] || "png"}`;
    a.click();
  }

  return (
    <div className="text-black transition-colors duration-300">
      <header className="pt-16 pb-8 px-4">
        <div className="w-full max-w-[1800px] mx-auto text-center">
          <div className="flex justify-center items-center gap-4 mb-6">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full neu-raised-sm">
              <span className={`w-2 h-2 rounded-full ${health === "ready" ? "bg-green-400" : health === "offline" ? "bg-[#F62440]" : "bg-yellow-400"} animate-pulse`}></span>
              <span className="text-xs font-medium tracking-wide uppercase text-black">
                {health === "ready" ? "Backend Ready" : health === "offline" ? "Backend Offline" : "Checking Backend"}
              </span>
            </div>
            <a 
              href="https://github.com/forex911/NeuraDepth" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full neu-btn neu-raised text-black transition-colors hover:text-[#F62440]"
            >
              <Github size={16} />
              <span className="text-xs font-semibold tracking-wide">Star on GitHub</span>
            </a>
          </div>
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-semibold tracking-tight text-black mb-4">
            DepthForge
          </h1>
          <p className="text-base md:text-lg max-w-2xl mx-auto text-black/70">
            Professional-grade local computer vision & 3D depth generator.
          </p>
        </div>
      </header>

      <main className="w-full max-w-[1800px] mx-auto px-4 lg:px-8 pb-20">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          
          {/* Controls Sidebar */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="lg:col-span-1 space-y-8"
          >
            <article className="neu-raised p-6 rounded-3xl">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-lg font-semibold text-black tracking-tight flex items-center gap-2">
                  <Settings2 size={20} />
                  Parameters
                </h2>
                {isProcessing && (
                  <span className="flex items-center gap-2 text-xs font-medium text-[#F62440] animate-pulse">
                    <Activity size={14} className="animate-spin" />
                    Syncing
                  </span>
                )}
              </div>

              <div className="space-y-6">
                <div>
                  <label className="text-sm font-medium text-black mb-3 block">Scan Mode</label>
                  <div className="grid grid-cols-2 gap-3">
                    {modes.map((item) => (
                      <button
                        key={item.id}
                        onClick={() => setMode(item.id)}
                        className={`neu-btn neu-focus p-3 rounded-xl text-xs font-medium transition-all ${
                          mode === item.id 
                            ? "neu-inset text-[#F62440]" 
                            : "neu-raised text-black"
                        }`}
                      >
                        {item.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="h-px w-full bg-gradient-to-r from-transparent via-[#FFE5BF] to-transparent my-4"></div>
                
                <div>
                  <label className="text-sm font-medium text-black mb-3 block">Professional Export</label>
                  <div className="grid grid-cols-2 gap-3">
                    <button onClick={() => downloadExport("export_16bit", "png")} className="neu-btn neu-raised p-2 rounded-xl text-[11px] font-medium text-black">16-Bit PNG</button>
                    <button onClick={() => downloadExport("export_obj", "obj")} className="neu-btn neu-raised p-2 rounded-xl text-[11px] font-medium text-black">3D Mesh (OBJ)</button>
                    <button onClick={() => downloadExport("export_ply", "ply")} className="neu-btn neu-raised p-2 rounded-xl text-[11px] font-medium text-black">Point Cloud (PLY)</button>
                  </div>
                </div>

                <div className="h-px w-full bg-gradient-to-r from-transparent via-[#FFE5BF] to-transparent my-4"></div>

                {controlMeta.map((control) => (
                  <div key={control.key}>
                    <div className="flex justify-between mb-3">
                      <label className="text-xs font-medium text-black">{control.label}</label>
                      <span className="text-xs font-semibold text-[#F62440]">{controls[control.key]}</span>
                    </div>
                    <div className="relative h-3 rounded-full neu-inset">
                      <div className="absolute h-full rounded-full progress-fill" style={{ width: `${controls[control.key]}%` }}></div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={controls[control.key]}
                        onChange={(e) => updateControl(control.key, e.target.value)}
                        className="absolute w-full h-full opacity-0 cursor-pointer z-10"
                      />
                      <div className="absolute top-1/2 -translate-y-1/2 w-5 h-5 rounded-full neu-raised pointer-events-none" style={{ left: `calc(${controls[control.key]}% - 10px)` }}></div>
                    </div>
                  </div>
                ))}
                
                {error && <p className="text-xs text-white mt-2 text-center bg-[#F62440]/80 p-2 rounded-lg">{error}</p>}
              </div>
            </article>
          </motion.div>

          {/* Viewers */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2, ease: "easeOut" }}
            className="lg:col-span-3 h-full relative"
          >
            <AnimatePresence>
              {progressMsg && (
                <motion.div 
                  initial={{ opacity: 0, y: -20 }} 
                  animate={{ opacity: 1, y: 0 }} 
                  exit={{ opacity: 0, y: -20 }}
                  className="absolute top-10 left-1/2 -translate-x-1/2 bg-black/80 text-[#32C5FF] px-6 py-3 rounded-full text-sm font-semibold z-50 backdrop-blur-md shadow-2xl flex items-center gap-3 border border-[#32C5FF]/30"
                >
                  <Activity className="w-4 h-4 animate-spin" />
                  {progressMsg}
                </motion.div>
              )}
            </AnimatePresence>

            <article className="neu-raised p-6 rounded-3xl h-full flex flex-col min-h-[600px]">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-lg font-semibold tracking-tight flex items-center gap-2 text-black">
                  <ImageIcon size={20} />
                  {resultUrl ? "Compare Mode" : "Workspace"}
                </h2>
                <div className="flex gap-2">
                  {sourceUrl && (
                    <button 
                      onClick={() => { setFile(null); setSourceUrl(""); setResultUrl(""); }}
                      className="neu-raised neu-btn px-4 py-2 rounded-xl text-xs font-medium text-black"
                    >
                      Clear
                    </button>
                  )}
                  {resultUrl && (
                    <button onClick={saveResult} className="neu-btn neu-focus px-4 py-2 rounded-xl text-sm font-medium text-white bg-gradient-to-br from-[#F62440] to-[#D91C35] shadow-lg flex items-center gap-2">
                      <Download size={16} />
                      Download
                    </button>
                  )}
                </div>
              </div>
              
              <div className="neu-inset p-4 rounded-2xl flex-grow flex items-center justify-center relative overflow-hidden bg-transparent">
                {!sourceUrl ? (
                  <div
                    className={`border-2 border-dashed ${isDragging ? "border-[#F62440] bg-[#FFF2DB]" : "border-[#FFE5BF]"} rounded-2xl p-8 w-full h-full text-center transition-all duration-300 cursor-pointer flex flex-col items-center justify-center`}
                    onClick={() => document.getElementById("file-input")?.click()}
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDrop={handleDrop}
                    onDragLeave={() => setIsDragging(false)}
                  >
                    <input type="file" id="file-input" className="hidden" accept="image/png,image/jpeg,image/webp" onChange={handleInput} />
                    <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4 neu-raised">
                      <Upload size={32} className="text-[#F62440]" />
                    </div>
                    <p className="text-sm font-medium mb-1 text-black">Drop image here or click to upload</p>
                    <p className="text-xs text-black/70">PNG, JPG, WEBP</p>
                  </div>
                ) : isProcessing && !resultUrl ? (
                  <div className="flex flex-col items-center gap-4">
                    <div className="w-12 h-12 rounded-full border-4 border-[#FFE5BF] border-t-[#F62440] animate-spin"></div>
                    <span className="text-sm font-medium animate-pulse text-black/70">Generating Map...</span>
                  </div>
                ) : resultUrl ? (
                  <div className="relative w-full h-full flex justify-center items-center select-none">
                    <div 
                      className="relative w-full overflow-hidden rounded-lg cursor-ew-resize shadow-lg neu-raised"
                      style={{ aspectRatio: imageAspect, maxHeight: '100%', maxWidth: `calc(100vh * ${imageAspect})` }}
                      onMouseMove={(e) => {
                        if (!isSliding) return;
                        const rect = e.currentTarget.getBoundingClientRect();
                        const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                        setSliderPos((x / rect.width) * 100);
                      }}
                      onMouseUp={() => setIsSliding(false)}
                      onMouseLeave={() => setIsSliding(false)}
                      onMouseDown={() => setIsSliding(true)}
                      onTouchMove={(e) => {
                        const rect = e.currentTarget.getBoundingClientRect();
                        const touch = e.touches[0];
                        const x = Math.max(0, Math.min(touch.clientX - rect.left, rect.width));
                        setSliderPos((x / rect.width) * 100);
                      }}
                    >
                      {/* Base Image (Depth Output) */}
                      <img src={resultUrl} className="absolute inset-0 w-full h-full object-cover pointer-events-none" draggable={false} alt="Depth Map" />
                      
                      {/* Foreground Image (Original Input) */}
                      <div 
                        className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none"
                        style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
                      >
                        <img src={sourceUrl} className="absolute inset-0 w-full h-full object-cover pointer-events-none" style={{ width: '100%', height: '100%', maxWidth: 'none' }} draggable={false} alt="Original Input" />
                      </div>

                      {/* Slider Handle */}
                      <div 
                        className="absolute top-0 bottom-0 w-1 bg-white pointer-events-none shadow-[0_0_5px_rgba(0,0,0,0.5)]"
                        style={{ left: `${sliderPos}%`, transform: 'translateX(-50%)' }}
                      >
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-[0_0_10px_rgba(0,0,0,0.3)] flex items-center justify-center">
                          <div className="flex gap-1">
                            <div className="w-0.5 h-3 bg-gray-400"></div>
                            <div className="w-0.5 h-3 bg-gray-400"></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="relative w-full h-full flex justify-center items-center">
                    <img 
                      src={sourceUrl} 
                      className="max-w-full max-h-full object-contain rounded-lg shadow-lg neu-raised" 
                      alt="Input Preview"
                      onLoad={(e) => setImageAspect(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)} 
                    />
                  </div>
                )}
              </div>
            </article>
          </motion.div>

        </div>
      </main>
    </div>
  );
}

export default App;
