# NeuraDepth

**NeuraDepth** is an advanced, professional-grade local web application that utilizes state-of-the-art deep learning to generate high-precision depth maps and interactive 3D structures from standard 2D input images.

Built with a React frontend and a FastAPI Python backend, NeuraDepth allows users to visualize, analyze, interact with, and export 3D approximations of any 2D image, running entirely locally with no reliance on cloud APIs or external servers.

![NeuraDepth Interface](https://raw.githubusercontent.com/forex911/NeuraDepth/main/frontend/public/screenshot.png)

---

## Key Features

- **Deep Learning Depth Estimation**: Utilizes the **MiDaS** model via PyTorch to estimate real-world depth from single 2D images.
- **Interactive 3D Viewer**: A fully integrated, in-browser 3D canvas built with `Three.js` and `@react-three/fiber`. Orbit, pan, and zoom around generated 3D meshes or dense point clouds.
- **Intuitive Controls**:
  - **Mouse-Wheel Comparison Slider**: Compare the original 2D input with the generated depth output by hovering and scrolling the mouse wheel.
  - **Hover Depth Inspection**: View the exact depth percentage at any pixel by hovering the cursor over the map.
- **Real-Time WebSockets**: Heavy ML processing runs asynchronously in the backend, communicating progress updates, success messages, and errors to the frontend via a toast notification system.
- **Neumorphic UI**: A soft-UI aesthetic featuring inset shadows, raised interactive components, smooth animations, and a high-contrast 3D viewport.
- **One-Click Start**: Includes a `start.bat` script for Windows users to launch both the backend and frontend simultaneously.

---

## Processing and Visual Modes

NeuraDepth includes optimized, OpenCV-driven processing modes for different visual analyses:

1. **Depth Map**: Generates an accurate, AI-driven grayscale depth map using MiDaS.
2. **LIDAR**: Simulates a LiDAR point cloud scan by selectively sampling depth data and rendering color-mapped points with realistic noise jitter.
3. **Topographic**: Generates contour lines and elevation bands using a custom water-to-snow terrain colormap.
4. **Wireframe**: Extracts high-frequency details and overlays a contour wireframe onto the depth map.
5. **Mesh**: Renders an interconnected network of vector lines simulating a structured 3D topological grid.
6. **Photogrammetry**: Simulates the artifacts, occlusion gaps, and noise of a photogrammetry reconstruction.
7. **Scanner**: A sci-fi-inspired visualization with sweeping scan lines and UI overlays.

## Adjustable Parameters

Fine-tune depth generation with interactive UI sliders that trigger debounced background regeneration:

- **Scan Density**: Controls the resolution and spacing of generated visual artifacts (such as LiDAR points or mesh lines).
- **Point Density**: Dictates the fill rate of simulated point clouds.
- **Depth Contrast**: Amplifies the perceived depth difference between foreground and background, flattening or exaggerating the topography.
- **Smoothing**: Applies Gaussian blurring to remove harsh artifacts and create organic transitions in the depth map.
- **Noise and Edge Sensitivity**: Advanced parameters to inject realistic noise or isolate sharp physical boundaries.

## Export Options

Export results for use in professional 3D software (Blender, Unity, Unreal Engine, Maya):

- **16-Bit PNG**: Lossless, high-precision depth map.
- **3D Mesh (.OBJ)**: Extrudes the original image into a physical 3D surface based on the depth map, generating a UV-mapped `.obj` file.
- **Point Cloud (.PLY)**: Generates a 3D point cloud file containing spatial coordinates and original RGB color data.

---

## Architecture Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Lucide React (icons), Framer Motion, `@react-three/fiber`, `@react-three/drei`, `three`.
- **Backend**: Python 3.9+, FastAPI, Uvicorn, PyTorch (`torch`, `torchvision`), OpenCV (`opencv-python`), NumPy, WebSockets.
- **Communication**: `multipart/form-data` REST API and standard WebSockets for telemetry.

---

## Getting Started

### Prerequisites

- Node.js (v16+)
- Python (3.9+)
- Git

### Quick Start (Windows)

Double-click the `start.bat` file in the root directory. It will activate the Python virtual environment, start the FastAPI backend, and launch the Vite frontend server in two separate terminal windows.

### Manual Startup

**1. Start the Backend (FastAPI + PyTorch)**

The backend handles ML inference. Note: the first time a scan is run, PyTorch will download the MiDaS model weights automatically.

```bash
cd backend
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server (runs on http://127.0.0.1:8000)
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**2. Start the Frontend (React + Vite)**

The frontend serves the UI and 3D canvas.

```bash
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

Visit `http://localhost:5173` (or the port provided by Vite) in a browser to start using NeuraDepth.

---

## Project Structure

```text
NeuraDepth/
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI endpoints, CORS, WebSockets
│   │   ├── processor.py        # OpenCV rendering & parameters logic
│   │   ├── services/
│   │   │   ├── depth_engine.py # Core PyTorch (MiDaS) inference
│   │   │   └── export_service.py # OBJ and PLY generation logic
│   │   └── __init__.py
│   └── requirements.txt        # Python ML dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Main UI, state management, sliders
│   │   ├── ThreeDViewer.tsx    # 3D canvas, point cloud & solid rendering
│   │   ├── main.tsx            # React root
│   │   └── styles.css          # Custom Neumorphic Tailwind configuration
│   ├── package.json            # Node dependencies
│   ├── tailwind.config.js      # Tailwind classes
│   └── vite.config.ts          # Vite bundler config
├── start.bat                   # Windows one-click startup script
└── README.md
```

## License

MIT License. Free to use, modify, and distribute.
