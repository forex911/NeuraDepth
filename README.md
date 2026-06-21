# 🌊 NeuraDepth

**NeuraDepth** is an advanced, professional-grade local web application that utilizes state-of-the-art Deep Learning to generate high-precision depth maps and interactive 3D structures from standard 2D input images.

Built with a lightning-fast React frontend and a powerful FastAPI Python backend, NeuraDepth allows users to instantly visualize, analyze, interact with, and export 3D approximations of any 2D image—running entirely locally with zero reliance on cloud APIs or external servers.

![NeuraDepth Interface](https://raw.githubusercontent.com/forex911/NeuraDepth/main/frontend/public/screenshot.png)

---

## 🌟 Key Features

*   **🧠 Deep Learning Depth Estimation**: Utilizes the powerful **MiDaS** model via PyTorch to intelligently estimate real-world depth from single 2D images.
*   **🎮 Interactive 3D Viewer**: A fully integrated, in-browser 3D canvas built with `Three.js` and `@react-three/fiber`. Instantly orbit, pan, and zoom around your generated 3D meshes or ultra-dense point clouds.
*   **🖱️ Intuitive UX & Controls**: 
    *   **Mouse-Wheel Comparison Slider**: Effortlessly compare the original 2D input with the generated depth output by hovering and scrolling your mouse wheel.
    *   **Hover Depth Inspection**: See the exact depth percentage at any pixel simply by hovering your cursor over the map.
*   **⚡ Real-Time WebSockets**: Heavy ML processing runs asynchronously in the backend, communicating real-time progress updates, success messages, and errors to the frontend via a sleek Neumorphic toast notification system.
*   **✨ Premium Neumorphic UI**: A breathtaking soft-UI aesthetic featuring deep inset shadows, raised interactive components, smooth animations, and a high-contrast 3D viewport.
*   **🛠️ One-Click Start**: Includes a `start.bat` script for Windows users to effortlessly spin up both the backend and frontend simultaneously.

---

## 📸 Processing & Visual Modes

NeuraDepth comes equipped with highly optimized, OpenCV-driven processing modes for different visual analyses:

1.  **Depth Map**: Generates a highly accurate, AI-driven grayscale depth map using MiDaS.
2.  **LIDAR**: Simulates a LiDAR point cloud scan by selectively sampling depth data and rendering color-mapped points with realistic noise jitter.
3.  **Topographic**: Generates authentic contour lines and elevation bands using a custom deep water-to-snow terrain colormap.
4.  **Wireframe**: Extracts high-frequency details and overlays a sleek, tech-inspired contour wireframe onto the depth map.
5.  **Mesh**: Renders an interconnected network of vector lines simulating a structured 3D topological grid.
6.  **Photogrammetry**: Simulates the artifacts, occlusion gaps, and noise of a photogrammetry reconstruction.
7.  **Scanner**: A sci-fi inspired visualization with sweeping scan lines and UI overlays.

## 🎛️ Adjustable Parameters

Fine-tune your depth generation with interactive UI sliders that immediately trigger debounced background regeneration:
*   **Scan Density**: Controls the resolution and spacing of the generated visual artifacts (like LiDAR points or Mesh lines).
*   **Point Density**: Dictates the fill-rate of simulated point clouds.
*   **Depth Contrast**: Amplifies the perceived depth difference between the foreground and background, flattening or exaggerating the topography.
*   **Smoothing**: Applies intelligent Gaussian blurring to remove harsh artifacts and create organic transitions in the depth map.
*   **Noise & Edge Sensitivity**: Advanced parameters to inject realistic noise or isolate sharp physical boundaries.

## 💾 Professional Export

Easily export your results for use in professional 3D software (Blender, Unity, Unreal Engine, Maya):
*   **16-Bit PNG**: Lossless high-precision depth map.
*   **3D Mesh (.OBJ)**: Extrudes the original image into a physical 3D surface based on the depth map, generating a UV-mapped `.obj` file.
*   **Point Cloud (.PLY)**: Generates a 3D point cloud file containing spatial coordinates and original RGB color data.

---

## 🏗️ Architecture Stack

*   **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Lucide React (Icons), Framer Motion, `@react-three/fiber`, `@react-three/drei`, `three`.
*   **Backend**: Python 3.9+, FastAPI, Uvicorn, PyTorch (`torch`, `torchvision`), OpenCV (`opencv-python`), Numpy, WebSockets.
*   **Communication**: `multipart/form-data` REST API & standard WebSockets for telemetry.

---

## 🚀 Getting Started

### Prerequisites
*   Node.js (v16+)
*   Python (3.9+)
*   Git

### The Easy Way (Windows)
Simply double-click the `start.bat` file in the root directory. It will automatically activate the Python virtual environment, start the FastAPI backend, and launch the Vite frontend server in two separate terminal windows.

### Manual Startup

**1. Start the Backend (FastAPI + PyTorch)**
The backend handles the heavy machine learning inference. Note: The first time you run a scan, PyTorch will download the MiDaS model weights automatically.

```bash
cd backend
python -m venv .venv

# Activate Virtual Environment
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
The frontend serves the Neumorphic UI and 3D Canvas.

```bash
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

Visit `http://localhost:5173` (or the port provided by Vite) in your browser to start using NeuraDepth!

---

## 📁 Project Structure

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
│   │   ├── App.tsx             # Main UI, state management, Sliders
│   │   ├── ThreeDViewer.tsx    # 3D Canvas, Point Cloud & Solid rendering
│   │   ├── main.tsx            # React root
│   │   └── styles.css          # Custom Neumorphic Tailwind configuration
│   ├── package.json            # Node dependencies
│   ├── tailwind.config.js      # Tailwind classes
│   └── vite.config.ts          # Vite bundler config
├── start.bat                   # Windows one-click startup script
└── README.md
```

## 📄 License
MIT License. Feel free to use, modify, and distribute the code.
