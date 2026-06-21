# DepthForge

**DepthForge** is an advanced, locally-hosted application that uses state-of-the-art Deep Learning to generate depth maps and 3D meshes from standard 2D input images. 

Built with a lightning-fast React frontend and a powerful FastAPI Python backend, DepthForge allows users to instantly visualize, interact with, and export 3D approximations of any 2D image, running entirely locally without the need for cloud APIs.

---

## 🌟 Key Features

*   **Deep Learning Depth Estimation**: Utilizes the powerful **MiDaS** model via PyTorch to intelligently estimate real-world depth from single 2D images.
*   **Interactive 3D Viewer**: A fully integrated, in-browser 3D canvas built with `Three.js` and `@react-three/fiber` that lets you orbit, pan, and zoom around your generated 3D meshes instantly.
*   **Real-Time Progress & WebSockets**: Heavy ML processing runs asynchronously in the backend, communicating real-time progress updates and status messages to the frontend via WebSockets.
*   **Modern Neuromorphic UI**: A breathtaking, premium soft-UI aesthetic featuring deep inset shadows, raised interactive components, and dynamic hover states.
*   **Interactive Comparison Slider**: Instantly compare the original 2D input with the generated depth output using a sleek, draggable overlay slider.
*   **One-Click Start**: Includes a `start.bat` script for Windows users to effortlessly spin up both the backend and frontend simultaneously.

## 🛠️ Processing Modes

DepthForge comes equipped with versatile processing and export modes:

1.  **Depth Map**: Generates a highly accurate, AI-driven grayscale depth map using MiDaS.
2.  **3D Mesh (.OBJ)**: Extrudes the original image into a physical 3D surface based on the depth map, generating a UV-mapped `.obj` file you can view in-app or import into Blender/Unity.
3.  **Experimental Modes**: Includes visual simulations like LiDAR point clouds, wireframes, and scanner overlays.

## 🎛️ Adjustable Parameters

Fine-tune your depth generation with interactive UI sliders:
*   **Scan Density**: Controls the resolution and spacing of the generated mesh vertices.
*   **Depth Contrast**: Amplifies the perceived depth difference between the foreground and background.
*   **Smoothing**: Applies intelligent blurring (Gaussian/Bilateral) to remove harsh artifacts and create organic transitions in the depth map.
*   **Noise & Edge Sensitivity**: Advanced parameters for experimental edge-detection filters.

---

## 🏗️ Architecture Stack

*   **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Lucide React (Icons), `@react-three/fiber`, `@react-three/drei`.
*   **Backend**: Python 3.9+, FastAPI, Uvicorn, PyTorch (`torch`, `torchvision`), OpenCV (`opencv-python`), WebSockets.
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
The frontend serves the Neuromorphic UI and 3D Canvas.

```bash
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

Visit `http://localhost:5173` (or the port provided by Vite) in your browser to start using DepthForge!

---

## 📁 Project Structure

```text
DepthForge/
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI endpoints, CORS, WebSockets
│   │   ├── processor.py    # Core PyTorch (MiDaS) & OpenCV logic
│   │   └── __init__.py
│   └── requirements.txt    # Python ML dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Main UI, state management, Sliders
│   │   ├── ThreeDViewer.tsx# 3D Canvas, OrbitControls, Scene Management
│   │   ├── main.tsx        # React root
│   │   └── styles.css      # Custom Neuromorphic Tailwind configuration
│   ├── package.json        # Node dependencies
│   ├── tailwind.config.js  # Tailwind classes
│   └── vite.config.ts      # Vite bundler config
├── start.bat               # Windows one-click startup script
└── README.md
```

## 📄 License
MIT License. Feel free to use, modify, and distribute the code.
