# NeuraDepth 👁️

**NeuraDepth** is an advanced, locally-hosted Neuromorphic UI tool that generates simulated depth maps, LiDAR point clouds, wireframes, and meshes directly from standard 2D input images using computer vision.

Built with a lightning-fast React/Vite frontend and a powerful FastAPI Python backend, NeuraDepth allows users to instantly visualize and compare spatial depth approximations of any image, all processed locally without needing cloud APIs.

---

## 🌟 Key Features

*   **Modern Neuromorphic Design**: A breathtaking, warm soft-UI aesthetic featuring deep inset shadows, raised interactive components, and a carefully curated cream and crimson color palette (`#FFFAF3`, `#FFF2DB`, `#FFE5BF`, `#F62440`).
*   **Interactive Comparison Slider**: Instantly compare the original 2D input with the generated depth output using a draggable overlay slider.
*   **Real-time Processing Tuning**: Adjust computer vision parameters via interactive sliders and see results generated on the fly.
*   **Drag & Drop Interface**: Seamlessly upload JPG, PNG, and WEBP files.
*   **Backend Health Monitoring**: Live API connection tracking indicator in the UI.

## 🛠️ Processing Modes

NeuraDepth comes equipped with 5 distinct computer vision generation modes:

1.  **Depth Map**: Generates a classic grayscale z-buffer estimation using luminance, contrast mapping, and edge detection.
2.  **LiDAR**: Simulates a high-density point cloud laser scan, projecting the depth map into a colored heatmap scatter plot.
3.  **Wireframe**: Extracts structural boundaries and prominent edges using Canny detection, generating an interconnected 3D-like wire mesh.
4.  **Mesh**: Simulates a triangulated surface mesh mapped over the detected depth contours.
5.  **Scanner**: Overlays an active horizontal/vertical laser scanner line simulation over the detected geometry.

## 🎛️ Adjustable Parameters

Fine-tune your depth generation with 6 precise sliders:
*   **Scan Density**: Controls the resolution and spacing of simulated LiDAR points and wireframe vertices.
*   **Noise Amount**: Introduces organic sensor noise for more realistic simulations.
*   **Edge Sensitivity**: Adjusts the threshold for edge-detection, preserving hard boundaries vs. smooth slopes.
*   **Depth Contrast**: Amplifies the perceived depth difference between the foreground and background.
*   **Smoothing**: Applies intelligent blurring (Gaussian/Bilateral) to remove harsh artifacts and create organic transitions.
*   **Point Density**: Specific to Mesh and LiDAR modes, increasing the density of the generated spatial points.

---

## 🏗️ Architecture

*   **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Lucide React (Icons).
*   **Backend**: Python 3.9+, FastAPI, Uvicorn, OpenCV (`opencv-python`), Numpy, Python-Multipart.
*   **Communication**: `multipart/form-data` REST API.

---

## 🚀 Getting Started

To run NeuraDepth locally, you will need to start both the backend server and the frontend development server.

### Prerequisites
*   Node.js (v16+)
*   Python (3.9+)
*   Git

### 1. Start the Backend (FastAPI + OpenCV)
The backend handles all heavy image processing algorithms.

```bash
cd backend
python -m venv venv

# Activate Virtual Environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server (runs on http://127.0.0.1:8000)
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Start the Frontend (React + Vite)
The frontend serves the Neuromorphic UI.

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
│   │   ├── main.py         # FastAPI endpoints and CORS setup
│   │   ├── processor.py    # Core OpenCV processing logic
│   │   └── __init__.py
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Main UI, state management, Comparison Slider
│   │   ├── main.tsx        # React root
│   │   └── styles.css      # Custom Neuromorphic Tailwind configuration
│   ├── index.html
│   ├── package.json        # Node dependencies
│   ├── tailwind.config.js  # Tailwind classes
│   └── vite.config.ts      # Vite bundler config
└── README.md
```

## 📝 License
MIT License. Feel free to use and modify the code.
