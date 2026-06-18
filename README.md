# NeuraDepth

NeuraDepth is a modern, locally-hosted Neuromorphic UI tool that generates depth maps, LiDAR point clouds, wireframes, and meshes from 2D input images.

It features a sleek React/Vite frontend built with an interactive Before/After comparison slider, powered by a FastAPI Python backend handling the image processing with OpenCV.

## Features
- **Modern Neuromorphic UI**: A stunning, warm aesthetic with smooth soft shadows.
- **Multiple Output Modes**: Depth Map, LiDAR, Wireframe, Mesh, and direct Scanner views.
- **Interactive Tuning**: Real-time adjustable sliders for Scan Density, Noise Amount, Edge Sensitivity, Depth Contrast, Smoothing, and Point Density.
- **Comparison Slider**: Intuitive drag-slider to instantly compare your input image with the generated depth output.
- **Local Processing**: Fully local, fast Python backend.

## Project Structure
- `/frontend`: React frontend using Vite, Tailwind CSS, and Lucide React.
- `/backend`: FastAPI Python backend for image processing.

## Getting Started

### Prerequisites
- Node.js & npm
- Python 3.9+

### Running the Backend
```bash
cd backend
python -m venv venv
# On Windows use: venv\Scripts\activate
# On Mac/Linux use: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Running the Frontend
```bash
cd frontend
npm install
npm run dev
```

Visit the local Vite server (typically `http://localhost:5173`) to use the application!
