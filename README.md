# NeuraDepth

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-000000?style=for-the-badge&logo=threedotjs&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### AI-Powered Depth Estimation, 3D Mesh Generation & Interactive Visualization

Generate professional-quality depth maps, point clouds, and textured meshes from ordinary images using modern deep learning—completely offline.

</div>

---

# Preview

<p align="center">
<img src="docs/images/hero.png" width="100%">
</p>

---

# Overview

NeuraDepth is a professional desktop web application that converts ordinary RGB images into accurate depth maps and interactive 3D reconstructions using deep learning.

Built with a React frontend and a FastAPI backend powered by PyTorch, the application performs all inference locally without requiring cloud services or external APIs.

The project is designed for developers, researchers, students, artists, and engineers who require fast, privacy-focused depth estimation and visualization.

---

# Features

## AI Depth Estimation

- MiDaS-based monocular depth estimation
- High-quality depth reconstruction
- GPU acceleration via PyTorch
- CPU fallback support
- Local inference

---

## Interactive 3D Viewer

- Solid mesh rendering
- Point cloud rendering
- Wireframe rendering
- Orbit controls
- Zoom
- Pan
- Automatic centering
- Real-time rendering

---

## Compare Mode

Compare the original image against the generated depth map using an interactive slider.

<p align="center">
<img src="docs/images/compare-mode.png" width="90%">
</p>

---

## Professional Processing Modes

| Mode | Description |
|------|-------------|
| Depth Map | AI-generated grayscale depth estimation |
| LiDAR | Simulated LiDAR point cloud |
| Mesh | Polygon mesh reconstruction |
| Wireframe | Structural topology visualization |
| Scanner | Futuristic scan visualization |
| Photogrammetry | Reconstruction artifact simulation |
| Topographic | Terrain contour generation |

---

## Interactive Controls

- Scan Density
- Point Density
- Edge Sensitivity
- Noise Amount
- Depth Contrast
- Smoothing
- Mesh Refresh
- Compare View
- 3D View Toggle

---

## Professional Export Formats

Export generated data for professional 3D software.

Supported formats:

- 16-bit PNG
- OBJ Mesh
- PLY Point Cloud

Compatible with

- Blender
- Unreal Engine
- Unity
- Maya
- MeshLab

---

# Screenshots

## Upload Workspace

<p align="center">
<img src="docs/images/upload-workspace.png" width="95%">
</p>

---

## Compare Mode

<p align="center">
<img src="docs/images/compare-mode.png" width="95%">
</p>

---

## Interactive Mesh Viewer

<p align="center">
<img src="docs/images/mesh-view.png" width="95%">
</p>

---

## Depth Visualization

<p align="center">
<img src="docs/images/depth-view.png" width="95%">
</p>

---

# Architecture

```
                User

                  │

          React + TypeScript

                  │

        REST API + WebSockets

                  │

              FastAPI

                  │

      OpenCV + NumPy + PyTorch

                  │

            MiDaS AI Model

                  │

        Depth Reconstruction

                  │

     Mesh / Point Cloud Engine

                  │

      Interactive Three.js Viewer
```

---

# Technology Stack

## Frontend

- React 18
- TypeScript
- Vite
- TailwindCSS
- Three.js
- React Three Fiber
- React Drei
- Framer Motion
- Lucide React

## Backend

- FastAPI
- Python
- Uvicorn
- PyTorch
- OpenCV
- NumPy
- Pillow

## AI

- MiDaS

## Communication

- REST API
- WebSockets

---

# Project Structure

```text
NeuraDepth/
├── backend/
├── frontend/
├── docs/
│   └── images/
├── Dockerfile
├── docker-compose.yml
├── start.bat
├── README.md
└── LICENSE
```

---

# Installation

## Clone

```bash
git clone https://github.com/forex-911/NeuraDepth.git

cd NeuraDepth
```

---

## Backend

```bash
cd backend

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload
```

---

## Frontend

```bash
cd frontend

npm install

npm run dev
```

---

# Docker

Pull

```bash
docker pull forex911/neuradepth
```

Run

```bash
docker run -p 8000:8000 forex911/neuradepth
```

Build locally

```bash
docker build -t neuradepth .
```

---

# Performance

| Hardware | Estimated Processing Time |
|------------|--------------------------:|
| RTX 3050 | 1–2 s |
| RTX 3060 | <1 s |
| GTX 1650 | 2–4 s |
| CPU | 8–20 s |

---

# Roadmap

- [x] AI Depth Estimation
- [x] Interactive 3D Viewer
- [x] Compare Mode
- [x] Mesh Generation
- [x] OBJ Export
- [x] PLY Export
- [x] Docker Support
- [ ] ONNX Runtime
- [ ] TensorRT Acceleration
- [ ] Video Depth Estimation
- [ ] Batch Processing
- [ ] Multi-image Reconstruction

---

# Why NeuraDepth?

- Fully offline execution
- No cloud dependency
- GPU accelerated
- Interactive 3D visualization
- Modern React interface
- Production-ready FastAPI backend
- Docker support
- Open-source architecture

---

# Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

# License

This project is licensed under the MIT License.

---

# Author

**Forex911**

GitHub: https://github.com/forex911

Docker Hub: https://hub.docker.com/r/forex911/neuradepth

---

If you find NeuraDepth useful, consider giving the repository a ⭐ to support future development.
