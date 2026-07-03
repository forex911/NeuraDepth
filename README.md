# NeuraDepth

<div align="center">

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg)
![React](https://img.shields.io/badge/React-18-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C.svg)

### AI-powered depth estimation, 3D reconstruction and visualization platform.

Generate accurate depth maps, point clouds and textured meshes from ordinary images using state-of-the-art deep learning.

</div>

---

# Overview

NeuraDepth is a modern AI-powered desktop web application designed for converting ordinary RGB images into high-quality depth maps, interactive point clouds, and textured 3D meshes.

The application integrates modern deep learning models with GPU acceleration, an interactive React interface, and real-time FastAPI services to deliver professional-grade depth estimation completely offline.

Unlike cloud-based services, every computation is performed locally, ensuring:

- Privacy
- Zero API costs
- Offline usage
- Full GPU utilization
- High performance

---

# Preview

<p align="center">
<img src="frontend/public/screenshot.png" width="100%">
</p>

---

# Features

## AI Depth Estimation

• MiDaS deep learning models

• Monocular depth prediction

• High precision depth reconstruction

• GPU acceleration

• CPU fallback

---

## Interactive 3D Visualization

- Three.js Rendering
- Point Cloud Viewer
- Mesh Viewer
- Orbit Controls
- Zoom
- Rotation
- Wireframe Rendering

---

## Processing Modes

| Mode | Description |
|------|-------------|
| Depth | AI generated grayscale depth |
| LiDAR | Simulated LiDAR scan |
| Mesh | Polygon mesh generation |
| Wireframe | Surface topology |
| Scanner | Futuristic scan visualization |
| Photogrammetry | Reconstruction artifacts |
| Topographic | Terrain contours |

---

## Export Formats

- PNG (16-bit)
- OBJ
- PLY

Compatible with

- Blender
- Unreal Engine
- Unity
- Maya
- MeshLab

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

      PyTorch + OpenCV + NumPy

                    │

             MiDaS Model

                    │

      Depth Generation Engine

                    │

    Mesh / Point Cloud Generator

                    │

       3D Visualization Engine
```

---

# Technology Stack

## Frontend

- React 18
- TypeScript
- Vite
- TailwindCSS
- Framer Motion
- Three.js
- React Three Fiber
- React Drei
- Lucide Icons

---

## Backend

- FastAPI
- Python
- Uvicorn
- PyTorch
- OpenCV
- NumPy
- Pillow

---

## AI Models

- MiDaS

---

## Communication

- REST API
- WebSockets

---

# Project Structure

```
NeuraDepth
│
├── backend
│   ├── app
│   ├── services
│   ├── models
│   ├── processor.py
│   └── main.py
│
├── frontend
│   ├── src
│   ├── assets
│   ├── components
│   ├── styles
│   └── App.tsx
│
├── Dockerfile
├── docker-compose.yml
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

## Pull

```bash
docker pull forex911/neuradepth
```

---

## Run

```bash
docker run -p 8000:8000 forex911/neuradepth
```

---

## Build

```bash
docker build -t neuradepth .
```

---

# API

## Upload Image

```
POST /api/upload
```

Returns

```json
{
  "job_id":"..."
}
```

---

## Job Status

```
GET /api/jobs/{id}
```

---

## WebSocket

```
ws://localhost:8000/ws
```

Provides

- Progress
- Completion
- Errors
- Notifications

---

# Performance

| Hardware | Processing Time |
|------------|----------------|
| RTX 3050 | ~1-2 seconds |
| GTX 1650 | ~2-4 seconds |
| CPU | ~8-20 seconds |

---

# Screenshots

## Upload

<img src="docs/upload.png">

---

## Depth Map

<img src="docs/depth.png">

---

## Mesh

<img src="docs/mesh.png">

---

## Point Cloud

<img src="docs/pointcloud.png">

---

# Roadmap

- [x] MiDaS Integration
- [x] Point Cloud
- [x] OBJ Export
- [x] Mesh Generation
- [x] LiDAR Mode
- [x] Docker Support
- [ ] ONNX Runtime
- [ ] TensorRT Acceleration
- [ ] Video Depth Estimation
- [ ] Batch Processing
- [ ] Multi-GPU Support

---

# Why NeuraDepth?

✅ Runs Completely Offline

✅ GPU Accelerated

✅ No Cloud Dependencies

✅ Open Source

✅ Cross Platform

✅ Docker Ready

✅ Modern React Interface

✅ Interactive 3D Rendering

---

# Contributing

Contributions are welcome.

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature/my-feature
```

3. Commit

```bash
git commit -m "Add feature"
```

4. Push

```bash
git push origin feature/my-feature
```

5. Open a Pull Request

---

# License

This project is licensed under the MIT License.

---

# Author

**Forex911**

AI & Full Stack Developer

GitHub

https://github.com/forex-911

Docker Hub

https://hub.docker.com/r/forex911/neuradepth

---

If NeuraDepth helped your work, consider giving the repository a ⭐.
