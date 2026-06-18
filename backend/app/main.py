import asyncio
import logging
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .processor import ProcessingParams, process_image

# Configure basic logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="DepthForge API",
    description="Professional-grade local AI 3D depth generator.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Websocket Connections
active_connections = set()

@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        active_connections.remove(websocket)

async def broadcast_progress(message: str):
    """Send progress updates to the frontend via WebSockets"""
    for connection in list(active_connections):
        try:
            await connection.send_text(message)
        except:
            active_connections.remove(connection)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready"}

@app.post("/scan")
async def scan(
    image: UploadFile = File(...),
    mode: str = Form("depth"),
    scan_density: float = Form(58),
    noise_level: float = Form(14),
    edge_sensitivity: float = Form(46),
    depth_contrast: float = Form(62),
    smoothing: float = Form(42),
    point_density: float = Form(50),
) -> Response:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WEBP image.")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    params = ProcessingParams(
        mode=mode,
        scan_density=scan_density,
        noise_level=noise_level,
        edge_sensitivity=edge_sensitivity,
        depth_contrast=depth_contrast,
        smoothing=smoothing,
        point_density=point_density,
    )

    try:
        # Notify the UI that processing has started
        await broadcast_progress("Initializing Depth Engine (PyTorch FP16)...")
        
        # Offload the heavy AI computation to a thread to keep the server responsive
        result_bytes, media_type, ext = await asyncio.to_thread(process_image, raw, params)
        
        await broadcast_progress("Depth scan generated successfully.")
    except Exception as exc:
        await broadcast_progress(f"Generation Error: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=result_bytes, 
        media_type=media_type,
        headers={"X-File-Extension": ext}
    )
