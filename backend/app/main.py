from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .processor import ProcessingParams, process_image


app = FastAPI(
    title="DepthScan API",
    description="Local classical computer vision depth scan generator.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        png = process_image(raw, params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=png, media_type="image/png")
