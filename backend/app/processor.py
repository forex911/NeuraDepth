from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

VALID_MODES = {"depth", "lidar", "wireframe", "mesh", "scanner"}

# Global model instance
_depth_estimator = None

def get_depth_estimator():
    global _depth_estimator
    if _depth_estimator is None:
        if pipeline is None:
            raise RuntimeError("transformers library is not installed.")
        # Load the state-of-the-art Depth Anything V2 model
        _depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    return _depth_estimator


@dataclass(frozen=True)
class ProcessingParams:
    mode: str
    scan_density: float
    noise_level: float
    edge_sensitivity: float
    depth_contrast: float
    smoothing: float
    point_density: float


def process_image(raw: bytes, params: ProcessingParams) -> bytes:
    mode = params.mode if params.mode in VALID_MODES else "depth"
    image = _decode_image(raw)
    
    depth, edges, gradients = _estimate_depth(image, params)

    if mode == "depth":
        output = _depth_map(depth)
    elif mode == "lidar":
        output = _lidar_scan(depth, edges, params)
    elif mode == "wireframe":
        output = _wireframe(depth, edges, image, params)
    elif mode == "mesh":
        output = _mesh_scan(depth, gradients, params)
    else:
        output = _scanner_visualization(depth, edges, gradients, params)

    success, encoded = cv2.imencode(".png", output)
    if not success:
        raise ValueError("Could not encode generated scan.")
    return encoded.tobytes()


def _decode_image(raw: bytes) -> np.ndarray:
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read the uploaded image.")
    return image


def _estimate_depth(image: np.ndarray, params: ProcessingParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)

    # 1. Classical Edge & Gradient extraction for rendering styles
    sensitivity = _unit(params.edge_sensitivity)
    low = int(18 + (1.0 - sensitivity) * 92)
    high = int(low + 70 + (1.0 - sensitivity) * 82)
    edges = cv2.Canny(clahe, low, high)

    sobel_x = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    gradient = _normalize(gradient)

    # 2. True AI Monocular Depth Estimation
    estimator = get_depth_estimator()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    
    # Inference
    result = estimator(pil_img)
    
    # 'depth' is typically a PIL Image containing the depth map
    ai_depth_img = result["depth"]
    ai_depth = np.array(ai_depth_img).astype(np.float32)
    
    # Resize AI depth to original resolution
    h, w = gray.shape
    ai_depth = cv2.resize(ai_depth, (w, h), interpolation=cv2.INTER_LINEAR)
    ai_depth = _normalize(ai_depth)

    # 3. Hybrid Approach: Guided Filter / Edge-Aware Sharpening
    # Use the crisp original image to snap the blurry AI edges to exact pixel boundaries
    radius = 8
    eps = 0.01
    try:
        refined_depth = cv2.ximgproc.createGuidedFilter(guide=clahe, radius=radius, eps=eps).filter(ai_depth)
    except AttributeError:
        # Fallback if opencv-contrib is not available
        refined_depth = cv2.bilateralFilter(ai_depth, d=9, sigmaColor=0.1, sigmaSpace=15.0)
    
    refined_depth = _normalize(refined_depth)

    # 4. Apply UI Control Parameters
    contrast = _unit(params.depth_contrast)
    smooth = _unit(params.smoothing)

    depth = refined_depth
    
    if smooth > 0.03:
        blur_kernel = int(3 + smooth * 10)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        depth = cv2.GaussianBlur(depth, (blur_kernel, blur_kernel), 0)

    # Contrast adjustment
    depth = np.clip((depth - 0.5) * (0.75 + contrast * 1.9) + 0.5, 0.0, 1.0)
    
    return depth.astype(np.float32), edges, gradient


def _depth_map(depth: np.ndarray) -> np.ndarray:
    depth_u8 = (depth * 255).astype(np.uint8)
    return cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)


def _lidar_scan(depth: np.ndarray, edges: np.ndarray, params: ProcessingParams) -> np.ndarray:
    height, width = depth.shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    rng = np.random.default_rng(12)

    row_step = max(2, int(13 - _unit(params.scan_density) * 10))
    col_step = max(2, int(13 - _unit(params.point_density) * 10))
    yy, xx = np.mgrid[0:height:row_step, 0:width:col_step]
    sampled_depth = depth[yy, xx]
    keep = rng.random(sampled_depth.shape) < (0.32 + _unit(params.point_density) * 0.62)

    jitter = np.round(rng.normal(0, 1.4 + _unit(params.noise_level) * 3.8, size=xx.shape)).astype(np.int32)
    px = np.clip(xx + jitter, 0, width - 1)
    py = np.clip(yy + jitter // 2, 0, height - 1)

    values = np.clip(sampled_depth * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    canvas[py[keep], px[keep]] = colors[keep, 0]

    edge_points = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    canvas[edge_points > 0] = np.maximum(canvas[edge_points > 0], np.array([70, 210, 255], dtype=np.uint8))

    for y in range(0, height, row_step * 2):
        cv2.line(canvas, (0, y), (width, y), (34, 88, 94), 1, cv2.LINE_AA)

    return _add_noise(canvas, params.noise_level)


def _wireframe(depth: np.ndarray, edges: np.ndarray, image: np.ndarray, params: ProcessingParams) -> np.ndarray:
    height, width = depth.shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    bands = np.floor(depth * (8 + _unit(params.depth_contrast) * 18)).astype(np.uint8)
    contours_image = cv2.convertScaleAbs(bands * 12)
    contours, _ = cv2.findContours(contours_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 18:
            color = (42, 220, 255) if len(contour) > 50 else (66, 140, 190)
            cv2.drawContours(canvas, [contour], -1, color, 1, cv2.LINE_AA)

    edge_glow = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    canvas[edge_glow > 0] = (235, 247, 255)
    shaded = cv2.cvtColor(cv2.convertScaleAbs(depth * 72), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(shaded, 0.45, canvas, 1.0, 0)


def _mesh_scan(depth: np.ndarray, gradients: np.ndarray, params: ProcessingParams) -> np.ndarray:
    height, width = depth.shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    shaded = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    canvas = cv2.addWeighted(canvas, 0.35, shaded, 0.65, 0)

    step = max(10, int(32 - _unit(params.scan_density) * 22))
    points = []
    for y in range(0, height, step):
        row = []
        for x in range(0, width, step):
            offset = int((depth[y, x] - 0.5) * step * 0.65)
            row.append((np.clip(x + offset, 0, width - 1), np.clip(y - offset, 0, height - 1)))
        points.append(row)

    for row_idx in range(len(points) - 1):
        for col_idx in range(len(points[row_idx]) - 1):
            p1 = tuple(map(int, points[row_idx][col_idx]))
            p2 = tuple(map(int, points[row_idx][col_idx + 1]))
            p3 = tuple(map(int, points[row_idx + 1][col_idx]))
            p4 = tuple(map(int, points[row_idx + 1][col_idx + 1]))
            color = (32, 205, 232) if gradients[p1[1], p1[0]] < 0.42 else (230, 245, 255)
            cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
            cv2.line(canvas, p1, p3, color, 1, cv2.LINE_AA)
            cv2.line(canvas, p2, p3, (38, 120, 150), 1, cv2.LINE_AA)
            if (row_idx + col_idx) % 2 == 0:
                cv2.line(canvas, p2, p4, (28, 96, 122), 1, cv2.LINE_AA)

    return _add_noise(canvas, params.noise_level * 0.55)


def _scanner_visualization(
    depth: np.ndarray,
    edges: np.ndarray,
    gradients: np.ndarray,
    params: ProcessingParams,
) -> np.ndarray:
    height, width = depth.shape
    base = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_OCEAN)
    base = cv2.addWeighted(base, 0.72, np.zeros_like(base), 0.28, 0)

    bands = np.sin(depth * np.pi * (10 + _unit(params.depth_contrast) * 22))
    band_mask = bands > 0.9
    base[band_mask] = np.maximum(base[band_mask], np.array([85, 230, 255], dtype=np.uint8))

    line_step = max(3, int(14 - _unit(params.scan_density) * 10))
    for y in range(0, height, line_step):
        alpha = 0.25 + 0.45 * float(np.mean(depth[y : min(y + 2, height), :]))
        color = (int(34 * alpha), int(190 * alpha), int(255 * alpha))
        cv2.line(base, (0, y), (width, y), color, 1, cv2.LINE_AA)

    missing = (gradients > 0.72) & (np.random.default_rng(24).random(depth.shape) < _unit(params.noise_level) * 0.18)
    base[missing] = 0
    base[edges > 0] = (245, 250, 255)

    _draw_overlay(base)
    return _add_noise(base, params.noise_level)


def _draw_overlay(canvas: np.ndarray) -> None:
    height, width = canvas.shape[:2]
    color = (64, 226, 255)
    cv2.rectangle(canvas, (18, 18), (width - 18, height - 18), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (28, 58), (190, 58), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (width - 190, 58), (width - 28, 58), color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "NEURADEPTH SYSTEM", (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "AI MONOCULAR DEPTH", (width - 184, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
    center = (width // 2, height // 2)
    cv2.drawMarker(canvas, center, color, markerType=cv2.MARKER_CROSS, markerSize=30, thickness=1, line_type=cv2.LINE_AA)


def _add_noise(image: np.ndarray, level: float) -> np.ndarray:
    amount = _unit(level)
    if amount <= 0:
        return image
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 4 + amount * 28, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


def _unit(value: float) -> float:
    return float(np.clip(value, 0, 100) / 100.0)
