from dataclasses import dataclass

import cv2
import numpy as np
VALID_MODES = {"depth", "lidar", "wireframe", "mesh", "scanner"}


@dataclass(frozen=True)
class ProcessingParams:
    mode: str
    scan_density: float
    noise_level: float
    edge_sensitivity: float
    depth_contrast: float
    smoothing: float
    point_density: float


@dataclass
class ScaleFeatures:
    """Features extracted at a single scale in the image pyramid"""
    scale: float  # Scale factor relative to original (1.0, 0.5, 0.25)
    edges: np.ndarray  # Binary edge map (H x W, uint8)
    gradients: np.ndarray  # Gradient magnitude (H x W, float32, [0, 1])
    texture: np.ndarray  # Texture strength (H x W, float32, [0, 1])
    distance_depth: np.ndarray  # Distance transform depth (H x W, float32, [0, 1])
    intensity_depth: np.ndarray  # Intensity-based depth (H x W, float32, [0, 1])
    confidence: np.ndarray  # Feature confidence (H x W, float32, [0, 1])


@dataclass
class AdaptiveWeights:
    """Spatially-varying weights for depth feature fusion"""
    distance_weight: np.ndarray  # Weight for distance transform (H x W, float32)
    intensity_weight: np.ndarray  # Weight for intensity depth (H x W, float32)
    gradient_weight: np.ndarray  # Weight for gradient info (H x W, float32)
    texture_weight: np.ndarray  # Weight for texture info (H x W, float32)


@dataclass
class DepthEstimationResult:
    """Complete result from enhanced depth estimation"""
    depth: np.ndarray  # Final depth map (H x W, float32, [0, 1])
    edges: np.ndarray  # Binary edge map (H x W, uint8)
    gradients: np.ndarray  # Gradient magnitude (H x W, float32, [0, 1])
    confidence: np.ndarray  # Depth confidence (H x W, float32, [0, 1])


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


def _fit_image(image: np.ndarray, max_size: int = 1320) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def _extract_scale_features(
    gray: np.ndarray,
    scale: float,
    params: ProcessingParams
) -> ScaleFeatures:
    """
    Extract all depth-relevant features at a specific scale.
    
    Args:
        gray: Grayscale image with CLAHE applied (H x W, uint8)
        scale: Scale factor (1.0 = full res, 0.5 = half res, 0.25 = quarter res)
        params: Processing parameters
        
    Returns:
        ScaleFeatures object with all extracted features
    """
    # Step 1: Resize to target scale
    h, w = gray.shape
    target_h, target_w = int(h * scale), int(w * scale)
    
    if scale < 1.0:
        scaled = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        scaled = gray
    
    # Step 2: Edge detection with adaptive thresholds
    sensitivity = _unit(params.edge_sensitivity)
    low_threshold = int(18 + (1.0 - sensitivity) * 92)
    high_threshold = int(low_threshold + 70 + (1.0 - sensitivity) * 82)
    edges = cv2.Canny(scaled, low_threshold, high_threshold)
    
    # Step 3: Gradient computation (Sobel)
    sobel_x = cv2.Sobel(scaled, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(scaled, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    gradients = _normalize(gradient_magnitude)
    
    # Step 4: Texture analysis (Laplacian)
    laplacian = cv2.Laplacian(scaled, cv2.CV_32F, ksize=3)
    texture = _normalize(np.abs(laplacian))
    
    # Step 5: Distance transform depth cue
    inverted_edges = cv2.bitwise_not(edges)
    distance_map = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)
    distance_depth = _normalize(distance_map)
    
    # Step 6: Intensity-based depth cue (darker = closer)
    intensity_depth = 1.0 - (scaled.astype(np.float32) / 255.0)
    
    # Step 7: Compute feature confidence
    # Confidence is high where multiple cues agree
    edge_confidence = edges.astype(np.float32) / 255.0
    gradient_confidence = gradients
    texture_confidence = texture
    
    # Combine confidences (high when features are strong and consistent)
    confidence = (
        edge_confidence * 0.4 +
        gradient_confidence * 0.3 +
        texture_confidence * 0.3
    )
    
    # Scale-based confidence adjustment (full resolution = highest confidence)
    confidence = confidence * scale
    
    return ScaleFeatures(
        scale=scale,
        edges=edges,
        gradients=gradients,
        texture=texture,
        distance_depth=distance_depth,
        intensity_depth=intensity_depth,
        confidence=confidence
    )


def _compute_adaptive_weights(
    features: ScaleFeatures,
    params: ProcessingParams
) -> AdaptiveWeights:
    """
    Compute spatially-varying weights for depth feature fusion.
    
    Weights are adjusted based on:
    - Edge proximity (edges favor distance transform)
    - Texture richness (texture areas favor gradient/texture cues)
    - Illumination uniformity (uniform areas favor intensity)
    
    Args:
        features: Extracted features at current scale
        params: Processing parameters
        
    Returns:
        AdaptiveWeights with spatially-varying weight maps
    """
    h, w = features.gradients.shape
    
    # Step 1: Compute edge proximity map
    # Dilate edges to create proximity zones
    edge_dilated = cv2.dilate(features.edges, np.ones((15, 15), np.uint8), iterations=1)
    edge_proximity = edge_dilated.astype(np.float32) / 255.0
    
    # Step 2: Compute texture richness
    # Areas with high texture variance
    texture_richness = features.texture
    
    # Step 3: Compute illumination uniformity
    # Low gradient = uniform illumination
    illumination_uniformity = 1.0 - features.gradients
    
    # Step 4: Assign base weights based on local characteristics
    # Distance transform works best near edges
    distance_weight = edge_proximity * 0.5 + 0.1
    
    # Intensity depth works best in uniform illumination areas
    intensity_weight = illumination_uniformity * 0.5 + 0.1
    
    # Gradient information works best in textured regions
    gradient_weight = texture_richness * 0.5 + 0.1
    
    # Texture analysis works best where texture exists
    texture_weight = texture_richness * 0.4 + 0.1
    
    # Step 5: Normalize weights to sum to 1.0 at each pixel
    weight_sum = (
        distance_weight + intensity_weight + 
        gradient_weight + texture_weight
    )
    
    # Avoid division by zero (should not happen with base weights)
    weight_sum = np.maximum(weight_sum, 1e-6)
    
    distance_weight = distance_weight / weight_sum
    intensity_weight = intensity_weight / weight_sum
    gradient_weight = gradient_weight / weight_sum
    texture_weight = texture_weight / weight_sum
    
    # ASSERT: ∀(i,j): |sum_of_weights[i,j] - 1.0| < 1e-5
    return AdaptiveWeights(
        distance_weight=distance_weight,
        intensity_weight=intensity_weight,
        gradient_weight=gradient_weight,
        texture_weight=texture_weight
    )


def _fuse_multiscale_depth(
    scale_depths: list[np.ndarray],
    scale_confidences: list[np.ndarray],
    target_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuse depth maps from multiple scales using confidence weighting.
    
    Args:
        scale_depths: List of depth maps at different scales
        scale_confidences: List of confidence maps at different scales
        target_shape: Target output shape (height, width)
        
    Returns:
        fused_depth: Combined depth map at target resolution (H x W, float32, [0, 1])
        fused_confidence: Combined confidence map (H x W, float32, [0, 1])
    """
    target_h, target_w = target_shape
    n_scales = len(scale_depths)
    
    # Initialize accumulators
    accumulated_depth = np.zeros((target_h, target_w), dtype=np.float32)
    accumulated_weight = np.zeros((target_h, target_w), dtype=np.float32)
    
    # LOOP INVARIANT: accumulated_depth ∈ [0, sum of weights so far]
    for i in range(n_scales):
        depth = scale_depths[i]
        confidence = scale_confidences[i]
        
        # Step 1: Resize depth and confidence to target resolution
        if depth.shape != target_shape:
            depth_resized = cv2.resize(
                depth, (target_w, target_h), 
                interpolation=cv2.INTER_LINEAR
            )
            confidence_resized = cv2.resize(
                confidence, (target_w, target_h),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            depth_resized = depth
            confidence_resized = confidence
        
        # Step 2: Use confidence as fusion weight
        # Add small epsilon to avoid zero weights
        weight = confidence_resized + 1e-6
        
        # Step 3: Accumulate weighted depth
        accumulated_depth += depth_resized * weight
        accumulated_weight += weight
        
        # ASSERT: accumulated_depth[i,j] ≥ 0 for all i,j
    
    # Step 4: Normalize by total weight
    fused_depth = accumulated_depth / accumulated_weight
    
    # Step 5: Compute fused confidence from weighted combination
    fused_confidence = accumulated_weight / n_scales
    fused_confidence = np.clip(fused_confidence, 0.0, 1.0)
    
    # ASSERT: fused_depth ∈ [0,1] AND fused_confidence ∈ [0,1]
    return fused_depth.astype(np.float32), fused_confidence.astype(np.float32)


def _detect_occlusions(
    depth: np.ndarray,
    edges: np.ndarray,
    gradient_threshold: float = 0.15
) -> np.ndarray:
    """
    Detect occlusion boundaries where depth changes rapidly.
    
    Args:
        depth: Initial depth map (H x W, float32, [0, 1])
        edges: Binary edge map from image (H x W, uint8)
        gradient_threshold: Threshold for depth gradient magnitude (default 0.15)
        
    Returns:
        occlusion_mask: Binary mask (H x W, uint8) where 255 indicates occlusion boundary, 0 no occlusion
    """
    # Step 1: Compute depth gradients using Sobel operators
    depth_grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    depth_grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_gradient = cv2.magnitude(depth_grad_x, depth_grad_y)
    
    # Step 2: Normalize depth gradient magnitude
    depth_gradient = _normalize(depth_gradient)
    
    # Step 3: Threshold depth gradient to find depth discontinuities
    depth_discontinuity = (depth_gradient > gradient_threshold).astype(np.uint8) * 255
    
    # Step 4: Combine with image edges (logical AND)
    # Occlusions occur where both depth changes rapidly AND image has edges
    edge_normalized = edges.astype(np.float32) / 255.0
    depth_disc_normalized = depth_discontinuity.astype(np.float32) / 255.0
    
    # Logical AND with soft weighting
    combined = edge_normalized * depth_disc_normalized
    
    # Step 5: Threshold to binary mask
    occlusion_mask = (combined > 0.3).astype(np.uint8) * 255
    
    # Step 6: Morphological cleanup - apply opening to remove isolated pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_OPEN, kernel)
    
    # Step 7: Dilate to capture boundary regions
    occlusion_mask = cv2.dilate(occlusion_mask, kernel, iterations=1)
    
    return occlusion_mask


def _estimate_depth(image: np.ndarray, params: ProcessingParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)

    sensitivity = _unit(params.edge_sensitivity)
    low = int(18 + (1.0 - sensitivity) * 92)
    high = int(low + 70 + (1.0 - sensitivity) * 82)
    edges = cv2.Canny(clahe, low, high)

    sobel_x = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    gradient = _normalize(gradient)

    laplacian = cv2.Laplacian(clahe, cv2.CV_32F, ksize=3)
    texture = _normalize(np.abs(laplacian))

    contrast = _unit(params.depth_contrast)
    smooth = _unit(params.smoothing)

    # Use multiple scales so the estimate can capture both large structures and fine detail.
    scales = [1.0, 0.75, 0.5, 0.25]
    fused_depth = np.zeros_like(clahe, dtype=np.float32)
    fused_confidence = np.zeros_like(clahe, dtype=np.float32)

    for scale in scales:
        features = _extract_scale_features(clahe, scale, params)
        weights = _compute_adaptive_weights(features, params)

        # Fuse cues with adaptive weights and a confidence term that favors stable regions.
        depth_scale = (
            features.distance_depth * weights.distance_weight
            + features.intensity_depth * weights.intensity_weight
            + (1.0 - features.gradients) * weights.gradient_weight
            + (1.0 - features.texture) * weights.texture_weight
        )
        depth_scale = _normalize(depth_scale)

        confidence = np.clip(
            features.confidence
            + 0.15 * (1.0 - np.abs(depth_scale - 0.5) * 2.0)
            + 0.10 * (1.0 - gradient if scale == 1.0 else 0.0),
            0.0,
            1.0,
        )

        if scale != 1.0:
            depth_scale = cv2.resize(
                depth_scale,
                (clahe.shape[1], clahe.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            confidence = cv2.resize(
                confidence,
                (clahe.shape[1], clahe.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        fused_depth += depth_scale * confidence
        fused_confidence += confidence

    depth = fused_depth / np.maximum(fused_confidence, 1e-6)
    depth = _normalize(depth)

    # Edge-aware cleanup to preserve contours while reducing noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
    depth = cv2.morphologyEx(depth, cv2.MORPH_OPEN, kernel)

    if smooth > 0.03:
        diameter = int(5 + smooth * 12)
        if diameter % 2 == 0:
            diameter += 1
        depth = cv2.bilateralFilter(
            depth.astype(np.float32),
            diameter,
            35 + smooth * 80,
            35 + smooth * 80,
        )
        blur_kernel = int(3 + smooth * 10)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        depth = cv2.GaussianBlur(depth, (blur_kernel, blur_kernel), 0)

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
    cv2.putText(canvas, "DEPTHSCAN SYSTEM", (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "CLASSICAL CV", (width - 134, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
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
