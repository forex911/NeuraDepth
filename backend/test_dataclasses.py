"""
Unit tests for data models: ScaleFeatures, AdaptiveWeights, and DepthEstimationResult
Tests Requirements: 1.5, 2.5, 3.1, 8.1-8.7
"""
import sys
import numpy as np

try:
    from app.processor import ScaleFeatures, AdaptiveWeights, DepthEstimationResult
    
    print("Testing ScaleFeatures dataclass...")
    # Create sample arrays
    h, w = 100, 100
    edges = np.zeros((h, w), dtype=np.uint8)
    gradients = np.random.rand(h, w).astype(np.float32)
    texture = np.random.rand(h, w).astype(np.float32)
    distance_depth = np.random.rand(h, w).astype(np.float32)
    intensity_depth = np.random.rand(h, w).astype(np.float32)
    confidence = np.random.rand(h, w).astype(np.float32)
    
    # Create ScaleFeatures instance
    features = ScaleFeatures(
        scale=1.0,
        edges=edges,
        gradients=gradients,
        texture=texture,
        distance_depth=distance_depth,
        intensity_depth=intensity_depth,
        confidence=confidence
    )
    
    # Verify fields
    assert features.scale == 1.0, "ScaleFeatures scale field failed"
    assert features.edges.shape == (h, w), "ScaleFeatures edges shape mismatch"
    assert features.gradients.shape == (h, w), "ScaleFeatures gradients shape mismatch"
    assert features.texture.shape == (h, w), "ScaleFeatures texture shape mismatch"
    assert features.distance_depth.shape == (h, w), "ScaleFeatures distance_depth shape mismatch"
    assert features.intensity_depth.shape == (h, w), "ScaleFeatures intensity_depth shape mismatch"
    assert features.confidence.shape == (h, w), "ScaleFeatures confidence shape mismatch"
    print("✓ ScaleFeatures tests passed")
    
    print("\nTesting AdaptiveWeights dataclass...")
    # Create weight arrays
    distance_weight = np.random.rand(h, w).astype(np.float32)
    intensity_weight = np.random.rand(h, w).astype(np.float32)
    gradient_weight = np.random.rand(h, w).astype(np.float32)
    texture_weight = np.random.rand(h, w).astype(np.float32)
    
    # Create AdaptiveWeights instance
    weights = AdaptiveWeights(
        distance_weight=distance_weight,
        intensity_weight=intensity_weight,
        gradient_weight=gradient_weight,
        texture_weight=texture_weight
    )
    
    # Verify fields
    assert weights.distance_weight.shape == (h, w), "AdaptiveWeights distance_weight shape mismatch"
    assert weights.intensity_weight.shape == (h, w), "AdaptiveWeights intensity_weight shape mismatch"
    assert weights.gradient_weight.shape == (h, w), "AdaptiveWeights gradient_weight shape mismatch"
    assert weights.texture_weight.shape == (h, w), "AdaptiveWeights texture_weight shape mismatch"
    print("✓ AdaptiveWeights tests passed")
    
    print("\nTesting DepthEstimationResult dataclass...")
    # Create result arrays
    depth = np.random.rand(h, w).astype(np.float32)
    result_edges = np.zeros((h, w), dtype=np.uint8)
    result_gradients = np.random.rand(h, w).astype(np.float32)
    result_confidence = np.random.rand(h, w).astype(np.float32)
    
    # Create DepthEstimationResult instance
    result = DepthEstimationResult(
        depth=depth,
        edges=result_edges,
        gradients=result_gradients,
        confidence=result_confidence
    )
    
    # Verify fields
    assert result.depth.shape == (h, w), "DepthEstimationResult depth shape mismatch"
    assert result.edges.shape == (h, w), "DepthEstimationResult edges shape mismatch"
    assert result.gradients.shape == (h, w), "DepthEstimationResult gradients shape mismatch"
    assert result.confidence.shape == (h, w), "DepthEstimationResult confidence shape mismatch"
    print("✓ DepthEstimationResult tests passed")
    
    print("\n" + "="*50)
    print("SUCCESS: All dataclass tests passed!")
    print("="*50)
    
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
