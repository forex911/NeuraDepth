"""
Unit tests for _compute_adaptive_weights() function
Tests Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""
import sys
import numpy as np

try:
    from app.processor import (
        _compute_adaptive_weights, 
        ScaleFeatures, 
        AdaptiveWeights,
        ProcessingParams
    )
    
    print("Testing _compute_adaptive_weights() function...")
    print("="*60)
    
    # Test 1: Basic functionality - weights sum to 1.0
    print("\nTest 1: Weights sum to 1.0 at every pixel")
    h, w = 100, 100
    edges = np.zeros((h, w), dtype=np.uint8)
    gradients = np.random.rand(h, w).astype(np.float32)
    texture = np.random.rand(h, w).astype(np.float32)
    distance_depth = np.random.rand(h, w).astype(np.float32)
    intensity_depth = np.random.rand(h, w).astype(np.float32)
    confidence = np.random.rand(h, w).astype(np.float32)
    
    features = ScaleFeatures(
        scale=1.0,
        edges=edges,
        gradients=gradients,
        texture=texture,
        distance_depth=distance_depth,
        intensity_depth=intensity_depth,
        confidence=confidence
    )
    
    params = ProcessingParams(
        mode="depth",
        scan_density=50.0,
        noise_level=10.0,
        edge_sensitivity=50.0,
        depth_contrast=50.0,
        smoothing=50.0,
        point_density=50.0
    )
    
    weights = _compute_adaptive_weights(features, params)
    
    # Check weights sum to 1.0
    weight_sum = (
        weights.distance_weight + weights.intensity_weight +
        weights.gradient_weight + weights.texture_weight
    )
    
    assert np.allclose(weight_sum, 1.0, atol=1e-5), f"Weights don't sum to 1.0: max diff = {np.max(np.abs(weight_sum - 1.0))}"
    print(f"✓ All weights sum to 1.0 (max deviation: {np.max(np.abs(weight_sum - 1.0)):.2e})")
    
    # Test 2: All weights are non-negative
    print("\nTest 2: All weights are non-negative")
    assert np.all(weights.distance_weight >= 0), "distance_weight has negative values"
    assert np.all(weights.intensity_weight >= 0), "intensity_weight has negative values"
    assert np.all(weights.gradient_weight >= 0), "gradient_weight has negative values"
    assert np.all(weights.texture_weight >= 0), "texture_weight has negative values"
    print("✓ All weights are non-negative")
    
    # Test 3: Output shape matches input shape
    print("\nTest 3: Output shape matches input shape")
    assert weights.distance_weight.shape == (h, w), "distance_weight shape mismatch"
    assert weights.intensity_weight.shape == (h, w), "intensity_weight shape mismatch"
    assert weights.gradient_weight.shape == (h, w), "gradient_weight shape mismatch"
    assert weights.texture_weight.shape == (h, w), "texture_weight shape mismatch"
    print(f"✓ All weight arrays have correct shape {(h, w)}")
    
    # Test 4: Edge proximity increases distance weight (Requirement 2.2)
    print("\nTest 4: Edge proximity increases distance weight")
    # Create features with strong edges
    edges_strong = np.ones((h, w), dtype=np.uint8) * 255
    edges_strong[40:60, 40:60] = 0  # Central region with no edges
    
    features_edges = ScaleFeatures(
        scale=1.0,
        edges=edges_strong,
        gradients=np.ones((h, w), dtype=np.float32) * 0.1,  # Low gradients
        texture=np.ones((h, w), dtype=np.float32) * 0.1,    # Low texture
        distance_depth=np.random.rand(h, w).astype(np.float32),
        intensity_depth=np.random.rand(h, w).astype(np.float32),
        confidence=np.random.rand(h, w).astype(np.float32)
    )
    
    weights_edges = _compute_adaptive_weights(features_edges, params)
    
    # Near edges should have higher distance weight
    mean_distance_weight_edges = np.mean(weights_edges.distance_weight[:30, :])
    mean_distance_weight_center = np.mean(weights_edges.distance_weight[45:55, 45:55])
    
    assert mean_distance_weight_edges > mean_distance_weight_center, \
        f"Edge regions should have higher distance weight: edge={mean_distance_weight_edges:.3f}, center={mean_distance_weight_center:.3f}"
    print(f"✓ Edge regions have higher distance weight: edge={mean_distance_weight_edges:.3f} > center={mean_distance_weight_center:.3f}")
    
    # Test 5: Texture richness increases gradient and texture weights (Requirement 2.3)
    print("\nTest 5: Texture richness increases gradient and texture weights")
    # Create features with high texture
    features_textured = ScaleFeatures(
        scale=1.0,
        edges=np.zeros((h, w), dtype=np.uint8),  # No edges
        gradients=np.ones((h, w), dtype=np.float32) * 0.2,
        texture=np.ones((h, w), dtype=np.float32) * 0.9,  # High texture
        distance_depth=np.random.rand(h, w).astype(np.float32),
        intensity_depth=np.random.rand(h, w).astype(np.float32),
        confidence=np.random.rand(h, w).astype(np.float32)
    )
    
    features_smooth = ScaleFeatures(
        scale=1.0,
        edges=np.zeros((h, w), dtype=np.uint8),  # No edges
        gradients=np.ones((h, w), dtype=np.float32) * 0.2,
        texture=np.ones((h, w), dtype=np.float32) * 0.1,  # Low texture
        distance_depth=np.random.rand(h, w).astype(np.float32),
        intensity_depth=np.random.rand(h, w).astype(np.float32),
        confidence=np.random.rand(h, w).astype(np.float32)
    )
    
    weights_textured = _compute_adaptive_weights(features_textured, params)
    weights_smooth = _compute_adaptive_weights(features_smooth, params)
    
    # High texture should increase gradient and texture weights
    mean_gradient_weight_textured = np.mean(weights_textured.gradient_weight)
    mean_gradient_weight_smooth = np.mean(weights_smooth.gradient_weight)
    
    mean_texture_weight_textured = np.mean(weights_textured.texture_weight)
    mean_texture_weight_smooth = np.mean(weights_smooth.texture_weight)
    
    assert mean_gradient_weight_textured > mean_gradient_weight_smooth, \
        f"Textured regions should have higher gradient weight: textured={mean_gradient_weight_textured:.3f}, smooth={mean_gradient_weight_smooth:.3f}"
    assert mean_texture_weight_textured > mean_texture_weight_smooth, \
        f"Textured regions should have higher texture weight: textured={mean_texture_weight_textured:.3f}, smooth={mean_texture_weight_smooth:.3f}"
    
    print(f"✓ Textured regions have higher gradient weight: {mean_gradient_weight_textured:.3f} > {mean_gradient_weight_smooth:.3f}")
    print(f"✓ Textured regions have higher texture weight: {mean_texture_weight_textured:.3f} > {mean_texture_weight_smooth:.3f}")
    
    # Test 6: Uniform illumination increases intensity weight (Requirement 2.4)
    print("\nTest 6: Uniform illumination increases intensity weight")
    # Create features with uniform illumination (low gradients)
    features_uniform = ScaleFeatures(
        scale=1.0,
        edges=np.zeros((h, w), dtype=np.uint8),
        gradients=np.ones((h, w), dtype=np.float32) * 0.1,  # Low gradients = uniform
        texture=np.ones((h, w), dtype=np.float32) * 0.1,    # Low texture
        distance_depth=np.random.rand(h, w).astype(np.float32),
        intensity_depth=np.random.rand(h, w).astype(np.float32),
        confidence=np.random.rand(h, w).astype(np.float32)
    )
    
    features_varied = ScaleFeatures(
        scale=1.0,
        edges=np.zeros((h, w), dtype=np.uint8),
        gradients=np.ones((h, w), dtype=np.float32) * 0.9,  # High gradients = varied
        texture=np.ones((h, w), dtype=np.float32) * 0.1,
        distance_depth=np.random.rand(h, w).astype(np.float32),
        intensity_depth=np.random.rand(h, w).astype(np.float32),
        confidence=np.random.rand(h, w).astype(np.float32)
    )
    
    weights_uniform = _compute_adaptive_weights(features_uniform, params)
    weights_varied = _compute_adaptive_weights(features_varied, params)
    
    mean_intensity_weight_uniform = np.mean(weights_uniform.intensity_weight)
    mean_intensity_weight_varied = np.mean(weights_varied.intensity_weight)
    
    assert mean_intensity_weight_uniform > mean_intensity_weight_varied, \
        f"Uniform illumination should have higher intensity weight: uniform={mean_intensity_weight_uniform:.3f}, varied={mean_intensity_weight_varied:.3f}"
    
    print(f"✓ Uniform illumination has higher intensity weight: {mean_intensity_weight_uniform:.3f} > {mean_intensity_weight_varied:.3f}")
    
    # Test 7: Different image sizes
    print("\nTest 7: Function works with different image sizes")
    for size in [(50, 50), (200, 150), (512, 512)]:
        h_test, w_test = size
        features_test = ScaleFeatures(
            scale=1.0,
            edges=np.zeros(size, dtype=np.uint8),
            gradients=np.random.rand(*size).astype(np.float32),
            texture=np.random.rand(*size).astype(np.float32),
            distance_depth=np.random.rand(*size).astype(np.float32),
            intensity_depth=np.random.rand(*size).astype(np.float32),
            confidence=np.random.rand(*size).astype(np.float32)
        )
        
        weights_test = _compute_adaptive_weights(features_test, params)
        
        # Check weights sum to 1.0
        weight_sum_test = (
            weights_test.distance_weight + weights_test.intensity_weight +
            weights_test.gradient_weight + weights_test.texture_weight
        )
        
        assert np.allclose(weight_sum_test, 1.0, atol=1e-5), \
            f"Size {size}: Weights don't sum to 1.0"
        assert weights_test.distance_weight.shape == size, \
            f"Size {size}: Output shape mismatch"
    
    print(f"✓ Function works correctly with image sizes: (50x50), (200x150), (512x512)")
    
    # Test 8: No NaN or Inf in outputs
    print("\nTest 8: No NaN or Inf values in outputs")
    assert np.all(np.isfinite(weights.distance_weight)), "distance_weight has NaN/Inf"
    assert np.all(np.isfinite(weights.intensity_weight)), "intensity_weight has NaN/Inf"
    assert np.all(np.isfinite(weights.gradient_weight)), "gradient_weight has NaN/Inf"
    assert np.all(np.isfinite(weights.texture_weight)), "texture_weight has NaN/Inf"
    print("✓ All outputs contain only finite values")
    
    # Test 9: Weights are between 0 and 1
    print("\nTest 9: All weights are in [0, 1] range")
    assert np.all(weights.distance_weight <= 1.0), "distance_weight exceeds 1.0"
    assert np.all(weights.intensity_weight <= 1.0), "intensity_weight exceeds 1.0"
    assert np.all(weights.gradient_weight <= 1.0), "gradient_weight exceeds 1.0"
    assert np.all(weights.texture_weight <= 1.0), "texture_weight exceeds 1.0"
    print("✓ All weights are in [0, 1] range")
    
    print("\n" + "="*60)
    print("SUCCESS: All _compute_adaptive_weights() tests passed!")
    print("="*60)
    
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
