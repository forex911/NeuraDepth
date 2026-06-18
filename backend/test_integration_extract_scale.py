"""Integration test demonstrating _extract_scale_features in context"""
import sys
import numpy as np
import cv2

from app.processor import _extract_scale_features, ProcessingParams


def test_realistic_image():
    """Test with a realistic synthetic image showing all features"""
    print("Integration Test: Realistic image processing...")
    
    # Create a synthetic scene with different features
    # 300x300 image with various depth cues
    image = np.zeros((300, 300), dtype=np.uint8)
    
    # Background gradient (depth cue)
    for i in range(300):
        image[i, :] = int(50 + (i / 300) * 150)
    
    # Add some objects with different intensities (closer objects are darker)
    image[50:100, 50:100] = 30      # Dark square (close)
    image[120:170, 120:170] = 180   # Light square (far)
    image[200:250, 80:130] = 100    # Medium square (medium depth)
    
    # Add texture to some regions
    noise = np.random.randint(-20, 20, (50, 50))
    image[50:100, 150:200] = np.clip(120 + noise, 0, 255).astype(np.uint8)
    
    # Apply CLAHE (as would be done in real pipeline)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    params = ProcessingParams(
        mode="depth",
        scan_density=50.0,
        noise_level=10.0,
        edge_sensitivity=50.0,
        depth_contrast=50.0,
        smoothing=50.0,
        point_density=50.0
    )
    
    print("  Processing at scale 1.0...")
    features_1_0 = _extract_scale_features(enhanced, 1.0, params)
    
    print("  Processing at scale 0.5...")
    features_0_5 = _extract_scale_features(enhanced, 0.5, params)
    
    print("  Processing at scale 0.25...")
    features_0_25 = _extract_scale_features(enhanced, 0.25, params)
    
    # Verify all features are present and valid
    print("\n  Feature Analysis:")
    print(f"    Scale 1.0: {features_1_0.edges.shape}, {np.sum(features_1_0.edges > 0)} edge pixels")
    print(f"    Scale 0.5: {features_0_5.edges.shape}, {np.sum(features_0_5.edges > 0)} edge pixels")
    print(f"    Scale 0.25: {features_0_25.edges.shape}, {np.sum(features_0_25.edges > 0)} edge pixels")
    
    print(f"\n    Gradient stats (1.0): min={features_1_0.gradients.min():.3f}, max={features_1_0.gradients.max():.3f}, mean={features_1_0.gradients.mean():.3f}")
    print(f"    Texture stats (1.0): min={features_1_0.texture.min():.3f}, max={features_1_0.texture.max():.3f}, mean={features_1_0.texture.mean():.3f}")
    print(f"    Distance depth (1.0): min={features_1_0.distance_depth.min():.3f}, max={features_1_0.distance_depth.max():.3f}, mean={features_1_0.distance_depth.mean():.3f}")
    print(f"    Intensity depth (1.0): min={features_1_0.intensity_depth.min():.3f}, max={features_1_0.intensity_depth.max():.3f}, mean={features_1_0.intensity_depth.mean():.3f}")
    print(f"    Confidence (1.0): min={features_1_0.confidence.min():.3f}, max={features_1_0.confidence.max():.3f}, mean={features_1_0.confidence.mean():.3f}")
    
    # Verify edges were detected (we have clear boundaries)
    assert np.sum(features_1_0.edges > 0) > 100, "Should detect edges in structured image"
    
    # Verify gradients are meaningful (not all zero)
    assert features_1_0.gradients.mean() > 0.01, "Gradients should be non-trivial"
    
    # Verify confidence is properly scaled
    conf_1_0 = features_1_0.confidence.mean()
    conf_0_5 = features_0_5.confidence.mean()
    conf_0_25 = features_0_25.confidence.mean()
    
    print(f"\n    Confidence scaling: 1.0={conf_1_0:.4f}, 0.5={conf_0_5:.4f}, 0.25={conf_0_25:.4f}")
    assert conf_1_0 >= conf_0_5 >= conf_0_25, "Confidence should decrease with scale"
    
    print("\n✓ Integration test passed: All features extracted correctly at multiple scales")
    return 0


def test_edge_cases():
    """Test edge cases"""
    print("\nEdge Case Tests:")
    
    # Uniform image (no features)
    print("  Test 1: Uniform image...")
    uniform = np.full((100, 100), 128, dtype=np.uint8)
    params = ProcessingParams(
        mode="depth",
        scan_density=50.0,
        noise_level=10.0,
        edge_sensitivity=50.0,
        depth_contrast=50.0,
        smoothing=50.0,
        point_density=50.0
    )
    
    features = _extract_scale_features(uniform, 1.0, params)
    assert features.edges.shape == (100, 100), "Shape should match input"
    assert np.sum(features.edges > 0) == 0, "Uniform image should have no edges"
    print("    ✓ Uniform image handled correctly")
    
    # Very small image
    print("  Test 2: Small image at quarter scale...")
    small = np.random.randint(0, 256, (60, 60), dtype=np.uint8)
    features_small = _extract_scale_features(small, 0.25, params)
    assert features_small.edges.shape == (15, 15), "Quarter scale of 60x60 should be 15x15"
    print("    ✓ Small image handled correctly")
    
    # All black image
    print("  Test 3: All black image...")
    black = np.zeros((100, 100), dtype=np.uint8)
    features_black = _extract_scale_features(black, 1.0, params)
    # Should still produce valid outputs
    assert features_black.intensity_depth.mean() == 1.0, "All black should have max intensity depth"
    print("    ✓ All black image handled correctly")
    
    # All white image
    print("  Test 4: All white image...")
    white = np.full((100, 100), 255, dtype=np.uint8)
    features_white = _extract_scale_features(white, 1.0, params)
    assert features_white.intensity_depth.mean() == 0.0, "All white should have min intensity depth"
    print("    ✓ All white image handled correctly")
    
    print("\n✓ All edge cases passed")
    return 0


def run_integration_tests():
    """Run all integration tests"""
    print("=" * 70)
    print("Integration Tests for _extract_scale_features()")
    print("=" * 70)
    
    try:
        result1 = test_realistic_image()
        result2 = test_edge_cases()
        
        print("\n" + "=" * 70)
        print("✓ All integration tests passed successfully!")
        print("=" * 70)
        return 0
    except AssertionError as e:
        print("\n" + "=" * 70)
        print(f"✗ Integration test failed: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ Unexpected error: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_integration_tests())
