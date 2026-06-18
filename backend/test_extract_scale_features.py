"""Unit tests for _extract_scale_features() function"""
import sys
import numpy as np
import cv2

from app.processor import _extract_scale_features, ProcessingParams, ScaleFeatures


def test_basic_functionality():
    """Test that _extract_scale_features returns valid ScaleFeatures object"""
    print("Test 1: Basic functionality...")
    
    # Create a test grayscale image with CLAHE applied
    gray = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    params = ProcessingParams(
        mode="depth",
        scan_density=50.0,
        noise_level=10.0,
        edge_sensitivity=50.0,
        depth_contrast=50.0,
        smoothing=50.0,
        point_density=50.0
    )
    
    # Extract features at full scale
    features = _extract_scale_features(enhanced, 1.0, params)
    
    # Verify return type
    assert isinstance(features, ScaleFeatures), "Should return ScaleFeatures object"
    
    # Verify scale
    assert features.scale == 1.0, "Scale should be 1.0"
    
    # Verify dimensions match
    assert features.edges.shape == (200, 200), f"Edges shape mismatch: {features.edges.shape}"
    assert features.gradients.shape == (200, 200), f"Gradients shape mismatch: {features.gradients.shape}"
    assert features.texture.shape == (200, 200), f"Texture shape mismatch: {features.texture.shape}"
    assert features.distance_depth.shape == (200, 200), f"Distance depth shape mismatch: {features.distance_depth.shape}"
    assert features.intensity_depth.shape == (200, 200), f"Intensity depth shape mismatch: {features.intensity_depth.shape}"
    assert features.confidence.shape == (200, 200), f"Confidence shape mismatch: {features.confidence.shape}"
    
    print("✓ Test 1 passed: Basic functionality works correctly")


def test_edges_binary():
    """Test that edges are binary (0 or 255)"""
    print("Test 2: Edges are binary...")
    
    gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    params = ProcessingParams(
        mode="depth",
        scan_density=50.0,
        noise_level=10.0,
        edge_sensitivity=50.0,
        depth_contrast=50.0,
        smoothing=50.0,
        point_density=50.0
    )
    
    features = _extract_scale_features(enhanced, 1.0, params)
    
    # Check that edges are binary
    unique_values = np.unique(features.edges)
    assert all(v in [0, 255] for v in unique_values), f"Edges should be binary (0 or 255), got: {unique_values}"
    assert features.edges.dtype == np.uint8, f"Edges dtype should be uint8, got: {features.edges.dtype}"
    
    print("✓ Test 2 passed: Edges are binary")


def test_normalized_ranges():
    """Test that normalized arrays are in [0, 1] range"""
    print("Test 3: Normalized ranges...")
    
    gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    params = ProcessingParams(
        mode="depth",
        scan_density=50.0,
        noise_level=10.0,
        edge_sensitivity=50.0,
        depth_contrast=50.0,
        smoothing=50.0,
        point_density=50.0
    )
    
    features = _extract_scale_features(enhanced, 1.0, params)
    
    # Check ranges
    assert np.all(features.gradients >= 0.0) and np.all(features.gradients <= 1.0), "Gradients out of [0, 1] range"
    assert np.all(features.texture >= 0.0) and np.all(features.texture <= 1.0), "Texture out of [0, 1] range"
    assert np.all(features.distance_depth >= 0.0) and np.all(features.distance_depth <= 1.0), "Distance depth out of [0, 1] range"
    assert np.all(features.intensity_depth >= 0.0) and np.all(features.intensity_depth <= 1.0), "Intensity depth out of [0, 1] range"
    assert np.all(features.confidence >= 0.0) and np.all(features.confidence <= 1.0), "Confidence out of [0, 1] range"
    
    print("✓ Test 3 passed: All normalized arrays in [0, 1] range")


def test_scale_half_resolution():
    """Test feature extraction at half resolution"""
    print("Test 4: Half resolution scale...")
    
    gray = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    params = ProcessingParams(
        mode="depth",
        scan_density=50.0,
        noise_level=10.0,
        edge_sensitivity=50.0,
        depth_contrast=50.0,
        smoothing=50.0,
        point_density=50.0
    )
    
    features = _extract_scale_features(enhanced, 0.5, params)
    
    # Check scale
    assert features.scale == 0.5, "Scale should be 0.5"
    
    # Check dimensions (should be half)
    assert features.edges.shape == (100, 100), f"Half resolution shape should be (100, 100), got: {features.edges.shape}"
    assert features.gradients.shape == (100, 100), f"Gradients shape mismatch at half scale: {features.gradients.shape}"
    
    print("✓ Test 4 passed: Half resolution works correctly")


def test_scale_quarter_resolution():
    """Test feature extraction at quarter resolution"""
    print("Test 5: Quarter resolution scale...")
    
    gray = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    params = ProcessingParams(
        mode="depth",
        scan_density=50.0,
        noise_level=10.0,
        edge_sensitivity=50.0,
        depth_contrast=50.0,
        smoothing=50.0,
        point_density=50.0
    )
    
    features = _extract_scale_features(enhanced, 0.25, params)
    
    # Check scale
    assert features.scale == 0.25, "Scale should be 0.25"
    
    # Check dimensions (should be quarter)
    assert features.edges.shape == (50, 50), f"Quarter resolution shape should be (50, 50), got: {features.edges.shape}"
    
    print("✓ Test 5 passed: Quarter resolution works correctly")


def test_confidence_decreases_with_scale():
    """Test that confidence decreases with lower scale (as per design)"""
    print("Test 6: Confidence decreases with scale...")
    
    # Create a test image with some structure
    gray = np.random.randint(50, 200, (200, 200), dtype=np.uint8)
    # Add some edges
    gray[50:60, :] = 255
    gray[100:110, :] = 0
    
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    params = ProcessingParams(
        mode="depth",
        scan_density=50.0,
        noise_level=10.0,
        edge_sensitivity=50.0,
        depth_contrast=50.0,
        smoothing=50.0,
        point_density=50.0
    )
    
    features_full = _extract_scale_features(enhanced, 1.0, params)
    features_half = _extract_scale_features(enhanced, 0.5, params)
    features_quarter = _extract_scale_features(enhanced, 0.25, params)
    
    # Mean confidence should decrease with scale (due to scale multiplier)
    mean_full = features_full.confidence.mean()
    mean_half = features_half.confidence.mean()
    mean_quarter = features_quarter.confidence.mean()
    
    # Since confidence is multiplied by scale, we expect:
    # mean_full > mean_half > mean_quarter
    assert mean_full >= mean_half, f"Full scale confidence ({mean_full}) should be >= half scale ({mean_half})"
    assert mean_half >= mean_quarter, f"Half scale confidence ({mean_half}) should be >= quarter scale ({mean_quarter})"
    
    print(f"✓ Test 6 passed: Confidence decreases with scale (full: {mean_full:.4f}, half: {mean_half:.4f}, quarter: {mean_quarter:.4f})")


def test_edge_sensitivity_parameter():
    """Test that edge_sensitivity parameter affects edge detection"""
    print("Test 7: Edge sensitivity parameter effect...")
    
    # Create a test image with moderate edges
    gray = np.random.randint(80, 180, (100, 100), dtype=np.uint8)
    gray[40:50, :] = 200  # Add a clear edge
    
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    params_low = ProcessingParams(
        mode="depth",
        scan_density=50.0,
        noise_level=10.0,
        edge_sensitivity=10.0,  # Low sensitivity
        depth_contrast=50.0,
        smoothing=50.0,
        point_density=50.0
    )
    
    params_high = ProcessingParams(
        mode="depth",
        scan_density=50.0,
        noise_level=10.0,
        edge_sensitivity=90.0,  # High sensitivity
        depth_contrast=50.0,
        smoothing=50.0,
        point_density=50.0
    )
    
    features_low = _extract_scale_features(enhanced, 1.0, params_low)
    features_high = _extract_scale_features(enhanced, 1.0, params_high)
    
    # Count edge pixels
    edges_low_count = np.sum(features_low.edges > 0)
    edges_high_count = np.sum(features_high.edges > 0)
    
    # Higher sensitivity should detect more edges
    assert edges_high_count >= edges_low_count, \
        f"High sensitivity ({edges_high_count} edges) should detect >= low sensitivity ({edges_low_count} edges)"
    
    print(f"✓ Test 7 passed: Edge sensitivity affects detection (low: {edges_low_count}, high: {edges_high_count} edge pixels)")


def test_no_nan_or_inf():
    """Test that outputs contain no NaN or Inf values"""
    print("Test 8: No NaN or Inf values...")
    
    gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    params = ProcessingParams(
        mode="depth",
        scan_density=50.0,
        noise_level=10.0,
        edge_sensitivity=50.0,
        depth_contrast=50.0,
        smoothing=50.0,
        point_density=50.0
    )
    
    features = _extract_scale_features(enhanced, 1.0, params)
    
    # Check for NaN and Inf
    assert np.all(np.isfinite(features.gradients)), "Gradients contain NaN or Inf"
    assert np.all(np.isfinite(features.texture)), "Texture contains NaN or Inf"
    assert np.all(np.isfinite(features.distance_depth)), "Distance depth contains NaN or Inf"
    assert np.all(np.isfinite(features.intensity_depth)), "Intensity depth contains NaN or Inf"
    assert np.all(np.isfinite(features.confidence)), "Confidence contains NaN or Inf"
    
    print("✓ Test 8 passed: No NaN or Inf values in outputs")


def test_intensity_depth_inversion():
    """Test that intensity_depth properly inverts (darker = closer)"""
    print("Test 9: Intensity depth inversion...")
    
    # Create images with different intensities
    gray_dark = np.full((50, 50), 50, dtype=np.uint8)  # Dark image
    gray_light = np.full((50, 50), 200, dtype=np.uint8)  # Light image
    
    params = ProcessingParams(
        mode="depth",
        scan_density=50.0,
        noise_level=10.0,
        edge_sensitivity=50.0,
        depth_contrast=50.0,
        smoothing=50.0,
        point_density=50.0
    )
    
    features_dark = _extract_scale_features(gray_dark, 1.0, params)
    features_light = _extract_scale_features(gray_light, 1.0, params)
    
    # Dark image should have higher intensity_depth (closer)
    # Light image should have lower intensity_depth (farther)
    mean_dark = features_dark.intensity_depth.mean()
    mean_light = features_light.intensity_depth.mean()
    
    assert mean_dark > mean_light, \
        f"Dark image intensity_depth ({mean_dark:.4f}) should be > light image ({mean_light:.4f})"
    
    print(f"✓ Test 9 passed: Intensity depth inverted correctly (dark: {mean_dark:.4f}, light: {mean_light:.4f})")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running _extract_scale_features() unit tests")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_edges_binary()
        test_normalized_ranges()
        test_scale_half_resolution()
        test_scale_quarter_resolution()
        test_confidence_decreases_with_scale()
        test_edge_sensitivity_parameter()
        test_no_nan_or_inf()
        test_intensity_depth_inversion()
        
        print("=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Unexpected error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
