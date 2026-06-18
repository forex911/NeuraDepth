"""Unit tests for _fuse_multiscale_depth() function"""
import sys
import numpy as np

from app.processor import _fuse_multiscale_depth


def test_single_scale_returns_that_scale():
    """Test that fusing a single scale returns that scale's depth and confidence"""
    print("Test 1: Single scale fusion...")
    
    # Create a single scale depth and confidence map
    depth = np.random.rand(100, 100).astype(np.float32)
    confidence = np.random.rand(100, 100).astype(np.float32)
    
    scale_depths = [depth]
    scale_confidences = [confidence]
    target_shape = (100, 100)
    
    fused_depth, fused_confidence = _fuse_multiscale_depth(
        scale_depths, scale_confidences, target_shape
    )
    
    # Should return approximately the same depth (with small epsilon for numerical stability)
    # Since weight = confidence + 1e-6, and we normalize by that same weight,
    # the result should be very close to the original depth
    assert fused_depth.shape == target_shape, f"Fused depth shape mismatch: {fused_depth.shape}"
    assert fused_confidence.shape == target_shape, f"Fused confidence shape mismatch: {fused_confidence.shape}"
    
    # Check that depth values are very close (within floating point tolerance)
    np.testing.assert_allclose(fused_depth, depth, rtol=1e-5, atol=1e-6,
                               err_msg="Single scale fusion should return original depth")
    
    print("✓ Test 1 passed: Single scale returns that scale")


def test_equal_confidence_produces_average():
    """Test that equal confidence scales produce weighted average"""
    print("Test 2: Equal confidence produces average...")
    
    # Create two scales with different depth values but equal confidence
    depth1 = np.full((50, 50), 0.2, dtype=np.float32)
    depth2 = np.full((50, 50), 0.8, dtype=np.float32)
    
    # Equal confidence
    confidence1 = np.full((50, 50), 0.5, dtype=np.float32)
    confidence2 = np.full((50, 50), 0.5, dtype=np.float32)
    
    scale_depths = [depth1, depth2]
    scale_confidences = [confidence1, confidence2]
    target_shape = (50, 50)
    
    fused_depth, fused_confidence = _fuse_multiscale_depth(
        scale_depths, scale_confidences, target_shape
    )
    
    # With equal confidence, fused depth should be approximately the average
    expected_average = (0.2 + 0.8) / 2.0  # 0.5
    
    # Check that the result is close to the average
    np.testing.assert_allclose(fused_depth, expected_average, rtol=1e-2, atol=1e-3,
                               err_msg="Equal confidence should produce average depth")
    
    print("✓ Test 2 passed: Equal confidence produces average")


def test_higher_confidence_scales_contribute_more():
    """Test that higher confidence scales have more influence on result"""
    print("Test 3: Higher confidence scales contribute more...")
    
    # Create two scales with very different confidence levels
    depth_low_conf = np.full((50, 50), 0.1, dtype=np.float32)
    depth_high_conf = np.full((50, 50), 0.9, dtype=np.float32)
    
    # One scale has very low confidence, other has very high confidence
    confidence_low = np.full((50, 50), 0.01, dtype=np.float32)
    confidence_high = np.full((50, 50), 0.99, dtype=np.float32)
    
    scale_depths = [depth_low_conf, depth_high_conf]
    scale_confidences = [confidence_low, confidence_high]
    target_shape = (50, 50)
    
    fused_depth, fused_confidence = _fuse_multiscale_depth(
        scale_depths, scale_confidences, target_shape
    )
    
    # The fused depth should be much closer to high confidence depth (0.9) than low confidence depth (0.1)
    # With weight ratio ~1:99, result should be ~0.89
    assert np.mean(fused_depth) > 0.8, f"Fused depth should be closer to high confidence value, got {np.mean(fused_depth)}"
    assert np.mean(fused_depth) < 0.95, f"Fused depth should not exceed high confidence value, got {np.mean(fused_depth)}"
    
    print("✓ Test 3 passed: Higher confidence scales contribute more")


def test_output_in_valid_range():
    """Test that output depth and confidence remain in [0, 1] range"""
    print("Test 4: Output in valid range...")
    
    # Create multiple scales with varying values
    np.random.seed(42)
    depth1 = np.random.rand(100, 80).astype(np.float32)
    depth2 = np.random.rand(50, 40).astype(np.float32)
    depth3 = np.random.rand(25, 20).astype(np.float32)
    
    confidence1 = np.random.rand(100, 80).astype(np.float32)
    confidence2 = np.random.rand(50, 40).astype(np.float32)
    confidence3 = np.random.rand(25, 20).astype(np.float32)
    
    scale_depths = [depth1, depth2, depth3]
    scale_confidences = [confidence1, confidence2, confidence3]
    target_shape = (100, 80)
    
    fused_depth, fused_confidence = _fuse_multiscale_depth(
        scale_depths, scale_confidences, target_shape
    )
    
    # Check output ranges
    assert np.all(fused_depth >= 0.0), "Fused depth has values below 0.0"
    assert np.all(fused_depth <= 1.0), "Fused depth has values above 1.0"
    
    assert np.all(fused_confidence >= 0.0), "Fused confidence has values below 0.0"
    assert np.all(fused_confidence <= 1.0), "Fused confidence has values above 1.0"
    
    # Check dtypes
    assert fused_depth.dtype == np.float32, f"Fused depth dtype should be float32, got {fused_depth.dtype}"
    assert fused_confidence.dtype == np.float32, f"Fused confidence dtype should be float32, got {fused_confidence.dtype}"
    
    # Check no NaN or Inf
    assert not np.any(np.isnan(fused_depth)), "Fused depth contains NaN values"
    assert not np.any(np.isinf(fused_depth)), "Fused depth contains Inf values"
    assert not np.any(np.isnan(fused_confidence)), "Fused confidence contains NaN values"
    assert not np.any(np.isinf(fused_confidence)), "Fused confidence contains Inf values"
    
    print("✓ Test 4 passed: Output in valid range [0, 1]")


def test_resizing_to_target_shape():
    """Test that all scales are properly resized to target shape"""
    print("Test 5: Resizing to target shape...")
    
    # Create scales with different resolutions
    depth1 = np.full((200, 150), 0.5, dtype=np.float32)
    depth2 = np.full((100, 75), 0.5, dtype=np.float32)
    depth3 = np.full((50, 38), 0.5, dtype=np.float32)
    
    confidence1 = np.full((200, 150), 0.6, dtype=np.float32)
    confidence2 = np.full((100, 75), 0.4, dtype=np.float32)
    confidence3 = np.full((50, 38), 0.3, dtype=np.float32)
    
    scale_depths = [depth1, depth2, depth3]
    scale_confidences = [confidence1, confidence2, confidence3]
    target_shape = (200, 150)
    
    fused_depth, fused_confidence = _fuse_multiscale_depth(
        scale_depths, scale_confidences, target_shape
    )
    
    # Output should have target shape
    assert fused_depth.shape == target_shape, f"Fused depth shape {fused_depth.shape} != target {target_shape}"
    assert fused_confidence.shape == target_shape, f"Fused confidence shape {fused_confidence.shape} != target {target_shape}"
    
    print("✓ Test 5 passed: Proper resizing to target shape")


def test_confidence_weighted_fusion():
    """Test that fusion correctly uses confidence as weights"""
    print("Test 6: Confidence-weighted fusion...")
    
    # Create a scenario with clear expected outcome
    # Two scales: one at 0.0, one at 1.0
    # If first has confidence 0.3 and second has confidence 0.7, 
    # result should be approximately 0.7
    depth1 = np.zeros((50, 50), dtype=np.float32)  # All 0.0
    depth2 = np.ones((50, 50), dtype=np.float32)   # All 1.0
    
    confidence1 = np.full((50, 50), 0.3, dtype=np.float32)
    confidence2 = np.full((50, 50), 0.7, dtype=np.float32)
    
    scale_depths = [depth1, depth2]
    scale_confidences = [confidence1, confidence2]
    target_shape = (50, 50)
    
    fused_depth, fused_confidence = _fuse_multiscale_depth(
        scale_depths, scale_confidences, target_shape
    )
    
    # Expected: (0.0 * 0.3 + 1.0 * 0.7) / (0.3 + 0.7) = 0.7 / 1.0 = 0.7
    # (plus small epsilon in weights, but should be negligible)
    expected_depth = 0.7
    
    np.testing.assert_allclose(fused_depth, expected_depth, rtol=1e-2, atol=1e-2,
                               err_msg=f"Confidence-weighted fusion incorrect, expected ~{expected_depth}")
    
    print("✓ Test 6 passed: Confidence-weighted fusion works correctly")


def test_zero_confidence_handling():
    """Test that zero confidence is handled gracefully with epsilon"""
    print("Test 7: Zero confidence handling...")
    
    # Create scales with zero confidence (edge case)
    depth1 = np.full((30, 30), 0.5, dtype=np.float32)
    depth2 = np.full((30, 30), 0.8, dtype=np.float32)
    
    # Zero confidence should still work due to epsilon (1e-6)
    confidence1 = np.zeros((30, 30), dtype=np.float32)
    confidence2 = np.zeros((30, 30), dtype=np.float32)
    
    scale_depths = [depth1, depth2]
    scale_confidences = [confidence1, confidence2]
    target_shape = (30, 30)
    
    fused_depth, fused_confidence = _fuse_multiscale_depth(
        scale_depths, scale_confidences, target_shape
    )
    
    # Should not produce NaN or Inf (epsilon prevents division by zero)
    assert not np.any(np.isnan(fused_depth)), "Zero confidence produced NaN values"
    assert not np.any(np.isinf(fused_depth)), "Zero confidence produced Inf values"
    
    # With equal (zero) confidence, should get approximately the average
    expected_average = (0.5 + 0.8) / 2.0
    np.testing.assert_allclose(fused_depth, expected_average, rtol=1e-1, atol=5e-2,
                               err_msg="Zero confidence should produce average via epsilon")
    
    print("✓ Test 7 passed: Zero confidence handled with epsilon")


if __name__ == "__main__":
    print("Running unit tests for _fuse_multiscale_depth()...\n")
    
    try:
        test_single_scale_returns_that_scale()
        test_equal_confidence_produces_average()
        test_higher_confidence_scales_contribute_more()
        test_output_in_valid_range()
        test_resizing_to_target_shape()
        test_confidence_weighted_fusion()
        test_zero_confidence_handling()
        
        print("\n" + "="*50)
        print("All tests passed! ✓")
        print("="*50)
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
