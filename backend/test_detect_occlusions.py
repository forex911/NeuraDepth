"""
Unit tests for _detect_occlusions() function.
Tests Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import numpy as np
import cv2
import pytest
from app.processor import _detect_occlusions


class TestDetectOcclusions:
    """Tests for the _detect_occlusions function"""
    
    def test_output_is_binary(self):
        """Test that output is binary (0 or 255)"""
        # Create simple depth map with discontinuity
        depth = np.ones((100, 100), dtype=np.float32) * 0.5
        depth[40:60, :] = 0.9  # Create a horizontal discontinuity
        
        # Create edges that align with discontinuity
        edges = np.zeros((100, 100), dtype=np.uint8)
        edges[39:41, :] = 255
        edges[59:61, :] = 255
        
        mask = _detect_occlusions(depth, edges)
        
        # Check binary output
        unique_values = np.unique(mask)
        assert all(v in [0, 255] for v in unique_values), f"Mask should be binary, got {unique_values}"
    
    def test_output_shape_matches_input(self):
        """Test that output shape matches input depth and edges shape"""
        depth = np.random.rand(120, 80).astype(np.float32)
        edges = np.random.randint(0, 2, (120, 80), dtype=np.uint8) * 255
        
        mask = _detect_occlusions(depth, edges)
        
        assert mask.shape == depth.shape, f"Expected shape {depth.shape}, got {mask.shape}"
        assert mask.shape == edges.shape, f"Expected shape {edges.shape}, got {mask.shape}"
    
    def test_uniform_depth_no_occlusions(self):
        """Test that uniform depth map produces no occlusions"""
        depth = np.ones((100, 100), dtype=np.float32) * 0.5
        edges = np.zeros((100, 100), dtype=np.uint8)
        
        mask = _detect_occlusions(depth, edges)
        
        # Uniform depth should have minimal or no occlusions
        occlusion_ratio = np.sum(mask > 0) / mask.size
        assert occlusion_ratio < 0.01, f"Uniform depth should have minimal occlusions, got {occlusion_ratio:.2%}"
    
    def test_detects_depth_discontinuities(self):
        """Test that depth discontinuities with realistic spatial extent are detected"""
        # Create depth map with a gradual but significant depth transition
        # (more realistic than step function)
        depth = np.ones((100, 100), dtype=np.float32) * 0.2
        
        # Create a smoother transition over several pixels
        for i, col in enumerate(range(48, 53)):
            depth[:, col] = 0.2 + (i / 5.0) * 0.6
        depth[:, 53:] = 0.8
        
        # Create edges around the transition  
        edges = np.zeros((100, 100), dtype=np.uint8)
        edges[:, 48:53] = 255
        
        mask = _detect_occlusions(depth, edges, gradient_threshold=0.10)  # Lower threshold for gradual transition
        
        # Should detect occlusion in the transition region
        occlusion_ratio = np.sum(mask > 0) / mask.size
        assert occlusion_ratio > 0.005, f"Should detect discontinuity, got {occlusion_ratio:.2%} occlusions"
    
    def test_threshold_affects_sensitivity(self):
        """Test that gradient_threshold parameter affects detection sensitivity"""
        # Create depth map with moderate discontinuity
        depth = np.ones((100, 100), dtype=np.float32) * 0.4
        depth[50:, :] = 0.6  # Moderate discontinuity
        
        edges = np.zeros((100, 100), dtype=np.uint8)
        edges[49:51, :] = 255
        
        # Low threshold (more sensitive)
        mask_low = _detect_occlusions(depth, edges, gradient_threshold=0.05)
        occlusions_low = np.sum(mask_low > 0)
        
        # High threshold (less sensitive)
        mask_high = _detect_occlusions(depth, edges, gradient_threshold=0.5)
        occlusions_high = np.sum(mask_high > 0)
        
        # Lower threshold should detect more or equal occlusions
        assert occlusions_low >= occlusions_high, \
            f"Lower threshold should detect more occlusions: {occlusions_low} vs {occlusions_high}"
    
    def test_requires_both_depth_change_and_edges(self):
        """Test that occlusions require both depth discontinuity and image edges"""
        # Depth discontinuity without edges (gradual transition)
        depth_only = np.ones((100, 100), dtype=np.float32) * 0.3
        for i, col in enumerate(range(48, 53)):
            depth_only[:, col] = 0.3 + (i / 5.0) * 0.4
        depth_only[:, 53:] = 0.7
        edges_none = np.zeros((100, 100), dtype=np.uint8)
        
        mask_depth_only = _detect_occlusions(depth_only, edges_none, gradient_threshold=0.10)
        occlusions_depth_only = np.sum(mask_depth_only > 0)
        
        # Same depth with edges
        edges_present = np.zeros((100, 100), dtype=np.uint8)
        edges_present[:, 48:53] = 255
        
        mask_both = _detect_occlusions(depth_only, edges_present, gradient_threshold=0.10)
        occlusions_both = np.sum(mask_both > 0)
        
        # Should detect more occlusions when edges are present
        assert occlusions_both > occlusions_depth_only, \
            f"Should detect more with edges: {occlusions_both} vs {occlusions_depth_only}"
    
    def test_no_isolated_pixels_after_morphological_opening(self):
        """Test that morphological opening removes isolated pixels"""
        # Create depth with scattered small discontinuities
        depth = np.random.rand(100, 100).astype(np.float32)
        edges = (np.random.rand(100, 100) > 0.95).astype(np.uint8) * 255
        
        mask = _detect_occlusions(depth, edges)
        
        # Check for isolated pixels (pixels with no neighbors)
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        
        # Any non-zero pixel in original mask should have at least one neighbor
        isolated = np.logical_and(mask > 0, dilated == mask)
        isolated_count = np.sum(isolated)
        
        # Should have few or no isolated pixels
        total_occlusions = np.sum(mask > 0)
        if total_occlusions > 0:
            isolated_ratio = isolated_count / total_occlusions
            assert isolated_ratio < 0.1, \
                f"Too many isolated pixels: {isolated_ratio:.2%}"
    
    def test_output_dtype_is_uint8(self):
        """Test that output dtype is uint8"""
        depth = np.random.rand(50, 50).astype(np.float32)
        edges = np.zeros((50, 50), dtype=np.uint8)
        
        mask = _detect_occlusions(depth, edges)
        
        assert mask.dtype == np.uint8, f"Expected dtype uint8, got {mask.dtype}"
    
    def test_default_threshold_value(self):
        """Test that default gradient_threshold value is 0.15"""
        depth = np.ones((100, 100), dtype=np.float32) * 0.3
        depth[50:, :] = 0.6
        edges = np.zeros((100, 100), dtype=np.uint8)
        edges[49:51, :] = 255
        
        # Call without threshold (should use default 0.15)
        mask_default = _detect_occlusions(depth, edges)
        
        # Call with explicit threshold 0.15
        mask_explicit = _detect_occlusions(depth, edges, gradient_threshold=0.15)
        
        # Should be identical
        assert np.array_equal(mask_default, mask_explicit), \
            "Default threshold should be 0.15"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
