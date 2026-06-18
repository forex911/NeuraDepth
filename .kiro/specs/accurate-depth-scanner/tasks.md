# Implementation Plan: Accurate Depth Scanner

## Overview

This plan implements an enhanced depth estimation system with multi-scale processing, adaptive feature weighting, confidence maps, occlusion detection and refinement, and confidence-aware smoothing. The implementation maintains backward compatibility with existing API interfaces and adds all new functions to backend/app/processor.py.

## Tasks

- [x] 1. Set up data models and helper functions
  - Create ScaleFeatures, AdaptiveWeights, and DepthEstimationResult dataclasses in processor.py
  - Ensure dataclasses are properly imported from dataclasses module
  - _Requirements: 1.5, 2.5, 3.1, 8.1-8.7_

- [ ] 2. Implement multi-scale feature extraction
  - [x] 2.1 Implement _extract_scale_features() function
    - Extract edges using Canny with adaptive thresholds based on edge_sensitivity
    - Compute gradients using Sobel operators (X and Y directions)
    - Analyze texture using Laplacian
    - Generate distance transform depth cues
    - Compute intensity-based depth (darker = closer)
    - Calculate feature confidence based on feature strength and scale
    - Return ScaleFeatures object with all fields populated
    - _Requirements: 1.4, 10.2, 10.3_

  - [ ]* 2.2 Write property test for feature extraction
    - **Property 6: Feature Extraction Completeness**
    - **Validates: Requirements 1.4**

  - [ ]* 2.3 Write unit tests for _extract_scale_features()
    - Test features at scale 1.0 match input dimensions
    - Test features at scale 0.5 have half dimensions
    - Test all normalized arrays in [0, 1] except edges
    - Test confidence decreases with scale
    - _Requirements: 1.4, 3.4_

- [ ] 3. Implement adaptive feature weighting
  - [x] 3.1 Implement _compute_adaptive_weights() function
    - Compute edge proximity map by dilating edges
    - Calculate texture richness from texture features
    - Determine illumination uniformity from gradients
    - Assign base weights based on local characteristics
    - Normalize weights to sum to 1.0 at each pixel
    - Return AdaptiveWeights object
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 3.2 Write property test for weight normalization
    - **Property 4: Adaptive Weight Normalization**
    - **Validates: Requirements 2.5**

  - [ ]* 3.3 Write unit tests for _compute_adaptive_weights()
    - Test weights sum to 1.0 at every pixel within tolerance
    - Test all weights are non-negative
    - Test weights adapt to local features
    - _Requirements: 2.5_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement multi-scale depth fusion
  - [x] 5.1 Implement _fuse_multiscale_depth() function
    - Resize all scale depths and confidences to target resolution
    - Use confidence values as fusion weights
    - Accumulate weighted depth values
    - Normalize by total weight to get fused depth
    - Compute fused confidence from weighted combination
    - Return (fused_depth, fused_confidence) tuple
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 5.2 Write property test for fusion range preservation
    - **Property 10: Fusion Preserves Valid Range**
    - **Validates: Requirements 4.4**

  - [ ]* 5.3 Write unit tests for _fuse_multiscale_depth()
    - Test single scale returns that scale
    - Test equal confidence produces average
    - Test higher confidence scales contribute more
    - Test output in [0, 1] range
    - _Requirements: 4.3, 4.4_

- [ ] 6. Implement occlusion detection and refinement
  - [ ] 6.1 Implement _detect_occlusions() function
    - Compute depth gradients using Sobel operators
    - Normalize depth gradient magnitude
    - Threshold to find depth discontinuities
    - Combine with image edges (logical AND)
    - Apply morphological opening to remove isolated pixels
    - Dilate to capture boundary regions
    - Return binary occlusion mask
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [~] 6.2 Implement _refine_depth_at_occlusions() function
    - Identify occlusion pixels from mask
    - Return unchanged depth if no occlusions detected
    - Blend original depth with distance-based depth at occlusions using 60:40 weight ratio
    - Apply edge-preserving bilateral filter with small kernel (5x5)
    - Ensure output remains in [0, 1] range
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 6.3 Write property test for occlusion detection
    - **Property 11: Occlusion Detection Binary Output**
    - **Property 13: Occlusion Detection Spatial Coherence**
    - **Validates: Requirements 5.4, 5.5**

  - [ ]* 6.4 Write unit tests for occlusion detection and refinement
    - Test output is binary (0 or 255)
    - Test no-op when mask is empty
    - Test refinement preserves [0, 1] range
    - _Requirements: 5.5, 6.2, 6.4_

- [ ] 7. Implement confidence-aware smoothing
  - [~] 7.1 Implement _confidence_aware_smoothing() function
    - Return input unchanged if smoothing parameter < 0.03
    - Map confidence to smoothing strength (inverse relationship)
    - Compute adaptive bilateral filter parameters
    - Create three confidence zones (high >0.7, low <0.3, medium)
    - Apply strong smoothing to low confidence regions
    - Apply moderate smoothing to medium confidence regions
    - Keep high confidence regions mostly unchanged
    - Apply Gaussian blur weighted by inverse confidence
    - Ensure output remains in [0, 1] range
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [ ]* 7.2 Write property test for confidence-aware smoothing
    - **Property 16: Confidence-Aware Smoothing Identity**
    - **Property 17: Confidence-Aware Smoothing Inverse Relationship**
    - **Property 18: Smoothing Preserves Valid Range**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.7_

  - [ ]* 7.3 Write unit tests for _confidence_aware_smoothing()
    - Test identity when smoothing < 0.03
    - Test high confidence smoothed less than low confidence
    - Test output in [0, 1] range
    - _Requirements: 7.1, 7.7_

- [~] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement enhanced depth estimator
  - [~] 9.1 Implement _estimate_depth_enhanced() function
    - Convert image to grayscale and apply CLAHE (clip limit 2.4, tile size 8x8)
    - Determine scales based on image dimensions: 3 scales for ≥200px, 2 scales for ≥100px and <200px, 1 scale for <100px
    - Loop through scales and extract features using _extract_scale_features()
    - Compute adaptive weights using _compute_adaptive_weights()
    - Fuse features into scale depth estimate with adaptive weighting
    - Normalize scale depth to [0, 1]
    - Store scale features, depths, and confidences
    - Fuse multi-scale depth estimates using _fuse_multiscale_depth()
    - Extract final edges and gradients from full resolution features
    - Detect occlusions using _detect_occlusions()
    - Refine depth at occlusions using _refine_depth_at_occlusions()
    - Apply confidence-aware smoothing using _confidence_aware_smoothing()
    - Apply morphological operations for cleanup (close then open with 5x5 ellipse kernel)
    - Apply final contrast adjustment based on depth_contrast parameter
    - Return (depth, edges, gradients, confidence) tuple
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.4, 10.5, 10.6_

  - [ ]* 9.2 Write property test for output validity
    - **Property 1: Output Range Preservation**
    - **Property 2: Output Validity and Finiteness**
    - **Property 3: Spatial Dimension Consistency**
    - **Validates: Requirements 1.5, 8.1-8.7**

  - [ ]* 9.3 Write property test for multi-scale processing
    - **Property 5: Multi-Scale Processing**
    - **Validates: Requirements 1.1, 1.2, 1.3**

  - [ ]* 9.4 Write unit tests for _estimate_depth_enhanced()
    - Test valid input produces valid outputs
    - Test all outputs have correct shape
    - Test depth in [0, 1], edges binary, no NaN/Inf
    - Test different parameters produce different results
    - Test small images handled gracefully
    - _Requirements: 8.1-8.7, 11.2_

- [ ] 10. Add error handling and input validation
  - [~] 10.1 Add input validation to _estimate_depth_enhanced()
    - Check for None or empty image and raise ValueError
    - Handle images smaller than 50x50 by using only full scale
    - Add epsilon to prevent division by zero in weight normalization
    - Catch MemoryError and raise descriptive ValueError
    - Check OpenCV function return values for None
    - Handle all-zero confidence by setting uniform value 0.1
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

  - [ ]* 10.2 Write unit tests for error handling
    - Test None image raises ValueError
    - Test empty image raises ValueError
    - Test extreme parameters are clamped
    - Test tiny images use available scales
    - _Requirements: 12.1, 12.2, 12.3_

- [ ] 11. Integrate enhanced depth estimator with existing API
  - [~] 11.1 Update process_image() to use _estimate_depth_enhanced()
    - Replace call to _estimate_depth() with _estimate_depth_enhanced()
    - Update to handle 4-element tuple (depth, edges, gradients, confidence)
    - Ensure all visualization modes work with new outputs
    - Maintain backward compatibility by ignoring confidence in existing modes
    - _Requirements: 11.1, 11.2, 11.3_

  - [ ]* 11.2 Write integration tests for API compatibility
    - Test all visualization modes (depth, lidar, wireframe, mesh, scanner)
    - Test API response format unchanged
    - Test processing time within acceptable limits
    - _Requirements: 11.3, 11.4_

- [~] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at reasonable breaks
- Property tests validate universal correctness properties from design
- Unit tests validate specific examples and edge cases
- All new functions are added to backend/app/processor.py
- Implementation maintains backward compatibility with existing API
