#!/usr/bin/env python3
"""
Electrical Receptacle Symbol Detector
=====================================
Automatically detects receptacle symbols from electrical drawing legend pages
and counts their occurrences in power plan pages.

Features:
- Automated symbol extraction from legend pages
- Multi-scale template matching for detection
- Contour-based fallback detection
- Non-maximum suppression for clean results
- Visualization of detected symbols

Author: [Your Name]
Date: 2026-07-21
License: MIT
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Data class for detection results"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    method: str


class ElectricalSymbolDetector:
    """
    Main detector class for electrical receptacle symbols.
    
    Attributes:
        legend_path (str): Path to legend page image
        power_path (str): Path to power plan page image
        receptacle_templates (list): Extracted receptacle templates
        min_match_threshold (float): Minimum confidence for template matching
        output_dir (str): Directory for output files
    """
    
    def __init__(
        self, 
        legend_image_path: str, 
        power_page_path: str,
        output_dir: str = "output",
        min_confidence: float = 0.7
    ):
        """
        Initialize the detector.
        
        Args:
            legend_image_path: Path to legend page image
            power_page_path: Path to power plan page image
            output_dir: Directory to save results
            min_confidence: Minimum detection confidence (0.0-1.0)
        """
        self.legend_path = legend_image_path
        self.power_path = power_page_path
        self.output_dir = output_dir
        self.receptacle_templates = []
        self.min_match_threshold = min_confidence
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Detection parameters
        self.scale_factors = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        self.min_symbol_area = 200
        self.max_symbol_area = 5000
        self.min_symbol_dimension = 15
        self.max_symbol_dimension = 200
        
        logger.info(f"Detector initialized with legend: {legend_image_path}")
        logger.info(f"Power page: {power_page_path}")
        logger.info(f"Output directory: {output_dir}")

    def preprocess_image(
        self, 
        image: np.ndarray, 
        apply_morphology: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for symbol detection.
        
        Args:
            image: Input image (BGR or grayscale)
            apply_morphology: Whether to apply morphological operations
            
        Returns:
            Tuple of (grayscale image, binary image)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive thresholding for varying lighting conditions
        binary = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        
        if apply_morphology:
            # Remove noise and connect broken components
            kernel_small = np.ones((2, 2), np.uint8)
            kernel_medium = np.ones((3, 3), np.uint8)
            
            # Close gaps
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
            
            # Remove small noise
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_medium)
        
        return gray, binary

    def extract_symbols_from_legend(self) -> List[Dict]:
        """
        Extract receptacle symbols from legend page.
        
        Returns:
            List of dictionaries containing symbol templates
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Extracting symbols from legend page...")
        logger.info("=" * 60)
        
        legend_img = cv2.imread(self.legend_path)
        if legend_img is None:
            raise FileNotFoundError(f"Could not load legend image: {self.legend_path}")
        
        logger.info(f"Legend image loaded: {legend_img.shape}")
        
        gray, binary = self.preprocess_image(legend_img)
        
        # Save preprocessed images
        cv2.imwrite(os.path.join(self.output_dir, "legend_binary.jpg"), binary)
        
        # Try multiple extraction methods
        templates = self._extract_symbols_near_text(binary, gray, legend_img)
        logger.info(f"Method 1 (text proximity): Found {len(templates)} symbols")
        
        if len(templates) < 2:
            additional = self._extract_symbols_by_size(binary, gray, legend_img)
            templates.extend(additional)
            logger.info(f"Method 2 (size-based): Found {len(additional)} additional symbols")
        
        if len(templates) < 2:
            additional = self._extract_symbols_by_contour(binary, gray, legend_img)
            templates.extend(additional)
            logger.info(f"Method 3 (contour-based): Found {len(additional)} additional symbols")
        
        # Remove duplicates
        templates = self._remove_duplicate_templates(templates)
        
        self.receptacle_templates = templates
        
        # Visualize extracted templates
        self._visualize_templates()
        
        logger.info(f"Total unique templates extracted: {len(templates)}")
        return templates

    def _extract_symbols_near_text(
        self, 
        binary: np.ndarray, 
        gray: np.ndarray, 
        original_img: np.ndarray
    ) -> List[Dict]:
        """
        Extract symbols by finding components near text regions.
        
        Args:
            binary: Binary image
            gray: Grayscale image
            original_img: Original color image
            
        Returns:
            List of template dictionaries
        """
        h, w = binary.shape
        templates = []
        
        # Detect text regions using horizontal projection
        row_sums = np.sum(binary > 0, axis=1)
        mean_sum = np.mean(row_sums)
        if mean_sum == 0:
            return templates
            
        text_rows = np.where(row_sums > mean_sum * 0.3)[0]
        
        if len(text_rows) == 0:
            return templates
        
        # Group consecutive text rows into bands
        text_bands = []
        current_band = [text_rows[0]]
        
        for r in text_rows[1:]:
            if r - current_band[-1] <= 5:
                current_band.append(r)
            else:
                if len(current_band) > 10:  # Minimum band height
                    text_bands.append((min(current_band), max(current_band)))
                current_band = [r]
        
        if len(current_band) > 10:
            text_bands.append((min(current_band), max(current_band)))
        
        logger.info(f"Found {len(text_bands)} text bands")
        
        # Process each text band
        for band_idx, (y1, y2) in enumerate(text_bands):
            # Extract band with padding
            y1_pad = max(0, y1 - 10)
            y2_pad = min(h, y2 + 10)
            band_binary = binary[y1_pad:y2_pad, :]
            band_gray = gray[y1_pad:y2_pad, :]
            
            # Look for symbols in the left portion (typically left-aligned)
            left_band = band_binary[:, :int(w * 0.4)]
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                left_band, connectivity=8
            )
            
            for i in range(1, num_labels):
                x, y, bw, bh, area = stats[i]
                
                # Filter by size
                if not self._is_valid_symbol_size(area, bw, bh):
                    continue
                
                # Extract symbol with padding
                pad = 8
                symbol_y1 = max(0, y1_pad + y - pad)
                symbol_y2 = min(h, y1_pad + y + bh + pad)
                symbol_x1 = max(0, x - pad)
                symbol_x2 = min(w, x + bw + pad)
                
                symbol_binary = binary[symbol_y1:symbol_y2, symbol_x1:symbol_x2]
                symbol_gray = gray[symbol_y1:symbol_y2, symbol_x1:symbol_x2]
                
                if symbol_binary.shape[0] > 20 and symbol_binary.shape[1] > 20:
                    # Check for receptacle-like features
                    features = self._extract_symbol_features(symbol_binary)
                    
                    templates.append({
                        'binary': symbol_binary.copy(),
                        'gray': symbol_gray.copy(),
                        'color': original_img[symbol_y1:symbol_y2, symbol_x1:symbol_x2].copy(),
                        'bbox': (symbol_x1, symbol_y1, symbol_x2, symbol_y2),
                        'features': features,
                        'method': 'text_proximity',
                        'band_idx': band_idx
                    })
        
        return templates

    def _extract_symbols_by_size(
        self, 
        binary: np.ndarray, 
        gray: np.ndarray, 
        original_img: np.ndarray
    ) -> List[Dict]:
        """
        Extract symbols based on typical size range (fallback method).
        
        Args:
            binary: Binary image
            gray: Grayscale image
            original_img: Original color image
            
        Returns:
            List of template dictionaries
        """
        h, w = binary.shape
        templates = []
        
        # Focus on left portion where symbols typically appear
        left_portion = binary[:, :int(w * 0.35)]
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            left_portion, connectivity=8
        )
        
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            
            # More lenient size filtering
            if area < 300 or area > 15000:
                continue
            if bw < 20 or bh < 20:
                continue
            if bw > 300 or bh > 300:
                continue
            
            pad = 10
            symbol_y1 = max(0, y - pad)
            symbol_y2 = min(h, y + bh + pad)
            symbol_x1 = max(0, x - pad)
            symbol_x2 = min(w, x + bw + pad)
            
            symbol_binary = binary[symbol_y1:symbol_y2, symbol_x1:symbol_x2]
            symbol_gray = gray[symbol_y1:symbol_y2, symbol_x1:symbol_x2]
            
            if symbol_binary.shape[0] > 20 and symbol_binary.shape[1] > 20:
                features = self._extract_symbol_features(symbol_binary)
                
                templates.append({
                    'binary': symbol_binary.copy(),
                    'gray': symbol_gray.copy(),
                    'color': original_img[symbol_y1:symbol_y2, symbol_x1:symbol_x2].copy(),
                    'bbox': (symbol_x1, symbol_y1, symbol_x2, symbol_y2),
                    'features': features,
                    'method': 'size_based'
                })
        
        return templates

    def _extract_symbols_by_contour(
        self, 
        binary: np.ndarray, 
        gray: np.ndarray, 
        original_img: np.ndarray
    ) -> List[Dict]:
        """
        Extract symbols using contour detection (second fallback).
        
        Args:
            binary: Binary image
            gray: Grayscale image
            original_img: Original color image
            
        Returns:
            List of template dictionaries
        """
        h, w = binary.shape
        templates = []
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < 500 or area > 10000:
                continue
            
            x, y, bw, bh = cv2.boundingRect(contour)
            
            if bw < 20 or bh < 20 or bw > 250 or bh > 250:
                continue
            
            # Check if it's likely a symbol (not a large rectangle or line)
            aspect_ratio = bw / bh if bh > 0 else 0
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                continue
            
            # Only consider contours in left portion
            if x > w * 0.35:
                continue
            
            pad = 10
            symbol_y1 = max(0, y - pad)
            symbol_y2 = min(h, y + bh + pad)
            symbol_x1 = max(0, x - pad)
            symbol_x2 = min(w, x + bw + pad)
            
            symbol_binary = binary[symbol_y1:symbol_y2, symbol_x1:symbol_x2]
            symbol_gray = gray[symbol_y1:symbol_y2, symbol_x1:symbol_x2]
            
            if symbol_binary.shape[0] > 20 and symbol_binary.shape[1] > 20:
                features = self._extract_symbol_features(symbol_binary)
                
                templates.append({
                    'binary': symbol_binary.copy(),
                    'gray': symbol_gray.copy(),
                    'color': original_img[symbol_y1:symbol_y2, symbol_x1:symbol_x2].copy(),
                    'bbox': (symbol_x1, symbol_y1, symbol_x2, symbol_y2),
                    'features': features,
                    'method': 'contour_based'
                })
        
        return templates

    def _is_valid_symbol_size(self, area: int, width: int, height: int) -> bool:
        """
        Check if component dimensions are valid for a receptacle symbol.
        
        Args:
            area: Component area in pixels
            width: Component width
            height: Component height
            
        Returns:
            Boolean indicating if size is valid
        """
        if area < self.min_symbol_area or area > self.max_symbol_area:
            return False
        if width < self.min_symbol_dimension or height < self.min_symbol_dimension:
            return False
        if width > self.max_symbol_dimension or height > self.max_symbol_dimension:
            return False
        
        # Check aspect ratio (receptacles are roughly square or slightly rectangular)
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return False
        
        return True

    def _extract_symbol_features(self, symbol_binary: np.ndarray) -> Dict:
        """
        Extract features from a symbol to identify if it's a receptacle.
        
        Args:
            symbol_binary: Binary image of the symbol
            
        Returns:
            Dictionary of features
        """
        h, w = symbol_binary.shape
        
        # Basic features
        total_pixels = h * w
        white_pixels = np.sum(symbol_binary > 0)
        density = white_pixels / total_pixels if total_pixels > 0 else 0
        
        # Contour features
        contours, _ = cv2.findContours(
            symbol_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        num_contours = len(contours)
        
        # Edge features
        edges = cv2.Canny(symbol_binary, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels if total_pixels > 0 else 0
        
        # Horizontal line detection (common in receptacles)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=10, 
            minLineLength=5, 
            maxLineGap=3
        )
        num_lines = len(lines) if lines is not None else 0
        
        # Circular feature detection
        circles = cv2.HoughCircles(
            symbol_binary, cv2.HOUGH_GRADIENT, 
            dp=1, minDist=10, 
            param1=50, param2=20, 
            minRadius=5, maxRadius=min(h, w)//2
        )
        num_circles = len(circles[0]) if circles is not None else 0
        
        return {
            'density': density,
            'num_contours': num_contours,
            'edge_density': edge_density,
            'num_lines': num_lines,
            'num_circles': num_circles,
            'aspect_ratio': w / h if h > 0 else 0
        }

    def _remove_duplicate_templates(self, templates: List[Dict]) -> List[Dict]:
        """
        Remove duplicate templates based on similarity.
        
        Args:
            templates: List of template dictionaries
            
        Returns:
            Deduplicated list of templates
        """
        if len(templates) <= 1:
            return templates
        
        unique_templates = []
        
        for i, template in enumerate(templates):
            is_duplicate = False
            
            for existing in unique_templates:
                # Compare sizes
                if (abs(template['binary'].shape[0] - existing['binary'].shape[0]) < 10 and
                    abs(template['binary'].shape[1] - existing['binary'].shape[1]) < 10):
                    
                    # Compare content using template matching
                    try:
                        result = cv2.matchTemplate(
                            template['binary'].astype(np.uint8),
                            existing['binary'].astype(np.uint8),
                            cv2.TM_CCOEFF_NORMED
                        )
                        if result[0][0] > 0.8:
                            is_duplicate = True
                            break
                    except Exception as e:
                        pass
            
            if not is_duplicate:
                unique_templates.append(template)
        
        return unique_templates

    def _visualize_templates(self):
        """Visualize and save extracted templates."""
        if not self.receptacle_templates:
            return
        
        # Create a combined visualization
        max_width = max(t['binary'].shape[1] for t in self.receptacle_templates)
        max_height = max(t['binary'].shape[0] for t in self.receptacle_templates)
        
        cols = min(5, len(self.receptacle_templates))
        rows = (len(self.receptacle_templates) + cols - 1) // cols
        
        canvas = np.ones(
            (rows * (max_height + 30), cols * (max_width + 10), 3), 
            dtype=np.uint8
        ) * 255
        
        for idx, template in enumerate(self.receptacle_templates):
            row = idx // cols
            col = idx % cols
            
            symbol = template['color']
            h, w = symbol.shape[:2]
            
            y_start = row * (max_height + 30)
            x_start = col * (max_width + 10)
            
            # Place symbol
            if len(symbol.shape) == 2:
                symbol = cv2.cvtColor(symbol, cv2.COLOR_GRAY2BGR)
            
            canvas[y_start:y_start+h, x_start:x_start+w] = symbol
            
            # Add label
            cv2.putText(
                canvas, f"Template {idx+1}",
                (x_start, y_start + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
        
        cv2.imwrite(os.path.join(self.output_dir, "extracted_templates.jpg"), canvas)
        logger.info(f"Templates visualization saved")

    def count_receptacles_in_power_page(self, visualize: bool = True) -> int:
        """
        Count receptacle symbols in the power page.
        
        Args:
            visualize: Whether to save visualization images
            
        Returns:
            Number of receptacles detected
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Counting receptacles in power page...")
        logger.info("=" * 60)
        
        power_img = cv2.imread(self.power_path)
        if power_img is None:
            raise FileNotFoundError(f"Could not load power page: {self.power_path}")
        
        logger.info(f"Power page loaded: {power_img.shape}")
        
        if not self.receptacle_templates:
            logger.warning("No templates available. Using contour-based detection instead.")
            return self._contour_based_detection(power_img, visualize)
        
        gray_power, binary_power = self.preprocess_image(power_img, apply_morphology=True)
        
        # Save preprocessed power page
        cv2.imwrite(os.path.join(self.output_dir, "power_binary.jpg"), binary_power)
        
        # Try template matching first
        template_matches = self._template_matching_detection(
            gray_power, binary_power, power_img
        )
        
        logger.info(f"Template matching found {len(template_matches)} candidates")
        
        # Apply non-maximum suppression
        final_matches = self._non_max_suppression(template_matches)
        
        logger.info(f"After NMS: {len(final_matches)} unique detections")
        
        # Also try contour-based detection as supplement
        contour_matches = self._contour_based_detection(power_img, visualize=False)
        logger.info(f"Contour-based detection found {len(contour_matches)} candidates")
        
        # Combine results (union of both methods)
        all_matches = final_matches + contour_matches
        final_combined = self._non_max_suppression(all_matches)
        
        # Visualize results
        if visualize:
            self._visualize_detections(power_img, final_combined, "final_detections.jpg")
        
        logger.info("=" * 60)
        logger.info(f"FINAL COUNT: {len(final_combined)} receptacle symbols detected")
        logger.info("=" * 60)
        
        # Save results to JSON
        self._save_results(final_combined)
        
        return len(final_combined)

    def _template_matching_detection(
        self, 
        gray_power: np.ndarray, 
        binary_power: np.ndarray,
        original_img: np.ndarray
    ) -> List[DetectionResult]:
        """
        Detect receptacles using template matching.
        
        Args:
            gray_power: Grayscale power page
            binary_power: Binary power page
            original_img: Original power page image
            
        Returns:
            List of DetectionResult objects
        """
        matches = []
        
        for template_idx, template in enumerate(self.receptacle_templates):
            logger.info(f"Testing template {template_idx + 1}/{len(self.receptacle_templates)}")
            
            # Try both binary and grayscale templates
            for img_type in ['binary', 'gray']:
                template_img = template[img_type].astype(np.uint8)
                
                if len(template_img.shape) == 2:
                    # Ensure correct format for matching
                    if img_type == 'binary':
                        template_img = 255 - template_img  # Invert if needed
                
                # Multi-scale detection
                for scale in self.scale_factors:
                    try:
                        scaled_template = cv2.resize(
                            template_img, None,
                            fx=scale, fy=scale,
                            interpolation=cv2.INTER_LINEAR
                        )
                    except Exception as e:
                        continue
                    
                    h, w = scaled_template.shape[:2]
                    
                    # Skip if template is too large
                    if h > gray_power.shape[0] or w > gray_power.shape[1]:
                        continue
                    if h < 20 or w < 20:
                        continue
                    
                    # Prepare images for matching
                    if len(scaled_template.shape) == 3:
                        scaled_template = cv2.cvtColor(scaled_template, cv2.COLOR_BGR2GRAY)
                    
                    search_img = gray_power.copy()
                    
                    try:
                        # Perform template matching
                        result = cv2.matchTemplate(
                            search_img, scaled_template,
                            cv2.TM_CCOEFF_NORMED
                        )
                        
                        # Find matches above threshold
                        locations = np.where(result >= self.min_match_threshold)
                        
                        for pt in zip(*locations[::-1]):
                            matches.append(DetectionResult(
                                x=pt[0],
                                y=pt[1],
                                width=w,
                                height=h,
                                confidence=result[pt[1], pt[0]],
                                method=f'template_{template_idx}_{img_type}_scale{scale:.1f}'
                            ))
                    except Exception as e:
                        logger.debug(f"Matching failed: {e}")
                        continue
        
        return matches

    def _contour_based_detection(
        self, 
        power_img: np.ndarray, 
        visualize: bool = True
    ) -> List[DetectionResult]:
        """
        Detect receptacles using contour analysis.
        
        Args:
            power_img: Power page image
            visualize: Whether to save visualization
            
        Returns:
            List of DetectionResult objects
        """
        gray, binary = self.preprocess_image(power_img)
        
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        matches = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < 100 or area > 10000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 15 or h < 15 or w > 200 or h > 200:
                continue
            
            aspect_ratio = w / h if h > 0 else 0
            
            # Receptacles are typically circular or square-ish
            if 0.5 <= aspect_ratio <= 2.0:
                # Extract ROI
                roi = binary[y:y+h, x:x+w]
                
                # Check for internal features (lines for receptacle prongs)
                edges = cv2.Canny(roi, 50, 150)
                lines = cv2.HoughLinesP(
                    edges, 1, np.pi/180,
                    threshold=10,
                    minLineLength=5,
                    maxLineGap=5
                )
                
                confidence = 0.5  # Base confidence
                
                if lines is not None:
                    # More lines suggest more complex symbol (likely a receptacle)
                    num_lines = len(lines)
                    if num_lines >= 2:
                        confidence = 0.7
                    if num_lines >= 4:
                        confidence = 0.8
                
                # Check for circular features
                circles = cv2.HoughCircles(
                    roi, cv2.HOUGH_GRADIENT,
                    dp=1, minDist=10,
                    param1=50, param2=20,
                    minRadius=5, maxRadius=min(h, w)//2
                )
                
                if circles is not None:
                    confidence = max(confidence, 0.75)
                
                if confidence >= 0.5:
                    matches.append(DetectionResult(
                        x=x, y=y,
                        width=w, height=h,
                        confidence=confidence,
                        method='contour_analysis'
                    ))
        
        return matches

    def _non_max_suppression(
        self, 
        detections: List[DetectionResult],
        iou_threshold: float = 0.3
    ) -> List[DetectionResult]:
        """
        Apply non-maximum suppression to remove overlapping detections.
        
        Args:
            detections: List of DetectionResult objects
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []
        
        # Convert to numpy arrays for efficiency
        boxes = np.array([[d.x, d.y, d.x + d.width, d.y + d.height] for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        # Sort by confidence
        idxs = np.argsort(scores)[::-1]
        
        picked = []
        
        while len(idxs) > 0:
            # Take the highest confidence detection
            current = idxs[0]
            picked.append(current)
            
            if len(idxs) == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(boxes[current, 0], boxes[idxs[1:], 0])
            yy1 = np.maximum(boxes[current, 1], boxes[idxs[1:], 1])
            xx2 = np.minimum(boxes[current, 2], boxes[idxs[1:], 2])
            yy2 = np.minimum(boxes[current, 3], boxes[idxs[1:], 3])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / (
                (boxes[current, 2] - boxes[current, 0] + 1) * 
                (boxes[current, 3] - boxes[current, 1] + 1)
            )
            
            # Remove overlapping boxes
            idxs = np.delete(
                idxs,
                np.concatenate(([0], np.where(overlap > iou_threshold)[0] + 1))
            )
        
        return [detections[i] for i in picked]

    def _visualize_detections(
        self, 
        image: np.ndarray, 
        detections: List[DetectionResult],
        filename: str
    ):
        """
        Visualize detected receptacles.
        
        Args:
            image: Original image
            detections: List of detections
            filename: Output filename
        """
        result_img = image.copy()
        
        colors = {
            'template': (0, 255, 0),  # Green
            'contour': (255, 0, 0),   # Blue
            'combined': (0, 0, 255)   # Red
        }
        
        for i, det in enumerate(detections):
            # Choose color based on method
            if 'template' in det.method:
                color = colors['template']
            elif 'contour' in det.method:
                color = colors['contour']
            else:
                color = colors['combined']
            
            # Draw bounding box
            cv2.rectangle(
                result_img,
                (det.x, det.y),
                (det.x + det.width, det.y + det.height),
                color, 2
            )
            
            # Draw label
            label = f"#{i+1} ({det.confidence:.2f})"
            cv2.putText(
                result_img, label,
                (det.x, det.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1
            )
        
        # Add legend
        cv2.putText(
            result_img, f"Total Receptacles: {len(detections)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2
        )
        
        cv2.putText(
            result_img, "Green: Template Match | Blue: Contour Analysis",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 0), 1
        )
        
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, result_img)
        logger.info(f"Detection visualization saved: {output_path}")

    def _save_results(self, detections: List[DetectionResult]):
        """
        Save detection results to JSON file.
        
        Args:
            detections: List of detected receptacles
        """
        results = {
            'total_count': len(detections),
            'detections': [
                {
                    'id': i + 1,
                    'x': d.x,
                    'y': d.y,
                    'width': d.width,
                    'height': d.height,
                    'confidence': d.confidence,
                    'method': d.method
                }
                for i, d in enumerate(detections)
            ],
            'legend_templates_used': len(self.receptacle_templates),
            'detection_threshold': self.min_match_threshold
        }
        
        output_path = os.path.join(self.output_dir, 'detection_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Detect and count electrical receptacle symbols in drawings.'
    )
    parser.add_argument(
        'legend', 
        type=str, 
        help='Path to legend page image'
    )
    parser.add_argument(
        'power', 
        type=str, 
        help='Path to power plan page image'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.7,
        help='Detection confidence threshold (0.0-1.0, default: 0.7)'
    )
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Disable visualization output'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.legend):
        logger.error(f"Legend file not found: {args.legend}")
        sys.exit(1)
    
    if not os.path.exists(args.power):
        logger.error(f"Power page file not found: {args.power}")
        sys.exit(1)
    
    try:
        # Initialize detector
        detector = ElectricalSymbolDetector(
            legend_image_path=args.legend,
            power_page_path=args.power,
            output_dir=args.output,
            min_confidence=args.confidence
        )
        
        # Extract symbols from legend
        templates = detector.extract_symbols_from_legend()
        
        if not templates:
            logger.warning("No templates extracted from legend!")
            logger.info("Proceeding with contour-based detection only...")
        
        # Count receptacles in power page
        count = detector.count_receptacles_in_power_page(
            visualize=not args.no_visualization
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Receptacles detected: {count}")
        logger.info(f"Results saved in: {args.output}/")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
