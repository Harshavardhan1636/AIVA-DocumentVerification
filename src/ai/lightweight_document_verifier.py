#!/usr/bin/env python3
"""
Lightweight Document Verifier for AIVA Document Verification System
Uses OpenCV and traditional computer vision techniques for document verification
"""

import os
import sys
import time
import json
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of document verification"""
    is_authentic: bool
    confidence_score: float
    document_type: str
    suspicious_regions: List[Dict[str, Any]]
    tampering_indicators: List[str]
    processing_time: float
    model_used: str
    features_detected: Dict[str, Any]
    document_hash: str


class LightweightDocumentVerifier:
    """
    Lightweight document verification system using OpenCV
    Performs document authenticity verification without deep learning
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize lightweight document verifier
        
        Args:
            confidence_threshold: Minimum confidence for authentic classification
        """
        self.confidence_threshold = confidence_threshold
        self.model_used = "opencv_lightweight"
        
        # Document type templates (simple patterns)
        self.document_templates = {
            "government_id": {
                "aspect_ratio_range": (1.2, 1.8),
                "color_dominance": "blue",
                "min_text_regions": 3,
                "expected_seals": 1
            },
            "passport": {
                "aspect_ratio_range": (1.4, 1.6),
                "color_dominance": "blue",
                "min_text_regions": 5,
                "expected_seals": 2
            },
            "bank_statement": {
                "aspect_ratio_range": (1.0, 1.5),
                "color_dominance": "white",
                "min_text_regions": 8,
                "expected_seals": 0
            },
            "certificate": {
                "aspect_ratio_range": (1.2, 1.4),
                "color_dominance": "white",
                "min_text_regions": 4,
                "expected_seals": 1
            },
            "pan_card": {
                "aspect_ratio_range": (1.5, 2.2),
                "color_dominance": "white",
                "min_text_regions": 4,
                "expected_seals": 1
            },
            "drivers_license": {
                "aspect_ratio_range": (1.5, 2.5),
                "color_dominance": "green",
                "min_text_regions": 5,
                "expected_seals": 1
            }
        }
    
    def verify_document(self, image_path: str) -> VerificationResult:
        """
        Complete document verification process
        
        Args:
            image_path: Path to the image file
            
        Returns:
            VerificationResult object
        """
        start_time = time.time()
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return self._create_error_result("Could not load image", start_time)
            
            # Calculate document hash
            document_hash = self._calculate_document_hash(img)
            
            # Step 1: Extract features
            features = self._extract_features(img)
            
            # Step 2: Classify document type
            document_type = self._classify_document_type(features)
            
            # Step 3: Detect tampering
            tampering_result = self._detect_tampering(img)
            
            # Step 4: Calculate authenticity score
            authenticity_score = self._calculate_authenticity_score(features, tampering_result)
            
            # Step 5: Determine final result
            is_authentic = authenticity_score >= self.confidence_threshold
            
            processing_time = time.time() - start_time
            
            return VerificationResult(
                is_authentic=is_authentic,
                confidence_score=authenticity_score,
                document_type=document_type,
                suspicious_regions=tampering_result["suspicious_regions"],
                tampering_indicators=tampering_result["indicators"],
                processing_time=processing_time,
                model_used=self.model_used,
                features_detected=features,
                document_hash=document_hash
            )
            
        except Exception as e:
            logger.error(f"❌ Error in document verification: {e}")
            return self._create_error_result(str(e), start_time)
    
    def _calculate_document_hash(self, img: np.ndarray) -> str:
        """Calculate SHA-256 hash of the document image"""
        try:
            # Convert to grayscale and resize for consistent hashing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (256, 256))
            
            # Calculate hash
            img_bytes = resized.tobytes()
            return hashlib.sha256(img_bytes).hexdigest()
            
        except Exception as e:
            logger.error(f"❌ Error calculating document hash: {e}")
            return "unknown"
    
    def _extract_features(self, img: np.ndarray) -> Dict[str, Any]:
        """Extract features from the document image"""
        try:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Basic features
            features = {
                "dimensions": {"width": w, "height": h},
                "aspect_ratio": w / h,
                "brightness": np.mean(gray),
                "contrast": np.std(gray),
                "color_channels": {
                    "blue": np.mean(img[:, :, 0]),
                    "green": np.mean(img[:, :, 1]),
                    "red": np.mean(img[:, :, 2])
                }
            }
            
            # Detect text regions
            text_regions = self._detect_text_regions(gray)
            features["text_regions"] = len(text_regions)
            features["text_density"] = len(text_regions) / (w * h) * 1000000  # Normalized
            
            # Detect seals/stamps
            seals = self._detect_seals(gray)
            features["seals_detected"] = len(seals)
            
            # Detect lines (for structured documents)
            lines = self._detect_lines(gray)
            features["lines_detected"] = len(lines)
            
            # Color analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            features["color_dominance"] = self._analyze_color_dominance(hsv)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting features: {e}")
            return {}
    
    def _detect_text_regions(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text regions in the image"""
        try:
            # Use edge detection to find text-like regions
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  # Filter by size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Text regions typically have specific aspect ratios
                    if 0.1 < aspect_ratio < 10:
                        text_regions.append({
                            "bbox": [x, y, w, h],
                            "area": area,
                            "aspect_ratio": aspect_ratio
                        })
            
            return text_regions
            
        except Exception as e:
            logger.error(f"❌ Error detecting text regions: {e}")
            return []
    
    def _detect_seals(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect seals/stamps in the image"""
        try:
            # Use Hough Circle Transform to detect circular seals
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=20,
                maxRadius=100
            )
            
            seals = []
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    seals.append({
                        "center": (circle[0], circle[1]),
                        "radius": circle[2],
                        "bbox": [circle[0] - circle[2], circle[1] - circle[2], circle[2] * 2, circle[2] * 2]
                    })
            
            return seals
            
        except Exception as e:
            logger.error(f"❌ Error detecting seals: {e}")
            return []
    
    def _detect_lines(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect lines in the image"""
        try:
            # Use Hough Line Transform
            lines = cv2.HoughLinesP(
                gray,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=50,
                maxLineGap=10
            )
            
            detected_lines = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    detected_lines.append({
                        "start": (x1, y1),
                        "end": (x2, y2),
                        "length": length
                    })
            
            return detected_lines
            
        except Exception as e:
            logger.error(f"❌ Error detecting lines: {e}")
            return []
    
    def _analyze_color_dominance(self, hsv: np.ndarray) -> str:
        """Analyze dominant colors in the image"""
        try:
            # Define color ranges
            blue_lower = np.array([100, 50, 50])
            blue_upper = np.array([130, 255, 255])
            green_lower = np.array([40, 50, 50])
            green_upper = np.array([80, 255, 255])
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 50, 50])
            red_upper2 = np.array([180, 255, 255])
            
            # Count pixels in each color range
            blue_pixels = cv2.inRange(hsv, blue_lower, blue_upper)
            green_pixels = cv2.inRange(hsv, green_lower, green_upper)
            red_pixels1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_pixels2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_pixels = cv2.bitwise_or(red_pixels1, red_pixels2)
            
            total_pixels = hsv.shape[0] * hsv.shape[1]
            blue_ratio = np.sum(blue_pixels > 0) / total_pixels
            green_ratio = np.sum(green_pixels > 0) / total_pixels
            red_ratio = np.sum(red_pixels > 0) / total_pixels
            
            # Determine dominant color
            if blue_ratio > 0.1:
                return "blue"
            elif green_ratio > 0.1:
                return "green"
            elif red_ratio > 0.1:
                return "red"
            else:
                return "white"
                
        except Exception as e:
            logger.error(f"❌ Error analyzing color dominance: {e}")
            return "unknown"
    
    def _classify_document_type(self, features: Dict[str, Any]) -> str:
        """Classify document type based on features (improved logic)"""
        try:
            if not features:
                return "unknown"
            aspect_ratio = features.get("aspect_ratio", 1.0)
            color_dominance = features.get("color_dominance", "unknown")
            text_regions = features.get("text_regions", 0)
            seals_detected = features.get("seals_detected", 0)
            best_match = "unknown"
            best_score = 0
            for doc_type, template in self.document_templates.items():
                score = 0
                ar_range = template["aspect_ratio_range"]
                if ar_range[0] <= aspect_ratio <= ar_range[1]:
                    score += 0.3
                if template["color_dominance"] == color_dominance:
                    score += 0.2
                if text_regions >= template["min_text_regions"]:
                    score += 0.3
                if seals_detected >= template["expected_seals"]:
                    score += 0.2
                if score > best_score:
                    best_score = score
                    best_match = doc_type
            # If no strong match, fallback to general_document
            return best_match if best_score > 0.4 else "general_document"
        except Exception as e:
            logger.error(f"❌ Error classifying document type: {e}")
            return "unknown"
    
    def _detect_tampering(self, img: np.ndarray) -> Dict[str, Any]:
        """Detect tampering in the document"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            indicators = []
            suspicious_regions = []
            
            # 1. Detect copy-paste artifacts (duplicate regions)
            duplicates = self._detect_duplicate_regions(gray)
            if duplicates:
                indicators.append("Duplicate regions detected")
                suspicious_regions.extend(duplicates)
            
            # 2. Detect text inconsistencies
            text_issues = self._detect_text_inconsistencies(gray)
            if text_issues:
                indicators.append("Text inconsistencies detected")
                suspicious_regions.extend(text_issues)
            
            # 3. Detect missing elements
            missing_elements = self._detect_missing_elements(gray)
            if missing_elements:
                indicators.append("Missing elements detected")
                suspicious_regions.extend(missing_elements)
            
            # 4. Detect noise inconsistencies
            noise_issues = self._detect_noise_inconsistencies(gray)
            if noise_issues:
                indicators.append("Noise inconsistencies detected")
                suspicious_regions.extend(noise_issues)
            
            return {
                "tampering_detected": len(indicators) > 0,
                "indicators": indicators,
                "suspicious_regions": suspicious_regions,
                "confidence": max(0.1, 1.0 - len(indicators) * 0.2)
            }
            
        except Exception as e:
            logger.error(f"❌ Error detecting tampering: {e}")
            return {
                "tampering_detected": False,
                "indicators": [str(e)],
                "suspicious_regions": [],
                "confidence": 0.5
            }
    
    def _detect_duplicate_regions(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect duplicate regions (copy-paste artifacts)"""
        try:
            regions = []
            h, w = gray.shape
            
            # Check for duplicate patterns at different scales
            for size in [30, 50, 70]:
                for y in range(0, h - size, size // 2):
                    for x in range(0, w - size, size // 2):
                        template = gray[y:y+size, x:x+size]
                        
                        # Search for similar regions
                        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                        locations = np.where(result >= 0.8)
                        
                        if len(locations[0]) > 1:  # Multiple similar regions found
                            regions.append({
                                "type": "duplicate_region",
                                "bbox": [x, y, size, size],
                                "confidence": 0.8
                            })
            
            return regions[:3]  # Return top 3 suspicious regions
            
        except Exception as e:
            logger.error(f"❌ Error detecting duplicate regions: {e}")
            return []
    
    def _detect_text_inconsistencies(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text inconsistencies"""
        try:
            regions = []
            
            # Use morphological operations to detect text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in morphological result
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 5000:  # Text-like regions
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check for unusual text patterns
                    if w > 0 and h > 0:
                        aspect_ratio = w / h
                        if aspect_ratio > 20 or aspect_ratio < 0.1:  # Unusual text
                            regions.append({
                                "type": "text_inconsistency",
                                "bbox": [x, y, w, h],
                                "confidence": 0.6
                            })
            
            return regions[:5]  # Return top 5 suspicious regions
            
        except Exception as e:
            logger.error(f"❌ Error detecting text inconsistencies: {e}")
            return []
    
    def _detect_missing_elements(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect missing elements (seals, signatures, etc.)"""
        try:
            regions = []
            h, w = gray.shape
            
            # Check for large empty regions (missing elements)
            for y in range(0, h, 100):
                for x in range(0, w, 100):
                    region = gray[y:y+100, x:x+100]
                    if np.mean(region) > 240:  # Very bright region (empty)
                        regions.append({
                            "type": "missing_element",
                            "bbox": [x, y, 100, 100],
                            "confidence": 0.5
                        })
            
            return regions[:3]  # Return top 3 suspicious regions
            
        except Exception as e:
            logger.error(f"❌ Error detecting missing elements: {e}")
            return []
    
    def _detect_noise_inconsistencies(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect noise inconsistencies (copy-paste artifacts)"""
        try:
            regions = []
            h, w = gray.shape
            
            # Calculate local noise levels
            for y in range(0, h - 50, 50):
                for x in range(0, w - 50, 50):
                    region = gray[y:y+50, x:x+50]
                    noise_level = np.std(region)
                    
                    # Check for unusually low or high noise
                    if noise_level < 5 or noise_level > 50:
                        regions.append({
                            "type": "noise_inconsistency",
                            "bbox": [x, y, 50, 50],
                            "confidence": 0.7
                        })
            
            return regions[:3]  # Return top 3 suspicious regions
            
        except Exception as e:
            logger.error(f"❌ Error detecting noise inconsistencies: {e}")
            return []
    
    def _calculate_authenticity_score(self, features: Dict[str, Any], tampering_result: Dict[str, Any]) -> float:
        """Calculate overall authenticity score"""
        try:
            score = 0.5  # Base score
            
            # Feature-based scoring
            if features:
                # Text density scoring
                text_density = features.get("text_density", 0)
                if 10 < text_density < 1000:  # Normal range
                    score += 0.1
                
                # Seal detection scoring
                seals = features.get("seals_detected", 0)
                if seals > 0:
                    score += 0.1
                
                # Line detection scoring (structured documents)
                lines = features.get("lines_detected", 0)
                if lines > 5:
                    score += 0.1
                
                # Aspect ratio scoring
                aspect_ratio = features.get("aspect_ratio", 1.0)
                if 0.5 < aspect_ratio < 2.0:  # Normal document ratios
                    score += 0.1
            
            # Tampering penalty
            if tampering_result["tampering_detected"]:
                score -= 0.3
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating authenticity score: {e}")
            return 0.5
    
    def _create_error_result(self, error_message: str, start_time: float) -> VerificationResult:
        """Create error result"""
        return VerificationResult(
            is_authentic=False,
            confidence_score=0.0,
            document_type="unknown",
            suspicious_regions=[],
            tampering_indicators=[error_message],
            processing_time=time.time() - start_time,
            model_used=self.model_used,
            features_detected={},
            document_hash="unknown"
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_type": self.model_used,
            "confidence_threshold": self.confidence_threshold,
            "document_templates": list(self.document_templates.keys())
        }


# Global lightweight verifier instance
_lightweight_verifier = None

def get_lightweight_verifier() -> LightweightDocumentVerifier:
    """Get global lightweight verifier instance"""
    global _lightweight_verifier
    if _lightweight_verifier is None:
        _lightweight_verifier = LightweightDocumentVerifier()
    return _lightweight_verifier

def verify_document_lightweight(image_path: str) -> VerificationResult:
    """Convenience function to verify a document using lightweight method"""
    verifier = get_lightweight_verifier()
    return verifier.verify_document(image_path) 