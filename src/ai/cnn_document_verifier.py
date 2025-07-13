#!/usr/bin/env python3
"""
CNN Document Verifier for AIVA Document Verification System
Uses MobileNetV2 and ResNet50 for document authenticity verification
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

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2, ResNet50
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")

try:
    import pytesseract
    # Set Tesseract path for Windows
    import platform
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Check if Tesseract is available
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except (ImportError, FileNotFoundError) as e:
    TESSERACT_AVAILABLE = False
    print("Warning: Pytesseract not found or Tesseract is not in your PATH. OCR features will be disabled.")
    print("Install Tesseract from: https://github.com/tesseract-ocr/tesseract and/or run: pip install pytesseract")
    print(f"Error details: {e}")


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


class CNNDocumentVerifier:
    """
    CNN-based document verification system
    Uses MobileNetV2 and ResNet50 for classification and tampering detection
    """
    
    def __init__(self, model_type: str = "mobilenet", confidence_threshold: float = 0.7):
        """
        Initialize CNN document verifier
        
        Args:
            model_type: 'mobilenet' or 'resnet'
            confidence_threshold: Minimum confidence for authentic classification
        """
        self.model_type = model_type.lower()
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.feature_extractor = None
        self.initialized = False
        
        if not TENSORFLOW_AVAILABLE:
            logger.error("❌ TensorFlow not available. Please install: pip install tensorflow")
            return
        
        if not TESSERACT_AVAILABLE:
            logger.warning(" OCR features will be limited.")

        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the CNN model"""
        try:
            model_path = Path("resnet_aadhar_trained.h5")
            if not model_path.exists():
                error_msg = f"CRITICAL_ERROR: AI model file not found at '{model_path.resolve()}'. Please ensure the model is in the root directory."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            logger.info(f"🔧 Loading custom trained model from {model_path}...")
            
            # Load your custom-trained model
            self.model = tf.keras.models.load_model(model_path)
            self.preprocess_func = resnet_preprocess # Assuming your custom model uses ResNet preprocessing
            
            # The feature extractor part might need adjustment depending on your custom model's architecture.
            # For now, we'll try to create it, but this might need refinement.
            try:
                # Find a suitable layer for feature extraction. 'conv5_block3_out' is typical for ResNet50.
                feature_layer_name = 'conv5_block3_out' 
                self.feature_extractor = Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer(feature_layer_name).output
                )
            except ValueError:
                logger.warning(f"Could not find layer '{feature_layer_name}'. Using last GlobalAveragePooling layer for features.")
                # Fallback to a layer that should exist
                pool_layers = [layer for layer in self.model.layers if isinstance(layer, GlobalAveragePooling2D)]
                if pool_layers:
                    self.feature_extractor = Model(
                        inputs=self.model.input,
                        outputs=pool_layers[-1].output
                    )
                else:
                    self.feature_extractor = None
                    logger.error("Could not create a feature extractor for the custom model.")

            self.initialized = True
            logger.info(f"✅ Custom model loaded and initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing model: {e}", exc_info=True)
            self.initialized = False
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for CNN input
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load and resize image
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Apply preprocessing
            img_array = self.preprocess_func(img_array)
            
            return img_array
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing image: {e}")
            return None
    
    def detect_tampering(self, image_path: str) -> Dict[str, Any]:
        """
        Detect tampering in document image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tampering detection results
        """
        try:
            # Load image for analysis
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not load image"}
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            tampering_indicators = []
            suspicious_regions = []
            
            # 1. Detect text inconsistencies (overwritten text)
            # Use OCR to detect text regions and check for inconsistencies
            text_inconsistencies = self._detect_text_inconsistencies(img_gray)
            if text_inconsistencies:
                tampering_indicators.append("Text inconsistencies detected")
                suspicious_regions.extend(text_inconsistencies)
            
            # 2. Detect missing or misplaced seals
            seal_issues = self._detect_seal_issues(img_gray)
            if seal_issues:
                tampering_indicators.append("Seal issues detected")
                suspicious_regions.extend(seal_issues)
            
            # 3. Detect layout anomalies
            layout_issues = self._detect_layout_anomalies(img_gray)
            if layout_issues:
                tampering_indicators.append("Layout anomalies detected")
                suspicious_regions.extend(layout_issues)
            
            # 4. Detect copy-paste artifacts
            copy_paste_artifacts = self._detect_copy_paste_artifacts(img_gray)
            if copy_paste_artifacts:
                tampering_indicators.append("Copy-paste artifacts detected")
                suspicious_regions.extend(copy_paste_artifacts)
            
            return {
                "tampering_detected": len(tampering_indicators) > 0,
                "indicators": tampering_indicators,
                "suspicious_regions": suspicious_regions,
                "confidence": max(0.1, 1.0 - len(tampering_indicators) * 0.2)
            }
            
        except Exception as e:
            logger.error(f"❌ Error detecting tampering: {e}")
            return {"error": str(e)}
    
    def _detect_text_inconsistencies(self, img_gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text inconsistencies in the image using Tesseract OCR"""
        if not TESSERACT_AVAILABLE:
            return []

        try:
            # Use OCR to get detailed data about words and their confidence
            ocr_data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT)
            
            inconsistent_regions = []
            num_boxes = len(ocr_data['level'])
            for i in range(num_boxes):
                confidence = int(ocr_data['conf'][i])
                # Low confidence might indicate a manipulated area
                if 0 < confidence < 60: # Confidence threshold can be tuned
                    (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                    inconsistent_regions.append({
                        "type": "low_confidence_text",
                        "bbox": [x, y, w, h],
                        "confidence": confidence,
                        "text": ocr_data['text'][i]
                    })
            
            return inconsistent_regions

        except Exception as e:
            logger.error(f"❌ Error during OCR-based text inconsistency detection: {e}")
            return []
    
    def _detect_seal_issues(self, img_gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect missing or misplaced seals"""
        try:
            # Detect circular regions (potential seals)
            circles = cv2.HoughCircles(
                img_gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=50
            )
            
            seal_regions = []
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # Draw the outer circle
                    seal_regions.append({
                        "type": "detected_seal",
                        "bbox": [i[0] - i[2], i[1] - i[2], i[2] * 2, i[2] * 2], # x, y, w, h
                        "confidence": 0.8
                    })
            
            return seal_regions
            
        except Exception as e:
            logger.error(f"❌ Error detecting seal issues: {e}")
            return []
    
    def _detect_layout_anomalies(self, img_gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect major layout anomalies like skewed lines"""
        try:
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Find regions with unusual line patterns
            combined_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
            
            # Detect regions with low line density (potential tampering)
            regions = []
            h, w = img_gray.shape
            for y in range(0, h, 50):
                for x in range(0, w, 50):
                    region = combined_lines[y:y+50, x:x+50]
                    if np.mean(region) < 10:  # Low line density
                        regions.append({
                            "type": "layout_anomaly",
                            "bbox": [x, y, 50, 50],
                            "confidence": 0.5
                        })
            
            return regions[:3]  # Return top 3 suspicious regions
            
        except Exception as e:
            logger.error(f"❌ Error detecting layout anomalies: {e}")
            return []
    
    def _detect_copy_paste_artifacts(self, img_gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect potential copy-paste artifacts using template matching on edges"""
        try:
            # Use Hough Line Transform to detect lines
            edges = cv2.Canny(img_gray, 50, 150) # Apply Canny edge detection
            lines = cv2.HoughLinesP(
                edges,
                cv2.HOUGH_PROBABILISTIC,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=50,
                maxLineGap=10
            )

            layout_issues = []
            if lines is not None:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                    angles.append(angle)
                
                # Check for significant deviation in line angles
                if np.std(angles) > 2: # High standard deviation might mean layout issues
                    layout_issues.append({
                        "type": "layout_skew",
                        "confidence": np.std(angles) / 10
                    })

            return layout_issues

        except Exception as e:
            logger.error(f"❌ Error detecting copy-paste artifacts: {e}")
            return []
    
    def classify_document(self, image_path: str) -> Dict[str, Any]:
        """
        Classify document as authentic or fake
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Classification results
        """
        try:
            if not self.initialized:
                return {"error": "Model not initialized"}
            
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            if img_array is None:
                return {"error": "Failed to preprocess image"}
            
            # Get prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Debug: Log prediction shape
            logger.info(f"Model prediction shape: {predictions.shape}")
            logger.info(f"Model prediction values: {predictions}")
            
            # Handle different prediction shapes robustly
            if len(predictions.shape) == 1:
                # Single prediction value
                prediction_value = float(predictions[0])
                authentic_prob = prediction_value
                fake_prob = 1.0 - prediction_value
            elif len(predictions.shape) == 2:
                # Multiple classes
                if predictions.shape[1] == 1:
                    # Single class prediction
                    prediction_value = float(predictions[0][0])
                    authentic_prob = prediction_value
                    fake_prob = 1.0 - prediction_value
                elif predictions.shape[1] == 2:
                    # Two classes (fake, authentic)
                    fake_prob = float(predictions[0][0])
                    authentic_prob = float(predictions[0][1])
                else:
                    # More than 2 classes, take the highest probability
                    max_idx = np.argmax(predictions[0])
                    authentic_prob = float(predictions[0][max_idx])
                    fake_prob = 1.0 - authentic_prob
            else:
                # Unexpected shape, use fallback
                logger.warning(f"Unexpected prediction shape: {predictions.shape}")
                authentic_prob = 0.5
                fake_prob = 0.5
            
            # Determine result
            is_authentic = authentic_prob > self.confidence_threshold
            confidence_score = max(authentic_prob, fake_prob)
            
            # Determine document type based on image analysis
            document_type = self._classify_document_type(image_path)
            
            return {
                "is_authentic": is_authentic,
                "confidence_score": float(confidence_score),
                "authentic_probability": float(authentic_prob),
                "fake_probability": float(fake_prob),
                "document_type": document_type,
                "model_used": self.model_type
            }
            
        except Exception as e:
            logger.error(f"❌ Error classifying document: {e}")
            return {"error": str(e)}
    
    def _classify_document_type(self, image_path: str) -> str:
        """Classify document type (e.g., Aadhaar, Passport) - Simplified for demo"""
        # This would typically be a separate classifier model
        # For the demo, we'll use a simple heuristic based on features
        try:
            img = cv2.imread(image_path)
            if img is None:
                return "unknown"
            
            h, w = img.shape[:2]
            
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Check for common document colors
            blue_pixels = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            green_pixels = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
            red_pixels = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            
            blue_ratio = np.sum(blue_pixels > 0) / (h * w)
            green_ratio = np.sum(green_pixels > 0) / (h * w)
            red_ratio = np.sum(red_pixels > 0) / (h * w)
            
            # Check for Aadhaar-like blue header
            blue_lower = np.array([100, 50, 50])
            blue_upper = np.array([130, 255, 255])
            blue_pixels_hsv = cv2.inRange(hsv, blue_lower, blue_upper)
            has_blue_header = np.sum(blue_pixels_hsv) > 5000 # Heuristic value
            
            # Check for tricolor logo
            green_pixels_hsv = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
            red_pixels_hsv = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            has_tricolor_logo = np.sum(green_pixels_hsv) > 1000 and np.sum(red_pixels_hsv) > 1000
            
            # Simple classification logic
            if has_blue_header and has_tricolor_logo:
                return "Aadhaar Card"
            elif blue_ratio > 0.1:
                return "Government ID"
            elif green_ratio > 0.1:
                return "Passport"
            else:
                return "Unknown Document"
            
        except Exception as e:
            logger.error(f"Color feature extraction failed: {e}")
            return "Unknown Document"

    def verify_document(self, image_path: str) -> Dict[str, Any]:
        """
        Verify the authenticity of a document image
        
        Args:
            image_path: Path to the document image
            
        Returns:
            A dictionary with detailed verification results
        """
        start_time = time.time()
        
        if not self.initialized:
            return {
                "error": "Verifier not initialized. Check logs for details.",
                "fraud_score": 100
            }

        # 1. Classify document authenticity
        classification_result = self.classify_document(image_path)
        if "error" in classification_result:
            return { "error": classification_result["error"], "fraud_score": 100 }

        is_authentic_cnn = classification_result.get("is_authentic", False)
        confidence_cnn = classification_result.get("confidence_score", 0.0)

        # 2. Detect tampering
        tampering_result = self.detect_tampering(image_path)
        if "error" in tampering_result:
            return { "error": tampering_result["error"], "fraud_score": 100 }

        # 3. Classify document type
        doc_type = self._classify_document_type(image_path)
        
        # 4. Calculate final fraud score
        fraud_score = (1 - confidence_cnn) * 50  # From classification
        if tampering_result.get("tampering_detected"):
            fraud_score += 50 # Add a heavy penalty for tampering
        
        fraud_score = min(100, int(fraud_score))

        # 5. Extract features and text for the report
        features = self._extract_features(image_path)

        # Deterministic document hash: use OCR text + color ratios + quality
        ocr_text = features.get("ocr_text", "")
        color_ratios = features.get("color_ratios", {})
        quality = features.get("quality", {})
        hash_input = ocr_text + json.dumps(color_ratios, sort_keys=True) + json.dumps(quality, sort_keys=True)
        document_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        processing_time = time.time() - start_time
        
        return {
            "document_type": doc_type,
            "is_authentic": is_authentic_cnn,
            "confidence": int(confidence_cnn * 100),
            "fraud_score": fraud_score,
            "analysis": {
                "tampering_indicators": tampering_result.get("indicators", []),
                "suspicious_regions": tampering_result.get("suspicious_regions", [])
            },
            "processing_time": round(processing_time, 2),
            "model_used": self.model_type,
            "features_detected": features,
            "extracted_text": ocr_text,
            "document_hash": document_hash
        }
    
    def _extract_features(self, image_path: str) -> Dict[str, Any]:
        """Extract visual and textual features from the document"""
        features = {}
        
        # OCR Text
        if TESSERACT_AVAILABLE:
            try:
                img = cv2.imread(image_path)
                features["ocr_text"] = pytesseract.image_to_string(img)
            except Exception as e:
                logger.error(f"Feature extraction OCR failed: {e}")
                features["ocr_text"] = "Error during OCR."
        else:
            features["ocr_text"] = "Tesseract OCR not available."
            
        # Color features (for document type classification)
        try:
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # Check for common document colors
                blue_pixels = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
                green_pixels = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
                red_pixels = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                
                blue_ratio = np.sum(blue_pixels > 0) / (h * w)
                green_ratio = np.sum(green_pixels > 0) / (h * w)
                red_ratio = np.sum(red_pixels > 0) / (h * w)
                
                # Check for Aadhaar-like blue header
                blue_lower = np.array([100, 50, 50])
                blue_upper = np.array([130, 255, 255])
                blue_pixels_hsv = cv2.inRange(hsv, blue_lower, blue_upper)
                features["has_blue_header"] = np.sum(blue_pixels_hsv) > 5000 # Heuristic value
                
                # Check for tricolor logo
                green_pixels_hsv = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
                red_pixels_hsv = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
                features["has_tricolor_logo"] = np.sum(green_pixels_hsv) > 1000 and np.sum(red_pixels_hsv) > 1000
                
                # Color ratios
                features["color_ratios"] = {
                    "blue": blue_ratio,
                    "green": green_ratio,
                    "red": red_ratio
                }
            
        except Exception as e:
            logger.error(f"Color feature extraction failed: {e}")

        # Image quality features
        try:
            img = cv2.imread(image_path)
            if img is not None:
                features["quality"] = {
                    "brightness": np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
                    "contrast": np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
                }
        except Exception as e:
            logger.error(f"Quality feature extraction failed: {e}")
            
        return features
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.initialized:
            return {"error": "Model not initialized"}
        return {
            "model_type": self.model_type,
            "confidence_threshold": self.confidence_threshold,
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape
        }


# Global CNN verifier instance
_cnn_verifier = None

def get_cnn_verifier(model_type: str = "resnet") -> CNNDocumentVerifier:
    """Get global CNN verifier instance"""
    global _cnn_verifier
    if _cnn_verifier is None or _cnn_verifier.model_type != model_type:
        _cnn_verifier = CNNDocumentVerifier(model_type=model_type)
    return _cnn_verifier

def verify_document_cnn(image_path: str, model_type: str = "resnet") -> Dict[str, Any]:
    """Convenience function to verify a document using CNN"""
    verifier = get_cnn_verifier(model_type)
    return verifier.verify_document(image_path) 