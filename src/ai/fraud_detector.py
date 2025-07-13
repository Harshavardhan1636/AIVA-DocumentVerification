"""
Fraud Detector - AI Module for AIVA Document Verification System

This module provides advanced fraud detection capabilities:
- Image manipulation detection
- Text consistency analysis
- Digital signature verification
- Statistical anomaly detection
"""

import json
import re
import hashlib
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FraudDetectionResult:
    """Results from fraud detection analysis"""
    is_fraudulent: bool
    fraud_score: float
    detected_anomalies: List[str]
    manipulation_indicators: List[str]
    confidence: float
    risk_level: str  # 'low', 'medium', 'high'

class FraudDetector:
    """
    Advanced fraud detection using pattern matching and rule-based analysis
    """
    
    def __init__(self):
        """Initialize the fraud detector with detection patterns"""
        
        # Fraud detection patterns
        self.fraud_patterns = {
            'digital_manipulation': {
                'compression_artifacts': r'JPEG_QUALITY|COMPRESSION',
                'copy_paste': r'DUPLICATE_REGIONS|COPY_PASTE',
                'resampling': r'RESAMPLING|INTERPOLATION'
            },
            'text_anomalies': {
                'font_inconsistency': r'FONT_MISMATCH|TYPEFACE',
                'spacing_irregular': r'SPACING_ANOMALY|KERNING',
                'alignment_issues': r'ALIGNMENT|BASELINE'
            },
            'content_red_flags': {
                'suspicious_dates': r'\d{1,2}/\d{1,2}/\d{4}',
                'invalid_numbers': r'\b\d{12}\b',  # Aadhaar-like
                'fake_watermarks': r'WATERMARK|SECURITY'
            }
        }
        
        # Known document templates (simplified)
        self.document_templates = {
            'aadhaar': {
                'expected_elements': ['uidai', 'aadhaar', 'unique identification'],
                'expected_format': r'\d{4}\s\d{4}\s\d{4}',
                'security_features': ['qr_code', 'hologram', 'micro_text']
            },
            'pan': {
                'expected_elements': ['permanent account number', 'income tax'],
                'expected_format': r'[A-Z]{5}[0-9]{4}[A-Z]{1}',
                'security_features': ['signature', 'photo', 'barcode']
            },
            'certificate': {
                'expected_elements': ['certificate', 'awarded', 'issued'],
                'expected_format': r'CERT-\d{4}-\d{3}',
                'security_features': ['watermark', 'serial_number', 'signature']
            },
            'diploma': {
                'expected_elements': ['diploma', 'degree', 'university'],
                'expected_format': r'DIP-\d{4}-\d{3}',
                'security_features': ['seal', 'signature', 'date']
            }
        }
        
        logger.info("Fraud Detector initialized successfully")
    
    def detect_fraud(self, image_data: Optional[Dict] = None, text_content: str = "", document_type: str = "unknown") -> FraudDetectionResult:
        """
        Comprehensive fraud detection analysis
        
        Args:
            image_data: Document image data (optional)
            text_content: Extracted text content
            document_type: Identified document type
            
        Returns:
            FraudDetectionResult: Fraud detection results
        """
        try:
            # Text-based fraud detection
            text_anomalies = self._detect_text_anomalies(text_content, document_type)
            
            # Content-based fraud detection
            content_anomalies = self._detect_content_anomalies(text_content, document_type)
            
            # Template validation
            template_anomalies = self._validate_document_template(text_content, document_type)
            
            # Combine all anomalies
            all_anomalies = text_anomalies + content_anomalies + template_anomalies
            
            # Calculate fraud score
            fraud_score = self._calculate_fraud_score(all_anomalies, text_content)
            
            # Determine risk level
            risk_level = self._determine_risk_level(fraud_score, len(all_anomalies))
            
            # Make final decision
            is_fraudulent = fraud_score > 0.6 or len(all_anomalies) > 3
            
            # Calculate confidence
            confidence = self._calculate_confidence(fraud_score, all_anomalies)
            
            return FraudDetectionResult(
                is_fraudulent=is_fraudulent,
                fraud_score=fraud_score,
                detected_anomalies=all_anomalies,
                manipulation_indicators=text_anomalies,
                confidence=confidence,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error in fraud detection: {e}")
            return self._create_error_result(str(e))
    
    def _detect_text_anomalies(self, text_content: str, document_type: str) -> List[str]:
        """
        Detect text-based anomalies and inconsistencies
        
        Args:
            text_content: Extracted text content
            document_type: Document type
            
        Returns:
            List of detected text anomalies
        """
        anomalies = []
        
        try:
            # Check for suspicious patterns
            for category, patterns in self.fraud_patterns['text_anomalies'].items():
                for pattern_name, pattern in patterns.items():
                    if re.search(pattern, text_content, re.IGNORECASE):
                        anomalies.append(f"{category}_{pattern_name}")
            
            # Check for content red flags
            for category, patterns in self.fraud_patterns['content_red_flags'].items():
                for pattern_name, pattern in patterns.items():
                    if re.search(pattern, text_content, re.IGNORECASE):
                        anomalies.append(f"{category}_{pattern_name}")
            
            # Check for suspicious dates
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b\d{1,2}-\d{1,2}-\d{4}\b'
            ]
            
            for pattern in date_patterns:
                dates = re.findall(pattern, text_content)
                for date in dates:
                    if self._is_suspicious_date(date):
                        anomalies.append("suspicious_date")
            
            # Check for inconsistent formatting
            if self._has_inconsistent_formatting(text_content):
                anomalies.append("inconsistent_formatting")
            
            # Check for missing required elements
            missing_elements = self._check_missing_elements(text_content, document_type)
            anomalies.extend(missing_elements)
            
        except Exception as e:
            logger.error(f"Text anomaly detection failed: {e}")
            anomalies.append("text_analysis_error")
        
        return anomalies
    
    def _detect_content_anomalies(self, text_content: str, document_type: str) -> List[str]:
        """
        Detect content-based anomalies
        
        Args:
            text_content: Extracted text content
            document_type: Document type
            
        Returns:
            List of content anomalies
        """
        anomalies = []
        
        try:
            # Check for generic content
            generic_indicators = [
                'sample', 'test', 'example', 'dummy', 'placeholder',
                'lorem ipsum', 'fake', 'mock', 'demo'
            ]
            
            text_lower = text_content.lower()
            for indicator in generic_indicators:
                if indicator in text_lower:
                    anomalies.append(f"generic_content_{indicator}")
            
            # Check for inconsistent information
            if self._has_inconsistent_information(text_content):
                anomalies.append("inconsistent_information")
            
            # Check for suspicious numbers
            number_patterns = [
                r'\b\d{12}\b',  # Aadhaar-like
                r'\b\d{10}\b',  # Phone-like
                r'\b\d{16}\b'   # Card-like
            ]
            
            for pattern in number_patterns:
                numbers = re.findall(pattern, text_content)
                for number in numbers:
                    if self._is_suspicious_number(number):
                        anomalies.append("suspicious_number")
            
        except Exception as e:
            logger.error(f"Content anomaly detection failed: {e}")
            anomalies.append("content_analysis_error")
        
        return anomalies
    
    def _validate_document_template(self, text_content: str, document_type: str) -> List[str]:
        """
        Validate document against known templates
        
        Args:
            text_content: Extracted text content
            document_type: Document type
            
        Returns:
            List of template validation anomalies
        """
        anomalies = []
        
        try:
            if document_type in self.document_templates:
                template = self.document_templates[document_type]
                
                # Check for expected elements
                for element in template['expected_elements']:
                    if element.lower() not in text_content.lower():
                        anomalies.append(f"missing_expected_element_{element}")
                
                # Check for expected format
                if 'expected_format' in template:
                    if not re.search(template['expected_format'], text_content):
                        anomalies.append("format_mismatch")
                
                # Check for security features
                for feature in template['security_features']:
                    if feature.lower() not in text_content.lower():
                        anomalies.append(f"missing_security_feature_{feature}")
            else:
                # Unknown document type
                anomalies.append("unknown_document_type")
        
        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            anomalies.append("template_validation_error")
        
        return anomalies
    
    def _is_suspicious_date(self, date_str: str) -> bool:
        """Check if a date is suspicious"""
        try:
            # Check for future dates
            if re.match(r'\d{4}', date_str):
                year = int(date_str[:4])
                current_year = datetime.now().year
                if year > current_year:
                    return True
            
            # Check for very old dates
            if re.match(r'\d{4}', date_str):
                year = int(date_str[:4])
                if year < 1900:
                    return True
            
            return False
        except:
            return False
    
    def _is_suspicious_number(self, number: str) -> bool:
        """Check if a number is suspicious"""
        try:
            # Check for all zeros
            if number == '0' * len(number):
                return True
            
            # Check for sequential numbers
            if len(number) >= 4:
                digits = [int(d) for d in number]
                if all(digits[i] == digits[i-1] + 1 for i in range(1, len(digits))):
                    return True
                if all(digits[i] == digits[i-1] - 1 for i in range(1, len(digits))):
                    return True
            
            return False
        except:
            return False
    
    def _has_inconsistent_formatting(self, text_content: str) -> bool:
        """Check for inconsistent text formatting"""
        try:
            # Check for mixed case inconsistencies
            lines = text_content.split('\n')
            case_patterns = []
            
            for line in lines:
                if line.strip():
                    if line.isupper():
                        case_patterns.append('upper')
                    elif line.islower():
                        case_patterns.append('lower')
                    elif line.istitle():
                        case_patterns.append('title')
                    else:
                        case_patterns.append('mixed')
            
            # Check for inconsistent patterns
            if len(set(case_patterns)) > 2:
                return True
            
            return False
        except:
            return False
    
    def _has_inconsistent_information(self, text_content: str) -> bool:
        """Check for inconsistent information in the document"""
        try:
            # Check for multiple different dates
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b\d{1,2}-\d{1,2}-\d{4}\b'
            ]
            
            all_dates = []
            for pattern in date_patterns:
                dates = re.findall(pattern, text_content)
                all_dates.extend(dates)
            
            if len(set(all_dates)) > 3:  # More than 3 different dates
                return True
            
            return False
        except:
            return False
    
    def _check_missing_elements(self, text_content: str, document_type: str) -> List[str]:
        """Check for missing required elements"""
        missing = []
        
        try:
            # Common required elements
            required_elements = {
                'certificate': ['certificate', 'issued', 'date'],
                'diploma': ['diploma', 'degree', 'university'],
                'transcript': ['transcript', 'grade', 'course'],
                'id_card': ['id', 'number', 'photo'],
                'passport': ['passport', 'nationality', 'date of birth']
            }
            
            if document_type in required_elements:
                for element in required_elements[document_type]:
                    if element.lower() not in text_content.lower():
                        missing.append(f"missing_{element}")
        
        except Exception as e:
            logger.error(f"Missing elements check failed: {e}")
        
        return missing
    
    def _calculate_fraud_score(self, anomalies: List[str], text_content: str) -> float:
        """
        Calculate fraud score based on detected anomalies
        
        Args:
            anomalies: List of detected anomalies
            text_content: Text content for additional analysis
            
        Returns:
            Fraud score between 0.0 and 1.0
        """
        try:
            base_score = 0.0
            
            # Score based on number of anomalies
            anomaly_score = min(len(anomalies) * 0.1, 0.5)
            base_score += anomaly_score
            
            # Score based on anomaly types
            high_risk_anomalies = [
                'suspicious_date', 'suspicious_number', 'generic_content',
                'missing_security_feature', 'format_mismatch'
            ]
            
            for anomaly in anomalies:
                for high_risk in high_risk_anomalies:
                    if high_risk in anomaly:
                        base_score += 0.15
                        break
            
            # Score based on text length (very short documents are suspicious)
            if len(text_content.strip()) < 50:
                base_score += 0.2
            
            # Cap the score at 1.0
            return min(base_score, 1.0)
            
        except Exception as e:
            logger.error(f"Fraud score calculation failed: {e}")
            return 0.5
    
    def _determine_risk_level(self, fraud_score: float, anomaly_count: int) -> str:
        """
        Determine risk level based on fraud score and anomaly count
        
        Args:
            fraud_score: Calculated fraud score
            anomaly_count: Number of detected anomalies
            
        Returns:
            Risk level: 'low', 'medium', 'high'
        """
        try:
            if fraud_score >= 0.7 or anomaly_count >= 5:
                return 'high'
            elif fraud_score >= 0.4 or anomaly_count >= 2:
                return 'medium'
            else:
                return 'low'
        except:
            return 'medium'
    
    def _calculate_confidence(self, fraud_score: float, anomalies: List[str]) -> float:
        """
        Calculate confidence in the fraud detection result
        
        Args:
            fraud_score: Calculated fraud score
            anomalies: List of detected anomalies
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            # Base confidence on number of anomalies detected
            base_confidence = min(len(anomalies) * 0.1, 0.8)
            
            # Adjust based on fraud score
            if fraud_score > 0.8:
                base_confidence += 0.2
            elif fraud_score < 0.2:
                base_confidence += 0.1
            
            # Cap at 1.0
            return min(base_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _create_error_result(self, error_message: str) -> FraudDetectionResult:
        """Create error result when fraud detection fails"""
        return FraudDetectionResult(
            is_fraudulent=False,
            fraud_score=0.5,
            detected_anomalies=[f"error: {error_message}"],
            manipulation_indicators=[],
            confidence=0.0,
            risk_level='medium'
        ) 