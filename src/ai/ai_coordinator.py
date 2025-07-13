"""
AI Coordinator - Main AI Module for AIVA Document Verification System

This module coordinates all AI components and provides a unified interface:
- Document Analyzer
- Fraud Detector  
- Decision Engine
- Integration with blockchain and vision modules
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib

# Import our AI components
from .document_analyzer import DocumentAnalyzer, AnalysisResult
from .fraud_detector import FraudDetector, FraudDetectionResult
from .decision_engine import DecisionEngine, DecisionResult, VerificationContext
from .blockchain_integration import BlockchainIntegration, MockBlockchainIntegration, VerificationRecord, BlockchainConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerificationRequest:
    """Request for document verification"""
    user_id: str
    document_image: np.ndarray
    document_hash: str
    blockchain_verified: bool = False
    previous_verifications: List[Dict] = None
    risk_level: str = 'medium'

@dataclass
class VerificationResponse:
    """Complete verification response"""
    is_verified: bool
    confidence_score: float
    document_type: str
    fraud_detected: bool
    fraud_score: float
    risk_level: str
    decision_factors: List[str]
    recommendations: List[str]
    verification_id: str
    timestamp: datetime
    processing_time: float
    ai_analysis: AnalysisResult
    fraud_analysis: FraudDetectionResult
    final_decision: DecisionResult

class AICoordinator:
    """
    Main AI coordinator for document verification system
    """
    
    def __init__(self, use_blockchain: bool = True, blockchain_config: Optional[BlockchainConfig] = None):
        """Initialize the AI coordinator with all components"""
        
        # Initialize AI components
        self.document_analyzer = DocumentAnalyzer()
        self.fraud_detector = FraudDetector()
        self.decision_engine = DecisionEngine()
        
        # Initialize blockchain integration
        self.use_blockchain = use_blockchain
        if use_blockchain:
            if blockchain_config:
                try:
                    self.blockchain_integration = BlockchainIntegration(blockchain_config)
                    logger.info("Blockchain integration initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize blockchain integration: {e}")
                    self.blockchain_integration = MockBlockchainIntegration()
            else:
                self.blockchain_integration = MockBlockchainIntegration()
                logger.info("Using mock blockchain integration")
        
        # Performance tracking
        self.processing_stats = {
            'total_verifications': 0,
            'successful_verifications': 0,
            'average_processing_time': 0.0,
            'blockchain_usage_count': 0
        }
        
        logger.info("AI Coordinator initialized successfully")
    
    def verify_document(self, request: VerificationRequest) -> VerificationResponse:
        """
        Main method to verify a document using all AI components
        
        Args:
            request: Verification request with document image and context
            
        Returns:
            VerificationResponse: Complete verification results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting verification for user: {request.user_id}")
            
            # Step 1: Document Analysis
            logger.info("Step 1: Performing document analysis...")
            ai_analysis = self.document_analyzer.analyze_document(request.document_image)
            
            # Step 2: Fraud Detection
            logger.info("Step 2: Performing fraud detection...")
            fraud_analysis = self.fraud_detector.detect_fraud(
                request.document_image,
                ai_analysis.features.text_content,
                ai_analysis.features.document_type
            )
            
            # Step 3: Create verification context
            logger.info("Step 3: Creating verification context...")
            context = VerificationContext(
                user_id=request.user_id,
                timestamp=datetime.now(),
                document_hash=request.document_hash,
                blockchain_verified=request.blockchain_verified,
                previous_verifications=request.previous_verifications or [],
                risk_level=request.risk_level
            )
            
            # Step 4: Make final decision
            logger.info("Step 4: Making final decision...")
            
            final_decision = self.decision_engine.make_verification_decision(
                ai_analysis, context
            )
            
            # Step 5: Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Step 5.5: Store on blockchain (if enabled)
            blockchain_tx_hash = None
            if self.use_blockchain:
                logger.info("Step 5.5: Storing verification record on blockchain...")
                try:
                    # Create verification record
                    verification_record = VerificationRecord(
                        document_hash=request.document_hash,
                        user_id=request.user_id,
                        verification_id=final_decision.verification_id,
                        is_verified=final_decision.is_verified,
                        confidence_score=final_decision.confidence_score,
                        fraud_score=fraud_analysis.fraud_score,
                        timestamp=int(context.timestamp.timestamp()),
                        ai_analysis_hash=self.blockchain_integration.create_analysis_hash({
                            'document_type': ai_analysis.features.document_type,
                            'fraud_detected': fraud_analysis.is_fraudulent,
                        })
                    )
                    
                    blockchain_tx_hash = self.blockchain_integration.store_verification_record(verification_record)
                    self.processing_stats['blockchain_usage_count'] += 1
                    logger.info(f"Verification record stored on blockchain: {blockchain_tx_hash}")
                except Exception as e:
                    logger.warning(f"Failed to store on blockchain: {e}")
            
            # Step 6: Update statistics
            self._update_statistics(processing_time, final_decision.is_verified)
            
            # Step 7: Create response
            response = VerificationResponse(
                is_verified=final_decision.is_verified,
                confidence_score=final_decision.confidence_score,
                document_type=ai_analysis.features.document_type,
                fraud_detected=fraud_analysis.is_fraudulent,
                fraud_score=fraud_analysis.fraud_score,
                risk_level=final_decision.risk_level,
                decision_factors=final_decision.decision_factors,
                recommendations=final_decision.recommendations,
                verification_id=final_decision.verification_id,
                timestamp=final_decision.timestamp,
                processing_time=processing_time,
                ai_analysis=ai_analysis,
                fraud_analysis=fraud_analysis,
                final_decision=final_decision
            )
            
            logger.info(f"Verification completed in {processing_time:.2f} seconds")
            logger.info(f"Result: {'VERIFIED' if response.is_verified else 'REJECTED'}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in document verification: {e}")
            return self._create_error_response(request, str(e), start_time)
    
    def verify_document_from_file(self, file_path: str, user_id: str, **kwargs) -> VerificationResponse:
        """
        Verify document from file path
        
        Args:
            file_path: Path to document image file
            user_id: User identifier
            **kwargs: Additional verification parameters
            
        Returns:
            VerificationResponse: Verification results
        """
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Could not load image from {file_path}")
            
            # Generate document hash
            document_hash = self._generate_document_hash(image)
            
            # Create request
            request = VerificationRequest(
                user_id=user_id,
                document_image=image,
                document_hash=document_hash,
                **kwargs
            )
            
            return self.verify_document(request)
            
        except Exception as e:
            logger.error(f"Error verifying document from file: {e}")
            return self._create_error_response(
                VerificationRequest(user_id=user_id, document_image=np.array([]), document_hash=""),
                str(e),
                datetime.now()
            )
    
    def verify_document_from_bytes(self, image_bytes: bytes, user_id: str, **kwargs) -> VerificationResponse:
        """
        Verify document from image bytes
        
        Args:
            image_bytes: Image data as bytes
            user_id: User identifier
            **kwargs: Additional verification parameters
            
        Returns:
            VerificationResponse: Verification results
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image from bytes")
            
            # Generate document hash
            document_hash = self._generate_document_hash(image)
            
            # Create request
            request = VerificationRequest(
                user_id=user_id,
                document_image=image,
                document_hash=document_hash,
                **kwargs
            )
            
            return self.verify_document(request)
            
        except Exception as e:
            logger.error(f"Error verifying document from bytes: {e}")
            return self._create_error_response(
                VerificationRequest(user_id=user_id, document_image=np.array([]), document_hash=""),
                str(e),
                datetime.now()
            )
    
    def get_verification_summary(self, response: VerificationResponse) -> Dict[str, any]:
        """
        Get a summary of verification results
        
        Args:
            response: Verification response
            
        Returns:
            Dictionary with verification summary
        """
        return {
            'verification_id': response.verification_id,
            'is_verified': response.is_verified,
            'confidence_score': response.confidence_score,
            'document_type': response.document_type,
            'fraud_detected': response.fraud_detected,
            'fraud_score': response.fraud_score,
            'risk_level': response.risk_level,
            'processing_time': response.processing_time,
            'timestamp': response.timestamp.isoformat(),
            'decision_factors': response.decision_factors,
            'recommendations': response.recommendations
        }
    
    def get_detailed_analysis(self, response: VerificationResponse) -> Dict[str, any]:
        """
        Get detailed analysis results
        
        Args:
            response: Verification response
            
        Returns:
            Dictionary with detailed analysis
        """
        return {
            'verification_summary': self.get_verification_summary(response),
            'ai_analysis': {
                'document_type': response.ai_analysis.features.document_type,
                'confidence_score': response.ai_analysis.features.confidence_score,
                'text_content': response.ai_analysis.features.text_content[:200] + "..." if len(response.ai_analysis.features.text_content) > 200 else response.ai_analysis.features.text_content,
                'image_quality_score': response.ai_analysis.features.image_quality_score,
                'suspicious_patterns': response.ai_analysis.features.suspicious_patterns,
                'fraud_indicators': response.ai_analysis.fraud_indicators,
                'metadata': response.ai_analysis.features.metadata
            },
            'fraud_analysis': {
                'is_fraudulent': response.fraud_analysis.is_fraudulent,
                'fraud_score': response.fraud_analysis.fraud_score,
                'detected_anomalies': response.fraud_analysis.detected_anomalies,
                'manipulation_indicators': response.fraud_analysis.manipulation_indicators,
                'confidence': response.fraud_analysis.confidence,
                'risk_level': response.fraud_analysis.risk_level
            },
            'decision_analysis': {
                'is_verified': response.final_decision.is_verified,
                'confidence_score': response.final_decision.confidence_score,
                'risk_level': response.final_decision.risk_level,
                'decision_factors': response.final_decision.decision_factors,
                'recommendations': response.final_decision.recommendations,
                'blockchain_status': response.final_decision.blockchain_status
            }
        }
    
    def get_performance_stats(self) -> Dict[str, any]:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance statistics
        """
        return {
            'total_verifications': self.processing_stats['total_verifications'],
            'successful_verifications': self.processing_stats['successful_verifications'],
            'success_rate': self.processing_stats['successful_verifications'] / max(1, self.processing_stats['total_verifications']),
            'average_processing_time': self.processing_stats['average_processing_time'],
            'blockchain_usage_count': self.processing_stats.get('blockchain_usage_count', 0),
            'components': {
                'document_analyzer': 'active',
                'fraud_detector': 'active',
                'decision_engine': 'active',
                'blockchain_integration': 'active' if self.use_blockchain else 'inactive'
            }
        }
    
    def update_decision_rules(self, new_rules: Dict[str, any]) -> None:
        """
        Update decision engine rules
        
        Args:
            new_rules: New decision rules
        """
        self.decision_engine.update_decision_rules(new_rules)
        logger.info(f"Updated decision rules: {new_rules}")
    
    def get_blockchain_verification(self, document_hash: str) -> Optional[VerificationRecord]:
        """
        Get verification record from blockchain
        
        Args:
            document_hash: Hash of the document
            
        Returns:
            VerificationRecord or None if not found
        """
        if not self.use_blockchain:
            logger.warning("Blockchain integration is not enabled")
            return None
        
        try:
            return self.blockchain_integration.get_verification_record(document_hash)
        except Exception as e:
            logger.error(f"Error retrieving blockchain verification: {e}")
            return None
    
    def get_user_blockchain_history(self, user_id: str) -> List[str]:
        """
        Get user's verification history from blockchain
        
        Args:
            user_id: User identifier
            
        Returns:
            List of verification IDs
        """
        if not self.use_blockchain:
            logger.warning("Blockchain integration is not enabled")
            return []
        
        try:
            return self.blockchain_integration.get_user_verifications(user_id)
        except Exception as e:
            logger.error(f"Error retrieving user blockchain history: {e}")
            return []
    
    def verify_document_on_blockchain(self, document_hash: str) -> bool:
        """
        Check if document exists on blockchain
        
        Args:
            document_hash: Hash of the document
            
        Returns:
            bool: True if document exists on blockchain
        """
        if not self.use_blockchain:
            logger.warning("Blockchain integration is not enabled")
            return False
        
        try:
            return self.blockchain_integration.verify_document_hash(document_hash)
        except Exception as e:
            logger.error(f"Error verifying document on blockchain: {e}")
            return False
    
    def get_blockchain_balance(self) -> float:
        """
        Get blockchain account balance
        
        Returns:
            float: Account balance in ETH
        """
        if not self.use_blockchain:
            logger.warning("Blockchain integration is not enabled")
            return 0.0
        
        try:
            return self.blockchain_integration.get_account_balance()
        except Exception as e:
            logger.error(f"Error getting blockchain balance: {e}")
            return 0.0
    
    def _generate_document_hash(self, image: np.ndarray) -> str:
        """
        Generate hash for document image
        
        Args:
            image: Document image
            
        Returns:
            Document hash string
        """
        # Convert image to bytes
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        
        # Generate SHA-256 hash
        return hashlib.sha256(image_bytes).hexdigest()
    
    def _update_statistics(self, processing_time: float, is_verified: bool) -> None:
        """
        Update processing statistics
        
        Args:
            processing_time: Processing time in seconds
            is_verified: Whether verification was successful
        """
        self.processing_stats['total_verifications'] += 1
        
        if is_verified:
            self.processing_stats['successful_verifications'] += 1
        
        # Update average processing time
        current_avg = self.processing_stats['average_processing_time']
        total_verifications = self.processing_stats['total_verifications']
        
        self.processing_stats['average_processing_time'] = (
            (current_avg * (total_verifications - 1) + processing_time) / total_verifications
        )
    
    def _create_error_response(
        self, 
        request: VerificationRequest, 
        error_message: str, 
        start_time: datetime
    ) -> VerificationResponse:
        """
        Create error response when verification fails
        
        Args:
            request: Original verification request
            error_message: Error description
            start_time: When verification started
            
        Returns:
            Error VerificationResponse
        """
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create error analysis results
        error_analysis = AnalysisResult(
            is_authentic=False,
            confidence=0.0,
            fraud_indicators=[f"verification_error: {error_message}"],
            recommendations=["Please try again or contact support"],
            features=request.document_image  # This will be empty for errors
        )
        
        error_fraud = FraudDetectionResult(
            is_fraudulent=True,
            fraud_score=1.0,
            detected_anomalies=[f"processing_error: {error_message}"],
            manipulation_indicators=[],
            confidence=0.0,
            risk_level='high'
        )
        
        error_decision = DecisionResult(
            is_verified=False,
            confidence_score=0.0,
            risk_level='high',
            decision_factors=[f"Error: {error_message}"],
            recommendations=["System error - please retry"],
            verification_id=f"ERROR_{request.user_id}_{int(start_time.timestamp())}",
            timestamp=datetime.now(),
            blockchain_status="error"
        )
        
        return VerificationResponse(
            is_verified=False,
            confidence_score=0.0,
            document_type="unknown",
            fraud_detected=True,
            fraud_score=1.0,
            risk_level='high',
            decision_factors=[f"Error: {error_message}"],
            recommendations=["System error - please retry"],
            verification_id=error_decision.verification_id,
            timestamp=datetime.now(),
            processing_time=processing_time,
            ai_analysis=error_analysis,
            fraud_analysis=error_fraud,
            final_decision=error_decision
        )

# Example usage and testing
if __name__ == "__main__":
    # Test the AI coordinator
    coordinator = AICoordinator()
    print("AI Coordinator initialized successfully!")
    print("Ready for document verification...")
    
    # Print performance stats
    print("\nInitial Performance Stats:")
    print(json.dumps(coordinator.get_performance_stats(), indent=2)) 