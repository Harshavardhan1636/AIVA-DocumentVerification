"""
Decision Engine - AI Module for AIVA Document Verification System

This module provides intelligent decision-making capabilities for document verification:
- Multi-factor decision analysis
- Risk assessment and scoring
- Integration with blockchain verification
- Confidence-based decision making
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import json

# Import our document analyzer
from .document_analyzer import AnalysisResult, DocumentFeatures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerificationContext:
    """Context information for verification decisions"""
    user_id: str
    timestamp: datetime
    document_hash: str
    blockchain_verified: bool
    previous_verifications: List[Dict]
    risk_level: str  # 'low', 'medium', 'high'

@dataclass
class DecisionResult:
    """Final decision result from the decision engine"""
    is_verified: bool
    confidence_score: float
    risk_level: str
    decision_factors: List[str]
    recommendations: List[str]
    verification_id: str
    timestamp: datetime
    blockchain_status: str

class DecisionEngine:
    """
    Intelligent decision engine for document verification
    """
    
    def __init__(self):
        """Initialize the decision engine with decision rules and weights"""
        
        # Decision weights for different factors
        self.weights = {
            'ai_analysis': 0.35,
            'blockchain_verification': 0.25,
            'image_quality': 0.15,
            'document_type_confidence': 0.15,
            'risk_assessment': 0.10
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 0.7,
            'medium': 0.5,
            'high': 0.3
        }
        
        # Decision rules
        self.decision_rules = {
            'minimum_confidence': 0.6,
            'maximum_fraud_indicators': 2,
            'minimum_image_quality': 0.4,
            'blockchain_required': True
        }
        
        logger.info("Decision Engine initialized successfully")
    
    def make_verification_decision(
        self, 
        ai_analysis: AnalysisResult, 
        verification_context: VerificationContext
    ) -> DecisionResult:
        """
        Make final verification decision based on AI analysis and context
        
        Args:
            ai_analysis: Results from document analyzer
            verification_context: Context information for verification
            
        Returns:
            DecisionResult: Final verification decision
        """
        try:
            # Calculate individual factor scores
            factor_scores = self._calculate_factor_scores(ai_analysis, verification_context)
            
            # Calculate weighted confidence score
            weighted_confidence = self._calculate_weighted_confidence(factor_scores)
            
            # Assess risk level
            risk_level = self._assess_risk_level(ai_analysis, verification_context)
            
            # Make final decision
            is_verified = self._make_final_decision(
                weighted_confidence, 
                ai_analysis, 
                verification_context,
                risk_level
            )
            
            # Generate decision factors and recommendations
            decision_factors = self._generate_decision_factors(factor_scores, ai_analysis)
            recommendations = self._generate_recommendations(
                is_verified, 
                weighted_confidence, 
                ai_analysis, 
                risk_level
            )
            
            # Create verification ID
            verification_id = self._generate_verification_id(verification_context)
            
            return DecisionResult(
                is_verified=is_verified,
                confidence_score=weighted_confidence,
                risk_level=risk_level,
                decision_factors=decision_factors,
                recommendations=recommendations,
                verification_id=verification_id,
                timestamp=datetime.now(),
                blockchain_status=self._get_blockchain_status(verification_context)
            )
            
        except Exception as e:
            logger.error(f"Error in decision making: {e}")
            return self._create_error_decision(str(e))
    
    def _calculate_factor_scores(
        self, 
        ai_analysis: AnalysisResult, 
        context: VerificationContext
    ) -> Dict[str, float]:
        """
        Calculate scores for different decision factors
        
        Args:
            ai_analysis: AI analysis results
            context: Verification context
            
        Returns:
            Dictionary of factor scores
        """
        scores = {}
        
        # AI Analysis Score
        scores['ai_analysis'] = ai_analysis.confidence
        
        # Blockchain Verification Score
        scores['blockchain_verification'] = 1.0 if context.blockchain_verified else 0.0
        
        # Image Quality Score
        scores['image_quality'] = ai_analysis.features.image_quality_score
        
        # Document Type Confidence Score
        scores['document_type_confidence'] = ai_analysis.features.confidence_score
        
        # Risk Assessment Score (inverse of risk)
        risk_scores = {'low': 1.0, 'medium': 0.7, 'high': 0.4}
        scores['risk_assessment'] = risk_scores.get(context.risk_level, 0.5)
        
        return scores
    
    def _calculate_weighted_confidence(self, factor_scores: Dict[str, float]) -> float:
        """
        Calculate weighted confidence score
        
        Args:
            factor_scores: Individual factor scores
            
        Returns:
            Weighted confidence score
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor, score in factor_scores.items():
            weight = self.weights.get(factor, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0
    
    def _assess_risk_level(
        self, 
        ai_analysis: AnalysisResult, 
        context: VerificationContext
    ) -> str:
        """
        Assess risk level based on analysis and context
        
        Args:
            ai_analysis: AI analysis results
            context: Verification context
            
        Returns:
            Risk level: 'low', 'medium', or 'high'
        """
        risk_score = 0.0
        
        # Base risk from AI analysis
        if len(ai_analysis.fraud_indicators) > 0:
            risk_score += 0.3
        
        if ai_analysis.features.image_quality_score < 0.5:
            risk_score += 0.2
        
        if ai_analysis.confidence < 0.6:
            risk_score += 0.2
        
        # Context-based risk
        if not context.blockchain_verified:
            risk_score += 0.3
        
        if context.risk_level == 'high':
            risk_score += 0.2
        elif context.risk_level == 'medium':
            risk_score += 0.1
        
        # Previous verification history
        if context.previous_verifications:
            failed_attempts = sum(1 for v in context.previous_verifications if not v.get('verified', True))
            if failed_attempts > 0:
                risk_score += min(failed_attempts * 0.1, 0.3)
        
        # Determine risk level
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.6:
            return 'medium'
        else:
            return 'high'
    
    def _make_final_decision(
        self, 
        weighted_confidence: float, 
        ai_analysis: AnalysisResult, 
        context: VerificationContext,
        risk_level: str
    ) -> bool:
        """
        Make final verification decision
        
        Args:
            weighted_confidence: Weighted confidence score
            ai_analysis: AI analysis results
            context: Verification context
            risk_level: Assessed risk level
            
        Returns:
            True if document is verified, False otherwise
        """
        # Check minimum confidence threshold
        if weighted_confidence < self.decision_rules['minimum_confidence']:
            return False
        
        # Check fraud indicators
        if len(ai_analysis.fraud_indicators) > self.decision_rules['maximum_fraud_indicators']:
            return False
        
        # Check image quality
        if ai_analysis.features.image_quality_score < self.decision_rules['minimum_image_quality']:
            return False
        
        # Check blockchain verification if required
        if self.decision_rules['blockchain_required'] and not context.blockchain_verified:
            return False
        
        # Adjust threshold based on risk level
        risk_threshold = self.risk_thresholds.get(risk_level, 0.5)
        if weighted_confidence < risk_threshold:
            return False
        
        return True
    
    def _generate_decision_factors(
        self, 
        factor_scores: Dict[str, float], 
        ai_analysis: AnalysisResult
    ) -> List[str]:
        """
        Generate list of factors that influenced the decision
        
        Args:
            factor_scores: Individual factor scores
            ai_analysis: AI analysis results
            
        Returns:
            List of decision factors
        """
        factors = []
        
        # Add factor scores
        for factor, score in factor_scores.items():
            if score > 0.8:
                factors.append(f"Strong {factor.replace('_', ' ')} performance")
            elif score < 0.4:
                factors.append(f"Poor {factor.replace('_', ' ')} performance")
        
        # Add fraud indicators
        if ai_analysis.fraud_indicators:
            factors.append(f"Detected {len(ai_analysis.fraud_indicators)} fraud indicators")
        
        # Add document type
        if ai_analysis.features.document_type != 'unknown':
            factors.append(f"Identified as {ai_analysis.features.document_type}")
        
        return factors
    
    def _generate_recommendations(
        self, 
        is_verified: bool, 
        confidence: float, 
        ai_analysis: AnalysisResult, 
        risk_level: str
    ) -> List[str]:
        """
        Generate recommendations based on decision
        
        Args:
            is_verified: Whether document was verified
            confidence: Confidence score
            ai_analysis: AI analysis results
            risk_level: Risk level
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if is_verified:
            if confidence > 0.9:
                recommendations.append("Document verified with high confidence")
            else:
                recommendations.append("Document verified - consider manual review for additional assurance")
        else:
            recommendations.append("Document verification failed")
            
            if confidence < 0.4:
                recommendations.append("Very low confidence - please provide clearer image")
            
            if ai_analysis.fraud_indicators:
                recommendations.append("Multiple fraud indicators detected - manual review required")
            
            if ai_analysis.features.image_quality_score < 0.5:
                recommendations.append("Poor image quality - retake photo with better lighting")
        
        # Risk-based recommendations
        if risk_level == 'high':
            recommendations.append("High-risk verification - additional verification steps recommended")
        elif risk_level == 'medium':
            recommendations.append("Medium-risk verification - proceed with caution")
        
        return recommendations
    
    def _generate_verification_id(self, context: VerificationContext) -> str:
        """
        Generate unique verification ID
        
        Args:
            context: Verification context
            
        Returns:
            Unique verification ID
        """
        timestamp = context.timestamp.strftime("%Y%m%d_%H%M%S")
        return f"VER_{context.user_id}_{timestamp}"
    
    def _get_blockchain_status(self, context: VerificationContext) -> str:
        """
        Get blockchain verification status
        
        Args:
            context: Verification context
            
        Returns:
            Blockchain status string
        """
        if context.blockchain_verified:
            return "verified_on_blockchain"
        else:
            return "not_verified_on_blockchain"
    
    def _create_error_decision(self, error_message: str) -> DecisionResult:
        """
        Create error decision when decision making fails
        
        Args:
            error_message: Error description
            
        Returns:
            Error DecisionResult
        """
        return DecisionResult(
            is_verified=False,
            confidence_score=0.0,
            risk_level='high',
            decision_factors=[f"Decision error: {error_message}"],
            recommendations=["Please try again or contact support"],
            verification_id="ERROR",
            timestamp=datetime.now(),
            blockchain_status="error"
        )
    
    def update_decision_rules(self, new_rules: Dict[str, any]) -> None:
        """
        Update decision rules dynamically
        
        Args:
            new_rules: New decision rules
        """
        self.decision_rules.update(new_rules)
        logger.info(f"Updated decision rules: {new_rules}")
    
    def get_decision_summary(self) -> Dict[str, any]:
        """
        Get summary of current decision engine configuration
        
        Returns:
            Dictionary with decision engine configuration
        """
        return {
            'weights': self.weights,
            'risk_thresholds': self.risk_thresholds,
            'decision_rules': self.decision_rules
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the decision engine
    engine = DecisionEngine()
    print("Decision Engine initialized successfully!")
    print("Ready for verification decisions...")
    
    # Print configuration
    print("\nDecision Engine Configuration:")
    print(json.dumps(engine.get_decision_summary(), indent=2)) 