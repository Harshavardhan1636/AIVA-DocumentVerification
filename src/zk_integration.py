"""
ZK Proof Integration for AIVA Document Verification System

This module integrates Zero-Knowledge Proofs with the AI and Blockchain modules
for enhanced security and privacy in document verification.
"""

import json
import hashlib
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# Import ZK Proof components
try:
    from .circuit_manager import CircuitManager
    from .proof_generator import ProofGenerator
    from .proof_verifier import ProofVerifier
except ImportError:
    from circuit_manager import CircuitManager
    from proof_generator import ProofGenerator
    from proof_verifier import ProofVerifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZKIntegration:
    """
    Integrates Zero-Knowledge Proofs with AI and Blockchain modules.
    
    This class handles:
    - ZK proof generation for document verification
    - ZK proof verification
    - Integration with AI analysis results
    - Integration with blockchain storage
    """
    
    def __init__(self, circuit_path: str = "circuits/"):
        """
        Initialize ZK Integration.
        
        Args:
            circuit_path: Path to circuit files
        """
        self.circuit_manager = CircuitManager(circuit_path)
        self.proof_generator = ProofGenerator(self.circuit_manager)
        self.proof_verifier = ProofVerifier(self.circuit_manager)
        
        # Statistics
        self.stats = {
            "total_proofs_generated": 0,
            "total_proofs_verified": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "processing_times": []
        }
    
    def generate_document_verification_proof(self, 
                                           document_data: Dict[str, Any],
                                           ai_analysis: Dict[str, Any],
                                           user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ZK proof for document verification.
        
        Args:
            document_data: Document data from vision module
            ai_analysis: AI analysis results
            user_context: User context information
            
        Returns:
            Dictionary containing ZK proof and metadata
        """
        try:
            start_time = time.time()
            
            # Generate document hash
            document_content = json.dumps(document_data, sort_keys=True)
            document_hash = hashlib.sha256(document_content.encode()).hexdigest()
            
            # Generate document hash proof
            document_proof = self.proof_generator.generate_document_hash_proof(
                document_hash, 
                document_data.get('document_type', 'general')
            )
            
            # Generate age verification proof if applicable
            age_proof = None
            if 'age' in document_data and 'min_age_requirement' in user_context:
                age = document_data['age']
                min_age = user_context['min_age_requirement']
                age_proof = self.proof_generator.generate_age_verification_proof(
                    age, min_age, document_data.get('document_type', 'id_card')
                )
            
            # Generate signature proof if signature data is available
            signature_proof = None
            if 'signature_data' in document_data:
                signature_data = document_data['signature_data'].encode()
                document_bytes = document_content.encode()
                signature_proof = self.proof_generator.generate_signature_proof(
                    document_bytes, signature_data
                )
            
            # Combine all proofs
            combined_proof = {
                "zk_proofs": {
                    "document_hash_proof": document_proof,
                    "age_verification_proof": age_proof,
                    "signature_verification_proof": signature_proof
                },
                "metadata": {
                    "document_hash": document_hash,
                    "document_type": document_data.get('document_type', 'unknown'),
                    "ai_analysis_hash": hashlib.sha256(
                        json.dumps(ai_analysis, sort_keys=True).encode()
                    ).hexdigest(),
                    "user_id": user_context.get('user_id', 'unknown'),
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "proof_version": "1.0.0"
                },
                "verification_data": {
                    "document_authentic": ai_analysis.get('analysis_result', {}).get('document_authentic', False),
                    "confidence_score": ai_analysis.get('analysis_result', {}).get('confidence_score', 0.0),
                    "fraud_indicators": ai_analysis.get('analysis_result', {}).get('fraud_indicators', []),
                    "blockchain_action": ai_analysis.get('recommended_actions', {}).get('blockchain_action', 'unknown')
                }
            }
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats("generated", processing_time)
            
            logger.info(f"✅ ZK proof generated successfully in {processing_time:.2f}s")
            return combined_proof
            
        except Exception as e:
            logger.error(f"❌ Error generating ZK proof: {str(e)}")
            return self._create_error_proof(str(e))
    
    def verify_document_proof(self, proof_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify ZK proof for document verification.
        
        Args:
            proof_data: ZK proof data to verify
            
        Returns:
            Dictionary containing verification results
        """
        try:
            start_time = time.time()
            
            # Check if proof_data is valid
            if not proof_data or not isinstance(proof_data, dict):
                logger.warning("Invalid proof data provided")
                return {
                    "verification_successful": False,
                    "error": "Invalid proof data structure",
                    "verified_at": datetime.utcnow().isoformat() + "Z"
                }
            
            zk_proofs = proof_data.get('zk_proofs', {})
            verification_results = {}
            
            # Verify document hash proof
            if 'document_hash_proof' in zk_proofs and zk_proofs['document_hash_proof']:
                try:
                    doc_proof_valid = self.proof_verifier.verify_document_hash_proof(
                        zk_proofs['document_hash_proof']
                    )
                    verification_results['document_hash_verified'] = doc_proof_valid
                except Exception as e:
                    logger.warning(f"Document hash proof verification failed: {e}")
                    verification_results['document_hash_verified'] = False
            else:
                verification_results['document_hash_verified'] = False
            
            # Verify age proof
            if 'age_verification_proof' in zk_proofs and zk_proofs['age_verification_proof']:
                try:
                    age_proof = zk_proofs['age_verification_proof']
                    min_age = age_proof.get('min_age', 18) if isinstance(age_proof, dict) else 18
                    age_proof_valid = self.proof_verifier.verify_age_proof(age_proof, min_age)
                    verification_results['age_verified'] = age_proof_valid
                except Exception as e:
                    logger.warning(f"Age proof verification failed: {e}")
                    verification_results['age_verified'] = False
            else:
                verification_results['age_verified'] = True  # No age requirement
            
            # Verify signature proof
            if 'signature_verification_proof' in zk_proofs and zk_proofs['signature_verification_proof']:
                try:
                    sig_proof_valid = self.proof_verifier.verify_signature_proof(
                        zk_proofs['signature_verification_proof']
                    )
                    verification_results['signature_verified'] = sig_proof_valid
                except Exception as e:
                    logger.warning(f"Signature proof verification failed: {e}")
                    verification_results['signature_verified'] = False
            else:
                verification_results['signature_verified'] = True  # No signature requirement
            
            # Overall verification result
            all_verified = all(verification_results.values())
            
            # Safely get metadata
            metadata = proof_data.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}
            
            # Create verification result
            result = {
                "verification_successful": all_verified,
                "verification_results": verification_results,
                "proof_integrity": self._verify_proof_integrity(proof_data),
                "metadata": {
                    "verified_at": datetime.utcnow().isoformat() + "Z",
                    "verification_time": time.time() - start_time,
                    "proof_version": metadata.get('proof_version', 'unknown')
                }
            }
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats("verified", processing_time, all_verified)
            
            logger.info(f"✅ ZK proof verification completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error verifying ZK proof: {str(e)}")
            return {
                "verification_successful": False,
                "error": str(e),
                "verified_at": datetime.utcnow().isoformat() + "Z"
            }
    
    def integrate_with_ai_analysis(self, ai_result: Dict[str, Any], 
                                 zk_proof: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate ZK proof with AI analysis results.
        
        Args:
            ai_result: AI analysis result
            zk_proof: ZK proof data
            
        Returns:
            Integrated result with ZK proof validation
        """
        try:
            # Verify the ZK proof
            verification_result = self.verify_document_proof(zk_proof)
            
            # Integrate verification results with AI analysis
            integrated_result = ai_result.copy()
            
            # Add ZK verification results
            integrated_result["zk_verification"] = verification_result
            
            # Update confidence based on ZK verification
            if verification_result["verification_successful"]:
                # Boost confidence if ZK proof is valid
                current_confidence = integrated_result.get('analysis_result', {}).get('confidence_score', 0.0)
                boosted_confidence = min(current_confidence * 1.1, 1.0)  # Boost by 10%
                integrated_result['analysis_result']['confidence_score'] = boosted_confidence
                
                # Add ZK verification as authenticity factor
                if 'authenticity_factors' not in integrated_result['analysis_result']:
                    integrated_result['analysis_result']['authenticity_factors'] = []
                integrated_result['analysis_result']['authenticity_factors'].append("zk_proof_verified")
            
            # Update blockchain action based on ZK verification
            if verification_result["verification_successful"]:
                if integrated_result.get('recommended_actions', {}).get('blockchain_action') == 'flag_for_manual_review':
                    integrated_result['recommended_actions']['blockchain_action'] = 'create_verification_record'
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"❌ Error integrating ZK proof with AI analysis: {str(e)}")
            return ai_result
    
    def integrate_with_blockchain(self, blockchain_data: Dict[str, Any], 
                                zk_proof: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate ZK proof with blockchain data.
        
        Args:
            blockchain_data: Blockchain transaction data
            zk_proof: ZK proof data
            
        Returns:
            Integrated blockchain data with ZK proof
        """
        try:
            # Add ZK proof to blockchain data
            integrated_data = blockchain_data.copy()
            
            # Add ZK proof hash to blockchain data
            zk_proof_hash = hashlib.sha256(
                json.dumps(zk_proof, sort_keys=True).encode()
            ).hexdigest()
            
            integrated_data["zk_proof_hash"] = zk_proof_hash
            integrated_data["zk_proof_metadata"] = zk_proof.get('metadata', {})
            
            # Add verification status
            verification_result = self.verify_document_proof(zk_proof)
            integrated_data["zk_verification_status"] = verification_result["verification_successful"]
            
            return integrated_data
            
        except Exception as e:
            logger.error(f"❌ Error integrating ZK proof with blockchain: {str(e)}")
            return blockchain_data
    
    def _verify_proof_integrity(self, proof_data: Dict[str, Any]) -> bool:
        """Verify the integrity of the proof data structure."""
        try:
            required_fields = ['zk_proofs', 'metadata', 'verification_data']
            
            for field in required_fields:
                if field not in proof_data:
                    return False
            
            # Check metadata
            metadata = proof_data['metadata']
            required_metadata = ['document_hash', 'generated_at', 'proof_version']
            
            for field in required_metadata:
                if field not in metadata:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _update_stats(self, operation: str, processing_time: float, success: bool = True):
        """Update statistics."""
        if operation == "generated":
            self.stats["total_proofs_generated"] += 1
        elif operation == "verified":
            self.stats["total_proofs_verified"] += 1
            if success:
                self.stats["successful_verifications"] += 1
            else:
                self.stats["failed_verifications"] += 1
        
        self.stats["processing_times"].append(processing_time)
        
        # Keep only last 100 processing times
        if len(self.stats["processing_times"]) > 100:
            self.stats["processing_times"] = self.stats["processing_times"][-100:]
    
    def _create_error_proof(self, error_message: str) -> Dict[str, Any]:
        """Create an error proof when generation fails."""
        return {
            "zk_proofs": {},
            "metadata": {
                "error": error_message,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "proof_version": "1.0.0"
            },
            "verification_data": {
                "document_authentic": False,
                "confidence_score": 0.0,
                "fraud_indicators": [f"zk_proof_error: {error_message}"],
                "blockchain_action": "flag_for_manual_review"
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ZK integration statistics."""
        avg_processing_time = 0
        if self.stats["processing_times"]:
            avg_processing_time = sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
        
        return {
            "total_proofs_generated": self.stats["total_proofs_generated"],
            "total_proofs_verified": self.stats["total_proofs_verified"],
            "successful_verifications": self.stats["successful_verifications"],
            "failed_verifications": self.stats["failed_verifications"],
            "average_processing_time": avg_processing_time,
            "circuit_status": self.circuit_manager.get_status()
        }
    
    def get_circuit_status(self) -> Dict[str, Any]:
        """Get circuit compilation status."""
        return self.circuit_manager.get_status() 