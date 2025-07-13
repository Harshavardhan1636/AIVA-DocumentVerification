"""
Main Document Verifier class for ZK Proof operations.

This class provides a high-level interface for generating and verifying
zero-knowledge proofs for document verification.
"""

import hashlib
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    from .circuit_manager import CircuitManager
    from .proof_generator import ProofGenerator
    from .proof_verifier import ProofVerifier
    from .utils import hash_document, validate_document_format
except ImportError:
    from circuit_manager import CircuitManager
    from proof_generator import ProofGenerator
    from proof_verifier import ProofVerifier
    from utils import hash_document, validate_document_format


@dataclass
class ProofResult:
    """Result of a ZK proof operation."""
    success: bool
    proof_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    verification_time: Optional[float] = None


class DocumentVerifier:
    """
    Main class for document verification using Zero-Knowledge Proofs.
    
    This class provides methods to:
    - Generate ZK proofs for document authenticity
    - Verify ZK proofs without revealing document content
    - Handle age verification proofs
    - Manage signature verification proofs
    """
    
    def __init__(self, circuit_path: str = "circuits/"):
        """
        Initialize the Document Verifier.
        
        Args:
            circuit_path: Path to the Circom circuit files
        """
        self.circuit_manager = CircuitManager(circuit_path)
        self.proof_generator = ProofGenerator(self.circuit_manager)
        self.proof_verifier = ProofVerifier(self.circuit_manager)
        
    def generate_document_proof(self, document_data: bytes, 
                               document_type: str = "general") -> ProofResult:
        """
        Generate a ZK proof for document authenticity.
        
        Args:
            document_data: Raw document data
            document_type: Type of document (aadhaar, passport, etc.)
            
        Returns:
            ProofResult containing the generated proof
        """
        try:
            # Validate document format
            if not validate_document_format(document_data, document_type):
                return ProofResult(
                    success=False,
                    error_message=f"Invalid document format for type: {document_type}"
                )
            
            # Generate document hash
            document_hash = hash_document(document_data)
            
            # Generate ZK proof
            proof_data = self.proof_generator.generate_document_hash_proof(
                document_hash, document_type
            )
            
            return ProofResult(
                success=True,
                proof_data=proof_data
            )
            
        except Exception as e:
            return ProofResult(
                success=False,
                error_message=f"Error generating proof: {str(e)}"
            )
    
    def verify_document_proof(self, proof_data: Dict[str, Any]) -> ProofResult:
        """
        Verify a ZK proof for document authenticity.
        
        Args:
            proof_data: The proof data to verify
            
        Returns:
            ProofResult containing verification status
        """
        try:
            import time
            start_time = time.time()
            
            # Verify the proof
            is_valid = self.proof_verifier.verify_document_hash_proof(proof_data)
            
            verification_time = time.time() - start_time
            
            return ProofResult(
                success=is_valid,
                verification_time=verification_time
            )
            
        except Exception as e:
            return ProofResult(
                success=False,
                error_message=f"Error verifying proof: {str(e)}"
            )
    
    def generate_age_proof(self, document_data: bytes, 
                          min_age: int, document_type: str = "id_card") -> ProofResult:
        """
        Generate a ZK proof for age verification without revealing exact age.
        
        Args:
            document_data: Raw document data
            min_age: Minimum age requirement
            document_type: Type of ID document
            
        Returns:
            ProofResult containing the age verification proof
        """
        try:
            # Extract birth date from document (this would be done by OCR in real system)
            # For demo purposes, we'll simulate this
            birth_date = self._extract_birth_date(document_data, document_type)
            
            if not birth_date:
                return ProofResult(
                    success=False,
                    error_message="Could not extract birth date from document"
                )
            
            # Calculate age
            age = self._calculate_age(birth_date)
            
            # Generate age verification proof
            proof_data = self.proof_generator.generate_age_verification_proof(
                age, min_age, document_type
            )
            
            return ProofResult(
                success=True,
                proof_data=proof_data
            )
            
        except Exception as e:
            return ProofResult(
                success=False,
                error_message=f"Error generating age proof: {str(e)}"
            )
    
    def verify_age_proof(self, proof_data: Dict[str, Any], 
                        min_age: int) -> ProofResult:
        """
        Verify an age verification ZK proof.
        
        Args:
            proof_data: The age verification proof data
            min_age: Minimum age requirement
            
        Returns:
            ProofResult containing verification status
        """
        try:
            import time
            start_time = time.time()
            
            # Verify the age proof
            is_valid = self.proof_verifier.verify_age_proof(proof_data, min_age)
            
            verification_time = time.time() - start_time
            
            return ProofResult(
                success=is_valid,
                verification_time=verification_time
            )
            
        except Exception as e:
            return ProofResult(
                success=False,
                error_message=f"Error verifying age proof: {str(e)}"
            )
    
    def generate_signature_proof(self, document_data: bytes,
                                signature: bytes) -> ProofResult:
        """
        Generate a ZK proof for signature verification.
        
        Args:
            document_data: The signed document data
            signature: The signature to verify
            
        Returns:
            ProofResult containing the signature verification proof
        """
        try:
            # Generate signature verification proof
            proof_data = self.proof_generator.generate_signature_proof(
                document_data, signature
            )
            
            return ProofResult(
                success=True,
                proof_data=proof_data
            )
            
        except Exception as e:
            return ProofResult(
                success=False,
                error_message=f"Error generating signature proof: {str(e)}"
            )
    
    def verify_signature_proof(self, proof_data: Dict[str, Any]) -> ProofResult:
        """
        Verify a signature verification ZK proof.
        
        Args:
            proof_data: The signature verification proof data
            
        Returns:
            ProofResult containing verification status
        """
        try:
            import time
            start_time = time.time()
            
            # Verify the signature proof
            is_valid = self.proof_verifier.verify_signature_proof(proof_data)
            
            verification_time = time.time() - start_time
            
            return ProofResult(
                success=is_valid,
                verification_time=verification_time
            )
            
        except Exception as e:
            return ProofResult(
                success=False,
                error_message=f"Error verifying signature proof: {str(e)}"
            )
    
    def _extract_birth_date(self, document_data: bytes, 
                           document_type: str) -> Optional[datetime]:
        """
        Extract birth date from document data.
        
        This is a placeholder implementation. In a real system,
        this would use OCR to extract the birth date from the document.
        
        Args:
            document_data: Raw document data
            document_type: Type of document
            
        Returns:
            Extracted birth date or None if not found
        """
        # For demo purposes, return a simulated birth date
        # In real implementation, this would use OCR to extract date
        if document_type == "aadhaar":
            # Simulate Aadhaar card birth date extraction
            return datetime(1990, 1, 1)  # Placeholder
        elif document_type == "passport":
            # Simulate passport birth date extraction
            return datetime(1985, 6, 15)  # Placeholder
        else:
            return datetime(1995, 12, 25)  # Default placeholder
    
    def _calculate_age(self, birth_date: datetime) -> int:
        """
        Calculate age from birth date.
        
        Args:
            birth_date: Date of birth
            
        Returns:
            Age in years
        """
        today = datetime.now()
        age = today.year - birth_date.year
        
        # Adjust age if birthday hasn't occurred this year
        if today.month < birth_date.month or (
            today.month == birth_date.month and today.day < birth_date.day
        ):
            age -= 1
            
        return age
    
    def get_proof_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about proof generation and verification.
        
        Returns:
            Dictionary containing proof statistics
        """
        gen_stats = self.proof_generator.get_stats()
        ver_stats = self.proof_verifier.get_stats()
        
        return {
            "total_proofs_generated": gen_stats.get("total_generated", 0),
            "total_proofs_verified": ver_stats.get("total_verified", 0),
            "average_verification_time": ver_stats.get("average_verification_time", 0.0),
            "circuit_compilation_status": self.circuit_manager.get_status()
        } 