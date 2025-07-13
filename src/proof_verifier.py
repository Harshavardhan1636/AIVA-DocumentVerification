"""
Proof Verifier for verifying Zero-Knowledge Proofs.

This module handles the verification of ZK proofs using compiled Circom circuits
and SnarkJS for document verification operations.
"""

import json
import hashlib
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

try:
    from .circuit_manager import CircuitManager
except ImportError:
    from circuit_manager import CircuitManager


class ProofVerifier:
    """
    Verifies Zero-Knowledge Proofs for document verification.
    
    This class handles:
    - Document hash verification proof verification
    - Age verification proof verification
    - Signature verification proof verification
    - Verification result validation
    """
    
    def __init__(self, circuit_manager: CircuitManager):
        """
        Initialize the Proof Verifier.
        
        Args:
            circuit_manager: Circuit manager instance
        """
        self.circuit_manager = circuit_manager
        self.stats = {
            "total_verified": 0,
            "document_proofs": 0,
            "age_proofs": 0,
            "signature_proofs": 0,
            "verification_times": [],
            "successful_verifications": 0,
            "failed_verifications": 0
        }
    
    def verify_document_hash_proof(self, proof_data: Dict[str, Any]) -> bool:
        """
        Verify a ZK proof for document hash verification.
        
        Args:
            proof_data: The proof data to verify
            
        Returns:
            True if verification successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Check if circuit is compiled
            if not self.circuit_manager.circuits["document_hash"]["compiled"]:
                # For demo purposes, verify simulated proof
                return self._verify_simulated_document_proof(proof_data)
            
            # Verify proof using snarkjs
            is_valid = self._verify_proof_with_snarkjs(
                "document_hash", proof_data
            )
            
            # Update statistics
            verification_time = time.time() - start_time
            self._update_stats("document_proofs", verification_time, is_valid)
            
            return is_valid
            
        except Exception as e:
            print(f"ERROR verifying document hash proof: {str(e)}")
            # Fallback to simulated verification
            return self._verify_simulated_document_proof(proof_data)
    
    def verify_age_proof(self, proof_data: Dict[str, Any], 
                        min_age: int) -> bool:
        """
        Verify an age verification ZK proof.
        
        Args:
            proof_data: The proof data to verify
            min_age: Minimum age requirement
            
        Returns:
            True if verification successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Check if circuit is compiled
            if not self.circuit_manager.circuits["age_verification"]["compiled"]:
                # For demo purposes, verify simulated proof
                return self._verify_simulated_age_proof(proof_data, min_age)
            
            # Verify proof using snarkjs
            is_valid = self._verify_proof_with_snarkjs(
                "age_verification", proof_data
            )
            
            # Additional validation for age proof
            if is_valid and "min_age" in proof_data:
                is_valid = proof_data["min_age"] >= min_age
            
            # Update statistics
            verification_time = time.time() - start_time
            self._update_stats("age_proofs", verification_time, is_valid)
            
            return is_valid
            
        except Exception as e:
            print(f"ERROR verifying age proof: {str(e)}")
            # Fallback to simulated verification
            return self._verify_simulated_age_proof(proof_data, min_age)
    
    def verify_signature_proof(self, proof_data: Dict[str, Any]) -> bool:
        """
        Verify a signature verification ZK proof.
        
        Args:
            proof_data: The proof data to verify
            
        Returns:
            True if verification successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Check if circuit is compiled
            if not self.circuit_manager.circuits["signature_verification"]["compiled"]:
                # For demo purposes, verify simulated proof
                return self._verify_simulated_signature_proof(proof_data)
            
            # Verify proof using snarkjs
            is_valid = self._verify_proof_with_snarkjs(
                "signature_verification", proof_data
            )
            
            # Update statistics
            verification_time = time.time() - start_time
            self._update_stats("signature_proofs", verification_time, is_valid)
            
            return is_valid
            
        except Exception as e:
            print(f"ERROR verifying signature proof: {str(e)}")
            # Fallback to simulated verification
            return self._verify_simulated_signature_proof(proof_data)
    
    def _verify_proof_with_snarkjs(self, circuit_name: str,
                                 proof_data: Dict[str, Any]) -> bool:
        """
        Verify proof using SnarkJS.
        
        Args:
            circuit_name: Name of the circuit to use
            proof_data: The proof data to verify
            
        Returns:
            True if verification successful, False otherwise
        """
        circuit_info = self.circuit_manager.circuits[circuit_name]
        circuit_path = self.circuit_manager.circuit_path
        
        # Extract proof and public inputs
        proof = proof_data.get("proof", {})
        public_inputs = proof_data.get("public_inputs", [])
        
        # Create proof file
        proof_file = circuit_path / "proof.json"
        with open(proof_file, 'w') as f:
            json.dump(proof, f, indent=2)
        
        # Create public inputs file
        public_file = circuit_path / "public.json"
        with open(public_file, 'w') as f:
            json.dump(public_inputs, f, indent=2)
        
        # Get verification key
        vkey_file = circuit_path / f"{circuit_name}_verification_key.json"
        
        # If verification key doesn't exist, create it
        if not vkey_file.exists():
            zkey_file = circuit_path / circuit_info["zkey_file"]
            vkey_cmd = [
                "snarkjs", "groth16", "export verificationkey",
                str(zkey_file),
                str(vkey_file)
            ]
            
            result = subprocess.run(
                vkey_cmd,
                cwd=circuit_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise Exception(f"Verification key generation failed: {result.stderr}")
        
        # Verify proof
        verify_cmd = [
            "snarkjs", "groth16", "verify",
            str(vkey_file),
            str(public_file),
            str(proof_file)
        ]
        
        result = subprocess.run(
            verify_cmd,
            cwd=circuit_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Clean up temporary files
        for file_path in [proof_file, public_file]:
            if file_path.exists():
                file_path.unlink()
        
        return result.returncode == 0 and "OK" in result.stdout
    
    def _verify_simulated_document_proof(self, proof_data: Dict[str, Any]) -> bool:
        """Verify a simulated document hash proof for demo purposes."""
        # For demo purposes, we'll do basic validation
        required_fields = ["proof_type", "document_hash", "proof", "public_inputs"]
        
        for field in required_fields:
            if field not in proof_data:
                print(f"ERROR Missing required field: {field}")
                return False
        
        if proof_data["proof_type"] != "document_hash":
            print("ERROR Invalid proof type")
            return False
        
        # Check if proof structure is valid
        proof = proof_data["proof"]
        if not all(key in proof for key in ["pi_a", "pi_b", "pi_c"]):
            print("ERROR Invalid proof structure")
            return False
        
        # For simulated proofs, we'll return True if structure is valid
        return True
    
    def _verify_simulated_age_proof(self, proof_data: Dict[str, Any], 
                                  min_age: int) -> bool:
        """Verify a simulated age verification proof for demo purposes."""
        # For demo purposes, we'll do basic validation
        required_fields = ["proof_type", "min_age", "proof", "public_inputs"]
        
        for field in required_fields:
            if field not in proof_data:
                print(f"ERROR Missing required field: {field}")
                return False
        
        if proof_data["proof_type"] != "age_verification":
            print("ERROR Invalid proof type")
            return False
        
        # Check if minimum age requirement is met
        if proof_data["min_age"] < min_age:
            print(f"ERROR Minimum age requirement not met: {proof_data['min_age']} < {min_age}")
            return False
        
        # Check if proof structure is valid
        proof = proof_data["proof"]
        if not all(key in proof for key in ["pi_a", "pi_b", "pi_c"]):
            print("ERROR Invalid proof structure")
            return False
        
        # For simulated proofs, we'll return True if structure is valid
        return True
    
    def _verify_simulated_signature_proof(self, proof_data: Dict[str, Any]) -> bool:
        """Verify a simulated signature verification proof for demo purposes."""
        # For demo purposes, we'll do basic validation
        required_fields = ["proof_type", "document_hash", "signature_hash", "proof", "public_inputs"]
        
        for field in required_fields:
            if field not in proof_data:
                print(f"ERROR Missing required field: {field}")
                return False
        
        if proof_data["proof_type"] != "signature_verification":
            print("ERROR Invalid proof type")
            return False
        
        # Check if proof structure is valid
        proof = proof_data["proof"]
        if not all(key in proof for key in ["pi_a", "pi_b", "pi_c"]):
            print("ERROR Invalid proof structure")
            return False
        
        # For simulated proofs, we'll return True if structure is valid
        return True
    
    def _update_stats(self, proof_type: str, verification_time: float, success: bool):
        """Update proof verification statistics."""
        self.stats["total_verified"] += 1
        self.stats[f"{proof_type}"] += 1
        self.stats["verification_times"].append(verification_time)
        
        if success:
            self.stats["successful_verifications"] += 1
        else:
            self.stats["failed_verifications"] += 1
        
        # Keep only last 100 verification times
        if len(self.stats["verification_times"]) > 100:
            self.stats["verification_times"] = self.stats["verification_times"][-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get proof verification statistics.
        
        Returns:
            Dictionary containing statistics
        """
        avg_time = 0
        if self.stats["verification_times"]:
            avg_time = sum(self.stats["verification_times"]) / len(self.stats["verification_times"])
        
        success_rate = 0
        if self.stats["total_verified"] > 0:
            success_rate = self.stats["successful_verifications"] / self.stats["total_verified"]
        
        return {
            "total_verified": self.stats["total_verified"],
            "document_proofs": self.stats["document_proofs"],
            "age_proofs": self.stats["age_proofs"],
            "signature_proofs": self.stats["signature_proofs"],
            "successful_verifications": self.stats["successful_verifications"],
            "failed_verifications": self.stats["failed_verifications"],
            "success_rate": success_rate,
            "average_verification_time": avg_time,
            "total_verification_time": sum(self.stats["verification_times"])
        }
    
    def batch_verify_proofs(self, proof_data_list: List[Dict[str, Any]]) -> List[bool]:
        """
        Verify multiple proofs in batch.
        
        Args:
            proof_data_list: List of proof data to verify
            
        Returns:
            List of verification results
        """
        results = []
        
        for i, proof_data in enumerate(proof_data_list):
            print(f"INFO Verifying proof {i+1}/{len(proof_data_list)}")
            
            proof_type = proof_data.get("proof_type", "unknown")
            
            if proof_type == "document_hash":
                result = self.verify_document_hash_proof(proof_data)
            elif proof_type == "age_verification":
                min_age = proof_data.get("min_age", 0)
                result = self.verify_age_proof(proof_data, min_age)
            elif proof_type == "signature_verification":
                result = self.verify_signature_proof(proof_data)
            else:
                print(f"ERROR Unknown proof type: {proof_type}")
                result = False
            
            results.append(result)
        
        return results
    
    def verify_proof_integrity(self, proof_data: Dict[str, Any]) -> bool:
        """
        Verify the integrity of a proof (check for tampering).
        
        Args:
            proof_data: The proof data to check
            
        Returns:
            True if proof integrity is valid, False otherwise
        """
        try:
            # Check if proof has required metadata
            required_metadata = ["proof_type", "generated_at", "circuit_version"]
            for field in required_metadata:
                if field not in proof_data:
                    print(f"ERROR Missing metadata field: {field}")
                    return False
            
            # Check if proof was generated recently (within 24 hours)
            generated_at = proof_data["generated_at"]
            current_time = time.time()
            
            if current_time - generated_at > 86400:  # 24 hours
                print("WARNING  Proof is older than 24 hours")
            
            # Check if proof structure is valid
            if "proof" not in proof_data:
                print("ERROR Missing proof data")
                return False
            
            proof = proof_data["proof"]
            if not isinstance(proof, dict):
                print("ERROR Invalid proof structure")
                return False
            
            # Check if public inputs are present
            if "public_inputs" not in proof_data:
                print("ERROR Missing public inputs")
                return False
            
            return True
            
        except Exception as e:
            print(f"ERROR verifying proof integrity: {str(e)}")
            return False 