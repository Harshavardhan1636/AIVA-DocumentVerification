"""
Proof Generator for creating Zero-Knowledge Proofs.

This module handles the generation of ZK proofs using compiled Circom circuits
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


class ProofGenerator:
    """
    Generates Zero-Knowledge Proofs for document verification.
    
    This class handles:
    - Document hash verification proofs
    - Age verification proofs
    - Signature verification proofs
    - Proof serialization and storage
    """
    
    def __init__(self, circuit_manager: CircuitManager):
        """
        Initialize the Proof Generator.
        
        Args:
            circuit_manager: Circuit manager instance
        """
        self.circuit_manager = circuit_manager
        self.stats = {
            "total_generated": 0,
            "document_proofs": 0,
            "age_proofs": 0,
            "signature_proofs": 0,
            "generation_times": []
        }
    
    def generate_document_hash_proof(self, document_hash: str, 
                                   document_type: str = "general") -> Dict[str, Any]:
        """
        Generate a ZK proof for document hash verification.
        
        Args:
            document_hash: Hash of the document
            document_type: Type of document
            
        Returns:
            Dictionary containing the generated proof data
        """
        try:
            start_time = time.time()
            
            # Check if circuit is compiled
            if not self.circuit_manager.circuits["document_hash"]["compiled"]:
                # For demo purposes, generate a simulated proof
                return self._generate_simulated_document_proof(document_hash, document_type)
            
            # Prepare witness data
            witness_data = {
                "documentHash": document_hash,
                "publicHash": document_hash,
                "privateDocumentData": self._generate_private_data(document_hash)
            }
            
            # Generate proof using snarkjs
            proof_data = self._generate_proof_with_snarkjs(
                "document_hash", witness_data
            )
            
            # Add metadata
            proof_data.update({
                "proof_type": "document_hash",
                "document_type": document_type,
                "generated_at": time.time(),
                "circuit_version": "1.0.0"
            })
            
            # Update statistics
            self._update_stats("document_proofs", time.time() - start_time)
            
            return proof_data
            
        except Exception as e:
            print(f"ERROR generating document hash proof: {str(e)}")
            # Fallback to simulated proof
            return self._generate_simulated_document_proof(document_hash, document_type)
    
    def generate_age_verification_proof(self, age: int, min_age: int,
                                      document_type: str = "id_card") -> Dict[str, Any]:
        """
        Generate a ZK proof for age verification.
        
        Args:
            age: Actual age of the person
            min_age: Minimum age requirement
            document_type: Type of ID document
            
        Returns:
            Dictionary containing the generated proof data
        """
        try:
            start_time = time.time()
            
            # Check if circuit is compiled
            if not self.circuit_manager.circuits["age_verification"]["compiled"]:
                # For demo purposes, generate a simulated proof
                return self._generate_simulated_age_proof(age, min_age, document_type)
            
            # Prepare witness data
            witness_data = {
                "minAge": min_age,
                "publicAgeProof": self._generate_age_proof_hash(age, min_age),
                "privateAge": age
            }
            
            # Generate proof using snarkjs
            proof_data = self._generate_proof_with_snarkjs(
                "age_verification", witness_data
            )
            
            # Add metadata
            proof_data.update({
                "proof_type": "age_verification",
                "document_type": document_type,
                "min_age": min_age,
                "generated_at": time.time(),
                "circuit_version": "1.0.0"
            })
            
            # Update statistics
            self._update_stats("age_proofs", time.time() - start_time)
            
            return proof_data
            
        except Exception as e:
            print(f"ERROR generating age verification proof: {str(e)}")
            # Fallback to simulated proof
            return self._generate_simulated_age_proof(age, min_age, document_type)
    
    def generate_signature_proof(self, document_data: bytes,
                               signature: bytes) -> Dict[str, Any]:
        """
        Generate a ZK proof for signature verification.
        
        Args:
            document_data: The signed document data
            signature: The signature to verify
            
        Returns:
            Dictionary containing the generated proof data
        """
        try:
            start_time = time.time()
            
            # Check if circuit is compiled
            if not self.circuit_manager.circuits["signature_verification"]["compiled"]:
                # For demo purposes, generate a simulated proof
                return self._generate_simulated_signature_proof(document_data, signature)
            
            # Calculate hashes
            document_hash = hashlib.sha256(document_data).hexdigest()
            signature_hash = hashlib.sha256(signature).hexdigest()
            
            # Prepare witness data
            witness_data = {
                "publicSignatureHash": signature_hash,
                "documentHash": document_hash,
                "privateSignature": signature.hex(),
                "privateDocumentData": document_data.hex()
            }
            
            # Generate proof using snarkjs
            proof_data = self._generate_proof_with_snarkjs(
                "signature_verification", witness_data
            )
            
            # Add metadata
            proof_data.update({
                "proof_type": "signature_verification",
                "document_hash": document_hash,
                "signature_hash": signature_hash,
                "generated_at": time.time(),
                "circuit_version": "1.0.0"
            })
            
            # Update statistics
            self._update_stats("signature_proofs", time.time() - start_time)
            
            return proof_data
            
        except Exception as e:
            print(f"ERROR generating signature proof: {str(e)}")
            # Fallback to simulated proof
            return self._generate_simulated_signature_proof(document_data, signature)
    
    def _generate_proof_with_snarkjs(self, circuit_name: str, 
                                   witness_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate proof using SnarkJS.
        
        Args:
            circuit_name: Name of the circuit to use
            witness_data: Witness data for the circuit
            
        Returns:
            Generated proof data
        """
        circuit_info = self.circuit_manager.circuits[circuit_name]
        circuit_path = self.circuit_manager.circuit_path
        
        # Create witness file
        witness_file = circuit_path / f"{circuit_name}_witness.json"
        with open(witness_file, 'w') as f:
            json.dump(witness_data, f, indent=2)
        
        # Generate witness
        wasm_file = circuit_path / circuit_info["wasm_file"]
        witness_cmd = [
            "node", str(wasm_file),
            str(witness_file)
        ]
        
        result = subprocess.run(
            witness_cmd,
            cwd=circuit_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise Exception(f"Witness generation failed: {result.stderr}")
        
        # Generate proof
        zkey_file = circuit_path / circuit_info["zkey_file"]
        proof_cmd = [
            "snarkjs", "groth16", "prove",
            str(zkey_file),
            str(witness_file),
            "proof.json",
            "public.json"
        ]
        
        result = subprocess.run(
            proof_cmd,
            cwd=circuit_path,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise Exception(f"Proof generation failed: {result.stderr}")
        
        # Read generated proof
        proof_file = circuit_path / "proof.json"
        public_file = circuit_path / "public.json"
        
        with open(proof_file, 'r') as f:
            proof_data = json.load(f)
        
        with open(public_file, 'r') as f:
            public_data = json.load(f)
        
        # Clean up temporary files
        for file_path in [witness_file, proof_file, public_file]:
            if file_path.exists():
                file_path.unlink()
        
        return {
            "proof": proof_data,
            "public_inputs": public_data
        }
    
    def _generate_simulated_document_proof(self, document_hash: str,
                                         document_type: str) -> Dict[str, Any]:
        """Generate a simulated document hash proof for demo purposes."""
        return {
            "proof_type": "document_hash",
            "document_type": document_type,
            "document_hash": document_hash,
            "proof": {
                "pi_a": ["123456789", "987654321"],
                "pi_b": [["111111111", "222222222"], ["333333333", "444444444"]],
                "pi_c": ["555555555", "666666666"]
            },
            "public_inputs": [document_hash],
            "generated_at": time.time(),
            "simulated": True,
            "circuit_version": "1.0.0"
        }
    
    def _generate_simulated_age_proof(self, age: int, min_age: int,
                                    document_type: str) -> Dict[str, Any]:
        """Generate a simulated age verification proof for demo purposes."""
        return {
            "proof_type": "age_verification",
            "document_type": document_type,
            "min_age": min_age,
            "proof": {
                "pi_a": ["123456789", "987654321"],
                "pi_b": [["111111111", "222222222"], ["333333333", "444444444"]],
                "pi_c": ["555555555", "666666666"]
            },
            "public_inputs": [min_age, self._generate_age_proof_hash(age, min_age)],
            "generated_at": time.time(),
            "simulated": True,
            "circuit_version": "1.0.0"
        }
    
    def _generate_simulated_signature_proof(self, document_data: bytes,
                                          signature: bytes) -> Dict[str, Any]:
        """Generate a simulated signature verification proof for demo purposes."""
        document_hash = hashlib.sha256(document_data).hexdigest()
        signature_hash = hashlib.sha256(signature).hexdigest()
        
        return {
            "proof_type": "signature_verification",
            "document_hash": document_hash,
            "signature_hash": signature_hash,
            "proof": {
                "pi_a": ["123456789", "987654321"],
                "pi_b": [["111111111", "222222222"], ["333333333", "444444444"]],
                "pi_c": ["555555555", "666666666"]
            },
            "public_inputs": [signature_hash, document_hash],
            "generated_at": time.time(),
            "simulated": True,
            "circuit_version": "1.0.0"
        }
    
    def _generate_private_data(self, document_hash: str) -> str:
        """Generate private data for the circuit."""
        # In a real implementation, this would be the actual document data
        # For demo purposes, we'll use a hash of the document hash
        return hashlib.sha256(document_hash.encode()).hexdigest()
    
    def _generate_age_proof_hash(self, age: int, min_age: int) -> str:
        """Generate a hash for age proof verification."""
        data = f"{age}:{min_age}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _update_stats(self, proof_type: str, generation_time: float):
        """Update proof generation statistics."""
        self.stats["total_generated"] += 1
        self.stats[f"{proof_type}"] += 1
        self.stats["generation_times"].append(generation_time)
        
        # Keep only last 100 generation times
        if len(self.stats["generation_times"]) > 100:
            self.stats["generation_times"] = self.stats["generation_times"][-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get proof generation statistics.
        
        Returns:
            Dictionary containing statistics
        """
        avg_time = 0
        if self.stats["generation_times"]:
            avg_time = sum(self.stats["generation_times"]) / len(self.stats["generation_times"])
        
        return {
            "total_generated": self.stats["total_generated"],
            "document_proofs": self.stats["document_proofs"],
            "age_proofs": self.stats["age_proofs"],
            "signature_proofs": self.stats["signature_proofs"],
            "average_generation_time": avg_time,
            "total_generation_time": sum(self.stats["generation_times"])
        }
    
    def save_proof(self, proof_data: Dict[str, Any], 
                  filename: str) -> bool:
        """
        Save a proof to a file.
        
        Args:
            proof_data: The proof data to save
            filename: Name of the file to save to
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            proof_file = Path(filename)
            with open(proof_file, 'w') as f:
                json.dump(proof_data, f, indent=2)
            
            print(f"OK Proof saved to: {proof_file}")
            return True
            
        except Exception as e:
            print(f"ERROR saving proof: {str(e)}")
            return False
    
    def load_proof(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load a proof from a file.
        
        Args:
            filename: Name of the file to load from
            
        Returns:
            Loaded proof data or None if failed
        """
        try:
            proof_file = Path(filename)
            if not proof_file.exists():
                print(f"ERROR Proof file not found: {proof_file}")
                return None
            
            with open(proof_file, 'r') as f:
                proof_data = json.load(f)
            
            print(f"OK Proof loaded from: {proof_file}")
            return proof_data
            
        except Exception as e:
            print(f"ERROR loading proof: {str(e)}")
            return None 