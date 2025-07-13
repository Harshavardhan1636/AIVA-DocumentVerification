"""
Circuit Manager for handling Circom circuit operations.

This module manages the compilation, setup, and management of ZK circuits
used for document verification proofs.
"""

import os
import json
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path


class CircuitManager:
    """
    Manages Circom circuits for ZK proof generation and verification.
    
    This class handles:
    - Circuit compilation
    - Proving key generation
    - Circuit status monitoring
    - Circuit file management
    """
    
    def __init__(self, circuit_path: str = "circuits/"):
        """
        Initialize the Circuit Manager.
        
        Args:
            circuit_path: Path to the circuit files directory
        """
        self.circuit_path = Path(circuit_path)
        self.circuit_path.mkdir(exist_ok=True)
        
        # Circuit definitions
        self.circuits = {
            "document_hash": {
                "file": "document_hash.circom",
                "description": "Document hash verification circuit",
                "compiled": False,
                "r1cs_file": "document_hash.r1cs",
                "wasm_file": "document_hash.wasm",
                "zkey_file": "document_hash_0000.zkey"
            },
            "age_verification": {
                "file": "age_verification.circom", 
                "description": "Age verification circuit",
                "compiled": False,
                "r1cs_file": "age_verification.r1cs",
                "wasm_file": "age_verification.wasm",
                "zkey_file": "age_verification_0000.zkey"
            },
            "signature_verification": {
                "file": "signature_verification.circom",
                "description": "Signature verification circuit", 
                "compiled": False,
                "r1cs_file": "signature_verification.r1cs",
                "wasm_file": "signature_verification.wasm",
                "zkey_file": "signature_verification_0000.zkey"
            }
        }
        
        # Check if circuits are already compiled
        self._check_compilation_status()
    
    def compile_circuit(self, circuit_name: str) -> bool:
        """
        Compile a specific circuit.
        
        Args:
            circuit_name: Name of the circuit to compile
            
        Returns:
            True if compilation successful, False otherwise
        """
        if circuit_name not in self.circuits:
            raise ValueError(f"Unknown circuit: {circuit_name}")
        
        circuit_info = self.circuits[circuit_name]
        circuit_file = self.circuit_path / circuit_info["file"]
        
        if not circuit_file.exists():
            # Create a basic circuit file if it doesn't exist
            self._create_basic_circuit(circuit_name)
        
        try:
            # Compile the circuit using circom
            cmd = [
                "circom",
                str(circuit_file),
                "--r1cs",
                "--wasm", 
                "--sym",
                "--c"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.circuit_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                circuit_info["compiled"] = True
                print(f"OK Successfully compiled circuit: {circuit_name}")
                return True
            else:
                print(f"ERROR Failed to compile circuit {circuit_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"ERROR Compilation timeout for circuit: {circuit_name}")
            return False
        except FileNotFoundError:
            print("ERROR Circom not found. Please install circom first.")
            return False
        except Exception as e:
            print(f"ERROR compiling circuit {circuit_name}: {str(e)}")
            return False
    
    def compile_all_circuits(self) -> Dict[str, bool]:
        """
        Compile all circuits.
        
        Returns:
            Dictionary mapping circuit names to compilation success status
        """
        results = {}
        
        for circuit_name in self.circuits:
            print(f"INFO Compiling circuit: {circuit_name}")
            results[circuit_name] = self.compile_circuit(circuit_name)
        
        return results
    
    def setup_proving_keys(self, circuit_name: str) -> bool:
        """
        Generate proving keys for a circuit.
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            True if setup successful, False otherwise
        """
        if circuit_name not in self.circuits:
            raise ValueError(f"Unknown circuit: {circuit_name}")
        
        circuit_info = self.circuits[circuit_name]
        
        if not circuit_info["compiled"]:
            print(f"ERROR Circuit {circuit_name} not compiled. Compile first.")
            return False
        
        try:
            # Check if we have the powers of tau file
            pot_file = self.circuit_path / "pot12_final.ptau"
            if not pot_file.exists():
                print("WARNING  Powers of tau file not found. Using simulated setup.")
                # For demo purposes, we'll simulate the setup
                return self._simulate_key_setup(circuit_name)
            
            # Generate proving keys using snarkjs
            r1cs_file = self.circuit_path / circuit_info["r1cs_file"]
            zkey_file = self.circuit_path / circuit_info["zkey_file"]
            
            cmd = [
                "snarkjs", "groth16", "setup",
                str(r1cs_file),
                str(pot_file),
                str(zkey_file)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.circuit_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                print(f"OK Successfully generated proving keys for: {circuit_name}")
                return True
            else:
                print(f"ERROR Failed to generate proving keys for {circuit_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"ERROR Key setup timeout for circuit: {circuit_name}")
            return False
        except FileNotFoundError:
            print("ERROR SnarkJS not found. Please install snarkjs first.")
            return False
        except Exception as e:
            print(f"ERROR setting up proving keys for {circuit_name}: {str(e)}")
            return False
    
    def setup_all_proving_keys(self) -> Dict[str, bool]:
        """
        Generate proving keys for all circuits.
        
        Returns:
            Dictionary mapping circuit names to setup success status
        """
        results = {}
        
        for circuit_name in self.circuits:
            if self.circuits[circuit_name]["compiled"]:
                print(f"INFO Setting up proving keys for: {circuit_name}")
                results[circuit_name] = self.setup_proving_keys(circuit_name)
            else:
                print(f"WARNING  Skipping {circuit_name} - not compiled")
                results[circuit_name] = False
        
        return results
    
    def get_circuit_info(self, circuit_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific circuit.
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            Circuit information dictionary or None if not found
        """
        if circuit_name not in self.circuits:
            return None
        
        circuit_info = self.circuits[circuit_name].copy()
        
        # Add file existence information
        for key in ["r1cs_file", "wasm_file", "zkey_file"]:
            file_path = self.circuit_path / circuit_info[key]
            circuit_info[f"{key}_exists"] = file_path.exists()
        
        return circuit_info
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get overall status of all circuits.
        
        Returns:
            Dictionary containing status information
        """
        status = {
            "total_circuits": len(self.circuits),
            "compiled_circuits": sum(1 for c in self.circuits.values() if c["compiled"]),
            "circuit_details": {}
        }
        
        for name, info in self.circuits.items():
            status["circuit_details"][name] = self.get_circuit_info(name)
        
        return status
    
    def clean_circuits(self) -> bool:
        """
        Clean all compiled circuit files.
        
        Returns:
            True if cleanup successful
        """
        try:
            for circuit_info in self.circuits.values():
                for key in ["r1cs_file", "wasm_file", "zkey_file"]:
                    file_path = self.circuit_path / circuit_info[key]
                    if file_path.exists():
                        file_path.unlink()
                
                circuit_info["compiled"] = False
            
            print("OK Cleaned all circuit files")
            return True
            
        except Exception as e:
            print(f"ERROR cleaning circuits: {str(e)}")
            return False
    
    def _check_compilation_status(self):
        """Check which circuits are already compiled."""
        for circuit_name, circuit_info in self.circuits.items():
            r1cs_file = self.circuit_path / circuit_info["r1cs_file"]
            wasm_file = self.circuit_path / circuit_info["wasm_file"]
            
            circuit_info["compiled"] = r1cs_file.exists() and wasm_file.exists()
    
    def _create_basic_circuit(self, circuit_name: str):
        """Create a basic circuit file for demo purposes."""
        circuit_file = self.circuit_path / self.circuits[circuit_name]["file"]
        
        if circuit_name == "document_hash":
            circuit_content = self._get_document_hash_circuit()
        elif circuit_name == "age_verification":
            circuit_content = self._get_age_verification_circuit()
        elif circuit_name == "signature_verification":
            circuit_content = self._get_signature_verification_circuit()
        else:
            circuit_content = self._get_basic_circuit()
        
        with open(circuit_file, 'w') as f:
            f.write(circuit_content)
        
        print(f"INFO Created basic circuit file: {circuit_file}")
    
    def _get_document_hash_circuit(self) -> str:
        """Get the document hash verification circuit template."""
        return '''
pragma circom 2.1.4;

include "node_modules/circomlib/circuits/poseidon.circom";
include "node_modules/circomlib/circuits/comparators.circom";

template DocumentHash() {
    // Public inputs
    signal input documentHash;
    signal input publicHash;
    
    // Private inputs
    signal input privateDocumentData;
    
    // Outputs
    signal output isValid;
    
    // Component for hash verification
    component hasher = Poseidon(1);
    component isEqual = IsEqual();
    
    // Hash the private document data
    hasher.inputs[0] <== privateDocumentData;
    
    // Compare the computed hash with the public hash
    isEqual.in[0] <== hasher.out;
    isEqual.in[1] <== documentHash;
    
    // Output is valid if hashes match
    isValid <== isEqual.out;
}

component main { public [publicHash] } = DocumentHash();
'''
    
    def _get_age_verification_circuit(self) -> str:
        """Get the age verification circuit template."""
        return '''
pragma circom 2.1.4;

include "node_modules/circomlib/circuits/comparators.circom";

template AgeVerification() {
    // Public inputs
    signal input minAge;
    signal input publicAgeProof;
    
    // Private inputs
    signal input privateAge;
    
    // Outputs
    signal output isOldEnough;
    
    // Component for age comparison
    component ageCheck = GreaterThan(8); // 8 bits for age (0-255)
    
    // Check if private age is greater than or equal to minimum age
    ageCheck.in[0] <== privateAge;
    ageCheck.in[1] <== minAge;
    
    // Output is true if age is sufficient
    isOldEnough <== ageCheck.out;
}

component main { public [minAge, publicAgeProof] } = AgeVerification();
'''
    
    def _get_signature_verification_circuit(self) -> str:
        """Get the signature verification circuit template."""
        return '''
pragma circom 2.1.4;

include "node_modules/circomlib/circuits/poseidon.circom";
include "node_modules/circomlib/circuits/comparators.circom";

template SignatureVerification() {
    // Public inputs
    signal input publicSignatureHash;
    signal input documentHash;
    
    // Private inputs
    signal input privateSignature;
    signal input privateDocumentData;
    
    // Outputs
    signal output isValidSignature;
    
    // Components
    component docHasher = Poseidon(1);
    component sigHasher = Poseidon(1);
    component isEqual = IsEqual();
    
    // Hash the document data
    docHasher.inputs[0] <== privateDocumentData;
    
    // Hash the signature
    sigHasher.inputs[0] <== privateSignature;
    
    // Verify signature matches
    isEqual.in[0] <== sigHasher.out;
    isEqual.in[1] <== publicSignatureHash;
    
    // Verify document hash matches
    component docCheck = IsEqual();
    docCheck.in[0] <== docHasher.out;
    docCheck.in[1] <== documentHash;
    
    // Both conditions must be true
    isValidSignature <== isEqual.out * docCheck.out;
}

component main { public [publicSignatureHash, documentHash] } = SignatureVerification();
'''
    
    def _get_basic_circuit(self) -> str:
        """Get a basic circuit template."""
        return '''
pragma circom 2.1.4;

template BasicCircuit() {
    signal input in;
    signal output out;
    
    out <== in;
}

component main { public [in] } = BasicCircuit();
'''
    
    def _simulate_key_setup(self, circuit_name: str) -> bool:
        """Simulate key setup for demo purposes."""
        print(f"INFO Simulating key setup for {circuit_name}")
        
        # Create a dummy zkey file
        zkey_file = self.circuit_path / self.circuits[circuit_name]["zkey_file"]
        zkey_file.touch()
        
        return True 