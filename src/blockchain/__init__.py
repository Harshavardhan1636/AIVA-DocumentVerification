"""
AIVA Document Verification - Blockchain Module
Module 4: Blockchain Module
Purpose: Store verification records on Ethereum blockchain
Location: src/blockchain/

Input:
- ZK Proof Module output
- AI Agent verification result  
- User's wallet address

Processing:
- Smart Contract Interaction: Call verification contract
- Transaction Creation: Build and sign transaction
- Gas Optimization: Optimize transaction costs
- IPFS Storage: Store large data off-chain
- Event Logging: Record verification events

Output: JSON with blockchain_result, stored_data, verification_certificate, and timestamp
"""

from .web3_connector import Web3Connector
from .smart_contract import SmartContract
from .transaction_manager import TransactionManager
from .ipfs_handler import IPFSHandler
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional


class BlockchainModule:
    """
    Main blockchain module for AIVA Document Verification System
    Handles all blockchain interactions and verification storage
    """
    
    def __init__(self, network: str = "localhost", 
                 contract_address: Optional[str] = None,
                 private_key: Optional[str] = None):
        """
        Initialize blockchain module components
        
        Args:
            network: Ethereum network name (e.g., 'localhost', 'sepolia', 'mainnet')
            contract_address: Smart contract address
            private_key: User's private key for transactions
        """
        self.web3_connector = Web3Connector(network, private_key)
        self.smart_contract = SmartContract(self.web3_connector, contract_address)
        self.transaction_manager = TransactionManager(self.web3_connector)
        self.ipfs_handler = IPFSHandler()
        
    def process_verification(self, 
                           zk_proof_output: Dict[str, Any],
                           ai_verification_result: Dict[str, Any],
                           user_wallet_address: str) -> Dict[str, Any]:
        """
        Main processing function for document verification
        
        Args:
            zk_proof_output: Output from ZK Proof Module
            ai_verification_result: Result from AI Agent verification
            user_wallet_address: User's wallet address
            
        Returns:
            JSON response with blockchain_result, stored_data, verification_certificate, and timestamp
        """
        try:
            # Step 1: Validate ZK Proof
            zk_validation = self._validate_zk_proof(zk_proof_output)
            
            # Step 2: Smart Contract Interaction
            verification_data = self._prepare_verification_data(
                zk_proof_output, ai_verification_result, user_wallet_address, zk_validation
            )
            
            # Step 3: IPFS Storage (Store large data off-chain)
            ipfs_hash = self.ipfs_handler.store_verification_data(verification_data)
            
            # Step 4: Transaction Creation and Gas Optimization
            transaction_result = self.transaction_manager.create_verification_transaction(
                verification_data, ipfs_hash, user_wallet_address
            )
            
            # Step 5: Event Logging
            self._log_verification_event(transaction_result, verification_data)
            
            # Step 6: Generate verification certificate
            certificate = self._generate_verification_certificate(transaction_result, zk_validation)
            
            # Step 7: Prepare final output
            result = {
                "blockchain_result": {
                    "transaction_hash": transaction_result["transaction_hash"],
                    "block_number": transaction_result["block_number"],
                    "contract_address": self.smart_contract.contract_address,
                    "gas_used": transaction_result["gas_used"],
                    "verification_id": transaction_result["verification_id"]
                },
                "stored_data": {
                    "on_chain": {
                        "document_hash": verification_data["document_hash"],
                        "verification_timestamp": int(time.time()),
                        "authenticity_score": ai_verification_result.get("confidence", 0),
                        "verifier_address": user_wallet_address,
                        "zk_proof_valid": zk_validation["zk_proof_valid"],
                        "zk_proof_hash": verification_data.get("zk_proof_hash")
                    },
                    "ipfs_hash": ipfs_hash,
                    "zk_proof_stored": True
                },
                "verification_certificate": certificate,
                "zk_validation": zk_validation,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
    
    def _prepare_verification_data(self, 
                                 zk_proof_output: Dict[str, Any],
                                 ai_verification_result: Dict[str, Any],
                                 user_wallet_address: str,
                                 zk_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare verification data for blockchain storage"""
        return {
            "document_hash": zk_proof_output.get("document_hash", ""),
            "zk_proof": zk_proof_output.get("proof", ""),
            "zk_proof_hash": zk_proof_output.get("proof_hash", ""), # Added zk_proof_hash
            "ai_confidence": ai_verification_result.get("confidence", 0),
            "ai_verdict": ai_verification_result.get("verdict", "unknown"),
            "user_address": user_wallet_address,
            "verification_timestamp": int(time.time()),
            "zk_proof_valid": zk_validation["zk_proof_valid"] # Added zk_proof_valid
        }
    
    def _log_verification_event(self, transaction_result: Dict[str, Any], 
                              verification_data: Dict[str, Any]):
        """Log verification event for audit trail"""
        event_data = {
            "event_type": "document_verification",
            "transaction_hash": transaction_result["transaction_hash"],
            "document_hash": verification_data["document_hash"],
            "user_address": verification_data["user_address"],
            "timestamp": datetime.utcnow().isoformat()
        }
        # In production, this would be logged to a database or external service
        print(f"Verification Event: {json.dumps(event_data, indent=2)}")
    
    def _generate_verification_certificate(self, transaction_result: Dict[str, Any], 
                                          zk_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate verification certificate"""
        certificate_id = f"AIVA_CERT_{datetime.now().strftime('%Y%m%d')}_{transaction_result['verification_id'][-3:]}"
        
        return {
            "certificate_id": certificate_id,
            "blockchain_proof": transaction_result["transaction_hash"],
            "validity_period": "permanent",
            "verification_url": f"https://etherscan.io/tx/{transaction_result['transaction_hash']}",
            "zk_proof_valid": zk_validation["zk_proof_valid"] # Added zk_proof_valid
        }
    
    def verify_document(self, document_hash: str) -> Dict[str, Any]:
        """
        Verify a document on the blockchain
        
        Args:
            document_hash: Hash of the document to verify
            
        Returns:
            Verification result from smart contract
        """
        return self.smart_contract.verify_document(document_hash)
    
    def get_verification_history(self, user_address: str) -> Dict[str, Any]:
        """
        Get verification history for a user
        
        Args:
            user_address: User's wallet address
            
        Returns:
            List of verifications for the user
        """
        return self.smart_contract.get_user_verifications(user_address)

    def _validate_zk_proof(self, zk_proof_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the ZK Proof output.
        This is a placeholder for actual ZK proof validation logic.
        In a real scenario, this would involve calling a ZK proof verification contract.
        """
        # For demonstration, we'll assume a simple check
        # In a real ZK proof, you'd verify the proof against a trusted setup
        # and check if it's valid for the given statement.
        # For now, we'll just return a dummy valid result.
        return {
            "zk_proof_valid": True,
            "message": "ZK proof validation successful (dummy check)"
        }

    def register_document(self, document_hash: str, document_type: str, is_authentic: bool, confidence_score: int) -> Dict[str, Any]:
        """
        Register a document on the blockchain
        
        Args:
            document_hash: Hash of the document
            document_type: Type of document (e.g., "Aadhaar", "Passport")
            is_authentic: Whether the document is authentic
            confidence_score: Confidence score (0-100)
            
        Returns:
            Registration result
        """
        try:
            # Convert document hash to bytes32 format
            doc_hash_bytes = self.web3_connector.w3.to_bytes(hexstr=document_hash)
            
            # Prepare transaction data
            transaction_data = {
                'document_hash': doc_hash_bytes.hex(),
                'document_type': document_type,
                'is_authentic': is_authentic,
                'confidence_score': confidence_score,
                'value': self.smart_contract.contract.functions.verificationFee().call()
            }
            
            # Create and send transaction
            transaction_result = self.transaction_manager.create_document_registration_transaction(
                transaction_data
            )
            
            return {
                "success": True,
                "transaction_hash": transaction_result.get("transaction_hash"),
                "block_number": transaction_result.get("block_number"),
                "document_hash": document_hash,
                "document_type": document_type,
                "is_authentic": is_authentic,
                "confidence_score": confidence_score,
                "gas_used": transaction_result.get("gas_used")
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def log_verification_result(self, document_hash: str, is_authentic: bool, confidence_score: int, tampering_indicators: str) -> Dict[str, Any]:
        """
        Log verification result on the blockchain
        
        Args:
            document_hash: Hash of the document
            is_authentic: Whether the document is authentic
            confidence_score: Confidence score (0-100)
            tampering_indicators: JSON string of tampering indicators
            
        Returns:
            Logging result
        """
        try:
            # Convert document hash to bytes32 format
            doc_hash_bytes = self.web3_connector.w3.to_bytes(hexstr=document_hash)
            
            # Prepare transaction data
            transaction_data = {
                'document_hash': doc_hash_bytes.hex(),
                'is_authentic': is_authentic,
                'confidence_score': confidence_score,
                'tampering_indicators': tampering_indicators
            }
            
            # Create and send transaction
            transaction_result = self.transaction_manager.create_verification_result_transaction(
                transaction_data
            )
            
            return {
                "success": True,
                "transaction_hash": transaction_result.get("transaction_hash"),
                "block_number": transaction_result.get("block_number"),
                "document_hash": document_hash,
                "is_authentic": is_authentic,
                "confidence_score": confidence_score,
                "tampering_indicators": tampering_indicators,
                "gas_used": transaction_result.get("gas_used")
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_contract_stats(self) -> Dict[str, Any]:
        """
        Get contract statistics
        
        Returns:
            Contract statistics
        """
        try:
            stats = self.smart_contract.get_contract_stats()
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_document_details(self, document_hash: str) -> Dict[str, Any]:
        """
        Get detailed information about a document
        
        Args:
            document_hash: Hash of the document
            
        Returns:
            Document details
        """
        try:
            # Convert document hash to bytes32 format
            doc_hash_bytes = self.web3_connector.w3.to_bytes(hexstr=document_hash)
            
            details = self.smart_contract.get_document_details(doc_hash_bytes.hex())
            return {
                "success": True,
                "details": details
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# Main module interface
def create_blockchain_module(network_url: str = "http://127.0.0.1:8545",
                           contract_address: Optional[str] = None,
                           private_key: Optional[str] = None) -> BlockchainModule:
    """
    Factory function to create blockchain module instance
    
    Args:
        network_url: Ethereum network URL
        contract_address: Smart contract address
        private_key: User's private key
        
    Returns:
        BlockchainModule instance
    """
    try:
        return BlockchainModule(network_url, contract_address, private_key)
    except Exception as e:
        print(f"Warning: Blockchain module not available, using mock implementation: {e}")
        # Return a mock blockchain module for testing
        return MockBlockchainModule()


class MockBlockchainModule:
    """
    Mock blockchain module for testing when real blockchain is not available
    """
    
    def __init__(self):
        self.web3_connector = None
        self.smart_contract = None
        self.transaction_manager = None
        self.ipfs_handler = None
        
    def process_verification(self, 
                           zk_proof_output: Dict[str, Any],
                           ai_verification_result: Dict[str, Any],
                           user_wallet_address: str) -> Dict[str, Any]:
        """Mock verification processing"""
        return {
            "blockchain_result": {
                "transaction_hash": "0x" + "a" * 64,
                "block_number": 12345,
                "contract_address": "0x" + "b" * 40,
                "gas_used": 21000,
                "verification_id": "mock_verification_123"
            },
            "stored_data": {
                "on_chain": {
                    "document_hash": zk_proof_output.get("document_hash", "0x" + "c" * 64),
                    "verification_timestamp": int(time.time()),
                    "authenticity_score": ai_verification_result.get("confidence", 0),
                    "verifier_address": user_wallet_address,
                    "zk_proof_valid": True,
                    "zk_proof_hash": zk_proof_output.get("proof_hash", "0x" + "d" * 64)
                },
                "ipfs_hash": "Qm" + "e" * 44,
                "zk_proof_stored": True
            },
            "verification_certificate": {
                "certificate_id": f"AIVA_CERT_{datetime.now().strftime('%Y%m%d')}_123",
                "blockchain_proof": "0x" + "a" * 64,
                "validity_period": "permanent",
                "verification_url": "https://etherscan.io/tx/0x" + "a" * 64,
                "zk_proof_valid": True
            },
            "zk_validation": {
                "zk_proof_valid": True,
                "validation_details": "Mock validation successful"
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    def verify_document(self, document_hash: str) -> Dict[str, Any]:
        """Mock document verification"""
        return {
            "verified": True,
            "document_hash": document_hash,
            "verification_timestamp": int(time.time())
        }
    
    def get_verification_history(self, user_address: str) -> Dict[str, Any]:
        """Mock verification history"""
        return {
            "user_address": user_address,
            "verifications": [],
            "total_count": 0
        }


# Export main class
__all__ = ['BlockchainModule', 'create_blockchain_module', 'BlockchainManager']


class BlockchainManager:
    """
    Simplified blockchain manager for testing and integration
    Provides hash generation and verification methods without network dependencies
    """
    
    def __init__(self):
        """Initialize blockchain manager"""
        self.verification_count = 0
        self.verifications = []
    
    def generate_document_hash(self, data: str) -> str:
        """Generate hash for document verification"""
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify_document_hash(self, data: str, hash_value: str) -> bool:
        """Verify document hash"""
        import hashlib
        expected_hash = hashlib.sha256(data.encode()).hexdigest()
        return expected_hash == hash_value
    
    def log_verification(self, verification_data):
        """Log verification to blockchain (mock implementation)"""
        import hashlib
        import time
        
        self.verification_count += 1
        verification_id = f"VER_{self.verification_count:04d}"
        record = {
            "verification_id": verification_id,
            "timestamp": int(time.time()),
            **verification_data
        }
        self.verifications.append(record)
        return {
            "hash": hashlib.sha256(str(record).encode()).hexdigest(),
            "verification_id": verification_id,
            "timestamp": record["timestamp"],
            "details": record
        } 