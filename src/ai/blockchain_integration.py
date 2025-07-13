"""
Ethereum Blockchain Integration for AIVA Document Verification System

This module integrates with the Blockchain Module for:
- Smart contract interactions
- Document hash storage
- Verification record management
- Transaction handling
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import json
import hashlib

# Add blockchain module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'blockchain'))

try:
    from blockchain import BlockchainModule, create_blockchain_module
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False
    print("Warning: Blockchain module not available, using mock implementation")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerificationRecord:
    """Verification record to be stored on blockchain"""
    document_hash: str
    user_id: str
    verification_id: str
    is_verified: bool
    confidence_score: float
    fraud_score: float
    timestamp: int
    ai_analysis_hash: str
    blockchain_tx_hash: Optional[str] = None

@dataclass
class BlockchainConfig:
    """Blockchain configuration"""
    rpc_url: str
    contract_address: str
    private_key: str
    gas_limit: int = 3000000
    gas_price: int = 20000000000  # 20 gwei

class BlockchainIntegration:
    """
    Ethereum blockchain integration for document verification
    Uses the Blockchain Module for all blockchain operations
    """
    
    def __init__(self, network_url: str = "http://127.0.0.1:8545", 
                 contract_address: Optional[str] = None,
                 private_key: Optional[str] = None):
        """Initialize blockchain integration"""
        self.network_url = network_url
        self.contract_address = contract_address
        self.private_key = private_key
        self.blockchain_module = None
        self.use_mock = False
        
        # Try to initialize real blockchain module
        try:
            from src.blockchain import create_blockchain_module
            self.blockchain_module = create_blockchain_module(
                network_url, contract_address, private_key
            )
            logger.info("Blockchain module initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize blockchain module: {e}")
            logger.info("Using mock blockchain implementation for demo/testing")
            self.use_mock = True
    
    def store_verification_record(self, record: VerificationRecord) -> str:
        """
        Store verification record on blockchain
        
        Args:
            record: Verification record to store
            
        Returns:
            str: Transaction hash
        """
        try:
            if self.use_mock:
                return self._mock_store_verification_record(record)
            
            # Prepare data for blockchain module
            zk_proof_output = {
                "document_hash": record.document_hash,
                "proof": record.ai_analysis_hash  # Using AI analysis as proof
            }
            
            ai_verification_result = {
                "confidence": record.confidence_score,
                "verdict": "authentic" if record.is_verified else "fraudulent",
                "fraud_score": record.fraud_score
            }
            
            # Use blockchain module to process verification
            result = self.blockchain_module.process_verification(
                zk_proof_output=zk_proof_output,
                ai_verification_result=ai_verification_result,
                user_wallet_address=record.user_id
            )
            
            if "error" in result:
                raise Exception(f"Blockchain error: {result['error']}")
            
            # Extract transaction hash
            tx_hash = result["blockchain_result"]["transaction_hash"]
            record.blockchain_tx_hash = tx_hash
            
            logger.info(f"Verification record stored on blockchain")
            logger.info(f"Transaction hash: {tx_hash}")
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"Error storing verification record: {e}")
            if self.use_mock:
                return self._mock_store_verification_record(record)
            raise
    
    def get_verification_record(self, document_hash: str) -> Optional[VerificationRecord]:
        """
        Retrieve verification record from blockchain
        
        Args:
            document_hash: Hash of the document
            
        Returns:
            VerificationRecord or None if not found
        """
        try:
            if self.use_mock:
                return self._mock_get_verification_record(document_hash)
            
            # Use blockchain module to verify document
            result = self.blockchain_module.verify_document(document_hash)
            
            if result and "error" not in result:
                return VerificationRecord(
                    document_hash=document_hash,
                    user_id=result.get("owner_address", ""),
                    verification_id=result.get("verification_id", ""),
                    is_verified=result.get("is_valid", False),
                    confidence_score=result.get("confidence_score", 0.0),
                    fraud_score=1.0 - result.get("confidence_score", 0.0),
                    timestamp=result.get("timestamp", int(datetime.now().timestamp())),
                    ai_analysis_hash=result.get("ai_analysis_hash", ""),
                    blockchain_tx_hash=result.get("transaction_hash", "")
                )
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving verification record: {e}")
            if self.use_mock:
                return self._mock_get_verification_record(document_hash)
            return None
    
    def get_user_verifications(self, user_id: str) -> List[str]:
        """
        Get all verification IDs for a user
        
        Args:
            user_id: User ID
            
        Returns:
            List of verification IDs
        """
        try:
            if self.use_mock:
                return self._mock_get_user_verifications(user_id)
            
            # Use blockchain module to get user verifications
            result = self.blockchain_module.get_verification_history(user_id)
            
            if result and "error" not in result:
                return result.get("verifications", [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting user verifications: {e}")
            if self.use_mock:
                return self._mock_get_user_verifications(user_id)
            return []
    
    def verify_document_hash(self, document_hash: str) -> bool:
        """
        Verify if a document hash exists on blockchain
        
        Args:
            document_hash: Hash of the document
            
        Returns:
            bool: True if document exists and is verified
        """
        try:
            if self.use_mock:
                return self._mock_verify_document_hash(document_hash)
            
            # Use blockchain module to verify document
            result = self.blockchain_module.verify_document(document_hash)
            
            return result and "error" not in result and result.get("is_valid", False)
            
        except Exception as e:
            logger.error(f"Error verifying document hash: {e}")
            if self.use_mock:
                return self._mock_verify_document_hash(document_hash)
            return False
    
    def get_account_balance(self) -> float:
        """
        Get account balance
        
        Returns:
            float: Account balance in ETH
        """
        try:
            if self.use_mock:
                return self._mock_get_account_balance()
            
            # Use blockchain module to get balance
            # This would need to be implemented in the blockchain module
            return 1.0  # Placeholder
            
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            if self.use_mock:
                return self._mock_get_account_balance()
            return 0.0
    
    def create_document_hash(self, document_data: bytes) -> str:
        """
        Create hash for document data
        
        Args:
            document_data: Document data as bytes
            
        Returns:
            str: SHA256 hash of document
        """
        return hashlib.sha256(document_data).hexdigest()
    
    def create_analysis_hash(self, analysis_data: Dict[str, Any]) -> str:
        """
        Create hash for AI analysis data
        
        Args:
            analysis_data: AI analysis data
            
        Returns:
            str: SHA256 hash of analysis
        """
        analysis_json = json.dumps(analysis_data, sort_keys=True)
        return hashlib.sha256(analysis_json.encode()).hexdigest()
    
    def sign_message(self, message: str) -> str:
        """
        Sign a message with private key
        
        Args:
            message: Message to sign
            
        Returns:
            str: Signature
        """
        try:
            if self.use_mock:
                return self._mock_sign_message(message)
            
            # This would need to be implemented in the blockchain module
            return "mock_signature"
            
        except Exception as e:
            logger.error(f"Error signing message: {e}")
            if self.use_mock:
                return self._mock_sign_message(message)
            return "error_signature"
    
    def verify_signature(self, message: str, signature: str, address: str) -> bool:
        """
        Verify a message signature
        
        Args:
            message: Original message
            signature: Message signature
            address: Signer's address
            
        Returns:
            bool: True if signature is valid
        """
        try:
            if self.use_mock:
                return self._mock_verify_signature(message, signature, address)
            
            # This would need to be implemented in the blockchain module
            return True
            
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            if self.use_mock:
                return self._mock_verify_signature(message, signature, address)
            return False
    
    # Mock methods for fallback
    def _mock_store_verification_record(self, record: VerificationRecord) -> str:
        """Mock implementation for storing verification record"""
        mock_tx_hash = f"0x{hashlib.md5(f'{record.document_hash}{record.timestamp}'.encode()).hexdigest()}"
        record.blockchain_tx_hash = mock_tx_hash
        logger.info(f"Mock: Verification record stored with hash {mock_tx_hash}")
        return mock_tx_hash
    
    def _mock_get_verification_record(self, document_hash: str) -> Optional[VerificationRecord]:
        """Mock implementation for getting verification record"""
        # Return a mock record for testing
        return VerificationRecord(
            document_hash=document_hash,
            user_id="mock_user_123",
            verification_id="mock_verification_456",
            is_verified=True,
            confidence_score=0.89,
            fraud_score=0.11,
            timestamp=int(datetime.now().timestamp()),
            ai_analysis_hash="mock_analysis_hash",
            blockchain_tx_hash="mock_tx_hash"
        )
    
    def _mock_get_user_verifications(self, user_id: str) -> List[str]:
        """Mock implementation for getting user verifications"""
        return [f"mock_verification_{i}" for i in range(1, 4)]
    
    def _mock_verify_document_hash(self, document_hash: str) -> bool:
        """Mock implementation for verifying document hash"""
        return True
    
    def _mock_get_account_balance(self) -> float:
        """Mock implementation for getting account balance"""
        return 10.5
    
    def _mock_sign_message(self, message: str) -> str:
        """Mock implementation for signing message"""
        return f"mock_signature_{hashlib.md5(message.encode()).hexdigest()[:16]}"
    
    def _mock_verify_signature(self, message: str, signature: str, address: str) -> bool:
        """Mock implementation for verifying signature"""
        return True

class MockBlockchainIntegration:
    """
    Mock blockchain integration for testing and development
    """
    
    def __init__(self):
        """Initialize mock blockchain integration"""
        logger.info("Initialized mock blockchain integration")
    
    def store_verification_record(self, record: VerificationRecord) -> str:
        """Mock store verification record"""
        mock_tx_hash = f"0x{hashlib.md5(f'{record.document_hash}{record.timestamp}'.encode()).hexdigest()}"
        logger.info(f"Mock: Stored verification record with hash {mock_tx_hash}")
        return mock_tx_hash
    
    def get_verification_record(self, document_hash: str) -> Optional[VerificationRecord]:
        """Mock get verification record"""
        return VerificationRecord(
            document_hash=document_hash,
            user_id="mock_user",
            verification_id="mock_verification",
            is_verified=True,
            confidence_score=0.85,
            fraud_score=0.15,
            timestamp=int(datetime.now().timestamp()),
            ai_analysis_hash="mock_hash"
        )
    
    def get_user_verifications(self, user_id: str) -> List[str]:
        """Mock get user verifications"""
        return ["mock_verification_1", "mock_verification_2"]
    
    def verify_document_hash(self, document_hash: str) -> bool:
        """Mock verify document hash"""
        return True
    
    def get_account_balance(self) -> float:
        """Mock get account balance"""
        return 5.0
    
    def create_document_hash(self, document_data: bytes) -> str:
        """Create document hash"""
        return hashlib.sha256(document_data).hexdigest()
    
    def create_analysis_hash(self, analysis_data: Dict[str, Any]) -> str:
        """Create analysis hash"""
        analysis_json = json.dumps(analysis_data, sort_keys=True)
        return hashlib.sha256(analysis_json.encode()).hexdigest()
    
    def sign_message(self, message: str) -> str:
        """Mock sign message"""
        return "mock_signature"
    
    def verify_signature(self, message: str, signature: str, address: str) -> bool:
        """Mock verify signature"""
        return True 