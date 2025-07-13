"""
Contract Interface for AIVA Document Verification System
Handles smart contract interactions for document verification
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Tuple, List
from web3 import Web3
from web3.contract import Contract
from eth_account.signers.local import LocalAccount
import hashlib

from .web3_connector import Web3Connector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContractInterface:
    """
    Interface for DocumentVerification smart contract interactions
    """
    
    def __init__(self, web3_connector: Web3Connector, contract_address: Optional[str] = None):
        """
        Initialize contract interface
        
        Args:
            web3_connector: Web3Connector instance
            contract_address: Contract address (optional, will load from deployment)
        """
        self.web3_connector = web3_connector
        self.w3 = web3_connector.get_web3()
        self.account = web3_connector.get_account()
        self.contract_address = contract_address
        self.contract = None
        
        self._load_contract()
    
    def _load_contract(self) -> None:
        """Load contract instance and ABI"""
        try:
            # Load contract address if not provided
            if not self.contract_address:
                self.contract_address = self._get_deployed_contract_address()
            
            if not self.contract_address:
                raise ValueError("Contract address not found. Please deploy contract first.")
            
            # Load contract ABI
            contract_abi = self._load_contract_abi()
            if not contract_abi:
                raise ValueError("Contract ABI not found. Please compile contracts first.")
            
            # Create contract instance
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=contract_abi
            )
            
            logger.info(f"📋 Contract loaded at address: {self.contract_address}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load contract: {str(e)}")
            raise
    
    def _get_deployed_contract_address(self) -> Optional[str]:
        """Get contract address from deployment files"""
        try:
            network = self.web3_connector.network
            deployment_path = f"deployments/{network}.json"
            
            if os.path.exists(deployment_path):
                with open(deployment_path, 'r') as f:
                    deployment_data = json.load(f)
                    return deployment_data.get('contractAddress')
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to load deployment info: {str(e)}")
            return None
    
    def _load_contract_abi(self) -> Optional[List]:
        """Load contract ABI from artifacts"""
        try:
            artifacts_path = "artifacts/contracts/DocumentVerification.sol/DocumentVerification.json"
            
            if os.path.exists(artifacts_path):
                with open(artifacts_path, 'r') as f:
                    contract_data = json.load(f)
                    return contract_data['abi']
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to load contract ABI: {str(e)}")
            return None
    
    def register_document(self, document_hash: str, document_type: str, ipfs_hash: str) -> Dict[str, Any]:
        """
        Register a new document on the blockchain
        
        Args:
            document_hash: Hash of the document
            document_type: Type of document (e.g., "Aadhaar", "Passport")
            ipfs_hash: IPFS hash for document storage
            
        Returns:
            Transaction details
        """
        try:
            # Convert document hash to bytes32
            doc_hash_bytes = Web3.to_bytes(hexstr=document_hash) if document_hash.startswith('0x') else Web3.keccak(text=document_hash)
            
            # Get verification fee
            verification_fee = self.contract.functions.verificationFee().call()
            
            # Build transaction
            transaction = self.contract.functions.registerDocument(
                doc_hash_bytes,
                document_type,
                ipfs_hash
            ).build_transaction({
                'from': self.account.address,
                'value': verification_fee,
                'gas': 300000,  # Estimated gas
                'gasPrice': self.web3_connector.get_gas_price(),
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Estimate gas
            estimated_gas = self.web3_connector.estimate_gas(transaction)
            transaction['gas'] = estimated_gas
            
            # Send transaction
            tx_hash = self.web3_connector.send_transaction(transaction)
            
            # Wait for confirmation
            receipt = self.web3_connector.wait_for_transaction(tx_hash)
            
            result = {
                "success": receipt.status == 1,
                "tx_hash": tx_hash,
                "block_number": receipt.blockNumber,
                "gas_used": receipt.gasUsed,
                "document_hash": document_hash,
                "document_type": document_type,
                "ipfs_hash": ipfs_hash
            }
            
            if result["success"]:
                logger.info(f"✅ Document registered successfully: {document_hash}")
            else:
                logger.error(f"❌ Document registration failed: {document_hash}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to register document: {str(e)}")
            raise
    
    def verify_document(self, document_hash: str) -> Dict[str, Any]:
        """
        Verify a document on the blockchain
        
        Args:
            document_hash: Hash of the document to verify
            
        Returns:
            Verification result
        """
        try:
            # Convert document hash to bytes32
            doc_hash_bytes = Web3.to_bytes(hexstr=document_hash) if document_hash.startswith('0x') else Web3.keccak(text=document_hash)
            
            # Call contract function
            result = self.contract.functions.verifyDocument(doc_hash_bytes).call()
            
            verification_result = {
                "is_valid": result[0],
                "owner_address": result[1],
                "timestamp": result[2],
                "document_hash": document_hash,
                "verified_at": self.w3.eth.get_block('latest').timestamp
            }
            
            logger.info(f"🔍 Document verification result: {verification_result}")
            return verification_result
            
        except Exception as e:
            logger.error(f"❌ Failed to verify document: {str(e)}")
            return {"error": str(e)}
    
    def get_document_details(self, document_hash: str) -> Dict[str, Any]:
        """
        Get detailed information about a document
        
        Args:
            document_hash: Hash of the document
            
        Returns:
            Document details
        """
        try:
            # Convert document hash to bytes32
            doc_hash_bytes = Web3.to_bytes(hexstr=document_hash) if document_hash.startswith('0x') else Web3.keccak(text=document_hash)
            
            # Call contract function
            result = self.contract.functions.getDocumentDetails(doc_hash_bytes).call()
            
            details = {
                "owner_address": result[0],
                "timestamp": result[1],
                "document_type": result[2],
                "ipfs_hash": result[3],
                "is_valid": result[4],
                "document_hash": document_hash
            }
            
            logger.info(f"📄 Document details retrieved: {details}")
            return details
            
        except Exception as e:
            logger.error(f"❌ Failed to get document details: {str(e)}")
            return {"error": str(e)}
    
    def grant_access(self, document_hash: str, user_address: str) -> Dict[str, Any]:
        """
        Grant access to a document for another user
        
        Args:
            document_hash: Hash of the document
            user_address: Address to grant access to
            
        Returns:
            Transaction result
        """
        try:
            # Convert document hash to bytes32
            doc_hash_bytes = Web3.to_bytes(hexstr=document_hash) if document_hash.startswith('0x') else Web3.keccak(text=document_hash)
            
            # Build transaction
            transaction = self.contract.functions.grantAccess(
                doc_hash_bytes,
                user_address
            ).build_transaction({
                'from': self.account.address,
                'gas': 100000,
                'gasPrice': self.web3_connector.get_gas_price(),
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Send transaction
            tx_hash = self.web3_connector.send_transaction(transaction)
            receipt = self.web3_connector.wait_for_transaction(tx_hash)
            
            result = {
                "success": receipt.status == 1,
                "tx_hash": tx_hash,
                "document_hash": document_hash,
                "user_address": user_address
            }
            
            logger.info(f"🔓 Access granted: {user_address} -> {document_hash}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to grant access: {str(e)}")
            raise
    
    def revoke_access(self, document_hash: str, user_address: str) -> Dict[str, Any]:
        """
        Revoke access to a document for a user
        
        Args:
            document_hash: Hash of the document
            user_address: Address to revoke access from
            
        Returns:
            Transaction result
        """
        try:
            # Convert document hash to bytes32
            doc_hash_bytes = Web3.to_bytes(hexstr=document_hash) if document_hash.startswith('0x') else Web3.keccak(text=document_hash)
            
            # Build transaction
            transaction = self.contract.functions.revokeAccess(
                doc_hash_bytes,
                user_address
            ).build_transaction({
                'from': self.account.address,
                'gas': 100000,
                'gasPrice': self.web3_connector.get_gas_price(),
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Send transaction
            tx_hash = self.web3_connector.send_transaction(transaction)
            receipt = self.web3_connector.wait_for_transaction(tx_hash)
            
            result = {
                "success": receipt.status == 1,
                "tx_hash": tx_hash,
                "document_hash": document_hash,
                "user_address": user_address
            }
            
            logger.info(f"🔒 Access revoked: {user_address} -> {document_hash}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to revoke access: {str(e)}")
            raise
    
    def get_user_documents(self, user_address: Optional[str] = None) -> List[str]:
        """
        Get all documents for a user
        
        Args:
            user_address: Address to check (defaults to connected account)
            
        Returns:
            List of document hashes
        """
        try:
            target_address = user_address or self.account.address
            
            # Call contract function
            document_hashes = self.contract.functions.getUserDocuments(target_address).call()
            
            # Convert bytes32 to hex strings
            hex_hashes = [Web3.to_hex(hash_bytes) for hash_bytes in document_hashes]
            
            logger.info(f"📚 Found {len(hex_hashes)} documents for {target_address}")
            return hex_hashes
            
        except Exception as e:
            logger.error(f"❌ Failed to get user documents: {str(e)}")
            return []
    
    def check_document_access(self, document_hash: str, user_address: str) -> bool:
        """
        Check if a user has access to a document
        
        Args:
            document_hash: Hash of the document
            user_address: Address to check
            
        Returns:
            True if user has access
        """
        try:
            # Convert document hash to bytes32
            doc_hash_bytes = Web3.to_bytes(hexstr=document_hash) if document_hash.startswith('0x') else Web3.keccak(text=document_hash)
            
            # Call contract function
            has_access = self.contract.functions.hasDocumentAccess(doc_hash_bytes, user_address).call()
            
            logger.info(f"🔍 Access check: {user_address} -> {document_hash} = {has_access}")
            return has_access
            
        except Exception as e:
            logger.error(f"❌ Failed to check document access: {str(e)}")
            return False
    
    def get_contract_stats(self) -> Dict[str, Any]:
        """
        Get contract statistics
        
        Returns:
            Contract statistics
        """
        try:
            # Call contract function
            result = self.contract.functions.getContractStats().call()
            
            stats = {
                "total_documents": result[0],
                "contract_balance": self.w3.from_wei(result[1], 'ether'),
                "verification_fee": self.w3.from_wei(result[2], 'ether')
            }
            
            logger.info(f"📊 Contract stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get contract stats: {str(e)}")
            return {"error": str(e)}
    
    def invalidate_document(self, document_hash: str) -> Dict[str, Any]:
        """
        Invalidate a document (mark as invalid)
        
        Args:
            document_hash: Hash of the document to invalidate
            
        Returns:
            Transaction result
        """
        try:
            # Convert document hash to bytes32
            doc_hash_bytes = Web3.to_bytes(hexstr=document_hash) if document_hash.startswith('0x') else Web3.keccak(text=document_hash)
            
            # Build transaction
            transaction = self.contract.functions.invalidateDocument(doc_hash_bytes).build_transaction({
                'from': self.account.address,
                'gas': 100000,
                'gasPrice': self.web3_connector.get_gas_price(),
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Send transaction
            tx_hash = self.web3_connector.send_transaction(transaction)
            receipt = self.web3_connector.wait_for_transaction(tx_hash)
            
            result = {
                "success": receipt.status == 1,
                "tx_hash": tx_hash,
                "document_hash": document_hash
            }
            
            logger.info(f"❌ Document invalidated: {document_hash}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to invalidate document: {str(e)}")
            raise
    
    def create_document_hash(self, document_data: str) -> str:
        """
        Create a hash for document data
        
        Args:
            document_data: Document data to hash
            
        Returns:
            Document hash
        """
        try:
            # Create SHA256 hash
            hash_object = hashlib.sha256(document_data.encode())
            document_hash = hash_object.hexdigest()
            
            logger.info(f"🔐 Created document hash: {document_hash}")
            return document_hash
            
        except Exception as e:
            logger.error(f"❌ Failed to create document hash: {str(e)}")
            raise
    
    def get_contract_address(self) -> str:
        """Get contract address"""
        return self.contract_address
    
    def get_contract(self) -> Contract:
        """Get contract instance"""
        return self.contract


# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize Web3 connector
        web3_connector = Web3Connector("localhost")
        
        # Initialize contract interface
        contract_interface = ContractInterface(web3_connector)
        
        # Test contract functions
        print("Contract Address:", contract_interface.get_contract_address())
        print("Contract Stats:", contract_interface.get_contract_stats())
        
        # Test document hash creation
        test_data = "Test document content"
        doc_hash = contract_interface.create_document_hash(test_data)
        print("Document Hash:", doc_hash)
        
    except Exception as e:
        print(f"Test failed: {e}") 