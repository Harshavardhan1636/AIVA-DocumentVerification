"""
Web3 Connector for AIVA Document Verification System
Handles Ethereum network connections, wallet management, and transaction utilities
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from web3 import Web3
from eth_account import Account
from eth_account.signers.local import LocalAccount
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Web3Connector:
    """
    Web3 connection manager for Ethereum blockchain interactions
    """
    
    def __init__(self, network: str = "localhost", private_key: Optional[str] = None):
        """
        Initialize Web3 connector
        
        Args:
            network: Network to connect to ('localhost', 'hardhat', 'sepolia', 'mainnet')
            private_key: Private key for transaction signing (optional)
        """
        # Map hardhat to localhost for compatibility
        if network == "hardhat":
            network = "localhost"
            
        self.network = network
        self.w3 = None
        self.account = None
        self.contract_address = None
        self.contract_abi = None
        
        # Network configurations
        self.network_configs = {
            "localhost": {
                "url": "http://127.0.0.1:8545",
                "chain_id": 1337,
                "explorer": None
            },
            "sepolia": {
                "url": os.getenv("SEPOLIA_URL", "https://sepolia.infura.io/v3/YOUR-PROJECT-ID"),
                "chain_id": 11155111,
                "explorer": "https://sepolia.etherscan.io"
            },
            "mainnet": {
                "url": os.getenv("MAINNET_URL", "https://mainnet.infura.io/v3/YOUR-PROJECT-ID"),
                "chain_id": 1,
                "explorer": "https://etherscan.io"
            }
        }
        
        self._connect_to_network()
        self._setup_account(private_key)
        self._load_contract_abi()
    
    def _connect_to_network(self) -> None:
        """Establish connection to Ethereum network"""
        try:
            config = self.network_configs.get(self.network)
            if not config:
                raise ValueError(f"Unsupported network: {self.network}")
            
            self.w3 = Web3(Web3.HTTPProvider(config["url"]))
            
            # Add POA middleware for testnets (using correct import)
            if self.network in ["sepolia", "localhost"]:
                try:
                    from web3.middleware import geth_poa_middleware
                    self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                except ImportError:
                    # Try alternative import for newer web3 versions
                    try:
                        from web3.middleware import construct_sign_and_send_raw_middleware
                        # For newer versions, POA middleware might not be needed
                        logger.info("POA middleware not available, continuing without it")
                    except ImportError:
                        logger.warning("Could not import web3 middleware, continuing without it")
            
            # Test connection
            if not self.w3.is_connected():
                raise ConnectionError(f"Failed to connect to {self.network}")
            
            logger.info(f"✅ Connected to {self.network} network")
            logger.info(f"🔗 Network URL: {config['url']}")
            logger.info(f"🆔 Chain ID: {config['chain_id']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to {self.network}: {str(e)}")
            raise
    
    def _setup_account(self, private_key: Optional[str] = None) -> None:
        """Setup account for transaction signing"""
        try:
            if private_key:
                # Use provided private key
                self.account = Account.from_key(private_key)
                logger.info(f"👤 Using provided account: {self.account.address}")
            else:
                # Use environment variable or create new account
                env_private_key = os.getenv("PRIVATE_KEY")
                if env_private_key:
                    self.account = Account.from_key(env_private_key)
                    logger.info(f"👤 Using account from environment: {self.account.address}")
                else:
                    # Create new account for testing
                    self.account = Account.create()
                    logger.warning(f"⚠️ Created new test account: {self.account.address}")
                    logger.warning("⚠️ This account has no funds. Use only for testing!")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup account: {str(e)}")
            raise
    
    def _load_contract_abi(self) -> None:
        """Load contract ABI from artifacts"""
        try:
            # Try to load from Hardhat artifacts
            artifacts_path = "artifacts/contracts/DocumentVerification.sol/DocumentVerification.json"
            if os.path.exists(artifacts_path):
                with open(artifacts_path, 'r') as f:
                    contract_data = json.load(f)
                    self.contract_abi = contract_data['abi']
                    logger.info("📋 Contract ABI loaded from artifacts")
            else:
                logger.warning("⚠️ Contract ABI not found. Please compile contracts first.")
                
        except Exception as e:
            logger.error(f"❌ Failed to load contract ABI: {str(e)}")
    
    def get_balance(self, address: Optional[str] = None) -> float:
        """
        Get ETH balance for an address
        
        Args:
            address: Address to check (defaults to connected account)
            
        Returns:
            Balance in ETH
        """
        try:
            target_address = address or self.account.address
            balance_wei = self.w3.eth.get_balance(target_address)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            
            logger.info(f"💰 Balance for {target_address}: {balance_eth} ETH")
            return float(balance_eth)
            
        except Exception as e:
            logger.error(f"❌ Failed to get balance: {str(e)}")
            return 0.0
    
    def get_gas_price(self) -> int:
        """
        Get current gas price
        
        Returns:
            Gas price in wei
        """
        try:
            gas_price = self.w3.eth.gas_price
            gas_price_gwei = self.w3.from_wei(gas_price, 'gwei')
            logger.info(f"⛽ Current gas price: {gas_price_gwei} Gwei")
            return gas_price
            
        except Exception as e:
            logger.error(f"❌ Failed to get gas price: {str(e)}")
            return 20000000000  # 20 Gwei default
    
    def estimate_gas(self, transaction: Dict[str, Any]) -> int:
        """
        Estimate gas for a transaction
        
        Args:
            transaction: Transaction parameters
            
        Returns:
            Estimated gas in wei
        """
        try:
            estimated_gas = self.w3.eth.estimate_gas(transaction)
            logger.info(f"⛽ Estimated gas: {estimated_gas}")
            return estimated_gas
            
        except Exception as e:
            logger.error(f"❌ Failed to estimate gas: {str(e)}")
            return 21000  # Default gas limit
    
    def send_transaction(self, transaction: Dict[str, Any]) -> str:
        """
        Send a signed transaction
        
        Args:
            transaction: Transaction parameters
            
        Returns:
            Transaction hash
        """
        try:
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            tx_hash_hex = tx_hash.hex()
            
            logger.info(f"📤 Transaction sent: {tx_hash_hex}")
            return tx_hash_hex
            
        except Exception as e:
            logger.error(f"❌ Failed to send transaction: {str(e)}")
            raise
    
    def wait_for_transaction(self, tx_hash: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Wait for transaction to be mined
        
        Args:
            tx_hash: Transaction hash
            timeout: Timeout in seconds
            
        Returns:
            Transaction receipt
        """
        try:
            logger.info(f"⏳ Waiting for transaction {tx_hash}...")
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            
            if receipt.status == 1:
                logger.info(f"✅ Transaction confirmed in block {receipt.blockNumber}")
            else:
                logger.error(f"❌ Transaction failed")
                
            return receipt
            
        except Exception as e:
            logger.error(f"❌ Failed to wait for transaction: {str(e)}")
            raise
    
    def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """
        Get transaction status and details
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction details
        """
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            
            status = {
                "hash": tx_hash,
                "from": tx['from'],
                "to": tx['to'],
                "value": self.w3.from_wei(tx['value'], 'ether'),
                "gas_used": receipt['gasUsed'],
                "gas_price": self.w3.from_wei(tx['gasPrice'], 'gwei'),
                "block_number": receipt['blockNumber'],
                "status": "success" if receipt['status'] == 1 else "failed",
                "confirmations": self.w3.eth.block_number - receipt['blockNumber']
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get transaction status: {str(e)}")
            return {"error": str(e)}
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get current network information
        
        Returns:
            Network details
        """
        try:
            config = self.network_configs[self.network]
            
            info = {
                "network": self.network,
                "chain_id": config["chain_id"],
                "block_number": self.w3.eth.block_number,
                "gas_price": self.w3.from_wei(self.w3.eth.gas_price, 'gwei'),
                "connected": self.w3.is_connected(),
                "account": self.account.address if self.account else None,
                "balance": self.get_balance() if self.account else 0
            }
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Failed to get network info: {str(e)}")
            return {"error": str(e)}
    
    def is_connected(self) -> bool:
        """Check if connected to network"""
        return self.w3.is_connected() if self.w3 else False
    
    def get_account(self) -> Optional[LocalAccount]:
        """Get current account"""
        return self.account
    
    def get_web3(self) -> Web3:
        """Get Web3 instance"""
        return self.w3


# Example usage and testing
if __name__ == "__main__":
    # Test connection
    try:
        connector = Web3Connector("localhost")
        print("Network Info:", connector.get_network_info())
        print("Balance:", connector.get_balance())
        print("Gas Price:", connector.get_gas_price())
        
    except Exception as e:
        print(f"Test failed: {e}") 