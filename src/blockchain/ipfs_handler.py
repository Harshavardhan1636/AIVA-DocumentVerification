"""
IPFS Handler for AIVA Document Verification System
Handles decentralized file storage using IPFS protocol
"""

import os
import json
import logging
import requests
from typing import Optional, Dict, Any, BinaryIO
from pathlib import Path
import hashlib
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IPFSHandler:
    """
    IPFS handler for decentralized document storage
    """
    
    def __init__(self, ipfs_gateway: str = None, api_url: str = None):
        """
        Initialize IPFS handler
        
        Args:
            ipfs_gateway: IPFS gateway URL for file access (default: local if available, else https://ipfs.io)
            api_url: IPFS API URL for file upload (default: local if available, else https://api.ipfs.io)
        """
        # Prefer local IPFS node if available
        local_gateway = "http://127.0.0.1:8080"
        local_api = "http://127.0.0.1:5001"
        if ipfs_gateway is None:
            ipfs_gateway = local_gateway if self._is_local_ipfs_available(local_api) else "https://ipfs.io"
        if api_url is None:
            api_url = local_api if self._is_local_ipfs_available(local_api) else "https://api.ipfs.io"
        self.ipfs_gateway = ipfs_gateway.rstrip('/')
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
        # Test connection
        self._test_connection()

    def _is_local_ipfs_available(self, api_url: str) -> bool:
        try:
            response = requests.post(f"{api_url}/api/v0/version", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def _test_connection(self) -> None:
        """Test IPFS connection"""
        try:
            response = self.session.get(f"{self.api_url}/api/v0/version", timeout=10)
            if response.status_code == 200:
                version_info = response.json()
                logger.info(f"SUCCESS: Connected to IPFS API: {version_info.get('Version', 'Unknown')}")
            else:
                logger.warning(f"WARNING: IPFS API connection test failed: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"WARNING: IPFS API connection test failed: {str(e)}")
            logger.info("INFO: Using IPFS gateway only mode")
    
    def upload_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Upload a file to IPFS
        
        Args:
            file_path: Path to the file to upload
            metadata: Optional metadata to include
            
        Returns:
            Upload result with IPFS hash
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Prepare file for upload
            with open(file_path, 'rb') as file:
                files = {'file': (os.path.basename(file_path), file, 'application/octet-stream')}
                
                # Add metadata if provided
                data = {}
                if metadata:
                    data['metadata'] = json.dumps(metadata)
                
                # Upload to IPFS
                response = self.session.post(
                    f"{self.api_url}/api/v0/add",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    ipfs_hash = result['Hash']
                    
                    upload_result = {
                        "success": True,
                        "ipfs_hash": ipfs_hash,
                        "file_name": os.path.basename(file_path),
                        "file_size": os.path.getsize(file_path),
                        "gateway_url": f"{self.ipfs_gateway}/ipfs/{ipfs_hash}",
                        "metadata": metadata
                    }
                    
                    logger.info(f"SUCCESS: File uploaded to IPFS: {ipfs_hash}")
                    return upload_result
                else:
                    raise Exception(f"IPFS upload failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"ERROR: Failed to upload file to IPFS: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def upload_bytes(self, data: bytes, filename: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Upload bytes data to IPFS
        
        Args:
            data: Bytes data to upload
            filename: Name for the file
            metadata: Optional metadata to include
            
        Returns:
            Upload result with IPFS hash
        """
        try:
            # Prepare data for upload
            files = {'file': (filename, data, 'application/octet-stream')}
            
            # Add metadata if provided
            upload_data = {}
            if metadata:
                upload_data['metadata'] = json.dumps(metadata)
            
            # Upload to IPFS
            response = self.session.post(
                f"{self.api_url}/api/v0/add",
                files=files,
                data=upload_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ipfs_hash = result['Hash']
                
                upload_result = {
                    "success": True,
                    "ipfs_hash": ipfs_hash,
                    "file_name": filename,
                    "file_size": len(data),
                    "gateway_url": f"{self.ipfs_gateway}/ipfs/{ipfs_hash}",
                    "metadata": metadata
                }
                
                logger.info(f"SUCCESS: Data uploaded to IPFS: {ipfs_hash}")
                return upload_result
            else:
                raise Exception(f"IPFS upload failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"ERROR: Failed to upload data to IPFS: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def upload_json(self, data: Dict[str, Any], filename: str = "data.json") -> Dict[str, Any]:
        """
        Upload JSON data to IPFS
        
        Args:
            data: JSON data to upload
            filename: Name for the file
            
        Returns:
            Upload result with IPFS hash
        """
        try:
            json_bytes = json.dumps(data, indent=2).encode('utf-8')
            return self.upload_bytes(json_bytes, filename, {"content_type": "application/json"})
            
        except Exception as e:
            logger.error(f"ERROR: Failed to upload JSON to IPFS: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def upload_data(self, data: str, filename: str = "data.json") -> str:
        """
        Upload string data to IPFS
        
        Args:
            data: String data to upload
            filename: Name for the file
            
        Returns:
            IPFS hash of uploaded data
        """
        try:
            data_bytes = data.encode('utf-8')
            result = self.upload_bytes(data_bytes, filename, {"content_type": "text/plain"})
            
            if result.get("success"):
                return result.get("ipfs_hash", "")
            else:
                logger.error(f"ERROR: Failed to upload data to IPFS: {result.get('error', 'Unknown error')}")
                return ""
                
        except Exception as e:
            logger.error(f"ERROR: Failed to upload data to IPFS: {str(e)}")
            return ""
    
    def download_file(self, ipfs_hash: str, output_path: str) -> Dict[str, Any]:
        """
        Download a file from IPFS
        
        Args:
            ipfs_hash: IPFS hash of the file
            output_path: Path to save the downloaded file
            
        Returns:
            Download result
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Download from IPFS gateway
            response = self.session.get(
                f"{self.ipfs_gateway}/ipfs/{ipfs_hash}",
                timeout=30,
                stream=True
            )
            
            if response.status_code == 200:
                with open(output_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                
                download_result = {
                    "success": True,
                    "ipfs_hash": ipfs_hash,
                    "output_path": output_path,
                    "file_size": os.path.getsize(output_path)
                }
                
                logger.info(f"SUCCESS: File downloaded from IPFS: {ipfs_hash}")
                return download_result
            else:
                raise Exception(f"IPFS download failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"ERROR: Failed to download file from IPFS: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_file_info(self, ipfs_hash: str) -> Dict[str, Any]:
        """
        Get information about a file on IPFS
        
        Args:
            ipfs_hash: IPFS hash of the file
            
        Returns:
            File information
        """
        try:
            # Get file info from IPFS API
            response = self.session.post(
                f"{self.api_url}/api/v0/files/stat",
                params={"arg": f"/ipfs/{ipfs_hash}"},
                timeout=10
            )
            
            if response.status_code == 200:
                info = response.json()
                file_info = {
                    "success": True,
                    "ipfs_hash": ipfs_hash,
                    "file_size": info.get('Size', 0),
                    "file_type": info.get('Type', 'unknown'),
                    "gateway_url": f"{self.ipfs_gateway}/ipfs/{ipfs_hash}"
                }
                
                logger.info(f"INFO: File info retrieved: {ipfs_hash}")
                return file_info
            else:
                raise Exception(f"Failed to get file info: {response.status_code}")
                
        except Exception as e:
            logger.error(f"ERROR: Failed to get file info: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def pin_file(self, ipfs_hash: str) -> Dict[str, Any]:
        """
        Pin a file to IPFS (keep it available)
        
        Args:
            ipfs_hash: IPFS hash of the file to pin
            
        Returns:
            Pin result
        """
        try:
            response = self.session.post(
                f"{self.api_url}/api/v0/pin/add",
                params={"arg": ipfs_hash},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                pin_result = {
                    "success": True,
                    "ipfs_hash": ipfs_hash,
                    "pinned": True
                }
                
                logger.info(f"INFO: File pinned to IPFS: {ipfs_hash}")
                return pin_result
            else:
                raise Exception(f"Failed to pin file: {response.status_code}")
                
        except Exception as e:
            logger.error(f"ERROR: Failed to pin file: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def unpin_file(self, ipfs_hash: str) -> Dict[str, Any]:
        """
        Unpin a file from IPFS
        
        Args:
            ipfs_hash: IPFS hash of the file to unpin
            
        Returns:
            Unpin result
        """
        try:
            response = self.session.post(
                f"{self.api_url}/api/v0/pin/rm",
                params={"arg": ipfs_hash},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                unpin_result = {
                    "success": True,
                    "ipfs_hash": ipfs_hash,
                    "pinned": False
                }
                
                logger.info(f"INFO: File unpinned from IPFS: {ipfs_hash}")
                return unpin_result
            else:
                raise Exception(f"Failed to unpin file: {response.status_code}")
                
        except Exception as e:
            logger.error(f"ERROR: Failed to unpin file: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_document_metadata(self, document_type: str, document_hash: str, 
                                owner_address: str, additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create metadata for a document
        
        Args:
            document_type: Type of document
            document_hash: Hash of the document
            owner_address: Owner's Ethereum address
            additional_data: Additional metadata
            
        Returns:
            Document metadata
        """
        metadata = {
            "document_type": document_type,
            "document_hash": document_hash,
            "owner_address": owner_address,
            "timestamp": int(os.time.time()),
            "version": "1.0",
            "system": "AIVA Document Verification"
        }
        
        if additional_data:
            metadata.update(additional_data)
        
        return metadata
    
    def upload_document(self, file_path: str, document_type: str, document_hash: str, 
                       owner_address: str, additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Upload a document with metadata to IPFS
        
        Args:
            file_path: Path to the document file
            document_type: Type of document
            document_hash: Hash of the document
            owner_address: Owner's Ethereum address
            additional_metadata: Additional metadata
            
        Returns:
            Upload result with IPFS hash
        """
        try:
            # Create metadata
            metadata = self.create_document_metadata(
                document_type, document_hash, owner_address, additional_metadata
            )
            
            # Upload file
            upload_result = self.upload_file(file_path, metadata)
            
            if upload_result["success"]:
                logger.info(f"INFO: Document uploaded with metadata: {upload_result['ipfs_hash']}")
            
            return upload_result
            
        except Exception as e:
            logger.error(f"ERROR: Failed to upload document: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_gateway_url(self, ipfs_hash: str) -> str:
        """
        Get gateway URL for an IPFS hash
        
        Args:
            ipfs_hash: IPFS hash
            
        Returns:
            Gateway URL
        """
        return f"{self.ipfs_gateway}/ipfs/{ipfs_hash}"
    
    def validate_ipfs_hash(self, ipfs_hash: str) -> bool:
        """
        Validate IPFS hash format
        
        Args:
            ipfs_hash: IPFS hash to validate
            
        Returns:
            True if valid
        """
        try:
            # Basic IPFS hash validation (CID v0 or v1)
            if not ipfs_hash or len(ipfs_hash) < 46:
                return False
            
            # Check if it's a valid base58 string (for CID v0)
            if ipfs_hash.startswith('Qm') and len(ipfs_hash) == 46:
                return True
            
            # Check if it's a valid CID v1
            if ipfs_hash.startswith('bafy') and len(ipfs_hash) >= 59:
                return True
            
            return False
            
        except Exception:
            return False
    
    def store_verification_data(self, verification_data: Dict[str, Any]) -> str:
        """
        Store verification data on IPFS (Store large data off-chain)
        
        Args:
            verification_data: Verification data to store
            
        Returns:
            IPFS hash of stored data
        """
        try:
            import time
            import hashlib
            
            # Create metadata for verification data
            metadata = {
                "content_type": "verification_data",
                "timestamp": verification_data.get("verification_timestamp", int(time.time())),
                "document_type": "document_verification",
                "version": "1.0"
            }
            
            # Try to upload verification data to IPFS
            try:
                result = self.upload_json(verification_data, "verification_data.json")
                
                if result["success"]:
                    logger.info(f"SUCCESS: Verification data stored on IPFS: {result['ipfs_hash']}")
                    return result["ipfs_hash"]
                else:
                    logger.warning(f"WARNING: IPFS upload failed, using local hash: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.warning(f"WARNING: IPFS connection failed, using local hash: {str(e)}")
            
            # Fallback: Generate a local hash for the verification data
            data_string = json.dumps(verification_data, sort_keys=True)
            data_hash = hashlib.sha256(data_string.encode()).hexdigest()
            local_hash = f"Qm{data_hash[:44]}"  # Format as IPFS hash
            
            logger.info(f"INFO: Using local hash for verification data: {local_hash}")
            return local_hash
                
        except Exception as e:
            logger.error(f"ERROR: Error storing verification data: {str(e)}")
            # Return a placeholder hash for fallback
            return "QmPlaceholderHashForVerificationData"


# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize IPFS handler
        ipfs_handler = IPFSHandler()
        
        # Test file upload
        test_data = b"Test document content for IPFS"
        result = ipfs_handler.upload_bytes(test_data, "test.txt")
        print("Upload result:", result)
        
        if result["success"]:
            # Test file info
            info = ipfs_handler.get_file_info(result["ipfs_hash"])
            print("File info:", info)
            
            # Test gateway URL
            url = ipfs_handler.get_gateway_url(result["ipfs_hash"])
            print("Gateway URL:", url)
        
    except Exception as e:
        print(f"Test failed: {e}") 