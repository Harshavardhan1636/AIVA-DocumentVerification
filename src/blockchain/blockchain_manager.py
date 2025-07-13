#!/usr/bin/env python3
"""
Blockchain Manager for AIVA Document Verification System
Simplified blockchain integration for demo purposes
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BlockchainManager:
    """
    Simplified blockchain manager for demo purposes
    Provides mock blockchain functionality for the hackathon demo
    """
    
    def __init__(self):
        """Initialize the blockchain manager"""
        self.verification_count = 0
        self.verifications = []
        logger.info("🔗 Blockchain Manager initialized (Demo Mode)")
    
    def generate_document_hash(self, data: str) -> str:
        """Generate a hash for document data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify_document_hash(self, data: str, hash_value: str) -> bool:
        """Verify document hash"""
        return self.generate_document_hash(data) == hash_value
    
    def log_verification(self, verification_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log a verification to the blockchain (mock implementation)
        Returns deterministic hash for same data and user.
        """
        try:
            # Deterministic hash: use only relevant fields, sorted
            hash_fields = {k: verification_data[k] for k in sorted(verification_data) if k not in ["timestamp", "verification_id"]}
            hash_input = json.dumps(hash_fields, sort_keys=True)
            blockchain_hash = hashlib.sha256(hash_input.encode()).hexdigest()

            self.verification_count += 1
            verification_id = f"VER_{self.verification_count:04d}"
            record = {
                "verification_id": verification_id,
                "timestamp": int(time.time()),
                **verification_data
            }
            self.verifications.append(record)
            return {
                "hash": blockchain_hash,
                "verification_id": verification_id,
                "timestamp": record["timestamp"],
                "details": record
            }
        except Exception as e:
            logger.error(f"❌ Error logging verification: {e}")
            return {
                "hash": "N/A",
                "verification_id": "ERROR",
                "timestamp": int(time.time()),
                "details": {"error": str(e)}
            }
    
    def get_verification_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent verification history"""
        recent_verifications = self.verifications[-limit:] if self.verifications else []
        
        return {
            "verifications": recent_verifications,
            "total_count": len(self.verifications),
            "timestamp": int(time.time())
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get blockchain system statistics"""
        return {
            "total_verifications": len(self.verifications),
            "last_verification": self.verifications[-1] if self.verifications else None,
            "system_status": "operational",
            "blockchain_network": "demo_local",
            "timestamp": int(time.time())
        } 