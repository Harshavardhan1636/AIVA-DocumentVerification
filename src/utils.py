"""
Utility functions for the ZK Proof module.

This module provides helper functions for document processing, validation,
and other common operations used throughout the ZK Proof system.
"""

import hashlib
import json
import re
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import mimetypes


def get_current_timestamp() -> int:
    """
    Get current timestamp as integer.
    
    Returns:
        Current Unix timestamp
    """
    return int(time.time())


def hash_text(text: str) -> str:
    """
    Generate SHA-256 hash of text.
    
    Args:
        text: Text to hash
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def hash_document(document_data: bytes) -> str:
    """
    Generate a SHA-256 hash of document data.
    
    Args:
        document_data: Raw document data
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(document_data).hexdigest()


def validate_document_format(document_data: bytes, document_type: str) -> bool:
    """
    Validate document format based on type.
    
    Args:
        document_data: Raw document data
        document_type: Type of document to validate
        
    Returns:
        True if format is valid, False otherwise
    """
    if not document_data:
        return False
    
    # Check minimum size (reduced for testing)
    if len(document_data) < 10:  # Minimum 10 bytes for testing
        return False
    
    # Check maximum size (10MB)
    if len(document_data) > 10 * 1024 * 1024:
        return False
    
    # Type-specific validation
    if document_type == "aadhaar":
        return _validate_aadhaar_format(document_data)
    elif document_type == "passport":
        return _validate_passport_format(document_data)
    elif document_type == "pan_card":
        return _validate_pan_format(document_data)
    elif document_type == "driving_license":
        return _validate_driving_license_format(document_data)
    else:
        # For general documents, just check basic format
        return _validate_general_format(document_data)


def _validate_aadhaar_format(document_data: bytes) -> bool:
    """Validate Aadhaar card format."""
    # Aadhaar cards are typically 12-digit numbers
    # For demo purposes, we'll check if the data contains numeric patterns
    try:
        text = document_data.decode('utf-8', errors='ignore')
        # Look for 12-digit patterns (Aadhaar numbers)
        aadhaar_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
        return bool(re.search(aadhaar_pattern, text))
    except:
        return False


def _validate_passport_format(document_data: bytes) -> bool:
    """Validate passport format."""
    try:
        text = document_data.decode('utf-8', errors='ignore')
        # Look for passport number patterns (alphanumeric, 6-9 characters)
        passport_pattern = r'\b[A-Z0-9]{6,9}\b'
        return bool(re.search(passport_pattern, text))
    except:
        return False


def _validate_pan_format(document_data: bytes) -> bool:
    """Validate PAN card format."""
    try:
        text = document_data.decode('utf-8', errors='ignore')
        # PAN format: ABCDE1234F (5 letters, 4 digits, 1 letter)
        pan_pattern = r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b'
        return bool(re.search(pan_pattern, text))
    except:
        return False


def _validate_driving_license_format(document_data: bytes) -> bool:
    """Validate driving license format."""
    try:
        text = document_data.decode('utf-8', errors='ignore')
        # Driving license numbers are typically alphanumeric
        dl_pattern = r'\b[A-Z0-9]{10,15}\b'
        return bool(re.search(dl_pattern, text))
    except:
        return False


def _validate_general_format(document_data: bytes) -> bool:
    """Validate general document format."""
    # For general documents, just check if it's not empty and has reasonable size
    return len(document_data) > 0 and len(document_data) < 50 * 1024 * 1024  # 50MB max


def extract_document_metadata(document_data: bytes, document_type: str) -> Dict[str, Any]:
    """
    Extract metadata from document data.
    
    Args:
        document_data: Raw document data
        document_type: Type of document
        
    Returns:
        Dictionary containing extracted metadata
    """
    metadata = {
        "document_type": document_type,
        "size_bytes": len(document_data),
        "hash": hash_document(document_data),
        "extracted_at": datetime.now().isoformat()
    }
    
    try:
        text = document_data.decode('utf-8', errors='ignore')
        
        # Extract common patterns
        metadata.update({
            "has_numbers": bool(re.search(r'\d', text)),
            "has_letters": bool(re.search(r'[A-Za-z]', text)),
            "word_count": len(text.split()),
            "character_count": len(text)
        })
        
        # Type-specific extraction
        if document_type == "aadhaar":
            metadata.update(_extract_aadhaar_metadata(text))
        elif document_type == "passport":
            metadata.update(_extract_passport_metadata(text))
        elif document_type == "pan_card":
            metadata.update(_extract_pan_metadata(text))
        elif document_type == "driving_license":
            metadata.update(_extract_driving_license_metadata(text))
            
    except Exception as e:
        metadata["extraction_error"] = str(e)
    
    return metadata


def _extract_aadhaar_metadata(text: str) -> Dict[str, Any]:
    """Extract Aadhaar-specific metadata."""
    metadata = {}
    
    # Extract Aadhaar number
    aadhaar_match = re.search(r'\b(\d{4}\s?\d{4}\s?\d{4})\b', text)
    if aadhaar_match:
        metadata["aadhaar_number"] = aadhaar_match.group(1).replace(" ", "")
    
    # Extract name (look for patterns like "Name: John Doe")
    name_match = re.search(r'Name[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
    if name_match:
        metadata["name"] = name_match.group(1).strip()
    
    # Extract date of birth
    dob_match = re.search(r'DOB[:\s]+(\d{2}[/-]\d{2}[/-]\d{4})', text, re.IGNORECASE)
    if dob_match:
        metadata["date_of_birth"] = dob_match.group(1)
    
    return metadata


def _extract_passport_metadata(text: str) -> Dict[str, Any]:
    """Extract passport-specific metadata."""
    metadata = {}
    
    # Extract passport number
    passport_match = re.search(r'\b([A-Z0-9]{6,9})\b', text)
    if passport_match:
        metadata["passport_number"] = passport_match.group(1)
    
    # Extract name
    name_match = re.search(r'Name[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
    if name_match:
        metadata["name"] = name_match.group(1).strip()
    
    # Extract nationality
    nationality_match = re.search(r'Nationality[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
    if nationality_match:
        metadata["nationality"] = nationality_match.group(1).strip()
    
    return metadata


def _extract_pan_metadata(text: str) -> Dict[str, Any]:
    """Extract PAN card-specific metadata."""
    metadata = {}
    
    # Extract PAN number
    pan_match = re.search(r'\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b', text)
    if pan_match:
        metadata["pan_number"] = pan_match.group(1)
    
    # Extract name
    name_match = re.search(r'Name[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
    if name_match:
        metadata["name"] = name_match.group(1).strip()
    
    return metadata


def _extract_driving_license_metadata(text: str) -> Dict[str, Any]:
    """Extract driving license-specific metadata."""
    metadata = {}
    
    # Extract license number
    license_match = re.search(r'\b([A-Z0-9]{10,15})\b', text)
    if license_match:
        metadata["license_number"] = license_match.group(1)
    
    # Extract name
    name_match = re.search(r'Name[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
    if name_match:
        metadata["name"] = name_match.group(1).strip()
    
    return metadata


def calculate_age_from_date(birth_date: str) -> Optional[int]:
    """
    Calculate age from birth date string.
    
    Args:
        birth_date: Birth date in DD/MM/YYYY or YYYY-MM-DD format
        
    Returns:
        Age in years or None if invalid
    """
    try:
        # Try different date formats
        for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y']:
            try:
                birth_dt = datetime.strptime(birth_date, fmt)
                today = datetime.now()
                age = today.year - birth_dt.year
                
                # Adjust if birthday hasn't occurred this year
                if today.month < birth_dt.month or (
                    today.month == birth_dt.month and today.day < birth_dt.day
                ):
                    age -= 1
                
                return age
            except ValueError:
                continue
        
        return None
    except Exception:
        return None


def validate_proof_data(proof_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate proof data structure.
    
    Args:
        proof_data: Proof data to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required fields
    required_fields = ["proof_type", "proof", "public_inputs"]
    for field in required_fields:
        if field not in proof_data:
            errors.append(f"Missing required field: {field}")
    
    # Check proof structure
    if "proof" in proof_data:
        proof = proof_data["proof"]
        if not isinstance(proof, dict):
            errors.append("Proof must be a dictionary")
        else:
            required_proof_fields = ["pi_a", "pi_b", "pi_c"]
            for field in required_proof_fields:
                if field not in proof:
                    errors.append(f"Missing proof field: {field}")
    
    # Check public inputs
    if "public_inputs" in proof_data:
        public_inputs = proof_data["public_inputs"]
        if not isinstance(public_inputs, list):
            errors.append("Public inputs must be a list")
    
    # Check proof type
    if "proof_type" in proof_data:
        valid_types = ["document_hash", "age_verification", "signature_verification"]
        if proof_data["proof_type"] not in valid_types:
            errors.append(f"Invalid proof type: {proof_data['proof_type']}")
    
    return len(errors) == 0, errors


def serialize_proof(proof_data: Dict[str, Any]) -> str:
    """
    Serialize proof data to JSON string.
    
    Args:
        proof_data: Proof data to serialize
        
    Returns:
        JSON string representation
    """
    return json.dumps(proof_data, indent=2)


def deserialize_proof(proof_json: str) -> Optional[Dict[str, Any]]:
    """
    Deserialize proof data from JSON string.
    
    Args:
        proof_json: JSON string to deserialize
        
    Returns:
        Deserialized proof data or None if failed
    """
    try:
        return json.loads(proof_json)
    except json.JSONDecodeError:
        return None


def generate_proof_id(proof_data: Dict[str, Any]) -> str:
    """
    Generate a unique ID for a proof.
    
    Args:
        proof_data: Proof data
        
    Returns:
        Unique proof ID
    """
    # Create a hash of the proof data
    proof_str = serialize_proof(proof_data)
    proof_hash = hashlib.sha256(proof_str.encode()).hexdigest()
    
    # Add timestamp for uniqueness
    timestamp = str(int(datetime.now().timestamp()))
    
    return f"proof_{proof_hash[:8]}_{timestamp}"


def get_supported_document_types() -> List[str]:
    """
    Get list of supported document types.
    
    Returns:
        List of supported document types
    """
    return [
        "aadhaar",
        "passport", 
        "pan_card",
        "driving_license",
        "voter_id",
        "general"
    ]


def get_document_type_info(document_type: str) -> Dict[str, Any]:
    """
    Get information about a document type.
    
    Args:
        document_type: Type of document
        
    Returns:
        Dictionary containing document type information
    """
    info = {
        "aadhaar": {
            "name": "Aadhaar Card",
            "description": "12-digit unique identification number",
            "format": "XXXX XXXX XXXX",
            "validation_rules": ["12 digits", "numeric only"],
            "supported_operations": ["hash_verification", "age_verification"]
        },
        "passport": {
            "name": "Passport",
            "description": "International travel document",
            "format": "Alphanumeric, 6-9 characters",
            "validation_rules": ["alphanumeric", "6-9 characters"],
            "supported_operations": ["hash_verification", "age_verification"]
        },
        "pan_card": {
            "name": "PAN Card",
            "description": "Permanent Account Number card",
            "format": "ABCDE1234F",
            "validation_rules": ["5 letters, 4 digits, 1 letter"],
            "supported_operations": ["hash_verification"]
        },
        "driving_license": {
            "name": "Driving License",
            "description": "Official driving permit",
            "format": "Alphanumeric, 10-15 characters",
            "validation_rules": ["alphanumeric", "10-15 characters"],
            "supported_operations": ["hash_verification", "age_verification"]
        },
        "voter_id": {
            "name": "Voter ID",
            "description": "Electoral photo identity card",
            "format": "Varies by state",
            "validation_rules": ["state-specific format"],
            "supported_operations": ["hash_verification", "age_verification"]
        },
        "general": {
            "name": "General Document",
            "description": "Generic document type",
            "format": "Any format",
            "validation_rules": ["non-empty", "size limits"],
            "supported_operations": ["hash_verification"]
        }
    }
    
    return info.get(document_type, {
        "name": "Unknown",
        "description": "Unknown document type",
        "format": "Unknown",
        "validation_rules": [],
        "supported_operations": []
    }) 