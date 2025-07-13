"""
ZK Proof Module for Document Verification System

This module provides zero-knowledge proof capabilities for secure document verification
without revealing sensitive document content.
"""

from .document_verifier import DocumentVerifier
from .circuit_manager import CircuitManager
from .proof_generator import ProofGenerator
from .proof_verifier import ProofVerifier

__version__ = "1.0.0"
__author__ = "AIVA Hackathon Team"

__all__ = [
    "DocumentVerifier",
    "CircuitManager", 
    "ProofGenerator",
    "ProofVerifier"
] 