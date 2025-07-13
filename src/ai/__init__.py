#!/usr/bin/env python3
"""
AIVA Document Verification System - AI Module
Provides AI-powered document verification capabilities
"""

from .cnn_document_verifier import (
    CNNDocumentVerifier,
    VerificationResult,
    get_cnn_verifier,
    verify_document_cnn
)
from .lightweight_document_verifier import (
    LightweightDocumentVerifier,
    get_lightweight_verifier,
    verify_document_lightweight
)

__all__ = [
    'CNNDocumentVerifier',
    'VerificationResult',
    'get_cnn_verifier',
    'verify_document_cnn',
    'LightweightDocumentVerifier',
    'get_lightweight_verifier',
    'verify_document_lightweight'
] 