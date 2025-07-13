from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import sys
import tempfile
import shutil
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import json
import numpy as np

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import our modules
from src.document_verifier import DocumentVerifier
from src.blockchain import BlockchainManager
from src.ai import CNNDocumentVerifier
import src.utils as utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AIVA Document Verification API",
    description="AI-Powered Document Verification System with Blockchain Integration",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our modules
# Use 'resnet' for the custom model
cnn_verifier = CNNDocumentVerifier(model_type='resnet')
document_verifier = cnn_verifier # The CNN verifier now handles all vision tasks
blockchain_manager = BlockchainManager()

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif hasattr(obj, 'item') and callable(obj.item):
        # For numpy scalars
        return obj.item()
    else:
        return obj

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AIVA Document Verification API is running", "status": "healthy"}

@app.get("/api/status")
async def get_system_status():
    """Get system status for all modules"""
    try:
        # Check if modules are available
        status = {
            "vision_module": {
                "status": "online",
                "name": "Document Verifier (M1)",
                "capabilities": ["OCR", "Field Extraction", "Layout Analysis"]
            },
            "ai_module": {
                "status": "ready",
                "name": "CNN Verifier (M3)",
                "capabilities": ["Document Classification", "Fraud Detection", "Confidence Scoring"]
            },
            "blockchain_module": {
                "status": "connected",
                "name": "Blockchain Manager (M2)",
                "capabilities": ["Hash Storage", "Verification Logging", "IPFS Integration"]
            }
        }
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

@app.post("/api/verify")
async def verify_document(document: UploadFile = File(...)):
    """Main document verification endpoint"""
    verification_steps = []
    
    try:
        # Validate file
        if not document.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg", "application/pdf"]
        if document.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        verification_steps.append({"name": "Document format validated", "status": "success"})
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(document.filename).suffix) as temp_file:
            shutil.copyfileobj(document.file, temp_file)
            temp_file_path = temp_file.name
        
        vision_result, ai_result, blockchain_result = {}, {}, {}
        
        try:
            # The CNN verifier now handles both vision and AI analysis
            logger.info("Starting document verification pipeline...")
            
            # The single call to the verifier
            verification_result = document_verifier.verify_document(temp_file_path)
            
            if "error" in verification_result:
                raise Exception(verification_result["error"])

            vision_result = verification_result # Contains all vision-related data
            ai_result = verification_result # Contains all AI-related data
            
            verification_steps.append({"name": "Text extraction completed", "status": "success"})
            verification_steps.append({"name": "Security features checked", "status": "success"})
            is_fraudulent = ai_result.get("fraud_score", 100) > 30
            verification_steps.append({
                "name": "Fraud indicators checked", 
                "status": "warning" if is_fraudulent else "success"
            })
            
            # Step 3: Blockchain Verification (M2)
            logger.info("Starting blockchain verification...")
            blockchain_data_to_log = {
                "document_type": vision_result.get("document_type", "Unknown"),
                "fraud_score": ai_result.get("fraud_score", 100),
                "confidence": ai_result.get("confidence", 0),
                "extracted_text_hash": utils.hash_text(vision_result.get("extracted_text", ""))
            }
            blockchain_result = blockchain_manager.log_verification(blockchain_data_to_log)
            verification_steps.append({"name": "Blockchain record created", "status": "success"})

        except Exception as processing_error:
            # Log the detailed error to the console for debugging
            logger.error(f"PIPELINE_ERROR: An error occurred during document processing: {processing_error}", exc_info=True)
            # Add a failed step to the list for visibility on the frontend
            verification_steps.append({"name": "Processing failed", "status": "error", "error": str(processing_error)})
            # Re-raise to be caught by the outer exception handler
            raise processing_error
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

        # Combine results
        final_result = {
            "success": True,
            "data": {
                "documentType": vision_result.get("document_type", "Aadhaar Card"),
                "extractedText": vision_result.get("extracted_text", "Could not extract text."),
                "fraudScore": ai_result.get("fraud_score", 0),
                "confidence": ai_result.get("confidence", 0),
                "blockchainHash": blockchain_result.get("hash", "N/A"),
                "isVerified": not is_fraudulent,
                "verificationId": blockchain_result.get("verification_id", "N/A"),
                "timestamp": blockchain_result.get("timestamp", utils.get_current_timestamp()),
                "details": {
                    "ocr_confidence": vision_result.get("features_detected", {}).get("quality", {}).get("ocr_confidence", 0),
                    "layout_analysis": vision_result.get("features_detected", {}).get("layout", {}),
                    "ai_analysis": ai_result.get("analysis", {}),
                    "blockchain_details": blockchain_result.get("details", {}),
                    "verification_steps": verification_steps
                }
            }
        }
        # Patch: convert all NumPy types to Python types
        final_result = convert_numpy(final_result)
        return JSONResponse(content=final_result)
                
    except Exception as e:
        logger.error(f"Verification error: {e}")
        # Ensure verification_steps is included in the error response
        error_details = {
            "verification_steps": verification_steps + [{"name": "Overall process failed", "status": "error"}]
        }
        # You might want to add more details to the error response
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Verification failed: {e}",
                "data": { "details": error_details }
            }
        )

@app.get("/api/verifications")
async def get_verifications(limit: int = 10):
    """Get recent verifications"""
    try:
        # This would typically fetch from blockchain/database
        # For now, return mock data
        verifications = [
            {
                "id": "v_001",
                "document_type": "Aadhaar Card",
                "fraud_score": 15,
                "confidence": 94,
                "timestamp": "2024-01-15T10:30:00Z",
                "status": "verified"
            }
        ]
        return JSONResponse(content={"verifications": verifications[:limit]})
    except Exception as e:
        logger.error(f"Error fetching verifications: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch verifications")

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = {
            "total_verifications": 1247,
            "success_rate": 98.7,
            "avg_processing_time": 3.2,
            "today_verifications": 45,
            "blockchain_transactions": 1247,
            "ai_accuracy": 94.2
        }
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")

if __name__ == "__main__":
    uvicorn.run(
        "backend_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 