# AIVA Document Verification System

AI-powered document verification with blockchain logging for tamper-proof audit trails.

## Features
- FastAPI backend with AI (TensorFlow, OpenCV, OCR)
- Deterministic document hash (same doc = same hash)
- Deterministic blockchain hash (same doc/user = same hash)
- Next.js or HTML frontend for uploads and results
- Mock blockchain for demo/hackathon

## Setup

### Backend
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python backend_server.py
```

### Frontend (Next.js)
```bash
cd "aiva frontend"
npm install
npm run dev
```
Or use `demo_frontend.html` with a static server:
```bash
python -m http.server 3000
```

## Usage
- Open http://localhost:3000
- Upload a document image (JPG/PNG)
- See real-time verification, AI confidence, and blockchain hash

## Demo Flow
1. Upload the same document multiple times
2. Observe the same document hash and blockchain hash in the results
3. Tampered/altered docs will show different hashes

## Team
- [Your Name(s)]
- [Contact info, if required]

## Notes
- Model file (`resnet_aadhar_trained.h5`) is not included due to size. Add your own if needed.
- For hackathon/demo, blockchain is mocked for speed and reliability.

---
Good luck and thank you for reviewing our project! 