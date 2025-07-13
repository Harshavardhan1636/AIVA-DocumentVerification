# Agentic Ethereum Hackathon India

# 🛠 Project Title - [BLockSightAI]

Welcome to our submission for the *Agentic Ethereum Hackathon* by Reskilll & Geodework! This repository includes our project code, documentation, and related assets.

---

## 📌 Problem Statement

In a country where everything from job offers to college admissions and government aid depends on paper-based documents, **fraudulent certificates and fake IDs are on the rise**. Manual verification is slow, inconsistent, and unscalable — and even digital platforms like DigiLocker don’t check document authenticity.

**There is no intelligent, decentralized system that can visually detect tampering, verify the content, and log proof on-chain.**

---

## 💡 Our Solution: AIVA

AIVA (Agentic Intelligent Vision Assistant) is a document verification assistant that:
- Uses **CNN-based AI** to detect document forgery (e.g., tampered names, missing seals)
- Applies **OCR** to extract key details (Name, DOB, Aadhaar number)
- Classifies documents as **Real / Fake with a confidence score**
- Stores the result on **Ethereum** via a smart contract for transparency and auditability

With AIVA, document trust becomes **fast, intelligent, and verifiable — forever**.

---

## 🧱 Tech Stack

| Layer        | Technology                                      |
|--------------|-------------------------------------------------|
| 🖥 Frontend   | Streamlit (Python-based demo UI)                |
| ⚙ Backend    | Flask (Python REST APIs for integration)        |
| 🧠 AI Model   | Custom CNN using ResNet50 for forgery detection |
| 🔗 Blockchain | Ethereum (Sepolia Testnet), Solidity, Hardhat  |
| 🔍 Storage    | (Optional) IPFS for document hash anchoring     |
| 🚀 Hosting    | Localhost (for demo); Deployable via Render/Heroku |


---

## 📽 Demo

- 🎥 *Video Link*: NA  
- 🖥 *Live App (if available)*: NA

---

## 📂 Repository Structure

```bash
.
├── frontend/           # Frontend code
├── backend/            # Backend code
├── contracts/          # Smart contracts
├── assets/             # PPT, video links, images
├── docs/               # Architecture diagram, notes
├── README.md           # A detailed description of your project
├── .env.example
├── package.json / requirements.txt
├── yourppt.ppt
``` 
