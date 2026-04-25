# RAG_AI_Based_Law_Helper
AI-powered legal assistant using RAG + local LLM (LLaMA 3 via Ollama). Combines FAISS + BM25 hybrid retrieval to generate accurate, citation-grounded legal responses from Indian laws—fully offline, privacy-first, and zero cloud dependency. Includes document generation, PDF summarization &amp; case analysis.


⚖️ AI-Powered Indian Legal Guidance Chatbot (Local RAG System)
<div align="center">










Privacy-first legal intelligence powered by Retrieval-Augmented Generation + Local AI

Features
 • Architecture
 • Quick Start
 • API Reference
 • Configuration

</div>
📌 Overview

This is an open-source, AI-powered legal guidance platform designed for Indian law. It combines hybrid RAG retrieval (dense + sparse + RRF fusion) with a locally hosted LLaMA 3 8B model to deliver grounded, citation-backed legal explanations — fully offline.

Users describe legal situations in plain language and receive structured guidance referencing exact sections from major Indian acts.

⚠️ Disclaimer: This is an informational tool only. It does not constitute professional legal advice.

✨ Features
🤖 RAG Legal Chatbot (FAISS + BM25 + RRF)
📋 Document Generation (Complaint, Notice, Cybercrime)
📄 PDF Legal Summarizer
🏷️ Legal Section Tagger
⚖️ Case Outcome Predictor
🎤 Voice Input (EN / HI / ML)
📍 Lawyer Finder (Google Maps)
✉️ Email Delivery
📱 WhatsApp Integration Ready
📰 Live Legal News
🌗 Light / Dark Theme
🌐 Multilingual UI
🏛️ Architecture

User → FastAPI → Hybrid Retrieval (FAISS + BM25 + RRF) → Local LLaMA 3 → Structured Response

📚 Knowledge Base
Bharatiya Nyaya Sanhita (2023)
Bharatiya Nagarik Suraksha Sanhita (2023)
Bharatiya Sakshya Adhiniyam (2023)
Information Technology Act (2000)
Motor Vehicles Act (1988)
Consumer Protection Act (2019)
Indian Contract Act (1872)
⚡ Quick Start

git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git

cd YOUR_REPO
pip install -r requirements.txt
ollama pull llama3:8b

Run:

ollama serve
uvicorn app:app --reload --port 8000
python -m http.server 3000

Open:
http://localhost:3000

🌐 API Endpoints

POST /chat → RAG response
POST /generate → DOCX generation
POST /summarize → PDF summary
POST /tag_sections → Legal tagging
POST /predict_outcome → Case analysis
POST /send_email → Email delivery
GET /news → Legal news

🛠️ Tech Stack

Backend: FastAPI, Python
LLM: LLaMA 3 (Ollama local)
Retrieval: FAISS + BM25 + RRF
Embeddings: BGE-base
Frontend: HTML, CSS, JS
Docs: python-docx
Voice: Web Speech API
Maps: Google Maps API

📊 Performance

Recall@7: 86.5%
Precision@7: 84.2%
F1 Score: 85.3%
Avg Response: ~2.4s

⚙️ Configuration

Create .env:

RESEND_API_KEY=your_key
WHATSAPP_ACCESS_TOKEN=your_token

📄 License

MIT License

⚠️ Disclaimer

This project provides general legal information and is not a substitute for professional legal advice.

Built with ❤️ using Local AI + RAG
