<div align="center">
 <h1>⚖️ AI-Powered Indian Legal Guidance Chatbot</h1>
 <h3>Excellent for project</h3>
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LLaMA](https://img.shields.io/badge/LLaMA_3_8B-Local_Inference-E6A817?style=for-the-badge)

**Democratizing legal awareness in India through Retrieval-Augmented Generation**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Configuration](#-configuration)

</div>

---

## 📌 Overview

This is an open-source, AI-powered legal guidance platform designed specifically for Indian citizens. It combines **hybrid RAG retrieval** (dense + sparse + RRF fusion) with **locally hosted LLaMA 3 8B** to deliver grounded, citation-verified legal explanations — all without requiring internet access for the core AI pipeline.

Users describe their legal situation in plain language and receive structured guidance citing the exact applicable sections from seven major Indian acts, including the newly introduced Bharatiya Nyaya Sanhita (BNS) 2023 that replaced the Indian Penal Code.

> ⚠️ **Disclaimer:** LegalEase is an informational tool only. It does not constitute professional legal advice. Always consult a qualified advocate for legal matters.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **RAG Legal Chatbot** | Hybrid BGE + BM25 + RRF retrieval over 7 Indian acts. 86.5% Recall@7 |
| 📋 **Document Generation** | Auto-generates Complaint Letters, Legal Notices, Cybercrime Complaints as DOCX |
| 📄 **PDF Legal Summarizer** | Upload any legal PDF — AI detects, validates, and summarizes in plain language |
| 🏷️ **Legal Section Tagger** | Maps your situation to exact applicable sections with relevance ratings |
| ⚖️ **Case Outcome Predictor** | 9-section analysis: strength, probability, timeline, recommended actions |
| 🎤 **Voice Input** | Speak in English, Hindi (hi-IN), or Malayalam (ml-IN) |
| 📍 **Lawyer Finder** | Google Maps integration — find advocates near you by practice area |
| ✉️ **Email Delivery** | Send generated legal documents directly to recipients via Resend API |
| 📱 **WhatsApp Integration** | Full RAG pipeline over WhatsApp via Meta Cloud API + Make.com |
| 📰 **Live Legal News** | Real-time updates from LiveLaw and Bar & Bench RSS feeds |
| 🌗 **Light / Dark Theme** | Toggle between dark (default) and light theme, saved across sessions |
| 🌐 **Multilingual UI** | Interface available in English, Hindi, and Malayalam |

---

## 🏛️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  USER (Browser / WhatsApp)               │
└───────────────────────┬──────────────────────────────────┘
                        │ HTTP / ngrok
┌───────────────────────▼──────────────────────────────────┐
│                  FastAPI Backend (app.py)                │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Sanitize   │  │   Greeting   │  │  Off-topic   │    │
│  │   & Expand   │→ │   Check      │→ │   Filter     │    │
│  └──────┬───────┘  └──────────────┘  └──────────────┘    │
│         │                                                │
│  ┌──────▼────────────────────────────────────────────┐   │
│  │              Hybrid Retrieval Pipeline            │   │
│  │                                                   │   │
│  │  BGE-base-en-v1.5         BM25 Sparse Retrieval   │   │
│  │  Dense Embeddings     +   Keyword Matching        │   │
│  │  (FAISS Index)            (rank-bm25)             │   │
│  │      └──────── RRF Fusion (k=60) ─────────┘       │   │
│  │                       +                           │   │
│  │        Adjacency Boost  → Top-7 Chunks            │   │
│  │                                                   │   │
│  └──────┬────────────────────────────────────────────┘   │
│         │                                                │
│  ┌──────▼────────────────────────────────────────────┐   │
│  │          LLaMA 3 8B via Ollama (Local)            │   │
│  │  Structured 6-section response generation         │   │
│  │                       +                           │   │
│  │              Ghost citation guard                 │   │
│  │                       +                           │   │
│  │            Document offer detection               │   │
│  └──────┬────────────────────────────────────────────┘   │
│         │                                                │
│  ┌──────▼────────────────────────────────────────────┐   │
│  │              Response to User                     │   │
│  │  1. Relevant Law    4. Possible Actions           │   │
│  │  2. Explanation     5. Advice / Next Steps        │   │
│  │  3. Example         6. Disclaimer                 │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### Knowledge Base — Indian Acts

| Act | Year |
|-----|------|
| Bharatiya Nyaya Sanhita (BNS) | 2023 |
| Bharatiya Nagarik Suraksha Sanhita (BNSS) | 2023 |
| Bharatiya Sakshya Adhiniyam (BSA) | 2023 |
| Information Technology Act | 2000 |
| Motor Vehicles Act | 1988 |
| Consumer Protection Act | 2019 |
| Indian Contract Act | 1872 |
etc..

---

## ⚡ Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | Backend runtime |
| [Ollama](https://ollama.com) | Latest | Local LLM serving |
| Git | Any | Clone repo |

<br><br><br>
### Step 1 — Clone the Repository

```bash
git clone https://github.com/Shahadkonnekkatt/RAG_AI_Based_Law_Helper.git
cd RAG_AI_Based_Law_Helper
cd src
```

<br><br><br>
### Step 2 — Install Python Dependencies
##### Option A — With Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> Every time you open a new terminal to run the project, activate the venv first with `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux). You will see `(venv)` appear in your terminal prompt when it is active.

##### Option B — Without Virtual Environment

```bash
pip install -r requirements.txt
```

> This installs packages globally. Works fine but may cause conflicts with other Python projects on your machine.

### ⚠️ CAUTION
The faiss-cpu package sometimes fails to install on certain systems. If someone gets an error on pip install -r requirements.txt, run:

```bash
pip install faiss-cpu --no-cache-dir
```

<br><br><br>
### Step 3 — Download the LLM

```bash
# Install Ollama from https://ollama.com first, then:
ollama pull llama3:8b
```

> This downloads ~4.7 GB. Only needed once.

<br><br><br>
### Step 4 — Build the Knowledge Base
Create a data directory and store the required legal Act PDFs within it. Ensure the files are preprocessed to include only section-wise content (e.g., “Section X: …”), removing non-essential elements like headers, footers, and supplementary material.
```
data/
├── BNS.pdf
├── BNSS.pdf
├── BSA.pdf
├── CONSUMER ACT.pdf
├── MOTOR VEHICLES ACT.pdf
├── etc..
```
<br>
Then:

```bash
# Run the indexing script to build FAISS index, BM25 corpus, and chunks
python ingest.py
```

> This creates `faiss_index.bin`, `bm25_corpus.pkl`, and `chunks.pkl` in the project folder. These are not included in the repo due to file size.

<br><br><br>
### Step 5 — Configure Environment

Edit `.env`:
```env
RESEND_API_KEY=re_your_resend_key_here
RESEND_FROM_EMAIL=onboarding@resend.dev
WHATSAPP_VERIFY_TOKEN=your_verify_token
WHATSAPP_ACCESS_TOKEN=your_meta_access_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
```
| Variable | Required | Description |
|----------|----------|-------------|
| `RESEND_API_KEY` | For email feature | Get from resend.com |
| `RESEND_FROM_EMAIL` | Optional | Default: `onboarding@resend.dev` |
| `WHATSAPP_VERIFY_TOKEN` | For WhatsApp | Set any string, paste same in Meta dashboard |
| `WHATSAPP_ACCESS_TOKEN` | For WhatsApp | From Meta developer console |
| `WHATSAPP_PHONE_NUMBER_ID` | For WhatsApp | From Meta developer console |

> For testing without email/WhatsApp, leave those fields as-is. Only `RESEND_API_KEY` is needed for the email feature.

<br><br><br>
### Step 6 — Start the Application

```bash
# Terminal 1 — Start the LLM server
ollama serve

# Terminal 2 — Start the backend
# The command "uvicorn app:app --reload --port 8000" is integrated at last lines of app.py, therefore we can run app.py to start the backend.
python app.py
```
<br>

> **Note:** The first time you run `app.py`, it will automatically download the 
> BAAI/bge-base-en-v1.5 embedding model (~500MB). This is a one-time download.

<br><br><br>
### Step 7 — Open in Browser

Open index.html file directly from the folder(Recommended), or serve them with Python's built-in server:

```bash
# Terminal 3 — Serve frontend
python -m http.server 3000
```

Then open `http://localhost:3000/index.html` in your browser.

### Accessing from Another Device (Optional cause it is a bit sketchy)

To use LegalEase from your phone while the laptop runs the backend:

```bash
# Install ngrok from https://ngrok.com (free)
ngrok http 8000    # Exposes the API
ngrok http 3000    # Exposes the frontend
```

Copy the ngrok HTTPS URL for port 8000 and update `const API` in all HTML files:
```javascript
// Find this line in chat.html, generate.html, summarize.html, documents.html
const API = "http://127.0.0.1:8000";
// Replace with your ngrok URL:
const API = "https://YOUR_NGROK_URL.ngrok.io";
```

---

Then if you need to run the project back on the computer you need to change the URL back to (const API = "http://127.0.0.1:8000";)

## 🔑 API Keys Required

| Service | Required For | Free Tier | Get It |
|---------|-------------|-----------|--------|
| Resend | Email delivery | 3,000 emails/month | [resend.com](https://resend.com) |
| Google Maps | Lawyer Finder | $200/month credit | [console.cloud.google.com](https://console.cloud.google.com) |
| Meta WhatsApp Cloud | WhatsApp bot | 1,000 conversations/month | [developers.facebook.com](https://developers.facebook.com) |
| Make.com | WhatsApp routing | Premium plan required | [make.com](https://make.com) |

> The core chatbot, document generation, PDF summarizer, section tagger, case predictor, and voice input work completely **without any API keys**.

---

## 📁 Project Structure

```
legalease/
├── app.py                  # FastAPI backend — all endpoints and RAG pipeline
├── doc_generator.py        # python-docx document builders
├── requirements.txt        # Python dependencies
├── .env.template           # Environment variable template
├── .gitignore              # Git exclusions
│
├── index.html              # Homepage with dark/light theme, live news
├── chat.html               # Main RAG chat interface with voice input
├── documents.html          # Tools hub — all features in one page
├── generate.html           # Document generation with calendar/time pickers
├── summarize.html          # PDF legal document summarizer
└── lawyer-finder.html      # Google Maps lawyer finder
```

---

## 🌐 API Reference

All endpoints are served at `http://localhost:8000`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Main RAG chat — returns answer + optional document offer |
| `POST` | `/extract_fields` | Extract form fields from user narrative (for document pre-fill) |
| `GET`  | `/doc_types` | List all document type definitions and field specs |
| `POST` | `/generate` | Generate DOCX document — returns binary file |
| `POST` | `/summarize` | Summarize uploaded PDF legal document text |
| `POST` | `/tag_sections` | Tag applicable law sections for a described situation |
| `POST` | `/predict_outcome` | Predict case outcome with 9-section analysis |
| `POST` | `/send_email` | Send generated document via email (Resend API) |
| `GET`  | `/news` | Fetch latest Indian legal news from RSS feeds |
| `GET`  | `/health` | Health check |

### Example — Chat Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "My employer has not paid my salary for 3 months", "session_id": null}'
```

### Example — Generate Document

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "doc_type": "complaint_letter",
    "fields": {
      "complainant_name": "Rahul Sharma",
      "incident_description": "Employer withheld salary for 3 months",
      "date": "15/03/2026"
    }
  }' \
  --output complaint.docx
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `RESEND_API_KEY` | For email feature | Get from resend.com |
| `RESEND_FROM_EMAIL` | Optional | Default: `onboarding@resend.dev` |
| `WHATSAPP_VERIFY_TOKEN` | For WhatsApp | Set any string, paste same in Meta dashboard |
| `WHATSAPP_ACCESS_TOKEN` | For WhatsApp | From Meta developer console |
| `WHATSAPP_PHONE_NUMBER_ID` | For WhatsApp | From Meta developer console |

### Google Maps API Key

In `lawyer-finder.html`, replace `YOUR_GOOGLE_MAPS_API_KEY`:
```html
<script src="https://maps.googleapis.com/maps/api/js?key=YOUR_GOOGLE_MAPS_API_KEY&libraries=places&callback=initMap">
```

Enable these APIs in Google Cloud Console:
- Maps JavaScript API
- Places API

> **Note:** A billing account must be linked to your Google Cloud project for Places API `nearbySearch` to work. The $200/month free credit covers all demo usage.

---

## 🛠️ Tech Stack

```
Backend        FastAPI · Python 3.10+ · Uvicorn
LLM            LLaMA 3 8B · Ollama (local inference)
Embeddings     BAAI/bge-base-en-v1.5 · sentence-transformers
Retrieval      FAISS (dense) · BM25 (sparse) · RRF fusion
Documents      python-docx
Email          Resend API
WhatsApp       Meta Cloud API · Make.com
Frontend       HTML5 · CSS3 · Vanilla JavaScript
Maps           Google Maps JavaScript API · Places API
Voice          Web Speech API (browser-native)
PDF            PDF.js (CDN, client-side)
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Recall@7 | **86.5%** |
| Precision@7 | **84.2%** |
| F1 Score | **85.3%** |
| Avg Response Time (CPU) | **2.4 seconds** |
| Avg Response Time (GPU) | **~0.9 seconds** |
| Document Generation Success | **98.7%** |
| Test Queries | **58** |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m "Add YourFeature"`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a Pull Request

---

## ⚠️ Legal Disclaimer

“LegalEase” is used in this project purely as a placeholder name for a college assignment. We do not claim any ownership, trademark, or rights to this name. If any existing company, product, or service uses the same or a similar name, it is purely coincidental.

This project is intended for academic and demonstration purposes only and runs locally without any external deployment or commercial use. It does not intend to represent, compete with, or impact any real-world entity.

Anyone cloning or using this repository is free to rename the project as they wish for their own use.

Now this is an educational and informational tool. It is **not a substitute for professional legal advice**. The AI-generated responses are grounded in statute text but may be incomplete or inaccurate. Always consult a qualified legal professional before taking any legal action.

---

<div align="center">

Built with ❤️ for Indian citizens · Powered by open-source AI

**[⬆ Back to top](#️-legalease--ai-powered-indian-legal-guidance-chatbot)**

</div>
