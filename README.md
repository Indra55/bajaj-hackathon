# ⚡ PolicyEval-GPT & Domain-Aware Q&A API 🚀

> AI-powered, ultra-fast, domain-aware document Q&A with precision answers and production-ready APIs.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-🚀-009688?logo=fastapi&logoColor=white)
![Google Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash%20Lite-1a73e8?logo=google)
![Vector DB](https://img.shields.io/badge/Vector%20DB-FAISS-2E77BC)
![License](https://img.shields.io/badge/License-Custom-blue)
![Version](https://img.shields.io/badge/Version-2.1.0-informational)
![Build](https://img.shields.io/badge/Build-Production%20Ready-success)

</div>

---

## 🌟 Highlights

- ⚡ Blazing-fast parallel Q&A (async processing, smart caching)
- 🧠 Domain-aware routing (Insurance, Legal, HR, Compliance, General)
- 📑 Smart document handling (PDF, DOCX, PPTX, XLSX, HTML, EML, JSON, ZIP, and more)
- 🧭 Intelligent retrieval with FAISS + Gemini embeddings
- 🪄 Small-doc optimization (no-chunk direct context for higher accuracy)
- 🔐 Secure by default (API key auth)
- 🛠️ Production-friendly (Render/Procfile support, configurable env)

---

## 🧭 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Server](#-running-the-server)
- [Usage](#-usage)
  - [Swagger UI](#swagger-ui)
  - [HackRx Q&A](#hackrx-qa)
  - [Domain-Aware Q&A](#domain-aware-qa)
  - [Multi-Document Q&A](#multi-document-qa)
  - [Domain Info & Classification](#domain-info--classification)
- [Examples](#-examples)
- [License](#-license)
- [Call to Action](#-call-to-action)

---

## 📘 About

PolicyEval-GPT is a FastAPI-based service that answers questions about documents with high precision. It intelligently detects document type and domain, routes queries to the right analysis pipeline, and uses FAISS vector search with Gemini embeddings to deliver concise, accurate answers. Optimized for both small and large documents, and built for real-world production use.

---

## ✨ Features

- 🔍 Domain classification and routing for higher accuracy
- 🧩 Multi-format document processing (PDF, DOCX, PPTX, XLSX, CSV, HTML/ASPX, XML, EML/MSG, JSON, TXT, ZIP)
- 🧱 Smart chunking for policies; direct full-context analysis for small docs
- 🧪 Policy-specific reasoning and general-doc zero-hallucination prompts
- 🧠 Query expansion and multi-strategy retrieval
- 📈 Parallel question answering via asyncio.gather
- 💾 Embedding cache (in-memory, size-managed) for speed
- 🔐 API-key secured endpoints
- 📄 OpenAPI docs at `/docs` and `/redoc`

---

## 🧰 Tech Stack

- Backend: FastAPI, Uvicorn
- LLM: Google Gemini 2.5 Flash Lite (Generative + Embeddings `text-embedding-004`)
- Retrieval: FAISS (cosine similarity)
- Parsing: PyPDF2/pypdf, python-docx, python-pptx, openpyxl/xlrd, BeautifulSoup, lxml, markdown, chardet, eml-parser, extract-msg
- Data: numpy, scikit-learn, nltk
- Config: pydantic-settings, python-dotenv
- Deploy: Render (render.yaml), Procfile compatible

---

## 🏗️ Architecture

```mermaid
flowchart TD
    C[Client] -->|Bearer Token| A[FastAPI]
    A --> R1[/hackrx/run/]
    A --> R2[/domain-qa/domain-aware-qa/]
    A --> R3[/domain-qa/multi-document-qa/]
    A --> R4[/domain-qa/domain-info/]
    A --> R5[/domain-qa/classify-document/]

    subgraph Core Services
      D[DocumentProcessor\n(multiformat extract + chunking)]
      G[GeminiPolicyProcessor\n(LLM + Embeddings + Cache)]
      V[VectorStoreService\n(FAISS)]
      Q[QAService\n(Parallel QA + Small-Doc Direct Mode)]
      DA[DomainAwareQAService\n(Routing + Stats)]
    end

    R1 --> Q
    R2 --> DA
    R3 --> DA

    Q --> D
    Q --> G
    Q --> V

    DA --> D
    DA --> G
    DA --> V
```

---

## ⚙️ Installation

```bash
# Clone
git clone <your-fork-or-repo-url>
cd bajaj-r4

# Python 3.10 recommended (per render.yaml)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt
```

---

## 🔧 Configuration

Create a `.env` file or set environment variables:

```bash
# .env
GEMINI_API_KEY=your_gemini_api_key
HACKRX_API_KEY=your_server_api_key  # used to protect endpoints
```

Notes:
- `GEMINI_API_KEY` is required to start the server.
- `HACKRX_API_KEY` is validated from the `Authorization: Bearer <token>` header.

---

## ▶️ Running the Server

```bash
# Local
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Procfile/Render (already configured)
# web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

- Open Swagger UI: http://localhost:8000/docs
- Open ReDoc: http://localhost:8000/redoc

---

## 🧪 Usage

All endpoints require an API key in the header:

```
Authorization: Bearer <HACKRX_API_KEY>
Content-Type: application/json
```

### Swagger UI

Navigate to `http://localhost:8000/docs` for interactive testing.

### HackRx Q&A

Endpoint: `POST /hackrx/run`

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer $HACKRX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/sample.pdf",
    "questions": [
      "What is the grace period?",
      "Is maternity covered?"
    ]
  }'
```

### Domain-Aware Q&A

Endpoint: `POST /domain-qa/domain-aware-qa`

```bash
curl -X POST "http://localhost:8000/domain-qa/domain-aware-qa" \
  -H "Authorization: Bearer $HACKRX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": ["What is the waiting period for cataract?"],
    "enable_domain_routing": true
  }'
```

### Multi-Document Q&A

Endpoint: `POST /domain-qa/multi-document-qa`

```bash
curl -X POST "http://localhost:8000/domain-qa/multi-document-qa" \
  -H "Authorization: Bearer $HACKRX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"content": "https://example.com/insurance.pdf", "metadata": {"filename": "insurance.pdf"}},
      {"content": "https://example.com/hr_policy.docx", "metadata": {"filename": "hr_policy.docx"}}
    ],
    "questions": ["What is the maternity waiting period?"],
    "enable_domain_routing": true
  }'
```

### Domain Info & Classification

- `GET /domain-qa/domain-info`
- `POST /domain-qa/classify-document`

```bash
# Info
curl -H "Authorization: Bearer $HACKRX_API_KEY" http://localhost:8000/domain-qa/domain-info

# Classify
curl -X POST "http://localhost:8000/domain-qa/classify-document" \
  -H "Authorization: Bearer $HACKRX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "https://example.com/somefile.pdf",
    "metadata": {"filename": "somefile.pdf"}
  }'
```

---

## 🧪 Examples

Using Python `requests`:

```python
import requests

API = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer YOUR_HACKRX_API_KEY", "Content-Type": "application/json"}

payload = {
  "documents": "https://example.com/sample.pdf",
  "questions": [
    "Does the policy cover AYUSH treatments?",
    "What is the waiting period for pre-existing diseases?"
  ]
}

r = requests.post(f"{API}/hackrx/run", headers=HEADERS, json=payload)
print(r.json())
```

Color-themed JSON (Nord-ish):

```jsonc
{
  // request
  "documents": "https://example.com/policy.pdf",
  "questions": ["What is the grace period?", "Is maternity covered?"]
}
```

---

## 📜 License

This project is provided under a Custom/Proprietary license for hackathon/demo purposes. Please contact the maintainers for production licensing terms.

---

## ⭐ Call to Action

If you find this project useful or inspiring, please consider giving it a star. It helps others discover it and motivates further development!

➡️ "Star" this repo now and stay tuned for updates! ⭐

---

## 🐳 Docker

This project includes production and development Docker setups.

Prerequisites:
- Create a `.env` file in the project root with the following at minimum:

```bash
GEMINI_API_KEY=your_gemini_api_key
HACKRX_API_KEY=your_server_api_key
```

### Build and Run (production image)

```bash
# Build
docker build -t bajaj-r4-api:prod .

# Run
docker run --name bajaj-r4-api \
  --env-file .env \
  -p 8000:8000 \
  bajaj-r4-api:prod

# Open http://localhost:8000/docs
```

### Docker Compose

Two services are defined in `docker-compose.yml`:
- `api` (production image)
- `api-dev` (development with auto-reload)

```bash
# Prod-like
docker compose up --build api

# Development (live reload, mounts source code)
docker compose up --build api-dev
```

URLs:
- Prod-like: http://localhost:8000/docs
- Dev: http://localhost:8001/docs

Notes:
- The server requires `GEMINI_API_KEY` at start; without it the app will fail fast.
- OCR (pytesseract) and various document parsers are available in the image.
