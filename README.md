# PolicyEval-GPT (Simplified)

This is a simplified, self-contained version of the PolicyEval-GPT API. It uses a synchronous architecture and an in-memory vector database (FAISS), requiring no external services like Redis, RabbitMQ, or Docker.

## Features

- **Synchronous API**: Simple request-response model.
- **In-Memory Vector Search**: Uses FAISS for local, fast semantic search. No API keys needed.
- **No Authentication**: The endpoint is open for easy testing and development.
- **AI-Powered**: Leverages Google's Gemini 1.5 Flash for analysis.

## Prerequisites

- Python 3.9+

## Setup and Installation

**1. Clone the repository**

```bash
git clone <repository-url>
cd policyeval-gpt
```

**2. Set up Environment Variables**

Copy the example environment file and add your Gemini API key.

```bash
cp .env.example .env
```

Now, edit the `.env` file with your `GEMINI_API_KEY`.

**3. Install Python Dependencies**

It's highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

In your terminal, run the main API server.

```bash
source venv/bin/activate
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000/docs`.

## How to Test the API

Since authentication has been removed, you can directly call the `/v1/evaluate` endpoint. Use a tool like `curl` or any API client.

```bash
curl -X POST "http://127.0.0.1:8000/v1/evaluate" \
-H "Content-Type: application/json" \
-d '{
  "query": "Is damage from a water leak covered under this policy?",
  "documents": [
    {
      "content_type": "url",
      "content": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    }
  ]
}'
```

The API will process the request and return the full JSON evaluation directly in the response.
