# Avelon LLM

AI Microservice for Document Verification and Credit Scoring.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Unix/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Run the server
uvicorn app.main:app --reload --port 8000
```

### Access
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## 📁 Project Structure

```
avelon_llm/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py    # Health checks
│   │   │   ├── verify.py    # Document verification
│   │   │   └── score.py     # Credit scoring
│   │   └── dependencies.py  # API key auth
│   ├── schemas/             # Pydantic models
│   └── services/            # AI model wrappers
├── requirements.txt
├── Dockerfile
└── .env.example
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/verify/document` | Verify single document |
| POST | `/api/v1/verify/complete` | Complete KYC verification |
| POST | `/api/v1/score/calculate` | Calculate credit score |

## 🔐 Authentication

All `/api/v1/*` endpoints require an API key header:

```
X-API-Key: your-api-key
```

## 📚 Documentation

See [global docs](../docs/) for complete project documentation.