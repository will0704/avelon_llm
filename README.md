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
│   ├── api/routes/          # API endpoints
│   ├── schemas/             # Pydantic models
│   ├── services/            # AI model wrappers
│   └── models/              # Trained model weights
├── notebooks/               # Training notebooks
├── data/                    # Training data
├── requirements.txt
├── Dockerfile
└── .env.example
```

## 🤖 AI Services

| Service | Technology | Status |
|---------|------------|--------|
| **Classifier** | PyTorch MobileNetV2 | ✅ Trained |
| **OCR/Extractor** | EasyOCR (EN + Tagalog) | ✅ Ready |
| **NER** | BERT (HuggingFace) | ✅ Ready |
| **Fraud Detector** | OpenCV + XGBoost | ⏳ Pending |
| **Credit Scorer** | XGBoost | ⏳ Pending |
| **ETH Volatility Predictor** | LSTM / Time-Series Model | 📋 Planned |

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/verify/document` | Verify single document |
| POST | `/api/v1/verify/complete` | Complete KYC verification |
| POST | `/api/v1/score/calculate` | Calculate credit score |
| GET | `/api/v1/predict/volatility` | ETH volatility prediction |

## 🔐 Authentication

All `/api/v1/*` endpoints require an API key header:

```
X-API-Key: your-api-key
```

## 📚 Documentation

See [global docs](../docs/) for complete project documentation.