"""
Avelon LLM - FastAPI Application Entry Point

AI Microservice for:
- Document Classification
- OCR Text Extraction  
- Named Entity Recognition
- Fraud Detection
- Credit Scoring
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import health, verify, score

settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="Avelon LLM",
    description="AI Microservice for Document Verification and Credit Scoring",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.backend_url,
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(verify.router, prefix="/api/v1", tags=["Verification"])
app.include_router(score.router, prefix="/api/v1", tags=["Scoring"])


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Avelon LLM",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs" if settings.debug else "disabled",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
