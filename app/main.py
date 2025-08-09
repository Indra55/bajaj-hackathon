import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.endpoints import hackrx, domain_aware_qa, semantic_qa
from .utils.error_handlers import APIException, api_exception_handler

def create_app() -> FastAPI:
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    app = FastAPI(
        title="PolicyEval-GPT & HackRx Q&A",
        description="An AI-powered API for answering questions about policy documents.",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Routers
    app.include_router(hackrx.router, prefix="/hackrx", tags=["Q&A"])
    app.include_router(domain_aware_qa.router, prefix="/domain-qa", tags=["Domain-Aware Q&A"])
    app.include_router(semantic_qa.router, prefix="/api/v1", tags=["Semantic Q&A"])

    # Exception Handler
    app.add_exception_handler(APIException, api_exception_handler)

    @app.get("/", tags=["Root"])
    async def read_root():
        return {
            "message": "Welcome to the Enhanced PolicyEval-GPT API with Domain-Aware Q&A",
            "docs": "/docs",
            "version": "2.1.0",
            "features": [
                "Domain-aware document classification",
                "Intelligent query routing",
                "Multi-document processing",
                "Insurance, Legal, HR, and Compliance domain support"
            ],
            "endpoints": {
                "legacy_qa": "/hackrx/run",
                "domain_aware_qa": "/domain-qa/domain-aware-qa",
                "multi_document_qa": "/domain-qa/multi-document-qa",
                "domain_info": "/domain-qa/domain-info",
                "classify_document": "/domain-qa/classify-document"
            }
        }

    return app

# Create the FastAPI app
app = create_app()

# This allows the app to be run directly with: python -m app.main
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
