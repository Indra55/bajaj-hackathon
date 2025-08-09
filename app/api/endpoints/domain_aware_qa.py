"""
Domain-Aware QA API Endpoint
Enhanced endpoint with domain classification and routing for better accuracy
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List, Dict, Any
from app.api.schemas.evaluation import HackRxRequest
from app.core.security import get_api_key
from app.services.domain_aware_qa_service import DomainAwareQAService, DomainAwareAnswer
from app.services.domain_classifier import Domain
from pydantic import BaseModel
import time
import uuid
from datetime import datetime, timezone

router = APIRouter()

# Initialize the domain-aware QA service
domain_qa_service = DomainAwareQAService()

class DomainAwareQARequest(BaseModel):
    """Request model for domain-aware QA."""
    documents: str  # URL or base64 content
    questions: List[str]
    enable_domain_routing: bool = True
    target_domain: str = None  # Optional: force specific domain

class DomainAwareQAResponse(BaseModel):
    """Response model for domain-aware QA."""
    request_id: str
    timestamp: str
    processing_time_ms: int
    answers: List[Dict[str, Any]]
    domain_statistics: Dict[str, Any]
    routing_enabled: bool

class MultiDocumentQARequest(BaseModel):
    """Request model for multi-document QA with domain awareness."""
    documents: List[Dict[str, Any]]  # List of documents with metadata
    questions: List[str]
    enable_domain_routing: bool = True

@router.post("/domain-aware-qa", response_model=DomainAwareQAResponse, tags=["Domain-Aware Q&A"])
async def run_domain_aware_qa(
    request: DomainAwareQARequest = Body(...),
    api_key: str = Depends(get_api_key)
):
    """
    Enhanced Q&A endpoint with domain classification and routing.
    
    This endpoint:
    1. Classifies the document(s) by domain (Insurance, Legal, HR, Compliance)
    2. Classifies each query by domain
    3. Routes queries only to relevant domain-specific documents
    4. Provides domain-aware answers with higher accuracy
    
    - **Authentication**: Requires a Bearer token in the `Authorization` header.
    - **Request**: Takes document(s) and questions with optional domain routing settings.
    - **Response**: Returns domain-aware answers with routing information.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Convert single document request to multi-document format
        hackrx_request = HackRxRequest(
            documents=request.documents,
            questions=request.questions
        )
        
        # Process with domain awareness
        if request.enable_domain_routing:
            domain_answers = await domain_qa_service.answer_questions_with_domain_routing(hackrx_request)
            
            # Convert domain answers to response format
            answers = []
            for i, domain_answer in enumerate(domain_answers):
                answers.append({
                    "question_id": i + 1,
                    "question": request.questions[i],
                    "answer": domain_answer.answer,
                    "source_domain": domain_answer.source_domain.value,
                    "confidence": domain_answer.confidence,
                    "sources_used": domain_answer.sources_used,
                    "domain_match_quality": domain_answer.domain_match_quality,
                    "routing_applied": True
                })
        else:
            # Fallback to regular QA service
            from app.services.qa_service import QAService
            regular_qa_service = QAService()
            regular_answers = await regular_qa_service.answer_questions(hackrx_request)
            
            answers = []
            for i, answer in enumerate(regular_answers):
                answers.append({
                    "question_id": i + 1,
                    "question": request.questions[i],
                    "answer": answer,
                    "source_domain": "general",
                    "confidence": 0.5,  # Default confidence
                    "sources_used": ["document"],
                    "domain_match_quality": "not_applied",
                    "routing_applied": False
                })
        
        # Get domain statistics
        domain_stats = domain_qa_service.get_domain_statistics()
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return DomainAwareQAResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            processing_time_ms=processing_time_ms,
            answers=answers,
            domain_statistics=domain_stats,
            routing_enabled=request.enable_domain_routing
        )
        
    except Exception as e:
        print(f"Domain-aware QA error: {e}")
        raise HTTPException(status_code=500, detail=f"Domain-aware QA processing failed: {str(e)}")

@router.post("/multi-document-qa", response_model=DomainAwareQAResponse, tags=["Multi-Document Q&A"])
async def run_multi_document_qa(
    request: MultiDocumentQARequest = Body(...),
    api_key: str = Depends(get_api_key)
):
    """
    Multi-document Q&A with domain awareness.
    
    This endpoint handles multiple documents simultaneously:
    1. Processes and classifies each document by domain
    2. Creates domain-specific vector stores
    3. Routes queries to appropriate domain documents
    4. Provides comprehensive answers from multiple sources
    
    - **Authentication**: Requires a Bearer token in the `Authorization` header.
    - **Request**: Takes multiple documents with metadata and questions.
    - **Response**: Returns domain-aware answers from multiple document sources.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Process multiple documents with domain awareness
        await domain_qa_service.process_documents_with_domain_awareness(request.documents)
        
        # Create a dummy HackRxRequest for compatibility
        hackrx_request = HackRxRequest(
            documents="",  # Not used in multi-document mode
            questions=request.questions
        )
        
        # Process questions with domain routing
        if request.enable_domain_routing:
            domain_answers = await domain_qa_service.answer_questions_with_domain_routing(hackrx_request)
            
            answers = []
            for i, domain_answer in enumerate(domain_answers):
                answers.append({
                    "question_id": i + 1,
                    "question": request.questions[i],
                    "answer": domain_answer.answer,
                    "source_domain": domain_answer.source_domain.value,
                    "confidence": domain_answer.confidence,
                    "sources_used": domain_answer.sources_used,
                    "domain_match_quality": domain_answer.domain_match_quality,
                    "routing_applied": True
                })
        else:
            # Process without domain routing (search all documents)
            answers = []
            for i, question in enumerate(request.questions):
                answers.append({
                    "question_id": i + 1,
                    "question": question,
                    "answer": "Multi-document processing without domain routing not implemented",
                    "source_domain": "general",
                    "confidence": 0.0,
                    "sources_used": [],
                    "domain_match_quality": "not_applied",
                    "routing_applied": False
                })
        
        # Get domain statistics
        domain_stats = domain_qa_service.get_domain_statistics()
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return DomainAwareQAResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            processing_time_ms=processing_time_ms,
            answers=answers,
            domain_statistics=domain_stats,
            routing_enabled=request.enable_domain_routing
        )
        
    except Exception as e:
        print(f"Multi-document QA error: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-document QA processing failed: {str(e)}")

@router.get("/domain-info", tags=["Domain Information"])
async def get_domain_info(api_key: str = Depends(get_api_key)):
    """
    Get information about available domains and their characteristics.
    
    Returns:
    - Available domains
    - Domain-specific keywords
    - Document classification patterns
    """
    try:
        from app.services.domain_classifier import DomainClassifier
        
        classifier = DomainClassifier()
        
        domain_info = {}
        for domain in Domain:
            domain_info[domain.value] = {
                "keywords": classifier.get_domain_specific_keywords(domain),
                "description": _get_domain_description(domain)
            }
        
        return {
            "available_domains": list(Domain.__members__.keys()),
            "domain_details": domain_info,
            "routing_benefits": [
                "Higher accuracy by matching queries to relevant documents",
                "Reduced noise from irrelevant document content",
                "Domain-specific answer formatting and terminology",
                "Better handling of specialized terminology"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get domain info: {str(e)}")

@router.post("/classify-document", tags=["Domain Classification"])
async def classify_document(
    document: Dict[str, Any] = Body(...),
    api_key: str = Depends(get_api_key)
):
    """
    Classify a document's domain without processing Q&A.
    
    Useful for:
    - Pre-processing document classification
    - Understanding document content before Q&A
    - Building domain-specific document collections
    """
    try:
        from app.services.enhanced_document_processor import EnhancedDocumentProcessor
        
        processor = EnhancedDocumentProcessor()
        processed_doc = processor.process_document_with_domain(document)
        
        return {
            "filename": processed_doc.filename,
            "primary_domain": processed_doc.domain_classification.primary_domain.value,
            "confidence": processed_doc.domain_classification.confidence,
            "secondary_domains": [
                {"domain": d.value, "confidence": conf} 
                for d, conf in processed_doc.domain_classification.secondary_domains
            ],
            "keywords_found": processed_doc.domain_classification.keywords_found,
            "reasoning": processed_doc.domain_classification.reasoning,
            "chunk_count": len(processed_doc.chunks),
            "document_length": processed_doc.total_length
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document classification failed: {str(e)}")

def _get_domain_description(domain: Domain) -> str:
    """Get human-readable description for each domain."""
    descriptions = {
        Domain.INSURANCE: "Insurance policies, coverage details, claims, premiums, and related documents",
        Domain.LEGAL: "Legal documents, constitutional provisions, laws, regulations, and court documents",
        Domain.HR: "Human resources policies, employee handbooks, procedures, and workplace guidelines",
        Domain.COMPLIANCE: "Regulatory compliance, governance frameworks, audit reports, and risk management",
        Domain.GENERAL: "General documents that don't fit specific domain categories"
    }
    return descriptions.get(domain, "Unknown domain")
