"""
Semantic Q&A API Endpoints
Enhanced endpoints using semantic chunking for better accuracy
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import base64
import asyncio

from ...services.semantic_qa_service import SemanticQAService

router = APIRouter(prefix="/semantic-qa", tags=["Semantic Q&A"])

# Initialize semantic Q&A service
semantic_qa_service = SemanticQAService()

# Pydantic models for request/response
class SemanticQARequest(BaseModel):
    question: str
    document_content: str  # Base64 encoded document content
    filename: str
    content_type: Optional[str] = None

@router.post("/ask")
async def semantic_question_answer(
    request: SemanticQARequest
) -> Dict[str, Any]:
    """
    Answer questions using semantic chunking for better context understanding.
    
    This endpoint uses advanced semantic chunking that groups content by meaning
    rather than arbitrary size limits, resulting in more accurate answers.
    """
    try:
        # Validate inputs
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if not request.document_content:
            raise HTTPException(status_code=400, detail="No document content provided")
        
        # Prepare document data
        document = {
            'content': request.document_content,
            'metadata': {
                'filename': request.filename,
                'content_type': request.content_type or 'application/octet-stream',
                'size': len(request.document_content)
            }
        }
        
        print(f"Processing semantic Q&A for: {request.filename}")
        print(f"Question: {request.question}")
        
        # Process with semantic chunking
        result = await semantic_qa_service.process_document_and_answer(document, request.question)
        
        return {
            "status": "success",
            "question": question,
            "document": file.filename,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "domain": result["domain"],
            "domain_confidence": result.get("domain_confidence", 0.0),
            "semantic_analysis": {
                "chunks_used": result["chunks_used"],
                "semantic_score": result["semantic_score"],
                "chunk_topics": result.get("chunk_topics", [])
            },
            "validation": result.get("validation", {}),
            "processing_method": "semantic_chunking"
        }
        
    except Exception as e:
        print(f"Error in semantic Q&A: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process semantic Q&A: {str(e)}"
        )

class DocumentAnalysisRequest(BaseModel):
    document_content: str  # Base64 encoded document content
    filename: str
    content_type: Optional[str] = None

@router.post("/analyze-document")
async def analyze_document_semantics(
    request: DocumentAnalysisRequest
) -> Dict[str, Any]:
    """
    Analyze document semantic structure and chunking for insights.
    
    Returns detailed information about how the document is semantically chunked,
    including topics, coherence scores, and domain classification.
    """
    try:
        if not request.document_content:
            raise HTTPException(status_code=400, detail="No document content provided")
        
        # Prepare document data
        document = {
            'content': request.document_content,
            'metadata': {
                'filename': request.filename,
                'content_type': request.content_type or 'application/octet-stream',
                'size': len(request.document_content)
            }
        }
        
        print(f"Analyzing semantic structure of: {request.filename}")
        
        # Analyze semantic structure
        analysis = await semantic_qa_service.analyze_document_semantics(document)
        
        return {
            "status": "success",
            "document": request.filename,
            "semantic_analysis": analysis,
            "processing_method": "semantic_chunking"
        }
        
    except Exception as e:
        print(f"Error in semantic analysis: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to analyze document semantics: {str(e)}"
        )

class ChunkingComparisonRequest(BaseModel):
    question: str
    document_content: str  # Base64 encoded document content
    filename: str
    content_type: Optional[str] = None

@router.post("/compare-chunking")
async def compare_chunking_methods(
    request: ChunkingComparisonRequest
) -> Dict[str, Any]:
    """
    Compare semantic chunking vs traditional chunking for the same question.
    
    This endpoint processes the same document and question using both methods
    to demonstrate the differences in accuracy and relevance.
    """
    try:
        # Validate inputs
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if not request.document_content:
            raise HTTPException(status_code=400, detail="No document content provided")
        
        # Prepare document data
        document = {
            'content': request.document_content,
            'metadata': {
                'filename': request.filename,
                'content_type': request.content_type or 'application/octet-stream',
                'size': len(request.document_content)
            }
        }
        
        print(f"Comparing chunking methods for: {request.filename}")
        
        # Process with semantic chunking
        semantic_result = await semantic_qa_service.process_document_and_answer(document, request.question)
        
        # Process with traditional chunking (create traditional processor)
        from ...services.enhanced_document_processor import EnhancedDocumentProcessor
        from ...services.qa_service import QAService
        
        traditional_processor = EnhancedDocumentProcessor(use_semantic_chunking=False)
        traditional_qa = QAService()
        traditional_qa.document_processor = traditional_processor
        
        traditional_result = await traditional_qa.process_document_and_answer(document, request.question)
        
        return {
            "status": "success",
            "question": question,
            "document": file.filename,
            "comparison": {
                "semantic_chunking": {
                    "answer": semantic_result["answer"],
                    "confidence": semantic_result["confidence"],
                    "chunks_used": semantic_result["chunks_used"],
                    "semantic_score": semantic_result["semantic_score"],
                    "domain": semantic_result["domain"]
                },
                "traditional_chunking": {
                    "answer": traditional_result.get("answer", ""),
                    "confidence": traditional_result.get("confidence", 0.0),
                    "chunks_used": traditional_result.get("chunks_used", 0),
                    "domain": traditional_result.get("domain", "unknown")
                }
            },
            "recommendation": "semantic" if semantic_result["confidence"] > traditional_result.get("confidence", 0.0) else "traditional"
        }
        
    except Exception as e:
        print(f"Error in chunking comparison: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to compare chunking methods: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint for semantic Q&A service."""
    return {
        "status": "healthy",
        "service": "Semantic Q&A",
        "features": [
            "semantic_chunking",
            "domain_classification", 
            "enhanced_retrieval",
            "response_validation"
        ]
    }
