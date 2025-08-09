from fastapi import APIRouter, Body, HTTPException
from ..schemas.evaluation import EvaluationRequest, EvaluationResponse
from ...services.policy_eval_pipeline import PolicyEvalPipeline
from ...utils.error_handlers import DocumentProcessingError
import uuid
from datetime import datetime, timezone
import time

router = APIRouter()

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_policy(
    request: EvaluationRequest = Body(...)
):
    """
    This is the primary endpoint for the simplified PolicyEval-GPT service.
    It processes the request synchronously and returns the full evaluation.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        pipeline = PolicyEvalPipeline()
        result = await pipeline.process_request(request.dict())
    except Exception as e:
        raise DocumentProcessingError(str(e))

    processing_time_ms = int((time.time() - start_time) * 1000)

    # Map the detailed pipeline result to the final response schema
    raw_analysis = result.get("analysis", [])
    justifications = []
    if isinstance(raw_analysis, list):
        for item in raw_analysis:
            justifications.append({
                "clause_id": item.get("clause_id", "N/A"),
                "text": item.get("reasoning", "No reasoning provided."),
                "matched_criteria": item.get("matched_criteria", []),
                "confidence": item.get("relevance_score", 0.0),
                "rule_type": item.get("clause_type", "general"), # Updated from decision_impact
                "source": {"document": "source_doc", "page": 1, "section": "Details"} # Placeholder
            })

    # The final decision from the pipeline now contains much richer data
    response_data = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_time_ms": processing_time_ms,
        "decision": {
            "status": result.get("decision", "requires_review"),
            "confidence_score": result.get("confidence_score", 0.5),
            "risk_level": "medium" # Placeholder, can be derived from risk_factors later
        },
        "coverage": {
            "approved_amount": result.get("approved_amount", 0),
            "maximum_eligible": 0, # Placeholder
            "currency": "INR",
            "breakdown": { # Placeholder
                "base_coverage": 0,
                "co_pay": 0,
                "deductible": 0,
                "additional_benefits": 0
            },
            "waiting_period": { # Placeholder
                "required_months": 0,
                "elapsed_months": 0,
                "status": "not_met"
            }
        },
        "justification": justifications,
        "risk_factors": result.get("risk_factors", []),
        "recommendations": result.get("recommendations", []),
        "fraud_analysis": { # Placeholder
            "risk_score": 0.0,
            "flags": [],
            "similar_claims": 0
        },
        "processing_metadata": {
            "documents_processed": len(request.documents),
            "documents_failed": 0,
            "clauses_evaluated": len(justifications),
            "ai_model": "gemini-1.5-flash-latest",
            "model_version": "v1",
            "business_rules_version": "v2", # Incremented for new logic
            "cache_hit_ratio": 0.0,
            "gemini_tokens_used": result.get("token_usage", 0) # Placeholder for now
        },
        "warnings": [],
        "error": result.get("error")
    }

    return EvaluationResponse(**response_data)
