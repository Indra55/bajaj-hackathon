from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

# Request Schemas
class StructuredQuery(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration_months: Optional[int] = None
    emergency: Optional[bool] = None
    pre_existing: Optional[bool] = None

class Query(BaseModel):
    raw_text: Optional[str] = Field(None, example="46M, knee surgery, Pune, 3-month policy")
    structured_data: Optional[StructuredQuery] = None

class DocumentMetadata(BaseModel):
    policy_type: Literal["individual", "family", "group"]
    effective_date: str = Field(..., example="2023-01-01")
    language: str = Field("en-IN", example="en-IN")
    version: str = Field("v1.0", example="v2.1")

class Document(BaseModel):
    type: Literal["url", "base64", "document_id"]
    content: str
    metadata: DocumentMetadata

class EvaluationOptions(BaseModel):
    confidence_threshold: float = Field(0.8, ge=0, le=1)
    max_clauses: int = Field(10, gt=0)
    include_explanations: bool = True
    enable_fraud_detection: bool = True
    processing_mode: Literal["fast", "accurate", "comprehensive"] = "fast"

class EvaluationRequest(BaseModel):
    query: Query
    documents: List[Document]
    options: Optional[EvaluationOptions] = None

# Response Schemas
class Decision(BaseModel):
    status: Literal["approved", "rejected", "requires_review", "insufficient_info"]
    confidence_score: float
    risk_level: Literal["low", "medium", "high"]

class CoverageBreakdown(BaseModel):
    base_coverage: float
    co_pay: float
    deductible: float
    additional_benefits: float

class WaitingPeriod(BaseModel):
    required_months: int
    elapsed_months: int
    status: Literal["met", "not_met"]

class Coverage(BaseModel):
    approved_amount: float
    maximum_eligible: float
    currency: str = "INR"
    breakdown: CoverageBreakdown
    waiting_period: WaitingPeriod

class JustificationSource(BaseModel):
    document: str
    page: int
    section: str

class Justification(BaseModel):
    clause_id: str
    text: str
    matched_criteria: List[str]
    confidence: float
    rule_type: str
    source: JustificationSource

class RiskFactor(BaseModel):
    factor: str
    severity: Literal["low", "medium", "high"]
    impact_on_decision: Literal["positive", "negative", "neutral"]
    description: str

class Recommendation(BaseModel):
    type: str
    priority: Literal["low", "medium", "high"]
    message: str

class FraudAnalysis(BaseModel):
    risk_score: float
    flags: List[str]
    similar_claims: int

class ProcessingMetadata(BaseModel):
    documents_processed: int
    documents_failed: int
    clauses_evaluated: int
    ai_model: str
    model_version: str
    business_rules_version: str
    cache_hit_ratio: float
    gemini_tokens_used: int

class Warning(BaseModel):
    code: str
    message: str
    severity: Literal["low", "medium", "high"]

class Error(BaseModel):
    code: str
    message: str

class EvaluationTask(BaseModel):
    task_id: str
    status: str
    message: str

class EvaluationResponse(BaseModel):
    request_id: str
    timestamp: str
    processing_time_ms: int
    decision: Decision
    coverage: Coverage
    justification: List[Justification]
    risk_factors: List[RiskFactor]
    recommendations: List[Recommendation]
    fraud_analysis: FraudAnalysis
    processing_metadata: ProcessingMetadata
    warnings: Optional[List[Warning]] = None
    error: Optional[Dict[str, Any]] = None

# --- Schemas for HackRx Q&A Endpoint --- #

class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the policy document.")
    questions: list[str] = Field(..., description="A list of questions to ask about the document.")

class HackRxResponse(BaseModel):
    answers: list[str] = Field(..., description="A list of answers corresponding to the questions.")
