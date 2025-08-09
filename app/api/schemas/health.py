from pydantic import BaseModel, Field
from typing import Dict

class ServiceStatus(BaseModel):
    database: str = "healthy"
    vector_store: str = "healthy"
    ml_models: str = "healthy"
    cache: str = "healthy"

class HealthMetrics(BaseModel):
    requests_per_minute: int = Field(..., example=45)
    average_response_time: float = Field(..., example=1.8)
    error_rate: float = Field(..., example=0.002)

class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    services: ServiceStatus
    metrics: HealthMetrics
