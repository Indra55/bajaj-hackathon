from fastapi import APIRouter
from datetime import datetime, timezone
from ..schemas.health import HealthCheckResponse, ServiceStatus, HealthMetrics

router = APIRouter()

@router.get("/health", response_model=HealthCheckResponse)
def health_check():
    """
    Checks the health of the application and its services.
    """
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=ServiceStatus(),
        metrics=HealthMetrics(
            requests_per_minute=0, # Placeholder
            average_response_time=0.0, # Placeholder
            error_rate=0.0 # Placeholder
        )
    )
