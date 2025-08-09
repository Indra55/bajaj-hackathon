from fastapi import Request, status
from fastapi.responses import JSONResponse

class APIException(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message

async def api_exception_handler(request: Request, exc: APIException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )

# Example of a specific exception
class DocumentProcessingError(APIException):
    def __init__(self, message: str = "Failed to process document"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="DOCUMENT_PROCESSING_FAILED",
            message=message
        )
