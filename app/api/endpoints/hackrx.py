from fastapi import APIRouter, Depends, HTTPException
from app.api.schemas.evaluation import HackRxRequest, HackRxResponse
from app.core.security import get_api_key
from app.services.qa_service import QAService
from app.core.exceptions import UnsupportedFileTypeError, DocumentProcessingError
import logging

router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)

qa_service = QAService()

@router.post("/run", response_model=HackRxResponse, tags=["Q&A"])
async def run_hackrx_evaluation(
    request: HackRxRequest,
    api_key: str = Depends(get_api_key)
):
    """
    This endpoint processes a document from a URL against a list of questions.

    - **Authentication**: Requires a Bearer token in the `Authorization` header.
    - **Request**: Takes a document URL and a list of questions.
    - **Response**: Returns a list of answers corresponding to the questions.
    """
    try:
        # Log the incoming request details
        print(f"=== HackRx Q&A Request ===")
        print(f"Document URL: {request.documents}")
        print(f"Questions: {request.questions}")
        print(f"=========================")
        logger.info(f"HackRx Q&A Request - Document URL: {request.documents}, Questions: {request.questions}")
        
        answers = await qa_service.answer_questions(request)
        
        # Log successful processing
        print(f"=== HackRx Q&A Success ===")
        print(f"Document URL: {request.documents}")
        print(f"Questions processed: {len(request.questions)}")
        print(f"=========================")
        logger.info(f"HackRx Q&A Success - Document URL: {request.documents}, Questions processed: {len(request.questions)}")
        
        return HackRxResponse(answers=answers)
    
    except UnsupportedFileTypeError as e:
        # Handle unsupported file types gracefully
        print(f"=== HackRx Q&A Unsupported File Type ===")
        print(f"Document URL: {request.documents}")
        print(f"File Type: {e.file_type}")
        print(f"Error: {str(e)}")
        print(f"=======================================")
        logger.warning(f"HackRx Q&A Unsupported File Type - Document URL: {request.documents}, File Type: {e.file_type}, Error: {str(e)}")
        
        # Return the same response format but with file type error message for all questions
        unsupported_message = f"File type '{e.file_type}' is not supported. Please use supported text-based formats like PDF, DOCX, PPTX, XLSX, TXT, HTML, etc."
        answers = [unsupported_message for _ in request.questions]
        return HackRxResponse(answers=answers)
    
    except DocumentProcessingError as e:
        # Handle document processing errors
        print(f"=== HackRx Q&A Processing Error ===")
        print(f"Document URL: {request.documents}")
        print(f"Questions: {request.questions}")
        print(f"Error: {str(e)}")
        print(f"==================================")
        logger.error(f"HackRx Q&A Processing Error - Document URL: {request.documents}, Questions: {request.questions}, Error: {str(e)}")
        
        # Return processing error message for all questions
        error_message = f"Document processing failed: {str(e)}"
        answers = [error_message for _ in request.questions]
        return HackRxResponse(answers=answers)
    
    except Exception as e:
        # Log the error with request details
        print(f"=== HackRx Q&A Error ===")
        print(f"Document URL: {request.documents}")
        print(f"Questions: {request.questions}")
        print(f"Error: {str(e)}")
        print(f"========================")
        logger.error(f"HackRx Q&A Error - Document URL: {request.documents}, Questions: {request.questions}, Error: {str(e)}")
        
        # Return generic error message for all questions
        error_message = "An unexpected error occurred while processing your request. Please try again."
        answers = [error_message for _ in request.questions]
        return HackRxResponse(answers=answers)
