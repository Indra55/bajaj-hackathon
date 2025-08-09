"""Custom exceptions for the application."""

class UnsupportedFileTypeError(Exception):
    """Raised when a file type is not supported for processing."""
    
    def __init__(self, file_type: str, message: str = None):
        self.file_type = file_type
        if message is None:
            message = f"File type '{file_type}' is not supported. Please use supported text-based formats."
        super().__init__(message)

class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass
