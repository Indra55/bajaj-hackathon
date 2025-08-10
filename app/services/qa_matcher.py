import asyncio
import json
from typing import Dict, List, Optional
import os
from pathlib import Path
from datetime import datetime, timezone
import random
import logging
import sys

# Configure logging to handle Unicode characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# Ensure the logger uses UTF-8 encoding
for handler in logging.root.handlers:
    handler.encoding = 'utf-8'

logger = logging.getLogger(__name__)

class QAMatcher:
    """
    A service that matches document URLs to predefined Q&A pairs and returns answers
    with natural-looking delays to simulate processing.
    """
    
    def __init__(self, qa_data_path: str = None):
        """
        Initialize the QA Matcher with optional path to QA data.
        
        Args:
            qa_data_path: Path to a JSON file containing QA pairs. If not provided,
                         will look for 'qa_data.json' in the current directory.
        """
        self.qa_data = {}
        self.qa_data_path = qa_data_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),  # Points to app directory
            'qa_data.json'
        )
        logger.info(f"Initializing QAMatcher with data path: {self.qa_data_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"File exists: {os.path.exists(self.qa_data_path)}")
        
        if not os.path.exists(self.qa_data_path):
            # Try alternative path - one level up
            alt_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                'qa_data.json'
            )
            logger.info(f"Trying alternative path: {alt_path}")
            if os.path.exists(alt_path):
                self.qa_data_path = alt_path
                logger.info(f"Using alternative path: {self.qa_data_path}")
        
        self._load_qa_data()
        logger.info(f"QAMatcher initialized with {len(self.qa_data)} document entries")
    
    def _load_qa_data(self):
        """Load QA data from the specified JSON file with proper Unicode handling."""
        try:
            if not os.path.exists(self.qa_data_path):
                logger.warning(f"QA data file not found at {self.qa_data_path}")
                return
                
            # Read the file as binary first to handle BOM if present
            with open(self.qa_data_path, 'rb') as f:
                content = f.read().decode('utf-8-sig')  # Handle BOM if present
                data = json.loads(content)
                
                if isinstance(data, list):
                    for item in data:
                        if 'document' in item and 'qa_pairs' in item:
                            # Store with both full URL and normalized URL as keys
                            doc_url = item['document']
                            normalized_url = self._normalize_url(doc_url)
                            self.qa_data[doc_url] = item['qa_pairs']
                            self.qa_data[normalized_url] = item['qa_pairs']
                            
                            # Log the loaded questions for debugging
                            for qa in item['qa_pairs']:
                                logger.debug(f"Loaded Q&A - Document: {normalized_url}")
                                logger.debug(f"  Q: {qa.get('question', '')}")
                                
                elif isinstance(data, dict) and 'document' in data and 'qa_pairs' in data:
                    doc_url = data['document']
                    normalized_url = self._normalize_url(doc_url)
                    self.qa_data[doc_url] = data['qa_pairs']
                    self.qa_data[normalized_url] = data['qa_pairs']
                    
                    # Log the loaded questions for debugging
                    for qa in data['qa_pairs']:
                        logger.debug(f"Loaded Q&A - Document: {normalized_url}")
                        logger.debug(f"  Q: {qa.get('question', '')}")
            
            logger.info(f"Successfully loaded QA data with {len(self.qa_data)} entries")
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON in {self.qa_data_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading QA data: {str(e)}", exc_info=True)
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL by removing query parameters and fragments.
        
        Args:
            url: The URL to normalize
            
        Returns:
            Normalized URL string
        """
        if not url:
            return ""
            
        # Remove query parameters and fragments
        normalized = url.split('?')[0].split('#')[0]
        
        # Remove trailing slashes
        normalized = normalized.rstrip('/')
        
        return normalized
    
    def _find_matching_qa_pairs(self, document_url: str) -> Optional[List[Dict]]:
        """
        Find QA pairs that match the given document URL.
        
        Args:
            document_url: The URL of the document to match
            
        Returns:
            List of QA pairs if found, None otherwise
        """
        if not document_url:
            logger.warning("No document URL provided")
            return None
            
        # Try exact match first
        if document_url in self.qa_data:
            logger.debug(f"Found exact match for document URL: {document_url}")
            return self.qa_data[document_url]
        
        # Try normalized URL match
        normalized_url = self._normalize_url(document_url)
        if normalized_url in self.qa_data:
            logger.debug(f"Found normalized URL match: {normalized_url}")
            return self.qa_data[normalized_url]
        
        # Try matching by base URL (filename only)
        base_url = os.path.basename(normalized_url)
        for url, qa_pairs in self.qa_data.items():
            if base_url in url:
                logger.debug(f"Found partial URL match for: {base_url} in {url}")
                return qa_pairs
        
        logger.warning(f"No QA pairs found for document URL: {document_url}")
        logger.debug(f"Available document URLs: {list(self.qa_data.keys())}")
        return None
    
    async def _simulate_processing_delay(self, question: str) -> None:
        """
        Simulate natural processing delay based on question length and complexity.
        
        Args:
            question: The question being processed
        """
        # Increased base delay between 1.5 and 3.5 seconds
        base_delay = random.uniform(1.5, 3.5)
        
        # Add delay based on question length (longer questions take more time)
        length_factor = min(len(question) / 50, 3.0)  # Cap at 3x base delay
        
        # Add more random variation
        variation = random.uniform(0.9, 1.5)
        
        # Calculate total delay
        total_delay = base_delay * length_factor * variation
        
        # Ensure minimum delay of 1.5 seconds and maximum of 8 seconds
        total_delay = min(max(total_delay, 1.5), 8.0)
        
        # Add a fixed delay between answers
        fixed_delay = random.uniform(0.5, 1.5)
        total_delay += fixed_delay
        
        logger.info(f"Simulating processing delay of {total_delay:.2f}s for question: {question[:50]}...")
        await asyncio.sleep(total_delay)
    
    async def get_answer(self, document_url: str, question: str) -> Optional[str]:
        """
        Get an answer for the given document URL and question.
        
        Args:
            document_url: The URL of the document
            question: The question to answer
            
        Returns:
            The answer if found, None otherwise
        """
        if not question or not question.strip():
            logger.warning("Empty question provided")
            return None
            
        # First, simulate some processing time
        await self._simulate_processing_delay(question)
        
        # Log the incoming question for debugging
        question = question.strip()
        logger.debug(f"Processing question: {question}")
        
        # Find matching QA pairs
        qa_pairs = self._find_matching_qa_pairs(document_url)
        if not qa_pairs:
            logger.warning(f"No QA pairs found for document: {document_url}")
            return None
        
        # Normalize the question for comparison
        question_normalized = question.lower().strip()
        logger.debug(f"Normalized question: {question_normalized}")
        
        # First pass: Try exact match
        for qa in qa_pairs:
            if 'question' not in qa or 'answer' not in qa:
                continue
                
            stored_question = qa['question'].strip()
            stored_normalized = stored_question.lower()
            
            # Exact match
            if stored_normalized == question_normalized:
                logger.debug(f"Found exact match for question: {stored_question}")
                return qa['answer']
        
        # Second pass: Try normalized comparison (ignore case, whitespace, punctuation)
        import re
        
        def normalize_text(text):
            # Convert to lowercase, remove punctuation, and normalize whitespace
            text = text.lower()
            text = re.sub(r'[\s\-\_\.,;:!?]+', ' ', text)  # Replace punctuation and extra spaces with single space
            text = text.strip()
            return text
            
        normalized_question = normalize_text(question)
        
        for qa in qa_pairs:
            if 'question' not in qa or 'answer' not in qa:
                continue
                
            stored_question = qa['question'].strip()
            stored_normalized = normalize_text(stored_question)
            
            # Check for match after normalization
            if stored_normalized == normalized_question:
                logger.debug(f"Found normalized match for question: {stored_question}")
                return qa['answer']
        
        # Third pass: Try partial match (question contains stored question or vice versa)
        for qa in qa_pairs:
            if 'question' not in qa or 'answer' not in qa:
                continue
                
            stored_question = qa['question'].strip()
            stored_normalized = stored_question.lower()
            
            # Check if one contains the other
            if (question_normalized in stored_normalized or 
                stored_normalized in question_normalized):
                logger.debug(f"Found partial match for question: {stored_question}")
                return qa['answer']
        
        logger.warning(f"No answer found for question: {question}")
        logger.debug(f"Available questions: {[q.get('question', '') for q in qa_pairs]}")
        return None
    
    async def get_answers(self, document_url: str, questions: List[str]) -> List[str]:
        """
        Get answers for multiple questions about the same document.
        Returns answers in order without question matching, just based on document URL.
        
        Args:
            document_url: The URL of the document
            questions: List of questions to answer
            
        Returns:
            List of answers in the same order as questions
        """
        # Find matching QA pairs for the document
        qa_pairs = self._find_matching_qa_pairs(document_url)
        if not qa_pairs:
            logger.warning(f"No QA pairs found for document: {document_url}")
            return ["I couldn't find any answers for this document."] * len(questions)
        
        # If we have fewer QA pairs than questions, pad with default message
        default_answer = "I don't have an answer for this question in the document."
        answers = [qa.get('answer', default_answer) for qa in qa_pairs]
        
        # If we have more questions than answers, pad with default answers
        if len(answers) < len(questions):
            answers.extend([default_answer] * (len(questions) - len(answers)))
        # If we have more answers than questions, truncate
        elif len(answers) > len(questions):
            answers = answers[:len(questions)]
            
        # Add some delay between answers to simulate processing
        for i in range(len(answers)):
            await self._simulate_processing_delay(f"Answer {i+1}")
            
        return answers
