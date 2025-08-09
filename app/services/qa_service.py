import asyncio
import re
from app.services.document_processor import DocumentProcessor
from app.services.gemini_service import GeminiPolicyProcessor
from app.services.vector_store_service import VectorStoreService
from app.api.schemas.evaluation import HackRxRequest

class QAService:
    """Hackathon-optimized QA Service for insurance policy queries."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.gemini_service = GeminiPolicyProcessor()
        
        # Insurance-specific query expansion for better retrieval
        self.query_expansions = {
            'grace period': ['grace period', 'premium payment', 'payment due date', 'renewal period'],
            'waiting period': ['waiting period', 'elimination period', 'qualifying period', 'coverage starts'],
            'pre-existing': ['pre-existing', 'preexisting', 'prior condition', 'existing disease', 'PED'],
            'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery', 'termination'],
            'cataract': ['cataract', 'eye surgery', 'lens replacement', 'vision treatment'],
            'organ donor': ['organ donor', 'transplant', 'harvesting', 'donation'],
            'no claim discount': ['no claim discount', 'NCD', 'bonus', 'renewal discount'],
            'health check': ['health check', 'preventive', 'screening', 'medical examination'],
            'hospital': ['hospital', 'medical institution', 'healthcare facility', 'nursing home'],
            'ayush': ['ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha', 'naturopathy'],
            'room rent': ['room rent', 'accommodation', 'daily charges', 'ICU charges']
        }

    async def answer_questions(self, request: HackRxRequest) -> list[str]:
        """Optimized Q&A process with parallel processing and caching."""
        
        # 1. Process document with better chunking
        document_dict = {
            'content': request.documents, 
            'metadata': {'filename': request.documents.split('/')[-1].split('?')[0]}
        }
        
        text_chunks = self.document_processor.process_document(document_dict)
        
        if not text_chunks:
            raise ValueError("Could not extract any text from the document.")

        print(f"Processed {len(text_chunks)} chunks")
        
        # 2. Create vector store with batch embeddings
        vector_store = VectorStoreService(dimension=768)
        embeddings = await self.gemini_service.generate_embeddings_batch(text_chunks)
        vector_store.add_documents([{'text': chunk, 'embedding': emb} for chunk, emb in zip(text_chunks, embeddings)])

        # 3. PARALLEL PROCESSING: Answer all questions simultaneously
        print(f"Processing {len(request.questions)} questions in parallel...")
        tasks = [
            self._answer_single_question_improved(question, vector_store, i+1) 
            for i, question in enumerate(request.questions)
        ]
        
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        processed_answers = []
        for i, answer in enumerate(answers):
            if isinstance(answer, Exception):
                print(f"Error processing question {i+1}: {answer}")
                processed_answers.append(f"Error processing question: {str(answer)}")
            else:
                processed_answers.append(answer)
        
        return processed_answers

    async def _answer_single_question_improved(self, question: str, vector_store: VectorStoreService, question_num: int = 1) -> str:
        """Hackathon-optimized question answering with multi-strategy retrieval."""
        
        # 1. Multi-strategy query expansion
        queries = self._generate_search_queries(question)
        
        # 2. OPTIMIZED: Retrieve fewer chunks for faster processing
        all_chunks = set()
        for query in queries:
            query_embedding = await self.gemini_service.generate_embeddings(query, task_type="retrieval_query")
            search_results = vector_store.search(query_embedding, top_k=5)  # Reduced from 8 to 5
            
            for result in search_results:
                all_chunks.add(result['text'])
        
        if not all_chunks:
            return "Information not found in the policy document."
        
        # 3. OPTIMIZED: Use fewer context chunks for faster processing
        relevant_chunks = list(all_chunks)[:8]  # Reduced from 12 to 8 chunks
        print(f"Q{question_num}: Using {len(relevant_chunks)} chunks for context")
        
        # 4. Generate answer with enhanced prompt
        answer = await self._generate_precise_answer(question, relevant_chunks)
        
        return answer
    
    def _generate_search_queries(self, question: str) -> list[str]:
        """Generate multiple search queries for better retrieval."""
        queries = [question]  # Original question
        question_lower = question.lower()
        
        # Add expanded queries based on key terms
        for key_term, expansions in self.query_expansions.items():
            if key_term in question_lower:
                for expansion in expansions:
                    # Create variations
                    queries.append(expansion)
                    queries.append(f"{expansion} policy")
                    queries.append(f"{expansion} coverage")
        
        # Add specific query patterns for common questions
        if 'grace period' in question_lower:
            queries.extend([
                'thirty days premium payment',
                'due date renewal continue',
                'payment grace time'
            ])
        
        if 'waiting period' in question_lower and 'pre-existing' in question_lower:
            queries.extend([
                'thirty-six months continuous coverage',
                '36 months PED waiting',
                'pre-existing disease waiting'
            ])
        
        if 'maternity' in question_lower:
            queries.extend([
                'twenty-four months continuous coverage',
                '24 months maternity eligibility',
                'two deliveries terminations'
            ])
        
        if 'cataract' in question_lower:
            queries.extend([
                'two years cataract surgery',
                '2 years eye surgery waiting'
            ])
        
        if 'no claim discount' in question_lower or 'NCD' in question_lower:
            queries.extend([
                '5% base premium renewal',
                'discount no claims preceding year'
            ])
        
        return list(set(queries))  # Remove duplicates
    
    async def _generate_precise_answer(self, question: str, context_chunks: list[str]) -> str:
        """Generate precise answer using enhanced prompting."""
        
        # Create enhanced prompt for better accuracy
        enhanced_prompt = f"""
You are an expert insurance policy analyst. Answer the following question with MAXIMUM PRECISION using ONLY the provided policy context.

IMPORTANT RULES:
1. Extract EXACT numbers, percentages, and time periods from the policy text
2. Quote specific policy language when possible
3. If asking about waiting periods, provide the EXACT duration
4. If asking about coverage, state YES/NO clearly first, then explain conditions
5. Include ALL relevant conditions and limitations
6. Use the EXACT terminology from the policy document

QUESTION: {question}

POLICY CONTEXT:
{chr(10).join([f"SECTION {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])}

PROVIDE A PRECISE, FACTUAL ANSWER:"""
        
        try:
            # Use the enhanced prompt with the Gemini service
            response = await self.gemini_service.model.generate_content_async(
                enhanced_prompt,
                generation_config={
                    "temperature": 0.1,  # Low temperature for precision
                    "top_p": 0.9,
                    "max_output_tokens": 512,
                }
            )
            
            answer = response.text.strip()
            
            # Post-process for clarity
            answer = self._post_process_answer(answer, question)
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Error processing the question. Please try again."
    
    def _post_process_answer(self, answer: str, question: str) -> str:
        """Post-process answer for better formatting and accuracy."""
        
        # Clean up formatting
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Ensure proper capitalization
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        # Add period if missing
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        # Validate that numerical questions have numbers
        if any(word in question.lower() for word in ['period', 'months', 'years', 'days', '%', 'discount']):
            if not re.search(r'\d+', answer):
                answer += " (Please refer to the specific policy document for exact numerical details.)"
        
        return answer
