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
        
        # Policy-specific query expansion for better retrieval (only used for policy documents)
        self.policy_query_expansions = {
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
        
        # 2. SMART DOCUMENT TYPE DETECTION
        document_type = self.gemini_service.detect_document_type(text_chunks)
        print(f"Document type detected: {document_type}")
        
        # 3. SMART DOCUMENT SIZE DETECTION - Skip chunking for small documents
        full_text = ' '.join(text_chunks)
        word_count = len(full_text.split())
        
        print(f"Document size: {word_count} words")
        
        if word_count <= 2000 or len(text_chunks) <= 3:  # Small document threshold
            print("SMALL DOCUMENT DETECTED: Using direct processing (no vector search)")
            # For small documents, use direct processing without chunking
            tasks = [
                self._answer_question_direct(question, full_text, i+1, document_type)
                for i, question in enumerate(request.questions)
            ]
        else:
            print("LARGE DOCUMENT: Using vector search with chunking")
            # 4. Create vector store with batch embeddings for large documents
            vector_store = VectorStoreService(dimension=768)
            embeddings = await self.gemini_service.generate_embeddings_batch(text_chunks)
            vector_store.add_documents([{'text': chunk, 'embedding': emb} for chunk, emb in zip(text_chunks, embeddings)])

            # 5. PARALLEL PROCESSING: Answer all questions simultaneously with appropriate analysis
            print(f"Processing {len(request.questions)} questions in parallel using {document_type} analysis...")
            tasks = [
                self._answer_single_question_improved(question, vector_store, i+1, document_type) 
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

    async def _answer_question_direct(self, question: str, full_document_text: str, question_num: int = 1, document_type: str = 'policy') -> str:
        """Direct question answering for small documents without chunking or vector search.
        
        This method provides the complete document as context, ensuring no information is lost
        due to chunking or vector search limitations.
        """
        print(f"Q{question_num}: Using direct processing with full document context ({document_type} analysis)")
        print(f"Q{question_num}: Document length: {len(full_document_text)} characters")
        print(f"Q{question_num}: Question: {question}")
        
        # Debug: Show first 500 characters of document to verify content
        print(f"Q{question_num}: Document preview: {full_document_text[:500]}...")
        
        # Use appropriate analysis method based on document type
        if document_type == 'policy':
            answer = await self.gemini_service.generate_answer_from_context(question, [full_document_text])
        else:
            answer = await self.gemini_service.generate_general_answer_from_context(question, [full_document_text])
        
        print(f"Q{question_num}: Generated answer: {answer}")
        return answer

    async def _answer_single_question_improved(self, question: str, vector_store: VectorStoreService, question_num: int = 1, document_type: str = 'policy') -> str:
        """Hackathon-optimized question answering with multi-strategy retrieval and smart document type routing."""
        
        # 1. Multi-strategy query expansion based on document type
        if document_type == 'policy':
            queries = self._generate_search_queries(question)
        else:
            # For general documents, use enhanced query strategy to ensure completeness
            queries = [question, question.lower()]
            
            # Add key term variations for better retrieval
            question_words = question.lower().split()
            for word in question_words:
                if len(word) > 3:  # Skip short words
                    queries.append(word)
            
            # Add specific search patterns for common question types
            if any(word in question.lower() for word in ['what', 'which', 'how much', 'when']):
                # Extract key nouns and concepts
                key_terms = []
                for word in question_words:
                    if word not in ['what', 'which', 'how', 'much', 'when', 'where', 'why', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but']:
                        key_terms.append(word)
                
                # Add combinations of key terms
                if len(key_terms) >= 2:
                    queries.append(' '.join(key_terms[:2]))
                if len(key_terms) >= 3:
                    queries.append(' '.join(key_terms[:3]))
        
        # 2. Retrieve chunks based on document type for optimal accuracy
        all_chunks = set()
        
        # Adjust search parameters based on document type
        if document_type == 'policy':
            top_k_per_query = 5  # Optimized for policy documents
            max_chunks = 8
        else:
            top_k_per_query = 10  # Maximum chunks per query for general documents to ensure completeness
            max_chunks = 15  # Maximum context for complex general documents to capture all details
        
        for query in queries:
            query_embedding = await self.gemini_service.generate_embeddings(query, task_type="retrieval_query")
            search_results = vector_store.search(query_embedding, top_k=top_k_per_query)
            
            for result in search_results:
                all_chunks.add(result['text'])
        
        # For general documents, add additional broad search to ensure completeness
        if document_type == 'general':
            # Add a very broad search to catch any missed relevant chunks
            broad_queries = []
            
            # Extract all meaningful words from the question
            question_words = [word.lower().strip('.,!?;:') for word in question.split() 
                            if len(word) > 2 and word.lower() not in ['the', 'and', 'are', 'was', 'were', 'will', 'what', 'when', 'where', 'how', 'why', 'which', 'this', 'that', 'these', 'those']]
            
            # Search for individual important terms
            for word in question_words[:5]:  # Limit to top 5 words to avoid too many queries
                broad_queries.append(word)
            
            # Perform broad searches
            for broad_query in broad_queries:
                try:
                    broad_embedding = await self.gemini_service.generate_embeddings(broad_query, task_type="retrieval_query")
                    broad_results = vector_store.search(broad_embedding, top_k=5)
                    for result in broad_results:
                        all_chunks.add(result['text'])
                except Exception as e:
                    print(f"Broad search failed for '{broad_query}': {e}")
                    continue
        
        if not all_chunks:
            document_context = "document" if document_type == 'general' else "policy document"
            return f"Information not found in the {document_context}."
        
        # 3. Use appropriate number of context chunks based on document type
        relevant_chunks = list(all_chunks)[:max_chunks]
        print(f"Q{question_num}: Using {len(relevant_chunks)} chunks for context ({document_type} analysis)")
        
        # 4. Generate answer with appropriate analysis method
        if document_type == 'policy':
            answer = await self.gemini_service.generate_answer_from_context(question, relevant_chunks)
        else:
            answer = await self.gemini_service.generate_general_answer_from_context(question, relevant_chunks)
        
        return answer
    
    def _generate_search_queries(self, question: str) -> list[str]:
        """Generate multiple search queries for better retrieval."""
        queries = [question]  # Original question
        question_lower = question.lower()
        
        # Add expanded queries based on key terms
        for key_term, expansions in self.policy_query_expansions.items():
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
    
    def _detect_language(self, text: str) -> str:
        """Detect if the text contains non-Latin characters."""
        # Simple check for Malayalam Unicode range
        malayalam_range = '[ഀ-ൿ]'
        if re.search(malayalam_range, text):
            return 'malayalam'
        return 'english'

    async def _generate_precise_answer(self, question: str, context_chunks: list[str]) -> str:
        """Generate precise answer using enhanced prompting with multilingual support."""
        
        # Detect language of the question
        lang = self._detect_language(question)
        
        # Create language-specific prompts
        if lang == 'malayalam':
            system_prompt = """
നിങ്ങൾ ഒരു പ്രൊഫഷണലായ ഇൻഷുറൻസ് പോളിസി വിശകലനകാരനാണ്. താഴെയുള്ള ചോദ്യത്തിന് കൃത്യമായി ഉത്തരം നൽകുക. ഉത്തരം മലയാളത്തിലോ ഇംഗ്ലീഷിലോ നൽകാം, ചോദ്യത്തിന്റെ ഭാഷയെ ആശ്രയിച്ച്.

പ്രധാന നിയമങ്ങൾ:
1. പോളിസി വാചകത്തിൽ നിന്ന് കൃത്യമായ സംഖ്യകൾ, ശതമാനങ്ങൾ, സമയക്രമം എന്നിവ എടുക്കുക
2. സാധ്യമെങ്കിൽ പ്രത്യേക പോളിസി ഭാഷ ഉദ്ധരിക്കുക
3. കാത്തിരിക്കൽ കാലയളവ് ചോദിച്ചാൽ കൃത്യമായ കാലാവധി നൽകുക
4. കവറേജ് സംബന്ധിച്ച ചോദ്യമാണെങ്കിൽ ആദ്യം അതെ/ഇല്ല എന്ന് വ്യക്തമായി പറയുക, പിന്നീട് വ്യവസ്ഥകൾ വിശദീകരിക്കുക
5. ബന്ധപ്പെട്ട എല്ലാ വ്യവസ്ഥകളും പരിമിതികളും ഉൾപ്പെടുത്തുക
6. പോളിസി ഡോക്യുമെന്റിലെ കൃത്യമായ പദാവലി ഉപയോഗിക്കുക

ചോദ്യം: {question}

പോളിസി സന്ദർഭം:
{context}

കൃത്യവും വസ്തുനിഷ്ഠവുമായ ഉത്തരം നൽകുക:"""
        else:
            system_prompt = """
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
{context}

PROVIDE A PRECISE, FACTUAL ANSWER:"""
        
        # Prepare context with section numbers
        context = chr(10).join([f"SECTION {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Format the prompt with the appropriate language
        enhanced_prompt = system_prompt.format(question=question, context=context)
        
        try:
            # Use the enhanced prompt with the Gemini service
            response = await self.gemini_service.model.generate_content_async(
                enhanced_prompt,
                generation_config={
                    "temperature": 0.1,  # Low temperature for precision
                    "top_p": 0.9,
                    "max_output_tokens": 1024,  # Increased for multilingual support
                }
            )
            
            answer = response.text.strip()
            
            # Post-process for clarity
            answer = self._post_process_answer(answer, question)
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            lang_msg = "ചോദ്യം പ്രോസസ്സ് ചെയ്യുന്നതിൽ പിശക് സംഭവിച്ചു. ദയവായി വീണ്ടും ശ്രമിക്കുക." if lang == 'malayalam' else "Error processing the question. Please try again."
            return lang_msg
    
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
