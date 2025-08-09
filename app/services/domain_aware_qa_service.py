"""
Domain-Aware QA Service for Intelligent Query-Retrieval System
Routes queries to appropriate domain-specific documents for better accuracy
"""

import asyncio
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from app.services.enhanced_document_processor import EnhancedDocumentProcessor, ProcessedDocument
from app.services.domain_classifier import DomainClassifier, Domain, DomainClassification
from app.services.gemini_service import GeminiPolicyProcessor
from app.services.vector_store_service import VectorStoreService
from app.api.schemas.evaluation import HackRxRequest

@dataclass
class DomainAwareAnswer:
    """Answer with domain context."""
    answer: str
    source_domain: Domain
    confidence: float
    sources_used: List[str]
    domain_match_quality: str

class DomainAwareQAService:
    """Enhanced QA Service with domain-aware query routing."""
    
    def __init__(self):
        self.document_processor = EnhancedDocumentProcessor()
        self.domain_classifier = DomainClassifier()
        self.gemini_service = GeminiPolicyProcessor()
        
        # Store processed documents by domain
        self.processed_documents: List[ProcessedDocument] = []
        self.domain_vector_stores: Dict[Domain, VectorStoreService] = {}
        self.document_index: Dict[Domain, List[int]] = {}
        
        # Domain-specific query enhancement patterns
        self.domain_query_patterns = {
            Domain.INSURANCE: {
                'coverage_questions': [
                    r'\b(?:cover|covered|coverage|include|eligible|benefit)\b',
                    r'\b(?:does.*cover|is.*covered|what.*covered)\b'
                ],
                'period_questions': [
                    r'\b(?:waiting period|grace period|how long|when.*eligible)\b',
                    r'\b(?:after.*months|immediate.*coverage)\b'
                ],
                'amount_questions': [
                    r'\b(?:how much|amount|cost|premium|limit|maximum)\b',
                    r'\b(?:percentage|percent|%|deductible)\b'
                ]
            },
            Domain.LEGAL: {
                'rights_questions': [
                    r'\b(?:right|rights|fundamental rights|constitutional rights)\b',
                    r'\b(?:can.*do|allowed to|permitted to)\b'
                ],
                'procedure_questions': [
                    r'\b(?:how to|procedure|process|steps|method)\b',
                    r'\b(?:court|legal process|filing)\b'
                ],
                'definition_questions': [
                    r'\b(?:what is|define|definition|meaning|means)\b',
                    r'\b(?:constitutional|legal term)\b'
                ]
            },
            Domain.HR: {
                'policy_questions': [
                    r'\b(?:policy|procedure|guideline|rule)\b',
                    r'\b(?:employee.*policy|HR.*policy)\b'
                ],
                'leave_questions': [
                    r'\b(?:leave|vacation|sick.*leave|maternity.*leave)\b',
                    r'\b(?:time off|absence|holiday)\b'
                ],
                'performance_questions': [
                    r'\b(?:performance|appraisal|review|evaluation)\b',
                    r'\b(?:promotion|salary|compensation)\b'
                ]
            },
            Domain.COMPLIANCE: {
                'regulatory_questions': [
                    r'\b(?:compliance|regulatory|regulation|requirement)\b',
                    r'\b(?:audit|governance|control)\b'
                ],
                'risk_questions': [
                    r'\b(?:risk|risk management|assessment|mitigation)\b'
                ],
                'reporting_questions': [
                    r'\b(?:report|reporting|disclosure|documentation)\b'
                ]
            }
        }

    async def process_documents_with_domain_awareness(self, documents: List[dict]) -> None:
        """Process multiple documents with domain classification."""
        self.processed_documents = []
        
        for doc in documents:
            processed_doc = self.document_processor.process_document_with_domain(doc)
            self.processed_documents.append(processed_doc)
            
            print(f"Document '{processed_doc.filename}' classified as: {processed_doc.domain_classification.primary_domain.value} "
                  f"(confidence: {processed_doc.domain_classification.confidence:.2f})")
        
        # Create domain index
        self.document_index = self.document_processor.create_document_index(self.processed_documents)
        
        # Create domain-specific vector stores
        await self._create_domain_vector_stores()

    async def _create_domain_vector_stores(self) -> None:
        """Create separate vector stores for each domain."""
        self.domain_vector_stores = {}
        
        for domain in Domain:
            if domain in self.document_index and self.document_index[domain]:
                # Collect all chunks for this domain
                domain_chunks = []
                for doc_idx in self.document_index[domain]:
                    doc = self.processed_documents[doc_idx]
                    domain_chunks.extend(doc.chunks)
                
                if domain_chunks:
                    # Create vector store for this domain
                    vector_store = VectorStoreService(dimension=768)
                    embeddings = await self.gemini_service.generate_embeddings_batch(domain_chunks)
                    
                    vector_store.add_documents([
                        {'text': chunk, 'embedding': emb} 
                        for chunk, emb in zip(domain_chunks, embeddings)
                    ])
                    
                    self.domain_vector_stores[domain] = vector_store
                    print(f"Created vector store for {domain.value} with {len(domain_chunks)} chunks")

    async def answer_questions_with_domain_routing(self, request: HackRxRequest) -> List[DomainAwareAnswer]:
        """Answer questions with domain-aware routing."""
        if not self.processed_documents:
            # Fallback to single document processing
            document_dict = {
                'content': request.documents, 
                'metadata': {'filename': request.documents.split('/')[-1].split('?')[0]}
            }
            await self.process_documents_with_domain_awareness([document_dict])
        
        answers = []
        for i, question in enumerate(request.questions):
            print(f"Processing question {i+1}: {question}")
            answer = await self._answer_single_question_with_domain_routing(question)
            answers.append(answer)
        
        return answers

    async def _answer_single_question_with_domain_routing(self, question: str) -> DomainAwareAnswer:
        """Answer a single question with domain routing."""
        
        # 1. Classify the query domain
        query_classification = self.domain_classifier.classify_query(question)
        query_domain = query_classification.primary_domain
        
        print(f"Query classified as: {query_domain.value} (confidence: {query_classification.confidence:.2f})")
        
        # 2. Find relevant documents for this domain
        relevant_docs = self._get_relevant_documents_for_domain(query_domain)
        
        if not relevant_docs:
            return DomainAwareAnswer(
                answer="No relevant documents found for this query domain.",
                source_domain=query_domain,
                confidence=0.0,
                sources_used=[],
                domain_match_quality="no_match"
            )
        
        # 3. Get domain-specific vector store
        if query_domain not in self.domain_vector_stores:
            return DomainAwareAnswer(
                answer="No indexed content found for this domain.",
                source_domain=query_domain,
                confidence=0.0,
                sources_used=[],
                domain_match_quality="no_index"
            )
        
        vector_store = self.domain_vector_stores[query_domain]
        
        # 4. Generate domain-enhanced queries
        enhanced_queries = self._generate_domain_enhanced_queries(question, query_domain)
        
        # 5. Retrieve relevant chunks
        all_chunks = set()
        for query in enhanced_queries:
            query_embedding = await self.gemini_service.generate_embeddings(query, task_type="retrieval_query")
            search_results = vector_store.search(query_embedding, top_k=6)
            
            for result in search_results:
                all_chunks.add(result['text'])
        
        if not all_chunks:
            return DomainAwareAnswer(
                answer="No relevant information found in the domain-specific documents.",
                source_domain=query_domain,
                confidence=0.0,
                sources_used=[],
                domain_match_quality="no_content"
            )
        
        # 6. Generate domain-specific answer
        relevant_chunks = list(all_chunks)[:10]  # Use top 10 chunks
        answer = await self._generate_domain_specific_answer(question, relevant_chunks, query_domain)
        
        # 7. Calculate confidence and sources
        confidence = min(query_classification.confidence + 0.2, 1.0)  # Boost confidence for domain routing
        sources_used = [doc.filename for doc in relevant_docs]
        
        return DomainAwareAnswer(
            answer=answer,
            source_domain=query_domain,
            confidence=confidence,
            sources_used=sources_used,
            domain_match_quality="good_match" if query_classification.confidence > 0.5 else "partial_match"
        )

    def _get_relevant_documents_for_domain(self, domain: Domain) -> List[ProcessedDocument]:
        """Get documents relevant to the specified domain."""
        if domain not in self.document_index:
            return []
        
        relevant_docs = []
        for doc_idx in self.document_index[domain]:
            relevant_docs.append(self.processed_documents[doc_idx])
        
        return relevant_docs

    def _generate_domain_enhanced_queries(self, question: str, domain: Domain) -> List[str]:
        """Generate domain-specific enhanced queries."""
        base_queries = [question]
        
        # Add domain-specific enhancements
        domain_keywords = self.domain_classifier.get_domain_specific_keywords(domain)
        enhanced_queries = self.domain_classifier.enhance_query_for_domain(question, domain)
        
        # Add pattern-based enhancements
        if domain in self.domain_query_patterns:
            question_lower = question.lower()
            patterns = self.domain_query_patterns[domain]
            
            for pattern_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, question_lower):
                        # Add domain-specific query variations
                        if domain == Domain.INSURANCE:
                            if 'coverage' in pattern_type:
                                enhanced_queries.extend([
                                    f"{question} policy benefits",
                                    f"{question} insurance coverage",
                                    f"{question} claim eligibility"
                                ])
                            elif 'period' in pattern_type:
                                enhanced_queries.extend([
                                    f"{question} waiting time",
                                    f"{question} eligibility period",
                                    f"{question} coverage starts"
                                ])
                        
                        elif domain == Domain.LEGAL:
                            if 'rights' in pattern_type:
                                enhanced_queries.extend([
                                    f"{question} constitutional provision",
                                    f"{question} legal rights",
                                    f"{question} fundamental rights"
                                ])
                            elif 'procedure' in pattern_type:
                                enhanced_queries.extend([
                                    f"{question} legal process",
                                    f"{question} court procedure",
                                    f"{question} legal steps"
                                ])
        
        return list(set(enhanced_queries))  # Remove duplicates

    async def _generate_domain_specific_answer(self, question: str, context_chunks: List[str], 
                                             domain: Domain) -> str:
        """Generate answer with domain-specific prompting."""
        
        # Domain-specific prompt templates
        domain_prompts = {
            Domain.INSURANCE: """
You are an expert insurance policy analyst. Answer the following question with MAXIMUM PRECISION using ONLY the provided insurance policy context.

IMPORTANT RULES:
1. Extract EXACT numbers, percentages, and time periods from the policy text
2. Quote specific policy language when possible
3. If asking about waiting periods, provide the EXACT duration
4. If asking about coverage, state YES/NO clearly first, then explain conditions
5. Include ALL relevant conditions and limitations
6. Use the EXACT terminology from the policy document
7. Focus on insurance-specific terms like premiums, deductibles, coverage limits, exclusions
""",
            Domain.LEGAL: """
You are an expert legal analyst specializing in constitutional and statutory law. Answer the following question with MAXIMUM PRECISION using ONLY the provided legal context.

IMPORTANT RULES:
1. Quote exact legal provisions and article numbers
2. Cite specific sections, clauses, or amendments
3. Use precise legal terminology
4. Explain the legal framework and hierarchy
5. Include relevant constitutional principles
6. Reference specific legal authorities or precedents mentioned
7. Focus on rights, duties, procedures, and legal requirements
""",
            Domain.HR: """
You are an expert HR policy analyst. Answer the following question with MAXIMUM PRECISION using ONLY the provided HR policy context.

IMPORTANT RULES:
1. Extract exact policy numbers, timeframes, and procedures
2. Quote specific policy language and guidelines
3. Include all relevant conditions and eligibility criteria
4. Reference specific forms, processes, or approval requirements
5. Use exact HR terminology from the documents
6. Focus on employee rights, responsibilities, and procedures
7. Include any relevant escalation or appeal processes
""",
            Domain.COMPLIANCE: """
You are an expert compliance and regulatory analyst. Answer the following question with MAXIMUM PRECISION using ONLY the provided compliance context.

IMPORTANT RULES:
1. Extract exact regulatory requirements and standards
2. Quote specific compliance provisions and control numbers
3. Include all relevant regulatory obligations
4. Reference specific frameworks, standards, or guidelines
5. Use precise compliance and regulatory terminology
6. Focus on requirements, controls, monitoring, and reporting
7. Include any relevant penalties or consequences for non-compliance
""",
            Domain.GENERAL: """
You are an expert document analyst. Answer the following question with MAXIMUM PRECISION using ONLY the provided context.

IMPORTANT RULES:
1. Extract exact information from the provided text
2. Quote specific passages when relevant
3. Use precise terminology from the source documents
4. Include all relevant conditions and limitations
5. Provide clear, factual answers based solely on the given context
"""
        }
        
        prompt_template = domain_prompts.get(domain, domain_prompts[Domain.GENERAL])
        
        enhanced_prompt = f"""{prompt_template}

QUESTION: {question}

CONTEXT:
{chr(10).join([f"SECTION {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])}

PROVIDE A PRECISE, FACTUAL ANSWER:"""
        
        try:
            response = await self.gemini_service.model.generate_content_async(
                enhanced_prompt,
                generation_config={
                    "temperature": 0.1,  # Low temperature for precision
                    "top_p": 0.9,
                    "max_output_tokens": 512,
                }
            )
            
            answer = response.text.strip()
            answer = self._post_process_domain_answer(answer, question, domain)
            
            return answer
            
        except Exception as e:
            print(f"Error generating domain-specific answer: {e}")
            return f"Error processing the {domain.value} question. Please try again."

    def _post_process_domain_answer(self, answer: str, question: str, domain: Domain) -> str:
        """Post-process answer with domain-specific formatting."""
        
        # Clean up formatting
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Ensure proper capitalization
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        # Add period if missing
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        # Domain-specific validations
        if domain == Domain.INSURANCE:
            # Validate insurance-specific responses
            if any(word in question.lower() for word in ['period', 'months', 'years', 'days']):
                if not re.search(r'\d+', answer):
                    answer += " (Please refer to the specific policy document for exact time periods.)"
            
            if 'cover' in question.lower() and not any(word in answer.lower() for word in ['yes', 'no', 'covered', 'not covered']):
                answer = "Coverage determination: " + answer
        
        elif domain == Domain.LEGAL:
            # Validate legal responses
            if 'right' in question.lower() and not any(word in answer.lower() for word in ['article', 'section', 'constitution']):
                answer += " (Refer to specific constitutional provisions for complete details.)"
        
        return answer

    def get_domain_statistics(self) -> Dict[str, any]:
        """Get statistics about processed documents by domain."""
        stats = {
            'total_documents': len(self.processed_documents),
            'domain_distribution': {},
            'vector_stores_created': len(self.domain_vector_stores),
            'domains_with_content': list(self.domain_vector_stores.keys())
        }
        
        for domain in Domain:
            count = len(self.document_index.get(domain, []))
            if count > 0:
                stats['domain_distribution'][domain.value] = count
        
        return stats
