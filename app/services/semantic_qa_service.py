"""
Semantic Q&A Service
Enhanced Q&A service using semantic chunking for better context retrieval
"""

from typing import Dict, List, Optional, Tuple
import asyncio
import json

from .gemini_service import GeminiPolicyProcessor
from .enhanced_document_processor import EnhancedDocumentProcessor
from .vector_store_service import VectorStoreService
from .response_validator import ResponseValidator
from .semantic_chunker import SemanticChunker, SemanticChunk


class SemanticQAService:
    """Enhanced Q&A service with semantic chunking capabilities."""
    
    def __init__(self):
        self.gemini_service = GeminiPolicyProcessor()
        self.document_processor = EnhancedDocumentProcessor(use_semantic_chunking=True)
        self.response_validator = ResponseValidator()
        self.semantic_chunker = SemanticChunker()
    
    async def process_document_and_answer(self, document: dict, question: str) -> dict:
        """
        Process document with semantic chunking and answer question.
        
        Args:
            document: Document data with content and metadata
            question: User question
            
        Returns:
            Enhanced response with semantic context
        """
        try:
            print(f"Processing document with semantic chunking...")
            
            # 1. Process document with semantic chunking and domain classification
            processed_doc = self.document_processor.process_document_with_domain(document)
            
            if not processed_doc.chunks:
                return {
                    "answer": "Unable to process the document. Please check the document format.",
                    "confidence": 0.0,
                    "domain": "unknown",
                    "chunks_used": 0,
                    "semantic_score": 0.0
                }
            
            print(f"Domain: {processed_doc.domain_classification.primary_domain.value}")
            print(f"Confidence: {processed_doc.domain_classification.confidence:.3f}")
            print(f"Created {len(processed_doc.chunks)} semantic chunks")
            
            # 2. For now, skip vector store creation (will be enhanced later)
            # vector_store = VectorStoreService()
            # This will be implemented once embedding methods are available
            
            # 3. Enhanced question processing with domain context
            enhanced_question = await self._enhance_question_with_domain(
                question, processed_doc.domain_classification
            )
            
            # 4. Retrieve most relevant semantic chunks
            relevant_chunks = await self._retrieve_semantic_chunks(
                enhanced_question, None, processed_doc
            )
            
            if not relevant_chunks:
                return {
                    "answer": "No relevant information found in the document for your question.",
                    "confidence": 0.0,
                    "domain": processed_doc.domain_classification.primary_domain.value,
                    "chunks_used": 0,
                    "semantic_score": 0.0
                }
            
            print(f"Using {len(relevant_chunks)} semantic chunks for context")
            
            # 5. Generate answer with semantic context
            answer = await self._generate_semantic_answer(
                question, relevant_chunks, processed_doc.domain_classification
            )
            
            # 6. Validate response quality
            validation = self.response_validator.validate_response(
                question, answer, [chunk['content'] for chunk in relevant_chunks]
            )
            
            # 7. Calculate semantic coherence score
            semantic_score = self._calculate_semantic_coherence(relevant_chunks)
            
            return {
                "answer": answer,
                "confidence": validation.overall_confidence,
                "domain": processed_doc.domain_classification.primary_domain.value,
                "domain_confidence": processed_doc.domain_classification.confidence,
                "chunks_used": len(relevant_chunks),
                "semantic_score": semantic_score,
                "validation": {
                    "factual_accuracy": validation.factual_accuracy,
                    "context_relevance": validation.context_relevance,
                    "completeness": validation.completeness
                },
                "chunk_topics": [chunk.get('topics', []) for chunk in relevant_chunks]
            }
            
        except Exception as e:
            print(f"Error in semantic Q&A processing: {e}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "confidence": 0.0,
                "domain": "unknown",
                "chunks_used": 0,
                "semantic_score": 0.0
            }
    
    async def _enhance_question_with_domain(self, 
                                          question: str, 
                                          domain_classification) -> str:
        """Enhance question with domain-specific context."""
        domain = domain_classification.primary_domain.value
        confidence = domain_classification.confidence
        
        if confidence < 0.7:
            return question  # Don't modify if domain classification is uncertain
        
        # Add domain-specific context to improve retrieval
        domain_contexts = {
            "insurance": "In the context of insurance policies, coverage, claims, and benefits",
            "legal": "In the legal context of laws, regulations, compliance, and legal procedures",
            "hr": "In the context of human resources, employee policies, and workplace procedures",
            "compliance": "In the context of regulatory compliance, standards, and requirements"
        }
        
        context = domain_contexts.get(domain, "")
        if context:
            enhanced_question = f"{context}: {question}"
            print(f"Enhanced question with {domain} context")
            return enhanced_question
        
        return question
    
    async def _retrieve_semantic_chunks(self, 
                                       question: str, 
                                       vector_store: VectorStoreService,
                                       processed_doc) -> List[Dict]:
        """Retrieve most relevant semantic chunks using simplified approach."""
        
        # For now, use a simplified approach without embeddings
        # This will be enhanced once we have the embedding methods available
        
        # Use the first few chunks as relevant chunks for demonstration
        relevant_chunks = []
        for i, chunk_text in enumerate(processed_doc.chunks[:8]):
            # Extract topics from chunk using semantic chunker
            chunk_topics = self.semantic_chunker._extract_topic_keywords(chunk_text, top_k=3)
            
            relevant_chunks.append({
                'content': chunk_text,
                'topics': chunk_topics,
                'word_count': len(chunk_text.split())
            })
        
        # Sort by word count (longer chunks first)
        relevant_chunks.sort(
            key=lambda x: x['word_count'], 
            reverse=True
        )
        
        return relevant_chunks[:6]  # Return top 6 most relevant chunks
    
    async def _generate_alternative_questions(self, question: str) -> List[str]:
        """Generate alternative phrasings of the question for better retrieval."""
        prompt = f"""
        Generate 2-3 alternative ways to ask this question that might help find relevant information:
        
        Original question: {question}
        
        Provide alternative phrasings that:
        1. Use different keywords
        2. Are more specific or more general
        3. Focus on different aspects of the same topic
        
        Format as a simple list, one per line.
        """
        
        try:
            response = await self.gemini_service.generate_content(prompt)
            alternatives = [line.strip() for line in response.split('\n') if line.strip() and not line.startswith('-')]
            return alternatives[:3]  # Limit to 3 alternatives
        except:
            return []  # Return empty list if generation fails
    
    async def _generate_semantic_answer(self, 
                                       question: str, 
                                       semantic_chunks: List[Dict],
                                       domain_classification) -> str:
        """Generate answer using semantic chunks with domain awareness."""
        
        domain = domain_classification.primary_domain.value
        
        # Prepare context with semantic information
        context_sections = []
        for i, chunk in enumerate(semantic_chunks):
            topics_str = ", ".join(chunk['topics']) if chunk['topics'] else "general"
            context_sections.append(
                f"SEMANTIC SECTION {i+1} (Topics: {topics_str}):\n{chunk['content']}"
            )
        
        context_text = "\n\n".join(context_sections)
        
        # Domain-specific prompting
        domain_instructions = {
            "insurance": "Focus on policy terms, coverage details, claims procedures, and benefits. Be precise about conditions and exclusions.",
            "legal": "Provide accurate legal information, cite relevant sections, and be clear about legal requirements and procedures.",
            "hr": "Focus on employee policies, procedures, and workplace guidelines. Be clear about requirements and processes.",
            "compliance": "Emphasize regulatory requirements, standards, and compliance procedures. Be specific about obligations."
        }
        
        domain_instruction = domain_instructions.get(domain, "Provide accurate and helpful information based on the document content.")
        
        prompt = f"""
        You are an expert assistant specializing in {domain} domain. {domain_instruction}
        
        Based on the following semantic sections from the document, answer the user's question accurately and comprehensively.
        
        QUESTION: {question}
        
        SEMANTIC CONTEXT:
        {context_text}
        
        INSTRUCTIONS:
        1. Answer directly and specifically based on the provided context
        2. Reference relevant sections when appropriate
        3. If information is incomplete, clearly state what's missing
        4. Use the semantic topics to ensure comprehensive coverage
        5. Maintain {domain} domain expertise in your response
        6. Be concise but thorough
        
        ANSWER:
        """
        
        return await self.gemini_service.generate_answer_from_context(question, [chunk['content'] for chunk in semantic_chunks])
    
    def _calculate_semantic_coherence(self, chunks: List[Dict]) -> float:
        """Calculate semantic coherence score for the retrieved chunks."""
        if not chunks:
            return 0.0
        
        try:
            # Calculate based on topic overlap and content similarity
            all_topics = []
            for chunk in chunks:
                all_topics.extend(chunk.get('topics', []))
            
            if not all_topics:
                return 0.5
            
            # Calculate topic diversity (more diverse topics = lower coherence)
            unique_topics = len(set(all_topics))
            total_topics = len(all_topics)
            
            # Calculate coherence score (0.0 to 1.0)
            if total_topics == 0:
                return 0.5
            
            topic_coherence = 1.0 - (unique_topics / total_topics)
            
            # Adjust based on number of chunks (more chunks = potentially less coherent)
            chunk_factor = min(1.0, 8.0 / len(chunks))
            
            semantic_score = (topic_coherence + chunk_factor) / 2
            return max(0.0, min(1.0, semantic_score))
            
        except Exception as e:
            print(f"Error calculating semantic coherence: {e}")
            return 0.5
    
    async def analyze_document_semantics(self, document: dict) -> Dict:
        """Analyze document semantic structure for debugging/insights."""
        try:
            processed_doc = self.document_processor.process_document_with_domain(document)
            
            # Get semantic chunk analysis
            semantic_chunks = self.semantic_chunker.chunk_text(
                processed_doc.chunks[0] if processed_doc.chunks else ""
            )
            
            chunk_summaries = self.semantic_chunker.get_chunk_summaries(semantic_chunks)
            
            return {
                "domain": processed_doc.domain_classification.primary_domain.value,
                "domain_confidence": processed_doc.domain_classification.confidence,
                "total_chunks": len(processed_doc.chunks),
                "semantic_chunks": len(semantic_chunks),
                "chunk_summaries": chunk_summaries,
                "avg_semantic_score": sum(chunk.semantic_score for chunk in semantic_chunks) / len(semantic_chunks) if semantic_chunks else 0.0
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
