from .document_processor import DocumentProcessor
from .gemini_service import GeminiPolicyProcessor as GeminiService
from .vector_store_service import VectorStoreService
import traceback

class PolicyEvalPipeline:
    async def process_request(self, request_data: dict):
        """
        Runs the end-to-end evaluation pipeline asynchronously.
        """
        query_data = request_data.get("query", {})
        documents = request_data.get("documents", [])

        if not query_data or not documents:
            return {"error": "Query and documents must be provided."}

        # Combine raw_text and structured_data for a more comprehensive query
        raw_query = query_data.get("raw_text", "")
        structured_query = query_data.get("structured_data", {})
        
        combined_query_parts = []
        if raw_query:
            combined_query_parts.append(raw_query)
        
        if structured_query:
            structured_text = ", ".join([f"{k.replace('_', ' ')} is {v}" for k, v in structured_query.items() if v is not None])
            if structured_text:
                combined_query_parts.append(structured_text)

        if not combined_query_parts:
            return {"error": "A query must be provided in either raw_text or structured_data."}
        
        combined_query = ". ".join(combined_query_parts)

        # 1. Initialize services for this request
        doc_processor = DocumentProcessor()
        gemini_service = GeminiService()
        vector_store = VectorStoreService(dimension=768)

        try:
            # 2. Extract structured entities from the query
            extracted_entities = await gemini_service.extract_entities(combined_query)

            # 3. Process all documents to get text chunks
            all_chunks = []
            for doc in documents:
                try:
                    # The document format now includes metadata
                    chunks = doc_processor.process_document(doc)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing document: {doc.get('filename', 'unknown')}")
                    traceback.print_exc()
                    pass

            if not all_chunks:
                return {"error": "No content could be extracted from the provided documents."}

            # 4. Generate embeddings for all chunks in a single batch for efficiency
            embeddings = await gemini_service.generate_embeddings_batch(all_chunks)
            documents_to_add = [{'text': chunk, 'embedding': emb} for chunk, emb in zip(all_chunks, embeddings)]
            vector_store.add_documents(documents_to_add)

            # 5. Perform semantic search for relevant clauses
            query_embedding = gemini_service.generate_embeddings(combined_query, task_type="retrieval_query")
            search_results = vector_store.search(query_embedding=query_embedding, top_k=5)
            relevant_clauses = [match['text'] for match in search_results]

            # 6. Analyze relevant clauses with Gemini
            analyzed_clauses = await gemini_service.analyze_policy_clauses(
                query=extracted_entities, 
                document_chunks=relevant_clauses
            )

            # 7. Generate the final decision with Gemini
            final_decision = gemini_service.final_decision_reasoning(extracted_entities, analyzed_clauses)

            # 8. Combine all results for the final response
            final_decision['entities'] = extracted_entities
            final_decision['analysis'] = analyzed_clauses

            return final_decision
        except Exception as e:
            print(f"An unexpected error occurred in the evaluation pipeline: {e}")
            traceback.print_exc()
            raise
