import faiss
import numpy as np

class VectorStoreService:
    """Simple, effective vector store for insurance documents."""
    
    def __init__(self, dimension: int):
        """Initialize vector store."""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.document_chunks = []

    def add_documents(self, documents: list[dict]) -> None:
        """Add document chunks to the index."""
        if not documents:
            return

        # Store the text chunks in the same order
        self.document_chunks.extend([doc['text'] for doc in documents])

        # Extract embeddings and add to FAISS index
        embeddings = np.array([doc['embedding'] for doc in documents]).astype('float32')
        self.index.add(embeddings)

    def search(self, query_embedding: list, top_k: int) -> list[dict]:
        """Search for the most similar document chunks."""
        if self.index.ntotal == 0:
            return []

        # Convert query embedding to numpy array
        query_vector = np.array([query_embedding]).astype('float32')

        # Perform search
        distances, indices = self.index.search(query_vector, top_k)

        # Process results
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i != -1:  # FAISS returns -1 if no neighbors found
                results.append({
                    'text': self.document_chunks[i],
                    'score': float(dist)  # Lower distance = more similar
                })
        return results
