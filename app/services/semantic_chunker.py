"""
Semantic Chunking Service
Groups content by semantic similarity rather than arbitrary size limits
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import asyncio

# Try to import optional dependencies
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Using simplified chunking.")

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Using basic sentence splitting.")

# Download required NLTK data if available
if NLTK_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt')
        except:
            NLTK_AVAILABLE = False

@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk of text."""
    content: str
    sentences: List[str]
    topic_keywords: List[str]
    semantic_score: float
    start_sentence: int
    end_sentence: int

class SemanticChunker:
    """
    Advanced semantic chunking that groups sentences by semantic similarity.
    Uses sentence embeddings and clustering to create meaningful chunks.
    """
    
    def __init__(self, 
                 min_sentences_per_chunk: int = 3,
                 max_sentences_per_chunk: int = 15,
                 similarity_threshold: float = 0.3,
                 max_chunk_words: int = 500):
        """
        Initialize semantic chunker.
        
        Args:
            min_sentences_per_chunk: Minimum sentences in a chunk
            max_sentences_per_chunk: Maximum sentences in a chunk
            similarity_threshold: Minimum similarity to group sentences
            max_chunk_words: Maximum words per chunk
        """
        self.min_sentences = min_sentences_per_chunk
        self.max_sentences = max_sentences_per_chunk
        self.similarity_threshold = similarity_threshold
        self.max_words = max_chunk_words
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        else:
            self.vectorizer = None
    
    def chunk_text(self, text: str, domain: str = None) -> List[SemanticChunk]:
        """
        Create semantic chunks from text.
        
        Args:
            text: Input text to chunk
            domain: Domain context for better chunking
            
        Returns:
            List of semantic chunks
        """
        # 1. Split into sentences
        sentences = self._split_into_sentences(text)
        if len(sentences) < self.min_sentences:
            return [self._create_single_chunk(sentences, text)]
        
        # 2. Calculate sentence similarities
        similarity_matrix = self._calculate_sentence_similarities(sentences)
        
        # 3. Group sentences by semantic similarity
        sentence_groups = self._group_sentences_semantically(
            sentences, similarity_matrix, domain
        )
        
        # 4. Create semantic chunks
        chunks = []
        for i, group in enumerate(sentence_groups):
            chunk = self._create_semantic_chunk(group, sentences, i)
            if chunk:
                chunks.append(chunk)
        
        # 5. Post-process chunks for quality
        chunks = self._post_process_chunks(chunks)
        
        print(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into clean sentences."""
        # Clean text first
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Use NLTK if available, otherwise use basic splitting
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            # Basic sentence splitting using regex
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sent in sentences:
            sent = sent.strip()
            # Filter out very short or header-like sentences
            if (len(sent.split()) >= 3 and 
                not re.match(r'^(page|table|figure|chart|section)\s*\d*:?', sent.lower()) and
                not sent.isupper()):
                clean_sentences.append(sent)
        
        return clean_sentences
    
    def _calculate_sentence_similarities(self, sentences: List[str]):
        """Calculate semantic similarities between sentences."""
        if len(sentences) < 2:
            if SKLEARN_AVAILABLE:
                return np.array([[1.0]])
            else:
                return [[1.0]]
        
        if not SKLEARN_AVAILABLE:
            # Fallback: use word overlap similarity
            n = len(sentences)
            similarity_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        similarity_matrix[i][j] = 1.0
                    else:
                        # Calculate word overlap similarity
                        words_i = set(sentences[i].lower().split())
                        words_j = set(sentences[j].lower().split())
                        intersection = len(words_i & words_j)
                        union = len(words_i | words_j)
                        similarity_matrix[i][j] = intersection / union if union > 0 else 0.0
            
            return similarity_matrix
        
        try:
            # Create TF-IDF vectors for sentences
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Calculate cosine similarities
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix
        except Exception as e:
            print(f"Error calculating similarities: {e}")
            # Return identity matrix as fallback
            n = len(sentences)
            if SKLEARN_AVAILABLE:
                return np.eye(n)
            else:
                return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    def _group_sentences_semantically(self, 
                                    sentences: List[str], 
                                    similarity_matrix,
                                    domain: str = None) -> List[List[int]]:
        """Group sentences into semantic clusters."""
        n_sentences = len(sentences)
        visited = [False] * n_sentences
        groups = []
        
        for i in range(n_sentences):
            if visited[i]:
                continue
            
            # Start new group with current sentence
            current_group = [i]
            visited[i] = True
            
            # Find similar sentences to add to group
            for j in range(i + 1, min(i + self.max_sentences, n_sentences)):
                if visited[j]:
                    continue
                
                # Check if sentence j is similar to any sentence in current group
                max_similarity = max(similarity_matrix[k][j] for k in current_group)
                
                if (max_similarity >= self.similarity_threshold and 
                    len(current_group) < self.max_sentences):
                    
                    # Check word count constraint
                    group_words = sum(len(sentences[k].split()) for k in current_group)
                    new_words = len(sentences[j].split())
                    
                    if group_words + new_words <= self.max_words:
                        current_group.append(j)
                        visited[j] = True
            
            # Only keep groups with minimum sentences
            if len(current_group) >= self.min_sentences:
                groups.append(current_group)
            else:
                # Try to merge small group with next group or previous group
                if groups and len(groups[-1]) < self.max_sentences:
                    groups[-1].extend(current_group)
                else:
                    groups.append(current_group)
        
        return groups
    
    def _create_semantic_chunk(self, 
                             sentence_indices: List[int], 
                             sentences: List[str], 
                             chunk_id: int) -> Optional[SemanticChunk]:
        """Create a semantic chunk from sentence indices."""
        if not sentence_indices:
            return None
        
        # Get sentences for this chunk
        chunk_sentences = [sentences[i] for i in sentence_indices]
        content = ' '.join(chunk_sentences)
        
        # Extract topic keywords using TF-IDF
        topic_keywords = self._extract_topic_keywords(content)
        
        # Calculate semantic coherence score
        semantic_score = self._calculate_semantic_score(chunk_sentences)
        
        return SemanticChunk(
            content=content,
            sentences=chunk_sentences,
            topic_keywords=topic_keywords,
            semantic_score=semantic_score,
            start_sentence=min(sentence_indices),
            end_sentence=max(sentence_indices)
        )
    
    def _extract_topic_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract key topics/keywords from chunk text."""
        try:
            # Use TF-IDF to find important terms
            tfidf = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = tfidf.fit_transform([text])
            feature_names = tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            top_indices = scores.argsort()[-top_k:][::-1]
            keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            return keywords
        except:
            # Fallback: extract important words manually
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            return sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:top_k]
    
    def _calculate_semantic_score(self, sentences: List[str]) -> float:
        """Calculate semantic coherence score for a chunk."""
        if len(sentences) < 2:
            return 1.0
        
        try:
            # Calculate average similarity between sentences in chunk
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            similarities = cosine_similarity(tfidf_matrix)
            
            # Get average similarity (excluding diagonal)
            n = len(sentences)
            total_similarity = 0
            count = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    total_similarity += similarities[i][j]
                    count += 1
            
            return total_similarity / count if count > 0 else 0.0
        except:
            return 0.5  # Default score
    
    def _create_single_chunk(self, sentences: List[str], text: str) -> SemanticChunk:
        """Create a single chunk when text is too short to split."""
        return SemanticChunk(
            content=text,
            sentences=sentences,
            topic_keywords=self._extract_topic_keywords(text),
            semantic_score=1.0,
            start_sentence=0,
            end_sentence=len(sentences) - 1
        )
    
    def _post_process_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Post-process chunks for quality and consistency."""
        if not chunks:
            return chunks
        
        processed_chunks = []
        
        for chunk in chunks:
            # Filter out very short chunks
            if len(chunk.content.split()) < 10:
                continue
            
            # Clean up content
            content = re.sub(r'\s+', ' ', chunk.content).strip()
            
            # Update chunk with cleaned content
            chunk.content = content
            processed_chunks.append(chunk)
        
        # Merge very small chunks with adjacent ones
        final_chunks = []
        i = 0
        while i < len(processed_chunks):
            current_chunk = processed_chunks[i]
            
            # If chunk is small and there's a next chunk, try to merge
            if (len(current_chunk.content.split()) < 50 and 
                i + 1 < len(processed_chunks)):
                
                next_chunk = processed_chunks[i + 1]
                merged_content = current_chunk.content + " " + next_chunk.content
                
                if len(merged_content.split()) <= self.max_words:
                    # Merge chunks
                    merged_chunk = SemanticChunk(
                        content=merged_content,
                        sentences=current_chunk.sentences + next_chunk.sentences,
                        topic_keywords=current_chunk.topic_keywords + next_chunk.topic_keywords,
                        semantic_score=(current_chunk.semantic_score + next_chunk.semantic_score) / 2,
                        start_sentence=current_chunk.start_sentence,
                        end_sentence=next_chunk.end_sentence
                    )
                    final_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk as it's merged
                    continue
            
            final_chunks.append(current_chunk)
            i += 1
        
        return final_chunks

    def get_chunk_summaries(self, chunks: List[SemanticChunk]) -> List[str]:
        """Get brief summaries of each chunk for debugging/analysis."""
        summaries = []
        for i, chunk in enumerate(chunks):
            summary = (f"Chunk {i+1}: {len(chunk.sentences)} sentences, "
                      f"{len(chunk.content.split())} words, "
                      f"Score: {chunk.semantic_score:.3f}, "
                      f"Keywords: {', '.join(chunk.topic_keywords[:3])}")
            summaries.append(summary)
        return summaries
