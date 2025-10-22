"""
Embedding service for generating vector representations using Cohere.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np

import cohere
from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using Cohere."""

    def __init__(self):
        """Initialize Cohere client."""
        self.client = cohere.Client(api_key=settings.COHERE_API_KEY)
        self.model = 'embed-english-v3.0'  # Cohere's latest embedding model

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return []

        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")

            # Cohere API call
            response = self.client.embed(
                texts=valid_texts,
                model=self.model,
                input_type='search_document'
            )

            embeddings = response.embeddings
            logger.info(f"Generated {len(embeddings)} embeddings")

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def embed_single_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_texts([text], input_type='search_query')
        return embeddings[0] if embeddings else []

    async def embed_texts(self, texts: List[str], input_type: str = 'search_document') -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            input_type: Type of input ('search_document' or 'search_query')

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return []

        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts with input_type={input_type}")

            # Cohere API call
            response = self.client.embed(
                texts=valid_texts,
                model=self.model,
                input_type=input_type
            )

            embeddings = response.embeddings
            logger.info(f"Generated {len(embeddings)} embeddings")

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Convert to numpy arrays
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    async def find_similar_texts(self, query: str, text_vectors: Dict[str, List[float]],
                               top_k: int = 5) -> List[tuple[str, float]]:
        """
        Find most similar texts to a query.

        Args:
            query: Query text
            text_vectors: Dict mapping text_id -> vector
            top_k: Number of similar texts to return

        Returns:
            List of (text_id, similarity_score) tuples
        """
        if not text_vectors:
            return []

        # Generate embedding for query
        query_embedding = await self.embed_single_text(query)
        if not query_embedding:
            return []

        # Calculate similarities
        similarities = []
        for text_id, vector in text_vectors.items():
            similarity = self.cosine_similarity(query_embedding, vector)
            similarities.append((text_id, similarity))

        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]