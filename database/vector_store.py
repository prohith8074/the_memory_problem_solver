"""
Qdrant vector database integration for storing and retrieving embeddings.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Match
from qdrant_client.http.exceptions import UnexpectedResponse

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Qdrant vector database wrapper."""

    def __init__(self, url: str = None, api_key: str = None):
        """Initialize Qdrant client."""
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection_name = settings.QDRANT_COLLECTION_NAME

        # Initialize client - use in-memory if localhost or no connection
        try:
            if self.url == "http://localhost:6333" or not self.url:
                # Use in-memory Qdrant for development
                self.client = QdrantClient(":memory:")
                logger.info("Using in-memory Qdrant client")
            elif self.api_key:
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
            else:
                self.client = QdrantClient(url=self.url)
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant at {self.url}, falling back to in-memory: {e}")
            self.client = QdrantClient(":memory:")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure the collection exists with correct configuration."""
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except (UnexpectedResponse, ValueError):
            # Create collection if it doesn't exist
            logger.info(f"Creating collection '{self.collection_name}'")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.QDRANT_VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )

            # Create payload indexes for filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.type",
                field_schema="keyword"
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.source",
                field_schema="keyword"
            )

    def _build_filter(self, filter_conditions: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from conditions."""
        conditions = []
        for field, value in filter_conditions.items():
            if field.startswith('metadata.'):
                field_name = field.split('.', 1)[1]
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{field_name}",
                        match=Match(value=value)
                    )
                )
        return Filter(should=conditions) if conditions else None

    async def store_chunks(self, chunks: List[Dict[str, Any]],
                         embeddings: List[List[float]]) -> bool:
        """
        Store text chunks with their embeddings in Qdrant.

        Args:
            chunks: List of chunk dictionaries
            embeddings: Corresponding embedding vectors

        Returns:
            True if successful
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        try:
            points = []

            for chunk, embedding in zip(chunks, embeddings):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        'text': chunk['content'],
                        'chunk_id': chunk['id'],
                        'metadata': chunk['metadata']
                    }
                )
                points.append(point)

            # Batch upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True  # Wait for completion
            )

            # Give Qdrant a moment to index
            time.sleep(2)

            logger.info(f"Stored {len(points)} chunks in vector store")
            return True

        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            return False
    

    async def search_similar(self, query_embedding: List[float],
                           limit: int = 5,
                           score_threshold: float = 0.1,
                           filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors with improved error handling.
        """
        try:
            # FIXED: Check if collection has vectors first
            collection_info = self.client.get_collection(self.collection_name)
            
            if collection_info.points_count == 0:
                logger.warning("Vector store is empty! No documents indexed yet.")
                return []
            
            logger.info(f"Searching in collection with {collection_info.points_count} vectors")
            
            # FIXED: Lower default threshold to 0.0 to get more results
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=max(0.0, score_threshold),  # Ensure non-negative
                query_filter=None if not filter_conditions else self._build_filter(filter_conditions)
            )
            
            # FIXED: Add detailed logging
            if not search_results:
                logger.warning(f"No results found for query. Collection has {collection_info.points_count} vectors")
                logger.warning(f"Query embedding dimension: {len(query_embedding)}")
                logger.warning(f"Score threshold: {score_threshold}")
            
            results = []
            for result in search_results:
                results.append({
                    'id': result.id,
                    'score': result.score,
                    'text': result.payload.get('text', ''),
                    'chunk_id': result.payload.get('chunk_id', ''),
                    'metadata': result.payload.get('metadata', {})
                })
            
            logger.info(f"Found {len(results)} similar chunks with scores: {[r['score'] for r in results]}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}", exc_info=True)
            return []

    # async def get_collection_info(self) -> Dict[str, Any]:
    #     """Get information about the collection."""
    #     try:
    #         collection_info = self.client.get_collection(self.collection_name)
    #         return {
    #             'name': self.collection_name,
    #             'vectors_count': collection_info.points_count,
    #             'status': 'active'
    #         }
    #     except Exception as e:
    #         logger.error(f"Error getting collection info: {e}")
    #         return {'status': 'error', 'error': str(e)}
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection with proper attribute access."""
        try:
            collection_info = self.client.get_collection(self.collection_name)

            # FIXED: Handle optimizer_status properly - it's a union type
            optimizer_status = 'unknown'
            if hasattr(collection_info, 'optimizer_status') and collection_info.optimizer_status is not None:
                # Try to get the status from the union type
                if hasattr(collection_info.optimizer_status, 'status'):
                    optimizer_status = collection_info.optimizer_status.status
                else:
                    # If it's a union type, try to get the actual status value
                    optimizer_status = str(collection_info.optimizer_status)

            return {
                'name': self.collection_name,  # Use stored name instead of attribute
                'vectors_count': collection_info.points_count,
                'vectors_size': getattr(collection_info.config.params.vectors, 'size', settings.QDRANT_VECTOR_SIZE),
                'status': getattr(collection_info, 'status', 'unknown'),
                'optimizer_status': optimizer_status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}", exc_info=True)
            return {
                'name': self.collection_name,
                'status': 'error',
                'error': str(e),
                'vectors_count': 0
            }


    async def clear_collection(self) -> bool:
        """Clear all vectors from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()  # Recreate empty collection
            logger.info("Collection cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

    def health_check(self) -> bool:
        """Check if the vector store is accessible."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False