import logging
from typing import List, Dict, Any, Optional

from database.vector_store import VectorStore
from services.embedding_service import EmbeddingService
from config.settings import settings

logger = logging.getLogger(__name__)

class VectorQueryInterface:
    """Handles vector-based queries by delegating to a vector database adapter."""

    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        """Initialize vector query interface."""
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    async def retrieve_similar_chunks(self, query: str, top_k: int = 10, score_threshold: float = 0.1, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve similar chunks by embedding the query and calling the adapter.
        """
        try:
            query_embedding = await self.embedding_service.embed_single_text(query)
            if not query_embedding:
                raise RuntimeError("Failed to generate query embedding")

            return await self.vector_store.search_similar(query_embedding, top_k, score_threshold, filters)
        except Exception as e:
            logger.error(f"Error retrieving similar chunks via vector store: {e}", exc_info=True)
            return []

    async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using the Cohere LLM based on the retrieved context.
        """
        if not context_chunks:
            return "No relevant information found in the vector store."

        context_text = "\n".join([chunk['text'] for chunk in context_chunks])
        prompt = f"""Based on the following context, answer the question:

Context:
{context_text}

Question: {query}

Answer:"""

        try:
            import cohere
            co = cohere.Client(api_key=settings.COHERE_API_KEY)
            response = co.chat(
                message=prompt,
                model=settings.COHERE_MODEL,
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {e}"

    async def query(self, user_query: str, top_k: int = 10, score_threshold: float = 0.1, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the complete vector query pipeline.
        """
        try:
            stats = await self.vector_store.get_collection_info()
            if stats.get('vectors_count', 0) == 0:
                logger.warning("Vector collection is empty")
                return {
                    'answer': "Vector database is empty. Please process a document first.",
                    'chunks': [], 'query': user_query, 'num_chunks': 0
                }

            chunks = await self.retrieve_similar_chunks(user_query, top_k, score_threshold, filters)
            if not chunks:
                return {
                    'answer': "No relevant information found in the vector store for the given query and filters.",
                    'chunks': [], 'query': user_query, 'num_chunks': 0
                }

            answer = await self.generate_answer(user_query, chunks)

            return {
                'answer': answer,
                'chunks': chunks,
                'query': user_query,
                'num_chunks': len(chunks)
            }
        except Exception as e:
            logger.error(f"Error in vector query pipeline: {e}", exc_info=True)
            return {
                'answer': f"Error processing vector query: {e}",
                'chunks': [], 'query': user_query, 'error': str(e), 'num_chunks': 0
            }

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection.
        """
        return await self.vector_store.get_collection_info()
