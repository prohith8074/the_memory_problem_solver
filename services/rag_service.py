"""
Main RAG service that orchestrates all components.
Coordinates document processing, vector storage, and query processing.
"""

import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Union

from core.document_processor import DocumentProcessor
from services.embedding_service import EmbeddingService
from database.vector_store import VectorStore
from services.query_router import QueryRouter
from services.memory_service import MemoryService
from services.vector_query_interface import VectorQueryInterface

logger = logging.getLogger(__name__)


class RAGService:
    """Main RAG service orchestrator."""

    def __init__(self):
        """Initialize all RAG components."""
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
        self.vector_query_interface = VectorQueryInterface(self.vector_store, self.embedding_service)
        self.query_router = QueryRouter(self.vector_query_interface)
        self.memory_service = MemoryService()

        # Service state
        self.processed_documents = {}
        self.is_ready = False

    async def initialize(self) -> bool:
        """Initialize the RAG service and check all components."""
        logger.info("Initializing RAG service...")

        try:
            # Check component health
            components_healthy = []

            # Check vector store
            if self.vector_store.health_check():
                components_healthy.append("vector_store")
            else:
                logger.error("Vector store health check failed")

            # Check query router
            if self.query_router.is_healthy():
                components_healthy.append("query_router")
            else:
                logger.error("Query router health check failed")

            # Check memory service
            if await self.memory_service.is_healthy():
                components_healthy.append("memory_service")
            else:
                logger.warning("Memory service not available (continuing without Redis)")

            if len(components_healthy) >= 2:  # At least vector store and query router
                self.is_ready = True
                logger.info("RAG service initialized successfully")
                return True
            else:
                logger.error("RAG service initialization failed")
                return False

        except Exception as e:
            logger.error(f"Error initializing RAG service: {e}")
            return False

    async def process_document(self, input_data: Union[str, bytes],
                             input_type: str = 'auto',
                             document_id: str = None) -> Dict[str, Any]:
        """
        Process a document and store it in the vector database.

        Args:
            input_data: File path/URL or raw bytes
            input_type: Type of input ('auto', 'pdf', 'image', 'url')
            document_id: Optional document identifier

        Returns:
            Processing results with statistics
        """
        if not self.is_ready:
            raise RuntimeError("RAG service not initialized")

        start_time = time.time()
        document_id = document_id or str(uuid.uuid4())

        try:
            logger.info(f"Processing document {document_id} of type {input_type}")

            # Detect input type if auto
            if input_type == 'auto':
                input_type = self.document_processor.detect_input_type(input_data)

            # Process document into chunks
            chunks = await self.document_processor.process_input(input_data, input_type)

            if not chunks:
                raise ValueError("No chunks generated from document")

            # Extract text content for embedding
            texts_to_embed = [chunk['content'] for chunk in chunks]

            # Generate embeddings
            embeddings = await self.embedding_service.embed_texts(texts_to_embed)

            if not embeddings:
                raise RuntimeError("No embeddings generated")

            # Store in vector database
            success = await self.vector_store.store_chunks(chunks, embeddings)

            if not success:
                raise RuntimeError("Failed to store chunks in vector database")

            # Store document metadata
            self.processed_documents[document_id] = {
                'chunks_count': len(chunks),
                'input_type': input_type,
                'processed_at': time.time(),
                'processing_time': time.time() - start_time
            }

            processing_time = time.time() - start_time

            logger.info(f"Document {document_id} processed successfully in {processing_time:.2f}s")

            return {
                'document_id': document_id,
                'status': 'success',
                'chunks_processed': len(chunks),
                'embeddings_generated': len(embeddings),
                'processing_time': processing_time,
                'input_type': input_type
            }

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            return {
                'document_id': document_id,
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    async def query(self, query_text: str,
                    conversation_id: str = None,
                    top_k: int = 5,
                    score_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Process a query and return relevant response.

        Args:
            query_text: The query string
            conversation_id: Optional conversation identifier
            top_k: Number of similar chunks to retrieve
            score_threshold: Minimum similarity score

        Returns:
            Query results with answer and metadata
        """
        if not self.is_ready:
            raise RuntimeError("RAG service not initialized")

        start_time = time.time()

        try:
            # Get conversation context if available
            conversation_context = None
            if conversation_id:
                conversation_context = await self.memory_service.get_conversation_context(conversation_id)
                if conversation_context:
                    # Get recent history
                    conversation_context['history'] = await self.memory_service.get_conversation_history(
                        conversation_id, limit=5
                    )

            # Use query router to process query (includes vector search and re-ranking)
            response_result = await self.query_router.process_query(
                query_text, None, conversation_context
            )

            # Get context chunks from query router response
            context_chunks = response_result.get("context_chunks", [])
            search_results = context_chunks

            answer = response_result['answer']

            # Store interaction in memory
            if conversation_id:
                await self.memory_service.add_to_conversation_history(conversation_id, {
                    'query': query_text,
                    'response': response_result['answer'],
                    'llm_used': response_result['llm_used'],
                    'chunks_used': len(search_results),
                    'timestamp': time.time()
                })

                # Update conversation context
                await self.memory_service.store_conversation_context(conversation_id, {
                    'last_query': query_text,
                    'last_response': response_result['answer'],
                    'llm_used': response_result['llm_used'],
                    'updated_at': time.time()
                })

            total_time = time.time() - start_time

            return {
                'query': query_text,
                'answer': response_result['answer'],
                'search_results': search_results,
                'conversation_id': conversation_id,
                'llm_used': response_result['llm_used'],
                'processing_time': total_time,
                'chunks_retrieved': len(search_results),
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'query': query_text,
                'answer': f"Sorry, I encountered an error processing your query: {str(e)}",
                'search_results': [],
                'conversation_id': conversation_id,
                'processing_time': time.time() - start_time,
                'status': 'error',
                'error': str(e)
            }

    async def query_stream(self, query_text: str,
                          conversation_id: str = None,
                          top_k: int = 5,
                          score_threshold: float = 0.1):
            """
            Process a query with streaming response.

            Args:
                query_text: The query string
                conversation_id: Optional conversation identifier
                top_k: Number of similar chunks to retrieve
                score_threshold: Minimum similarity score

            Yields:
                Response chunks for streaming
            """
            if not self.is_ready:
                yield "RAG service not initialized"
                return

            start_time = time.time()

            
            
            try:
                # Get conversation context if available
                conversation_context = None
                if conversation_id:
                    conversation_context = await self.memory_service.get_conversation_context(conversation_id)
                    if conversation_context:
                        # Get recent history
                        conversation_context['history'] = await self.memory_service.get_conversation_history(
                            conversation_id, limit=5
                        )
    
                # Use query router to process query with streaming
                async for response_chunk in self.query_router.process_query_stream(
                    query_text, None, conversation_context
                ):
                    yield response_chunk
    
                # Store interaction in memory after completion (handled by QueryRouter now)
    
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                yield f"Sorry, I encountered an error processing your query: {str(e)}"

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'service_ready': self.is_ready,
            'processed_documents': len(self.processed_documents),
            'vector_store': await self.vector_store.get_collection_info(),
            'memory_service': await self.memory_service.get_memory_stats(),
            'query_router_healthy': self.query_router.is_healthy(),
            'components': {
                'document_processor': 'ready',
                'embedding_service': 'ready',
                'vector_store': 'ready' if self.vector_store.health_check() else 'error',
                'query_router': 'ready' if self.query_router.is_healthy() else 'error',
                'memory_service': 'ready' if self.memory_service.is_healthy() else 'degraded'
            }
        }

    async def clear_all_data(self) -> bool:
        """Clear all stored data (for testing/reset)."""
        try:
            # Clear vector store
            vector_cleared = await self.vector_store.clear_collection()

            # Clear processed documents
            self.processed_documents.clear()

            # Note: We don't clear Redis data as it may contain user conversations

            logger.info("All data cleared")
            return vector_cleared

        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            return False