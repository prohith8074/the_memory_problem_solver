"""
Query routing service using Groq as primary LLM with Cohere fallback.
Routes queries and generates responses using vector search results.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
import json

import cohere
import groq
from config.settings import settings
from services.vector_query_interface import VectorQueryInterface

logger = logging.getLogger(__name__)

class QueryRouter:
    """Routes queries and generates responses using LLM services."""

    def __init__(self, vector_query_interface: VectorQueryInterface):
        """Initialize LLM clients and vector query interface."""
        self.cohere_client = cohere.Client(api_key=settings.COHERE_API_KEY)
        self.groq_client = groq.Groq(api_key=settings.GROQ_API_KEY)
        self.vector_query_interface = vector_query_interface

        logger.info("QueryRouter initialized with Groq, Cohere, and VectorQueryInterface")

    async def analyze_query_and_plan(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine the optimal set of retrieval tools and a core query.
        """
        try:
            # First, expand the query with related terms
            expanded_query = await self._expand_query_with_cohere(query)

            prompt = f"""
            Analyze the user's query to create an execution plan. Decompose it into:
            1. A `core_query` for semantic vector search (use the expanded query provided).
            2. A list of `tools` to use. Available tools are: `search_vector`.
            3. A structured `filters` object for the `search_vector` tool (if applicable).

            Expanded Query: "{expanded_query}"
            Original Query: "{query}"

            Example:
            User Query: "show me papers about attention mechanism by Vaswani after 2016"
            Response:
            ```json
            {{
                "core_query": "attention mechanism papers by Vaswani transformer architecture self-attention",
                "tools": ["search_vector"],
                "filters": {{
                    "must": [
                        {{ "key": "metadata.author", "match": {{ "value": "Vaswani" }} }},
                        {{ "key": "metadata.year", "range": {{ "gte": 2017 }} }}
                    ]
                }},
                "reasoning": "The user has specific filters (author, year) and a core semantic query, so a filtered vector search is optimal."
            }}
            ```

            Now, parse the given user query and use the expanded query for the core_query.
            Response:
            """

            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.GROQ_MODEL,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content
            plan = json.loads(response_text)

            # Ensure plan has required keys
            if 'tools' not in plan or 'core_query' not in plan:
                raise ValueError("LLM response missing 'tools' or 'core_query' keys.")

            logger.info(f"Generated retrieval plan: {plan}")
            return plan

        except Exception as e:
            logger.error(f"Failed to generate retrieval plan: {e}")
            # Fallback to a default plan
            return {
                "core_query": query,
                "tools": ["search_vector"],
                "filters": {},
                "reasoning": "Fallback to default vector search due to planning error."
            }

    async def process_query(self, query: str, context_chunks: Optional[List[Dict[str, Any]]] = None, conversation_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query using available context.

        Args:
            query: User query string
            context_chunks: Pre-retrieved context chunks (optional)
            conversation_context: Previous conversation context

        Returns:
            Response dictionary with answer and metadata
        """
        start_time = time.time()

        try:
            if context_chunks is None:
                # 1. Plan: Analyze the query to create a retrieval plan
                plan = await self.analyze_query_and_plan(query)
                core_query = plan.get("core_query", query)
                filters = plan.get("filters", {})

                # 2. Execute: Perform the planned vector search
                vector_result = await self.vector_query_interface.query(
                    user_query=core_query,
                    filters=filters
                )

                context_chunks = vector_result.get("chunks", [])
                answer = vector_result.get("answer", "No answer generated.")
                llm_used = vector_result.get("llm_used", "cohere")
            else:
                # Use provided context chunks
                plan = {"core_query": query, "tools": ["provided_context"], "filters": {}}
                llm_used = 'cohere'

            # Re-rank chunks for better relevance
            relevant_chunks = await self._rerank_chunks(query, context_chunks)

            # Use the LLM to generate response with re-ranked chunks
            context_text = self._prepare_context_text(relevant_chunks)
            response = await self._generate_response_cohere(query, context_text, conversation_context)

            processing_time = time.time() - start_time

            return {
                'query': query,
                'answer': response,
                'llm_used': llm_used,
                'processing_time': processing_time,
                'context_chunks': relevant_chunks,
                'context_chunks_used': len(relevant_chunks),
                'original_chunks': len(context_chunks),
                'avg_similarity': sum(c.get('score', 0) for c in relevant_chunks) / len(relevant_chunks) if relevant_chunks else 0,
                'timestamp': time.time(),
                'plan': plan
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'query': query,
                'answer': f"Sorry, I encountered an error processing your query: {str(e)}",
                'llm_used': 'error',
                'processing_time': time.time() - start_time,
                'context_chunks_used': 0,
                'timestamp': time.time()
            }

    async def process_query_stream(self, query: str, context_chunks: Optional[List[Dict[str, Any]]] = None, conversation_context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """
        Process a user query with streaming response.

        Args:
            query: User query string
            context_chunks: Pre-retrieved context chunks (optional)
            conversation_context: Previous conversation context

        Yields:
            Response chunks for streaming
        """
        start_time = time.time()

        try:
            if context_chunks is None:
                yield "Analyzing query and planning retrieval..."
                # 1. Plan: Analyze the query to create a retrieval plan
                plan = await self.analyze_query_and_plan(query)
                core_query = plan.get("core_query", query)
                filters = plan.get("filters", {})

                yield f"Executing vector search for '{core_query}' with filters {filters}..."
                # 2. Execute: Perform the planned vector search
                vector_result = await self.vector_query_interface.query(
                    user_query=core_query,
                    filters=filters
                )

                context_chunks = vector_result.get("chunks", [])

                if not context_chunks:
                    yield "I don't have enough relevant information in my knowledge base to answer this question."
                    return

            yield f"Found {len(context_chunks)} potential contexts. Re-ranking for relevance..."

            # Re-rank chunks for better relevance
            relevant_chunks = await self._rerank_chunks(query, context_chunks)

            if not relevant_chunks:
                yield "No sufficiently relevant context found after re-ranking."
                return

            yield f"Using {len(relevant_chunks)} most relevant contexts for response generation..."

            # Generate response with re-ranked chunks
            context_text = self._prepare_context_text(relevant_chunks)
            response = await self._generate_response_cohere(query, context_text, conversation_context)

            # Yield the complete answer as a single chunk for streaming
            yield response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            yield f"Sorry, I encountered an error processing your query: {str(e)}"

    async def _generate_response_groq(self, query: str, context: str,
                                    conversation_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using Groq."""
        try:
            # Build conversation history if available
            messages = []

            # Add conversation history
            if conversation_context and 'history' in conversation_context:
                for interaction in conversation_context['history'][-3:]:  # Last 3 interactions
                    messages.append({
                        'role': 'user',
                        'content': interaction.get('query', '')
                    })
                    messages.append({
                        'role': 'assistant',
                        'content': interaction.get('response', '')
                    })

            # Add current query
            messages.append({
                'role': 'user',
                'content': f"Context: {context}\n\nQuery: {query}"
            })

            # Generate response
            response = self.groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    async def _generate_response_cohere(self, query: str, context: str,
                                       conversation_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using Cohere with streaming support."""
        try:
            # Prepare conversation history
            conversation_history = ""
            if conversation_context and 'history' in conversation_context:
                history = conversation_context['history'][-3:]  # Last 3 interactions
                for interaction in history:
                    conversation_history += f"User: {interaction.get('query', '')}\n"
                    conversation_history += f"Assistant: {interaction.get('response', '')}\n\n"

            # Prepare message for chat API
            message = f"{conversation_history}Context information:\n{context}\n\nBased on the above context, please answer the following query. If the context doesn't contain enough information to fully answer the query, please say so clearly.\n\nQuery: {query}"

            # Generate response using chat API with streaming
            response = self.cohere_client.chat_stream(
                model=settings.COHERE_MODEL,
                message=message,
                max_tokens=1000,
                temperature=0.7
            )

            # Collect the full response from streaming
            full_response = ""
            for chunk in response:
                if hasattr(chunk, 'event_type') and chunk.event_type == 'text-generation':
                    full_response += chunk.text
                elif hasattr(chunk, 'text'):
                    full_response += chunk.text

            return full_response.strip()

        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            raise

    async def _generate_response_cohere_stream(self, query: str, context: str,
                                             conversation_context: Optional[Dict[str, Any]] = None):
        """Generate streaming response using Cohere."""
        try:
            # Prepare conversation history
            conversation_history = ""
            if conversation_context and 'history' in conversation_context:
                history = conversation_context['history'][-3:]  # Last 3 interactions
                for interaction in history:
                    conversation_history += f"User: {interaction.get('query', '')}\n"
                    conversation_history += f"Assistant: {interaction.get('response', '')}\n\n"

            # Prepare message for chat API
            message = f"{conversation_history}Context information:\n{context}\n\nBased on the above context, please answer the following query. If the context doesn't contain enough information to fully answer the query, please say so clearly.\n\nQuery: {query}"

            # Generate response using chat API with streaming
            response = self.cohere_client.chat_stream(
                model=settings.COHERE_MODEL,
                message=message,
                max_tokens=1000,
                temperature=0.7
            )

            # Yield chunks for streaming
            current_response = ""
            for chunk in response:
                if hasattr(chunk, 'event_type') and chunk.event_type == 'text-generation':
                    current_response += chunk.text
                elif hasattr(chunk, 'text'):
                    current_response += chunk.text
                yield current_response

        except Exception as e:
            logger.error(f"Cohere streaming API error: {e}")
            yield f"Sorry, I encountered an error: {str(e)}"

    def _prepare_context_text(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context text from search results."""
        if not context_chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            text = chunk.get('text', chunk.get('content', ''))
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', 'Unknown')

            context_parts.append(f"[Context {i}] (Source: {source})\n{text}")

        return "\n\n".join(context_parts)

    async def _expand_query_with_cohere(self, query: str) -> str:
        """Expand query with related terms using Cohere."""
        try:
            expansion_prompt = f"""
            Given the user's query, generate 3-5 related terms or phrases that could help find more relevant information.
            Focus on synonyms, related concepts, and broader terms.

            Query: "{query}"

            Respond with only the expanded query (original + related terms), separated by spaces.
            Example: "machine learning AI neural networks deep learning"

            Expanded query:
            """

            response = self.cohere_client.chat(
                model=settings.COHERE_MODEL,
                message=expansion_prompt,
                max_tokens=50,
                temperature=0.3
            )

            expanded_terms = response.text.strip()
            if expanded_terms:
                return f"{query} {expanded_terms}"
            else:
                return query

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query

    async def _rerank_chunks(self, query: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank context chunks based on relevance to the query using Cohere chat."""
        if not context_chunks:
            return []

        try:
            # Use Cohere chat to score relevance of each chunk
            relevance_scores = []

            for chunk in context_chunks:
                text = chunk.get('text', chunk.get('content', ''))
                if not text.strip():
                    continue

                # Create a prompt to score relevance
                relevance_prompt = f"""
                Rate the relevance of the following text to the query on a scale of 0.0 to 1.0.
                Consider semantic similarity, topical relevance, and usefulness for answering the query.

                Query: {query}
                Text: {text[:1000]}...

                Respond with only a number between 0.0 and 1.0 representing the relevance score.
                Higher scores mean more relevant.
                """

                response = self.cohere_client.chat(
                    model=settings.COHERE_MODEL,
                    message=relevance_prompt,
                    max_tokens=10,
                    temperature=0.0
                )

                try:
                    score = float(response.text.strip())
                    score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
                except ValueError:
                    score = 0.5  # Default score if parsing fails

                relevance_scores.append((chunk, score))

            # Sort by relevance score (highest first)
            relevance_scores.sort(key=lambda x: x[1], reverse=True)

            # Return top 5 most relevant
            top_chunks = relevance_scores[:5]
            reranked_chunks = [chunk for chunk, score in top_chunks]

            # Update chunks with relevance scores
            for chunk, score in top_chunks:
                chunk['relevance_score'] = score

            logger.info(f"Re-ranked {len(reranked_chunks)} chunks with scores: {[c['relevance_score'] for c in reranked_chunks]}")
            return reranked_chunks

        except Exception as e:
            logger.warning(f"Re-ranking failed, using original similarity scores: {e}")
            # Fallback: sort by original similarity score
            context_chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
            # Update with fallback scores
            for chunk in context_chunks[:5]:
                chunk['relevance_score'] = chunk.get('score', 0.5)
            return context_chunks[:5]

    async def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze query intent to determine search strategy.
        This is a simplified version since we're doing vector-only search.
        """
        try:
            # Simple intent analysis
            query_lower = query.lower()

            # Determine if it's a factual question, conversational, or search query
            if any(word in query_lower for word in ['what', 'when', 'where', 'who', 'how', 'explain']):
                intent = 'factual'
            elif any(word in query_lower for word in ['tell me about', 'describe', 'summarize']):
                intent = 'informational'
            else:
                intent = 'conversational'

            return {
                'intent': intent,
                'requires_context': True,  # Most queries benefit from context
                'confidence': 0.8,
                'reasoning': f"Query appears to be {intent} in nature"
            }

        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return {
                'intent': 'unknown',
                'requires_context': True,
                'confidence': 0.5,
                'reasoning': f"Error in analysis: {str(e)}"
            }

    def is_healthy(self) -> bool:
        """Check if query router services are available."""
        # Check Cohere using chat API
        try:
            test_response = self.cohere_client.chat(
                model=settings.COHERE_MODEL,
                message='Test',
                max_tokens=1
            )
            if not test_response.text:
                logger.warning("Cohere chat health check failed: No response text")
                return False
        except Exception as e:
            logger.warning(f"Cohere chat health check failed: {e}")
            return False

        logger.info("Cohere client is healthy")
        return True
