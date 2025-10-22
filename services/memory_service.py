"""
Redis-based memory service for storing conversation history and context.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
import redis.asyncio as redis

from config.settings import settings

logger = logging.getLogger(__name__)


# class MemoryService:
#     """Redis-b
# ased memory management for conversations."""

#     def __init__(self):
#         """Initialize Redis connection."""
#         self.redis_client = None
#         self._connect()

#     def _connect(self) -> None:
#         """Establish Redis connection."""
#         try:
#             if settings.REDIS_PASSWORD:
#                 self.redis_client = redis.Redis(
#                     host=settings.REDIS_HOST,
#                     port=settings.REDIS_PORT,
#                     db=settings.REDIS_DB,
#                     password=settings.REDIS_PASSWORD,
#                     decode_responses=True
#                 )
#             else:
#                 self.redis_client = redis.Redis(
#                     host=settings.REDIS_HOST,
#                     port=settings.REDIS_PORT,
#                     db=settings.REDIS_DB,
#                     decode_responses=True
#                 )
#             logger.info("Connected to Redis memory service")
#         except Exception as e:
#             logger.error(f"Failed to connect to Redis: {e}")
#             self.redis_client = None

#     async def store_conversation_context(self, conversation_id: str,
#                                        context: Dict[str, Any]) -> bool:
#         """
#         Store conversation context in Redis.

#         Args:
#             conversation_id: Unique conversation identifier
#             context: Context data to store

#         Returns:
#             True if successful
#         """
#         if not self.redis_client:
#             logger.warning("Redis not available, skipping context storage")
#             return False

#         try:
#             key = f"conv_context:{conversation_id}"

#             # Store context with TTL
#             await self.redis_client.setex(
#                 key,
#                 settings.SHORT_TERM_MEMORY_TTL,
#                 json.dumps(context, default=str)
#             )

#             logger.debug(f"Stored context for conversation {conversation_id}")
#             return True

#         except Exception as e:
#             logger.error(f"Error storing conversation context: {e}")
#             return False

#     async def get_conversation_context(self, conversation_id: str) -> Optional[Dict[str, Any]]:
#         """
#         Retrieve conversation context from Redis.

#         Args:
#             conversation_id: Unique conversation identifier

#         Returns:
#             Context data or None if not found
#         """
#         if not self.redis_client:
#             return None

#         try:
#             key = f"conv_context:{conversation_id}"
#             context_data = await self.redis_client.get(key)

#             if context_data:
#                 return json.loads(context_data)
#             return None

#         except Exception as e:
#             logger.error(f"Error retrieving conversation context: {e}")
#             return None

#     async def add_to_conversation_history(self, conversation_id: str,
#                                         interaction: Dict[str, Any]) -> bool:
#         """
#         Add an interaction to conversation history.

#         Args:
#             conversation_id: Unique conversation identifier
#             interaction: Interaction data (query, response, metadata)

#         Returns:
#             True if successful
#         """
#         if not self.redis_client:
#             logger.warning("Redis not available, skipping history storage")
#             return False

#         try:
#             # Store in list with max length
#             history_key = f"conv_history:{conversation_id}"

#             # Add timestamp if not present
#             if 'timestamp' not in interaction:
#                 interaction['timestamp'] = time.time()

#             # Add to list (left push for chronological order)
#             await self.redis_client.lpush(
#                 history_key,
#                 json.dumps(interaction, default=str)
#             )

#             # Trim to max history length
#             await self.redis_client.ltrim(
#                 history_key,
#                 0,
#                 settings.MAX_CONVERSATION_HISTORY - 1
#             )

#             # Set TTL on the list
#             await self.redis_client.expire(
#                 history_key,
#                 settings.SHORT_TERM_MEMORY_TTL
#             )

#             logger.debug(f"Added interaction to history for conversation {conversation_id}")
#             return True

#         except Exception as e:
#             logger.error(f"Error adding to conversation history: {e}")
#             return False

#     async def get_conversation_history(self, conversation_id: str,
#                                      limit: int = None) -> List[Dict[str, Any]]:
#         """
#         Get conversation history.

#         Args:
#             conversation_id: Unique conversation identifier
#             limit: Maximum number of interactions to return

#         Returns:
#             List of conversation interactions
#         """
#         if not self.redis_client:
#             return []

#         try:
#             history_key = f"conv_history:{conversation_id}"

#             # Get history items
#             if limit:
#                 history_items = await self.redis_client.lrange(
#                     history_key,
#                     0,
#                     limit - 1
#                 )
#             else:
#                 history_items = await self.redis_client.lrange(
#                     history_key,
#                     0,
#                     -1
#                 )

#             # Parse JSON and return in chronological order (oldest first)
#             history = []
#             for item in reversed(history_items):  # Reverse to get chronological order
#                 try:
#                     history.append(json.loads(item))
#                 except json.JSONDecodeError:
#                     logger.warning(f"Invalid JSON in history: {item}")
#                     continue

#             return history

#         except Exception as e:
#             logger.error(f"Error retrieving conversation history: {e}")
#             return []

#     async def clear_conversation_memory(self, conversation_id: str) -> bool:
#         """
#         Clear all memory for a conversation.

#         Args:
#             conversation_id: Unique conversation identifier

#         Returns:
#             True if successful
#         """
#         if not self.redis_client:
#             return False

#         try:
#             # Clear context
#             context_key = f"conv_context:{conversation_id}"
#             await self.redis_client.delete(context_key)

#             # Clear history
#             history_key = f"conv_history:{conversation_id}"
#             await self.redis_client.delete(history_key)

#             logger.info(f"Cleared memory for conversation {conversation_id}")
#             return True

#         except Exception as e:
#             logger.error(f"Error clearing conversation memory: {e}")
#             return False

#     async def get_memory_stats(self) -> Dict[str, Any]:
#         """Get memory usage statistics."""
#         if not self.redis_client:
#             return {'status': 'redis_unavailable'}

#         try:
#             # Get Redis info
#             info = await self.redis_client.info()

#             return {
#                 'status': 'connected',
#                 'redis_version': info.get('redis_version', 'unknown'),
#                 'connected_clients': info.get('connected_clients', 0),
#                 'used_memory_human': info.get('used_memory_human', 'unknown'),
#                 'keyspace_hits': info.get('keyspace_hits', 0),
#                 'keyspace_misses': info.get('keyspace_misses', 0)
#             }

#         except Exception as e:
#             logger.error(f"Error getting memory stats: {e}")
#             return {'status': 'error', 'error': str(e)}

#     async def cleanup_expired_sessions(self) -> int:
#         """
#         Clean up expired conversation sessions.

#         Returns:
#             Number of sessions cleaned up
#         """
#         if not self.redis_client:
#             return 0

#         try:
#             # Use SCAN to find all conversation keys
#             pattern = "conv_*:*"
#             keys = await self.redis_client.scan(0, pattern, 1000)

#             expired_count = 0
#             for key in keys[1]:  # scan returns (cursor, keys)
#                 try:
#                     # Check TTL for each key
#                     ttl = await self.redis_client.ttl(key)
#                     if ttl == -2:  # Key doesn't exist or expired
#                         await self.redis_client.delete(key)
#                         expired_count += 1
#                 except Exception as e:
#                     logger.warning(f"Error checking TTL for key {key}: {e}")

#             if expired_count > 0:
#                 logger.info(f"Cleaned up {expired_count} expired conversation sessions")

#             return expired_count

#         except Exception as e:
#             logger.error(f"Error during cleanup: {e}")
#             return 0

#     async def is_healthy(self) -> bool:
#         """Check if Redis connection is healthy."""
#         if not self.redis_client:
#             return False

#         try:
#             # Simple ping command
#             return await self.redis_client.ping()
#         except Exception:
#             return False
        



# -----------------------------------------------------------------------------


import asyncio
import logging
from typing import List, Dict, Optional
import json
import redis.asyncio as aioredis
from config.settings import settings

logger = logging.getLogger(__name__)


class MemoryService:
    """Redis-based conversation memory service with fixed async handling."""
    
    def __init__(self):
        self.redis_client = None
        self._loop = None
    
    async def _ensure_connection(self):
        """Ensure Redis connection exists and event loop is valid."""
        try:
            # Get or create event loop
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            
            # Create Redis connection if needed
            if self.redis_client is None:
                if settings.REDIS_PASSWORD:
                    self.redis_client = aioredis.from_url(
                        f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                        password=settings.REDIS_PASSWORD,
                        decode_responses=True,
                        socket_connect_timeout=5
                    )
                else:
                    self.redis_client = aioredis.from_url(
                        f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                        decode_responses=True,
                        socket_connect_timeout=5
                    )
                
                # Test connection
                await self.redis_client.ping()
                logger.info("Connected to Redis memory service")
                
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            self.redis_client = None
            raise
    
    async def store_conversation(self, conversation_id: str, message: Dict) -> bool:
        """Store conversation message with proper error handling."""
        try:
            await self._ensure_connection()
            
            if not self.redis_client:
                logger.warning("Redis not available, skipping memory storage")
                return False
            
            key = f"conversation:{conversation_id}"
            
            # Get existing conversation
            existing = await self.redis_client.lrange(key, 0, -1)
            conversation_history = [json.loads(msg) for msg in existing] if existing else []
            
            # Add new message
            conversation_history.append(message)
            
            # Keep only last N messages
            if len(conversation_history) > settings.MAX_CONVERSATION_HISTORY:
                conversation_history = conversation_history[-settings.MAX_CONVERSATION_HISTORY:]
            
            # Clear and store updated conversation
            await self.redis_client.delete(key)
            for msg in conversation_history:
                await self.redis_client.rpush(key, json.dumps(msg))
            
            # Set expiration
            await self.redis_client.expire(key, settings.SHORT_TERM_MEMORY_TTL)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing conversation: {e}", exc_info=True)
            return False
    
    async def get_conversation_context(self, conversation_id: str) -> List[Dict]:
        """Retrieve conversation context with proper error handling."""
        try:
            await self._ensure_connection()
            
            if not self.redis_client:
                logger.warning("Redis not available, returning empty context")
                return []
            
            key = f"conversation:{conversation_id}"
            messages = await self.redis_client.lrange(key, 0, -1)
            
            if not messages:
                return []
            
            return [json.loads(msg) for msg in messages]
            
        except Exception as e:
            logger.error(f"Error retrieving conversation context: {e}", exc_info=True)
            return []
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history."""
        try:
            await self._ensure_connection()

            if not self.redis_client:
                return False

            key = f"conversation:{conversation_id}"
            await self.redis_client.delete(key)
            return True

        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return False

    async def is_healthy(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            await self._ensure_connection()
            if not self.redis_client:
                return False
            # Simple ping command
            return await self.redis_client.ping()
        except Exception:
            return False

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            await self._ensure_connection()
            if not self.redis_client:
                return {'status': 'redis_unavailable'}

            # Get Redis info
            info = await self.redis_client.info()

            return {
                'status': 'connected',
                'redis_version': info.get('redis_version', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', 'unknown'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {'status': 'error', 'error': str(e)}
