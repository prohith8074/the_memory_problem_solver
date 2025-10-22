#!/usr/bin/env python3
"""
Main entry point for the RAG system.
Supports both CLI and web interface modes.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config.settings import settings
from services.rag_service import RAGService
from ui.gradio_app import create_app

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class RAGSystem:
    """Main RAG system class."""

    def __init__(self):
        """Initialize the RAG system."""
        self.rag_service = RAGService()
        self.conversation_id = None

    async def initialize(self) -> bool:
        """Initialize the RAG service."""
        logger.info("Initializing RAG System...")

        try:
            success = await self.rag_service.initialize()
            if success:
                logger.info("RAG System initialized successfully")
                return True
            else:
                logger.error("❌ Failed to initialize RAG System")
                return False
        except Exception as e:
            logger.error(f"❌ Error during initialization: {e}")
            return False

    def start_new_conversation(self) -> str:
        """Start a new conversation."""
        import uuid
        self.conversation_id = str(uuid.uuid4())
        return f"🆕 Started new conversation: {self.conversation_id[:8]}..."

    async def process_query(self, query: str) -> str:
        """Process a user query."""
        if not query.strip():
            return "Please enter a question."

        try:
            # Start conversation if needed
            if not self.conversation_id:
                self.start_new_conversation()

            # Process query
            result = await self.rag_service.query(query, self.conversation_id)

            if result['status'] == 'success':
                return result['answer']
            else:
                return f"❌ {result['answer']}"

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"❌ Error: {str(e)}"

    def get_system_status(self) -> str:
        """Get system status."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                status = loop.run_until_complete(self.rag_service.get_system_status())

                status_text = "📊 System Status\n"
                status_text += "=" * 30 + "\n"
                status_text += f"Service Ready: {'✅' if status['service_ready'] else '❌'}\n"
                status_text += f"Documents Processed: {status['processed_documents']}\n"
                status_text += f"Vector Store: {status['vector_store'].get('vectors_count', 0)} vectors\n"

                return status_text

            finally:
                loop.close()

        except Exception as e:
            return f"❌ Error getting status: {str(e)}"


async def run_cli_mode():
    """Run in CLI mode."""
    print("RAG System - CLI Mode")
    print("=" * 30)

    # Initialize system
    rag_system = RAGSystem()
    if not await rag_system.initialize():
        print("❌ Failed to initialize system")
        return

    print("System ready!")
    print("Commands: 'exit' to quit, 'status' for system status, 'new' for new conversation")
    print()

    try:
        while True:
            # Get user input
            query = input("Your question: ").strip()

            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            elif query.lower() == 'status':
                status = rag_system.get_system_status()
                print(f"\n{status}\n")
                continue
            elif query.lower() == 'new':
                conv_msg = rag_system.start_new_conversation()
                print(f"{conv_msg}\n")
                continue
            elif not query:
                continue

            # Process query
            print("Thinking...")
            response = await rag_system.process_query(query)

            print("\n" + "=" * 50)
            print(f"Answer: {response}")
            print("=" * 50 + "\n")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Error in CLI mode: {e}")
        print(f"❌ Error: {e}")


async def run_web_mode():
    """Run in web interface mode."""
    print("RAG System - Web Interface Mode")
    print("=" * 30)

    # Initialize system
    rag_system = RAGSystem()
    if not await rag_system.initialize():
        print("❌ Failed to initialize system")
        return

    print("System ready!")
    print("Starting web interface at http://localhost:7861")
    print("Press Ctrl+C to stop")

    try:
        # Create and launch Gradio app
        app = create_app()
        app.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=True,
            debug=True
        )

    except KeyboardInterrupt:
        print("\n👋 Web interface stopped")
    except Exception as e:
        logger.error(f"Error in web mode: {e}")
        print(f"❌ Error: {e}")


async def main():
    """Main entry point."""
    print("RAG System")
    print("Built with Qdrant, Cohere, Groq, and LlamaParse")

    # Validate configuration
    try:
        settings.validate()
        print("Configuration validated")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Please check your .env file")
        return

    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == '--cli':
            await run_cli_mode()
        elif mode == '--web':
            await run_web_mode()
        elif mode == '--status':
            rag_system = RAGSystem()
            await rag_system.initialize()
            print(rag_system.get_system_status())
        else:
            print("❌ Unknown mode. Use --cli or --web")
            print("💡 Defaulting to web mode")
            await run_web_mode()
    else:
        # Default to web mode
        await run_web_mode()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())





# ============================================================================
# FIX 1: Import Error - Restructure imports in gradio_app.py
# ============================================================================

# FILE: ui/gradio_app.py
# REPLACE relative imports with absolute imports

"""
Gradio web interface for the RAG system.
"""

# import asyncio
# import logging
# import os
# import sys
# from pathlib import Path
# from typing import Optional, Tuple
# import uuid

# import gradio as gr

# # Add parent directory to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent))

# # FIXED: Use absolute imports instead of relative
# from services.rag_service import RAGService
# from config.settings import settings

# logger = logging.getLogger(__name__)


# ============================================================================
# FIX 2: Vector Search Returning No Results - Fix vector_store.py
# ============================================================================

# FILE: database/vector_store.py
# ADD proper search with debugging

# async def search_similar(self, query_embedding: List[float],
#                        limit: int = 5,
#                        score_threshold: float = 0.0,
#                        filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
#     """
#     Search for similar vectors with improved error handling.
#     """
#     try:
#         # FIXED: Check if collection has vectors first
#         collection_info = self.client.get_collection(self.collection_name)
        
#         if collection_info.points_count == 0:
#             logger.warning("Vector store is empty! No documents indexed yet.")
#             return []
        
#         logger.info(f"Searching in collection with {collection_info.points_count} vectors")
        
#         # FIXED: Lower default threshold to 0.0 to get more results
#         search_results = self.client.search(
#             collection_name=self.collection_name,
#             query_vector=query_embedding,
#             limit=limit,
#             score_threshold=max(0.0, score_threshold),  # Ensure non-negative
#             query_filter=None if not filter_conditions else self._build_filter(filter_conditions)
#         )
        
#         # FIXED: Add detailed logging
#         if not search_results:
#             logger.warning(f"No results found for query. Collection has {collection_info.points_count} vectors")
#             logger.warning(f"Query embedding dimension: {len(query_embedding)}")
#             logger.warning(f"Score threshold: {score_threshold}")
        
#         results = []
#         for result in search_results:
#             results.append({
#                 'id': result.id,
#                 'score': result.score,
#                 'text': result.payload.get('text', ''),
#                 'chunk_id': result.payload.get('chunk_id', ''),
#                 'metadata': result.payload.get('metadata', {})
#             })
        
#         logger.info(f"Found {len(results)} similar chunks with scores: {[r['score'] for r in results]}")
#         return results
        
#     except Exception as e:
#         logger.error(f"Error searching vectors: {e}", exc_info=True)
#         return []


# ============================================================================
# FIX 3: Redis Event Loop Closed Error - Fix memory_service.py
# ============================================================================

# FILE: services/memory_service.py
# ADD proper async event loop handling

# import asyncio
# import logging
# from typing import List, Dict, Optional
# import json
# import redis.asyncio as aioredis
# from config.settings import settings

# logger = logging.getLogger(__name__)


# class MemoryService:
#     """Redis-based conversation memory service with fixed async handling."""
    
#     def __init__(self):
#         self.redis_client = None
#         self._loop = None
    
#     async def _ensure_connection(self):
#         """Ensure Redis connection exists and event loop is valid."""
#         try:
#             # Get or create event loop
#             try:
#                 self._loop = asyncio.get_running_loop()
#             except RuntimeError:
#                 self._loop = asyncio.new_event_loop()
#                 asyncio.set_event_loop(self._loop)
            
#             # Create Redis connection if needed
#             if self.redis_client is None:
#                 if settings.REDIS_PASSWORD:
#                     self.redis_client = aioredis.from_url(
#                         f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
#                         password=settings.REDIS_PASSWORD,
#                         decode_responses=True,
#                         socket_connect_timeout=5
#                     )
#                 else:
#                     self.redis_client = aioredis.from_url(
#                         f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
#                         decode_responses=True,
#                         socket_connect_timeout=5
#                     )
                
#                 # Test connection
#                 await self.redis_client.ping()
#                 logger.info("Connected to Redis memory service")
                
#         except Exception as e:
#             logger.error(f"Redis connection error: {e}")
#             self.redis_client = None
#             raise
    
#     async def store_conversation(self, conversation_id: str, message: Dict) -> bool:
#         """Store conversation message with proper error handling."""
#         try:
#             await self._ensure_connection()
            
#             if not self.redis_client:
#                 logger.warning("Redis not available, skipping memory storage")
#                 return False
            
#             key = f"conversation:{conversation_id}"
            
#             # Get existing conversation
#             existing = await self.redis_client.lrange(key, 0, -1)
#             conversation_history = [json.loads(msg) for msg in existing] if existing else []
            
#             # Add new message
#             conversation_history.append(message)
            
#             # Keep only last N messages
#             if len(conversation_history) > settings.MAX_CONVERSATION_HISTORY:
#                 conversation_history = conversation_history[-settings.MAX_CONVERSATION_HISTORY:]
            
#             # Clear and store updated conversation
#             await self.redis_client.delete(key)
#             for msg in conversation_history:
#                 await self.redis_client.rpush(key, json.dumps(msg))
            
#             # Set expiration
#             await self.redis_client.expire(key, settings.SHORT_TERM_MEMORY_TTL)
            
#             return True
            
#         except Exception as e:
#             logger.error(f"Error storing conversation: {e}", exc_info=True)
#             return False
    
#     async def get_conversation_context(self, conversation_id: str) -> List[Dict]:
#         """Retrieve conversation context with proper error handling."""
#         try:
#             await self._ensure_connection()
            
#             if not self.redis_client:
#                 logger.warning("Redis not available, returning empty context")
#                 return []
            
#             key = f"conversation:{conversation_id}"
#             messages = await self.redis_client.lrange(key, 0, -1)
            
#             if not messages:
#                 return []
            
#             return [json.loads(msg) for msg in messages]
            
#         except Exception as e:
#             logger.error(f"Error retrieving conversation context: {e}", exc_info=True)
#             return []
    
#     async def clear_conversation(self, conversation_id: str) -> bool:
#         """Clear conversation history."""
#         try:
#             await self._ensure_connection()
            
#             if not self.redis_client:
#                 return False
            
#             key = f"conversation:{conversation_id}"
#             await self.redis_client.delete(key)
#             return True
            
#         except Exception as e:
#             logger.error(f"Error clearing conversation: {e}")
#             return False


# ============================================================================
# FIX 4: Collection Info Error - Fix get_collection_info in vector_store.py
# ============================================================================

# async def get_collection_info(self) -> Dict[str, Any]:
#     """Get information about the collection with proper attribute access."""
#     try:
#         collection_info = self.client.get_collection(self.collection_name)
        
#         # FIXED: Access attributes correctly based on Qdrant client version
#         return {
#             'name': self.collection_name,  # Use stored name instead of attribute
#             'vectors_count': collection_info.points_count,
#             'vectors_size': collection_info.config.params.vectors.size,
#             'status': collection_info.status,
#             'optimizer_status': collection_info.optimizer_status.status if hasattr(collection_info, 'optimizer_status') else 'unknown'
#         }
#     except Exception as e:
#         logger.error(f"Error getting collection info: {e}", exc_info=True)
#         return {
#             'name': self.collection_name,
#             'status': 'error',
#             'error': str(e),
#             'vectors_count': 0
#         }


# ============================================================================
# FIX 5: Gradio Async Handling - Fix gradio_app.py file upload
# ============================================================================

# FILE: ui/gradio_app.py

# def create_app():
#     """Create Gradio interface with fixed async handling."""
    
#     # Initialize RAG service
#     rag_service = RAGService()
#     conversation_id = None
    
#     # FIXED: Proper async wrapper for Gradio
#     def process_file_upload_sync(file):
#         """Synchronous wrapper for async file upload."""
#         try:
#             # Create new event loop for this thread
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
            
#             try:
#                 # Initialize service if needed
#                 if not rag_service._initialized:
#                     loop.run_until_complete(rag_service.initialize())
                
#                 # Process the file
#                 result = loop.run_until_complete(
#                     rag_service.process_document(file.name, 'pdf')
#                 )
                
#                 if result['status'] == 'success':
#                     return f"✅ Document processed successfully!\n" \
#                            f"📄 Chunks: {result.get('chunks_processed', 0)}\n" \
#                            f"⏱️  Time: {result.get('processing_time', 0):.2f}s"
#                 else:
#                     return f"❌ Error: {result.get('error', 'Unknown error')}"
                    
#             finally:
#                 loop.close()
                
#         except Exception as e:
#             logger.error(f"Error processing uploaded file: {e}", exc_info=True)
#             return f"❌ Error: {str(e)}"
    
#     def process_url_sync(url):
#         """Synchronous wrapper for async URL processing."""
#         try:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
            
#             try:
#                 if not rag_service._initialized:
#                     loop.run_until_complete(rag_service.initialize())
                
#                 result = loop.run_until_complete(
#                     rag_service.process_document(url, 'url')
#                 )
                
#                 if result['status'] == 'success':
#                     return f"✅ URL processed successfully!\n" \
#                            f"📄 Chunks: {result.get('chunks_processed', 0)}"
#                 else:
#                     return f"❌ Error: {result.get('error', 'Unknown error')}"
                    
#             finally:
#                 loop.close()
                
#         except Exception as e:
#             logger.error(f"Error handling URL: {e}", exc_info=True)
#             return f"❌ Error: {str(e)}"
    
#     def query_documents_sync(query, history):
#         """Synchronous wrapper for async query."""
#         nonlocal conversation_id
        
#         if not query.strip():
#             return history + [["", "Please enter a question."]]
        
#         try:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
            
#             try:
#                 if not rag_service._initialized:
#                     loop.run_until_complete(rag_service.initialize())
                
#                 if not conversation_id:
#                     conversation_id = str(uuid.uuid4())
                
#                 result = loop.run_until_complete(
#                     rag_service.query(query, conversation_id)
#                 )
                
#                 answer = result.get('answer', 'No answer generated')
#                 history = history + [[query, answer]]
#                 return history
                
#             finally:
#                 loop.close()
                
#         except Exception as e:
#             logger.error(f"Error processing query: {e}", exc_info=True)
#             return history + [[query, f"❌ Error: {str(e)}"]]
    
#     # Create Gradio interface
#     with gr.Blocks(title="RAG System", theme=gr.themes.Soft()) as app:
#         gr.Markdown("# 🚀 RAG System - Document Q&A")
        
#         with gr.Tabs():
#             # Document Upload Tab
#             with gr.Tab("📄 Document Upload"):
#                 with gr.Row():
#                     with gr.Column():
#                         file_upload = gr.File(
#                             label="Upload PDF",
#                             file_types=[".pdf"],
#                             type="filepath"
#                         )
#                         upload_btn = gr.Button("Process Document", variant="primary")
#                         upload_output = gr.Textbox(
#                             label="Upload Status",
#                             lines=5
#                         )
                
#                 with gr.Row():
#                     with gr.Column():
#                         url_input = gr.Textbox(
#                             label="Or Enter URL",
#                             placeholder="https://example.com/article"
#                         )
#                         url_btn = gr.Button("Process URL", variant="primary")
#                         url_output = gr.Textbox(
#                             label="URL Status",
#                             lines=5
#                         )
            
#             # Chat Tab
#             with gr.Tab("💬 Ask Questions"):
#                 chatbot = gr.Chatbot(label="Conversation", height=500)
#                 msg = gr.Textbox(
#                     label="Your Question",
#                     placeholder="Ask anything about your documents..."
#                 )
#                 with gr.Row():
#                     submit_btn = gr.Button("Submit", variant="primary")
#                     clear_btn = gr.Button("Clear Chat")
#                     new_conv_btn = gr.Button("New Conversation")
        
#         # Event handlers
#         upload_btn.click(
#             fn=process_file_upload_sync,
#             inputs=[file_upload],
#             outputs=[upload_output]
#         )
        
#         url_btn.click(
#             fn=process_url_sync,
#             inputs=[url_input],
#             outputs=[url_output]
#         )
        
#         submit_btn.click(
#             fn=query_documents_sync,
#             inputs=[msg, chatbot],
#             outputs=[chatbot]
#         ).then(
#             lambda: "",
#             outputs=[msg]
#         )
        
#         msg.submit(
#             fn=query_documents_sync,
#             inputs=[msg, chatbot],
#             outputs=[chatbot]
#         ).then(
#             lambda: "",
#             outputs=[msg]
#         )
        
#         clear_btn.click(lambda: [], outputs=[chatbot])
        
#         def new_conversation():
#             nonlocal conversation_id
#             conversation_id = str(uuid.uuid4())
#             return []
        
#         new_conv_btn.click(fn=new_conversation, outputs=[chatbot])
    
#     return app


# ============================================================================
# FIX 6: Update main.py to handle imports correctly
# ============================================================================

# FILE: main.py

#!/usr/bin/env python3
"""
Main entry point for the RAG system.
Supports both CLI and web interface modes.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# FIXED: Ensure proper path setup
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from config.settings import settings
from services.rag_service import RAGService
from ui.gradio_app import create_app

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# FIX 7: Add requirements.txt fixes for missing dependencies
# ============================================================================

# FILE: requirements.txt
# ADD these lines:

"""
# Fixed dependencies
redis[hiredis]>=5.0.0  # Updated Redis with async support
qdrant-client>=1.7.3  # Updated Qdrant client
python-dotenv>=1.0.0
asyncio>=3.4.3
aiohttp>=3.9.0
nest-asyncio>=1.6.0

# AI and LLM services
cohere>=5.0.0  # Updated Cohere
groq>=0.4.0  # Added Groq SDK
llama-index>=0.10.0
llama-parse>=0.4.0

# Vector database
qdrant-client>=1.7.3

# Memory and caching  
redis[hiredis]>=5.0.0  # Async Redis support

# Web scraping
trafilatura>=1.6.0
firecrawl-py>=0.0.5
beautifulsoup4>=4.12.0

# Document processing
PyMuPDF>=1.23.0
pytesseract>=0.3.10  # Added OCR
Pillow>=10.0.0
python-docx>=1.1.0

# Web UI
gradio>=4.19.0  # Updated Gradio

# Development
pytest>=7.0.0
pytest-asyncio>=0.23.0
black>=24.0.0
"""


# ============================================================================
# FIX 8: Create a diagnostic script to check all systems
# ============================================================================

# FILE: diagnostic.py

"""
Diagnostic script to check RAG system health.
"""
 