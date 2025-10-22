"""
Gradio web interface for the RAG system.
Provides document upload and query interface.
"""

import asyncio
import gradio as gr
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple
import uuid

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# FIXED: Use absolute imports instead of relative
from services.rag_service import RAGService
from config.settings import settings

logger = logging.getLogger(__name__)


import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import uuid

import gradio as gr

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# FIXED: Use absolute imports instead of relative
from services.rag_service import RAGService
from config.settings import settings

logger = logging.getLogger(__name__)

class RAGInterface:
    """Gradio interface for the RAG system."""

    def __init__(self):
        """Initialize the RAG interface."""
        self.rag_service = RAGService()
        self.conversation_id = None

    async def initialize_service(self) -> str:
        """Initialize the RAG service."""
        try:
            success = await self.rag_service.initialize()
            if success:
                return "RAG service initialized successfully!"
            else:
                return "Failed to initialize RAG service. Check logs for details."
        except Exception as e:
            return f"Error initializing service: {str(e)}"

    def handle_file_upload(self, file):
        """Handle file upload and processing with streaming updates."""
        if file is None:
            yield "Please upload a file first."
            return

        try:
            # Ensure service is initialized
            if not self.rag_service.is_ready:
                yield "🔄 Initializing service..."
                init_msg = asyncio.run(self.initialize_service())
                if "successfully" not in init_msg:
                    yield "Service initialization failed. Please try again."
                    return

            # Save uploaded file temporarily
            file_path = file.name

            # Detect file type
            if file_path.lower().endswith('.pdf'):
                input_type = 'pdf'
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_type = 'image'
            else:
                yield "Unsupported file type. Please upload PDF or image files."
                return

            yield f"🔄 Processing {input_type.upper()} file: {file.name}. This may take a few minutes..."

            # Process document
            result = asyncio.run(self._process_uploaded_file(file_path, input_type))

            if result and result.get('status') == 'success':
                yield "✅ File processed successfully!"
            else:
                yield "❌ File processing failed. Check logs for details."

        except Exception as e:
            logger.error(f"Error handling file upload: {e}")
            yield f"Error processing file: {str(e)}"

    async def _process_uploaded_file(self, file_path: str, input_type: str):
        """Process uploaded file in background."""
        try:
            # Process the document
            result = await self.rag_service.process_document(file_path, input_type)

            if result['status'] == 'success':
                logger.info(f"Document processed successfully: {result}")
                return result
            else:
                logger.error(f"Document processing failed: {result}")
                return result

        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            return {'status': 'error', 'error': str(e)}

    async def handle_url_submit(self, url: str) -> str:
        """Handle URL submission and processing."""
        if not url:
            return "Please enter a URL first."

        try:
            # Ensure service is initialized
            if not self.rag_service.is_ready:
                init_msg = await self.initialize_service()
                if "successfully" not in init_msg:
                    return "Service initialization failed. Please try again."

            # Process URL asynchronously
            asyncio.create_task(self._process_url(url))

            return f"🔄 Processing URL: {url}. This may take a few minutes..."

        except Exception as e:
            logger.error(f"Error handling URL: {e}")
            return f"Error processing URL: {str(e)}"

    async def _process_url(self, url: str) -> None:
        """Process URL in background."""
        try:
            # Process the URL
            result = await self.rag_service.process_document(url, 'url')

            if result['status'] == 'success':
                logger.info(f"URL processed successfully: {result}")
            else:
                logger.error(f"URL processing failed: {result}")

        except Exception as e:
            logger.error(f"Error processing URL: {e}")

    def start_new_conversation(self) -> str:
        """Start a new conversation."""
        import uuid
        self.conversation_id = str(uuid.uuid4())
        return f"🆕 Started new conversation: {self.conversation_id[:8]}..."

    async def ask_question_stream(self, question: str, history: list):
        """Process a user question with streaming response."""
        if not question.strip():
            yield history + [{"role": "assistant", "content": "Please enter a question."}]
            return

        if not self.rag_service.is_ready:
            yield history + [{"role": "assistant", "content": "RAG service not ready. Please wait for initialization."}]
            return

        try:
            # Start conversation if not exists
            if not self.conversation_id:
                self.start_new_conversation()

            # Process query with streaming
            async for response_chunk in self.rag_service.query_stream(question, self.conversation_id):
                # Update history with current chunk
                updated_history = history + [[question, response_chunk]]
                yield updated_history

            # Store interaction in memory after completion (already handled within rag_service.query_stream)

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            error_history = history + [[question, f"Error: {str(e)}"]]
            yield error_history

    def ask_question(self, question: str, history: list) -> tuple[list, str]:
        """Process a user question (non-streaming for compatibility)."""
        # For now, use non-streaming to avoid complexity
        # In a full implementation, this would use the streaming version
        if not question.strip():
            return history + [{"role": "assistant", "content": "Please enter a question."}], ""

        if not self.rag_service.is_ready:
            return history + [{"role": "assistant", "content": "RAG service not ready. Please wait for initialization."}], ""

        try:
            # Start conversation if not exists
            if not self.conversation_id:
                self.start_new_conversation()

            # Process query using asyncio.run to avoid manual loop management
            result = asyncio.run(self.rag_service.query(question, self.conversation_id))

            if result['status'] == 'success':
                response = result['answer']

                # Format response for chatbot messages
                updated_history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": response}]

                # Show search results info
                info = f"✅ Found {result['chunks_retrieved']} relevant chunks using {result['llm_used'].upper()}"
                if result['processing_time']:
                    info += f" ({result['processing_time']:.2f}s)"

                return updated_history, info
            else:
                error_history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": result['answer']}]
                return error_history, "Query processing failed"

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            error_history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": f"Error: {str(e)}"}]
            return error_history, "Error occurred"

    def get_system_status(self) -> str:
        """Get current system status."""
        try:
            status = asyncio.run(self.rag_service.get_system_status())

            status_text = "📊 **System Status**\n\n"
            status_text += f"**Service Ready:** {'✅' if status['service_ready'] else '❌'}\n"
            status_text += f"**Documents Processed:** {status['processed_documents']}\n\n"

            # Vector store status
            vector_info = status['vector_store']
            status_text += "**Vector Store:**\n"
            if vector_info.get('status') == 'active':
                status_text += f"- Vectors: {vector_info.get('vectors_count', 0)}\n"
                status_text += f"- Collection: {vector_info.get('name', 'unknown')}\n"
            else:
                status_text += "- Not available\n"

            # Component status
            status_text += "\n**Components:**\n"
            for component, status_value in status['components'].items():
                status_text += f"- {component.replace('_', ' ').title()}: {'' if status_value == 'ready' else '' if status_value == 'error' else ''}\n"

            return status_text

        except Exception as e:
            return f"❌ Error getting status: {str(e)}"

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(
            title="RAG System - Document Q&A",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 800px;
                margin: auto;
            }
            """
        ) as app:
            gr.Markdown("# RAG System - Document Q&A")
            gr.Markdown("Upload documents (PDF, Image) or provide URLs, then ask questions!")

            with gr.Tabs():
                # Document Upload Tab
                with gr.TabItem("Document Upload"):
                    gr.Markdown("### Upload Documents")

                    with gr.Row():
                        file_input = gr.File(
                            label="Upload PDF or Image",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg"]
                        )
                        url_input = gr.Textbox(
                            label="Or enter URL",
                            placeholder="https://example.com/article"
                        )

                    with gr.Row():
                        process_file_btn = gr.Button("Process File", variant="primary")
                        process_url_btn = gr.Button("Process URL", variant="secondary")

                    file_status = gr.Textbox(label="Processing Status", interactive=False)
                    url_status = gr.Textbox(label="URL Status", interactive=False)

                    # File processing event
                    process_file_btn.click(
                        fn=self.handle_file_upload,
                        inputs=[file_input],
                        outputs=[file_status]
                    )

                    # URL processing event
                    process_url_btn.click(
                        fn=self.handle_url_submit,
                        inputs=[url_input],
                        outputs=[url_status]
                    ).then(
                        fn=lambda: "✅ URL processed successfully!",
                        inputs=[],
                        outputs=[url_status]
                    )

                # Query Tab
                with gr.TabItem("Ask Questions"):
                    gr.Markdown("### Ask Questions About Your Documents")

                    with gr.Row():
                        new_conv_btn = gr.Button("New Conversation")
                        status_btn = gr.Button("System Status")

                    conv_status = gr.Textbox(
                        label="Conversation Status",
                        interactive=False,
                        value="No active conversation"
                    )

                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask me anything about your uploaded documents...",
                        lines=3
                    )

                    answer_output = gr.Textbox(
                        label="Answer",
                        interactive=False,
                        lines=5
                    )

                    info_output = gr.Textbox(
                        label="Query Information",
                        interactive=False
                    )

                    # Conversation management
                    new_conv_btn.click(
                        fn=self.start_new_conversation,
                        outputs=[conv_status]
                    )

                    status_btn.click(
                        fn=self.get_system_status,
                        outputs=[info_output]
                    )

                    # Question submission with streaming
                    question_input.submit(
                        fn=self.ask_question_stream,
                        inputs=[question_input, conv_status],
                        outputs=[conv_status]
                    )

                    # Ask button for longer questions
                    ask_btn = gr.Button("Ask Question", variant="primary")
                    ask_btn.click(
                        fn=self.ask_question_stream,
                        inputs=[question_input, conv_status],
                        outputs=[conv_status]
                    )

            # Footer
            gr.Markdown("---")
            gr.Markdown("*Powered by Qdrant, Cohere, Groq, and LlamaParse*")

        return app


def create_app():
    """Create Gradio interface with fixed async handling."""
    
    # Initialize RAG service
    rag_service = RAGService()
    conversation_id = None
    
    # FIXED: Proper async wrapper for Gradio
    def process_file_upload_sync(file):
        """Synchronous wrapper for async file upload."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Initialize service if needed
                if not rag_service.is_ready:
                    loop.run_until_complete(rag_service.initialize())
                
                # Process the file
                result = loop.run_until_complete(
                    rag_service.process_document(file.name, 'pdf')
                )
                
                if result['status'] == 'success':
                    return f"✅ Document processed successfully!\n" \
                           f"📄 Chunks: {result.get('chunks_processed', 0)}\n" \
                           f"⏱️  Time: {result.get('processing_time', 0):.2f}s"
                else:
                    return f"❌ Error: {result.get('error', 'Unknown error')}"
                    
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}", exc_info=True)
            return f"❌ Error: {str(e)}"
    
    def process_url_sync(url):
        """Synchronous wrapper for async URL processing."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                if not rag_service.is_ready:
                    loop.run_until_complete(rag_service.initialize())
                
                result = loop.run_until_complete(
                    rag_service.process_document(url, 'url')
                )
                
                if result['status'] == 'success':
                    return f"✅ URL processed successfully!\n" \
                           f"📄 Chunks: {result.get('chunks_processed', 0)}"
                else:
                    return f"❌ Error: {result.get('error', 'Unknown error')}"
                    
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error handling URL: {e}", exc_info=True)
            return f"❌ Error: {str(e)}"
    
    async def query_documents_stream(query, history):
        """Streaming wrapper for async query."""
        nonlocal conversation_id

        if not query.strip():
            yield history + [{"role": "user", "content": query}, {"role": "assistant", "content": "Please enter a question."}]
            return

        try:
            if not rag_service.is_ready:
                await rag_service.initialize()

            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Process query with streaming
            async for response_chunk in rag_service.query_stream(query, conversation_id):
                updated_history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": response_chunk}]
                yield updated_history

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            error_history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": f"❌ Error: {str(e)}"}]
            yield error_history
    
    # Create Gradio interface
    with gr.Blocks(title="RAG System", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 RAG System - Document Q&A")
        
        with gr.Tabs():
            # Document Upload Tab
            with gr.Tab("📄 Document Upload"):
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(
                            label="Upload PDF",
                            file_types=[".pdf"],
                            type="filepath"
                        )
                        upload_btn = gr.Button("Process Document", variant="primary")
                        upload_output = gr.Textbox(
                            label="Upload Status",
                            lines=5
                        )
                
                with gr.Row():
                    with gr.Column():
                        url_input = gr.Textbox(
                            label="Or Enter URL",
                            placeholder="https://example.com/article"
                        )
                        url_btn = gr.Button("Process URL", variant="primary")
                        url_output = gr.Textbox(
                            label="URL Status",
                            lines=5
                        )
            
            # Chat Tab
            with gr.Tab("💬 Ask Questions"):
                chatbot = gr.Chatbot(label="Conversation", height=500, type='messages')
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about your documents..."
                )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear Chat")
                    new_conv_btn = gr.Button("New Conversation")
        
        # Event handlers
        upload_btn.click(
            fn=process_file_upload_sync,
            inputs=[file_upload],
            outputs=[upload_output]
        )
        
        url_btn.click(
            fn=process_url_sync,
            inputs=[url_input],
            outputs=[url_output]
        )
        
        submit_btn.click(
            fn=query_documents_stream,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )

        msg.submit(
            fn=query_documents_stream,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )
        
        clear_btn.click(lambda: [], outputs=[chatbot])
        
        def new_conversation():
            nonlocal conversation_id
            conversation_id = str(uuid.uuid4())
            return []
        
        new_conv_btn.click(fn=new_conversation, outputs=[chatbot])
    
    return app



# def create_app() -> gr.Blocks:
#     """Create and return the Gradio app."""
#     interface = RAGInterface()
#     app = interface.create_interface()

#     return app


async def initialize_app():
    """Initialize the RAG service when the app starts."""
    interface = RAGInterface()
    await interface.initialize_service()
    return interface


if __name__ == "__main__":
    # Initialize the service
    interface = asyncio.run(initialize_app())

    # Create and launch the app
    app = interface.create_interface()
    app.launch(
                server_name="127.0.0.1",        server_port=7861,
        share=True,  # Set to True if you want a public link
        debug=True
    )