"""
AI Journal Analysis Watcher
Monitors Obsidian daily journal entries and appends AI-generated analysis.
"""
import argparse
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from ollama import Client

from config import (
    VAULT_PATH,
    DAILY_DIR,
    PERSIST_DIR,
    LOG_PATH,
    DAILY_NOTES_PATH,
    INDEX_PERSIST_PATH,
    LOG_FILE_PATH,
    LLM_MODEL,
    EMBED_MODEL,
    OLLAMA_URL,
    OLLAMA_MODE,
    OLLAMA_API_KEY,
    READY_MARKERS,
    IDLE_SECONDS,
    MIN_LENGTH,
    TOP_K,
    AI_ANALYSIS_DELIMITER,
    ANALYSIS_SYSTEM_PROMPT,
)

# ============================================================================
# Logging Setup
# ============================================================================
logger = logging.getLogger(__name__)


def setup_logging(log_path: Path) -> None:
    """Setup structured logging with file handler."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger.info("Logging initialized")


# ============================================================================
# Helper Functions
# ============================================================================

def is_ready(file_path: Path) -> bool:
    """
    Check if a journal entry is ready for AI analysis.
    
    Args:
        file_path: Path to the journal entry file
        
    Returns:
        True if the entry is ready for processing, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if AI Analysis already exists (idempotence)
        if AI_ANALYSIS_DELIMITER in content:
            logger.debug(f"Skipping {file_path.name}: AI Analysis already exists")
            return False
        
        # Check for ready markers
        has_marker = any(marker in content for marker in READY_MARKERS)
        if not has_marker:
            logger.debug(f"Skipping {file_path.name}: No ready marker found")
            return False
        
        # Extract user text and check minimum length
        user_text = split_user_text(content)
        
        # Remove YAML frontmatter for length calculation
        text_without_frontmatter = remove_frontmatter(user_text)
        
        if len(text_without_frontmatter.strip()) < MIN_LENGTH:
            logger.debug(f"Skipping {file_path.name}: User text too short ({len(text_without_frontmatter)} chars)")
            return False
        
        logger.info(f"Entry {file_path.name} is ready for processing")
        return True
        
    except Exception as e:
        logger.error(f"Error checking if {file_path} is ready: {e}")
        return False


def remove_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from content."""
    if content.startswith('---\n'):
        # Find the closing ---
        parts = content.split('---\n', 2)
        if len(parts) >= 3:
            return parts[2]
    return content


def split_user_text(content: str) -> str:
    """
    Split content to extract only user-written text.
    
    Args:
        content: Full file content
        
    Returns:
        User-written portion (everything before AI Analysis delimiter)
    """
    if AI_ANALYSIS_DELIMITER in content:
        return content.split(AI_ANALYSIS_DELIMITER)[0]
    return content


def append_analysis(file_path: Path, analysis: str) -> None:
    """
    Append AI analysis to a journal entry file.
    
    Args:
        file_path: Path to the journal entry
        analysis: Generated AI analysis text
    """
    try:
        # Read existing content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if there's already a populated AI Analysis section
        # (checking for the delimiter + some content after it)
        if AI_ANALYSIS_DELIMITER in content:
            parts = content.split(AI_ANALYSIS_DELIMITER, 1)
            if len(parts) > 1 and parts[1].strip():
                logger.warning(f"AI Analysis already exists in {file_path.name}, skipping append")
                return
            # If AI Analysis exists but is empty, we'll replace it
            content = parts[0].rstrip()
        
        # Prepare the analysis section (without timestamp)
        analysis_section = f"{AI_ANALYSIS_DELIMITER}\n\n{analysis}\n"
        
        # Atomic write
        new_content = content + analysis_section
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.info(f"Successfully appended AI analysis to {file_path.name}")
        
    except Exception as e:
        logger.error(f"Error appending analysis to {file_path}: {e}")
        raise


# ============================================================================
# RAG Components
# ============================================================================

def retrieve_context(
    query_text: str,
    index: VectorStoreIndex,
    docstore: SimpleDocumentStore,
    llm: Ollama,
    min_journal_entries: int = 5,
    max_context_files: int = 5,
    max_total_nodes: int = 20
) -> List[str]:
    """
    Retrieve relevant past journal entries and context files using hybrid search.
    
    Ensures at least min_journal_entries journal entries are retrieved, with
    up to max_context_files context files, up to max_total_nodes total.
    
    Args:
        query_text: The current entry text to find similar entries for
        index: Vector store index
        docstore: BM25 document store
        llm: LLM instance for fusion retriever
        min_journal_entries: Minimum number of journal entries to retrieve (default: 5)
        max_context_files: Maximum number of context files to include (default: 5)
        max_total_nodes: Maximum total nodes to return (default: 20)
        
    Returns:
        List of formatted context snippets from past entries and context files
    """
    try:
        # Retrieve more nodes than needed to ensure we have good pool to filter from
        retrieve_k = 25
        logger.info(f"Retrieving up to {retrieve_k} nodes (target: {min_journal_entries} min journals, {max_context_files} max context, {max_total_nodes} max total)...")
        
        # Vector retriever
        vector_retriever = index.as_retriever(similarity_top_k=retrieve_k)
        
        # BM25 retriever
        num_docs = len(docstore.docs)
        bm25_top_k = min(retrieve_k, num_docs)
        
        if num_docs == 0:
            logger.info("No documents in store yet, using vector retrieval only")
            nodes = vector_retriever.retrieve(query_text)
        else:
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=docstore,
                similarity_top_k=bm25_top_k
            )
            
            # Fusion retriever
            retriever = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                similarity_top_k=retrieve_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
                llm=llm
            )
            
            nodes = retriever.retrieve(query_text)
        
        # Separate journal entries from context files
        journal_nodes = [n for n in nodes if n.metadata.get('doc_type') == 'journal']
        context_nodes = [n for n in nodes if n.metadata.get('doc_type') == 'context']
        
        # Take minimum required journal entries
        selected_journals = journal_nodes[:min_journal_entries]
        
        # Take up to max_context_files context files
        selected_context = context_nodes[:max_context_files]
        
        # Calculate remaining slots up to max_total_nodes
        current_count = len(selected_journals) + len(selected_context)
        remaining_slots = max_total_nodes - current_count
        
        if remaining_slots > 0:
            # Fill remaining slots with additional journal entries
            additional_journals = journal_nodes[min_journal_entries:min_journal_entries + remaining_slots]
            final_nodes = selected_journals + selected_context + additional_journals
        else:
            final_nodes = selected_journals + selected_context
        
        # Format the retrieved nodes
        context_snippets = []
        journal_count = 0
        context_count = 0
        
        for node in final_nodes:
            doc_type = node.metadata.get('doc_type', 'journal')
            filename = node.metadata.get('filename', 'Unknown')
            text = node.get_content(metadata_mode='none')
            
            if doc_type == 'journal':
                journal_count += 1
                date = node.metadata.get('date', filename.replace('.md', ''))
                snippet = f"[Past Journal Entry - [[{date}]]]\n{text[:500]}..."
            else:
                context_count += 1
                file_basename = filename.replace('.md', '')
                title = file_basename.replace('-', ' ').replace('_', ' ').title()
                snippet = f"[Background Context: {title} - link as [[{file_basename}]]]\n{text[:500]}..."
            
            context_snippets.append(snippet)
        
        logger.info(f"Retrieved {journal_count} journal entries and {context_count} context files (total: {len(final_nodes)} nodes)")
        return context_snippets
        
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return []


def generate_analysis(
    entry_text: str,
    context_snippets: List[str],
    llm: Ollama
) -> str:
    """
    Generate AI analysis for a journal entry.
    
    Args:
        entry_text: The current journal entry text
        context_snippets: Retrieved context from past entries
        llm: LLM instance
        
    Returns:
        Generated analysis text
    """
    try:
        logger.info("Generating AI analysis...")
        
        # Build context section
        if context_snippets:
            context_section = "\n\n".join(context_snippets)
            context_intro = f"Here are some relevant past entries for context:\n\n{context_section}\n\n---\n\n"
        else:
            context_intro = "This is the first entry being analyzed, so no past context is available.\n\n---\n\n"
        
        # Build the prompt
        user_prompt = f"""{context_intro}Current journal entry to analyze:

{entry_text}

Please provide your analysis following the exact structure specified in your system prompt."""
        
        # Generate response
        messages = [
            ChatMessage(role="system", content=ANALYSIS_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt)
        ]
        
        response = llm.chat(messages)
        analysis = response.message.content
        
        # Log thinking activity if present
        thinking_blocks = [block for block in response.message.blocks if hasattr(block, '__class__') and block.__class__.__name__ == 'ThinkingBlock']
        if thinking_blocks:
            thinking_length = sum(len(block.content) for block in thinking_blocks)
            logger.info(f"AI generated {thinking_length} chars of internal reasoning")
        
        logger.info("AI analysis generated successfully")
        return analysis
        
    except Exception as e:
        logger.error(f"Error generating analysis: {e}")
        raise


# ============================================================================
# Index Management
# ============================================================================

def upsert_index(
    file_path: Path,
    index: VectorStoreIndex,
    docstore: SimpleDocumentStore
) -> None:
    """
    Update the vector index and BM25 docstore with a journal entry.
    
    Args:
        file_path: Path to the journal entry
        index: Vector store index
        docstore: BM25 document store
    """
    try:
        logger.info(f"Upserting {file_path.name} to index...")
        
        # Read file and extract user text only
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        user_text = split_user_text(content)
        
        # Create document with absolute path as ID
        doc_id = str(file_path.absolute())
        filename = file_path.name
        
        document = Document(
            text=user_text,
            doc_id=doc_id,
            metadata={
                'doc_type': 'journal',
                'filename': filename,
                'file_path': str(file_path),
                'date': filename.replace('.md', ''),  # YYYY-MM-DD format
            }
        )
        
        # Check if document already exists
        existing_doc_ids = list(docstore.docs.keys()) if hasattr(docstore, 'docs') else []
        
        if doc_id in existing_doc_ids:
            logger.info(f"Updating existing document: {filename}")
            index.refresh_ref_docs([document])
        else:
            logger.info(f"Inserting new document: {filename}")
            index.insert(document)
        
        # Update BM25 docstore
        docstore.add_documents([document])
        
        # Persist both stores
        index.storage_context.persist(persist_dir=str(INDEX_PERSIST_PATH))
        docstore.persist(persist_path=str(INDEX_PERSIST_PATH / "bm25_docstore.json"))
        
        logger.info(f"Successfully upserted {filename} to index")
        
    except Exception as e:
        logger.error(f"Error upserting {file_path} to index: {e}")
        raise


# ============================================================================
# Entry Processing Pipeline
# ============================================================================

def process_entry(
    file_path: Path,
    index: VectorStoreIndex,
    docstore: SimpleDocumentStore,
    llm: Ollama,
    fusion_llm: Ollama
) -> bool:
    """
    Process a journal entry: retrieve context, generate analysis, append, and update index.
    
    Args:
        file_path: Path to the journal entry
        index: Vector store index
        docstore: BM25 document store
        llm: LLM instance for generation
        fusion_llm: LLM instance for retrieval fusion
    """
    try:
        logger.info(f"Processing entry: {file_path.name}")
        
        # 1. Check if ready
        if not is_ready(file_path):
            logger.info(f"Entry {file_path.name} not ready yet; will retry on next modification")
            return False
        
        # 2. Read file and extract user text
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        user_text = split_user_text(content)
        
        # 3. Retrieve context from past entries
        context_snippets = retrieve_context(user_text, index, docstore, fusion_llm)
        
        # 4. Generate AI analysis
        analysis = generate_analysis(user_text, context_snippets, llm)
        
        # 5. Append analysis to file
        append_analysis(file_path, analysis)
        
        # 6. Update index with the entry (before AI analysis was added)
        upsert_index(file_path, index, docstore)
        
        logger.info(f"Successfully processed entry: {file_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing entry {file_path}: {e}", exc_info=True)
        # Don't re-raise - we want the watcher to continue
        return False


# ============================================================================
# File Watcher
# ============================================================================

class JournalWatcherHandler(FileSystemEventHandler):
    """Watches for journal file modifications with debounce logic."""
    
    def __init__(
        self,
        index: VectorStoreIndex,
        docstore: SimpleDocumentStore,
        llm: Ollama,
        fusion_llm: Ollama,
        idle_seconds: int = IDLE_SECONDS
    ):
        super().__init__()
        self.index = index
        self.docstore = docstore
        self.llm = llm
        self.fusion_llm = fusion_llm
        self.idle_seconds = idle_seconds
        self.file_timestamps: Dict[str, float] = {}
        self.processed_files: set = set()
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        # Only process markdown files
        if not event.src_path.endswith('.md'):
            return
        
        file_path = Path(event.src_path)
        
        # Update timestamp
        self.file_timestamps[event.src_path] = time.time()
        # Allow re-processing after subsequent modifications
        self.processed_files.discard(event.src_path)
        logger.debug(f"File modified: {file_path.name}")
        
        # Schedule check after idle period
        # This will be handled by the periodic check in main()
    
    def check_idle_files(self) -> None:
        """Check for files that have been idle and process them."""
        current_time = time.time()
        files_to_process = []
        
        for file_path_str, last_modified in list(self.file_timestamps.items()):
            idle_time = current_time - last_modified
            
            if file_path_str in self.processed_files:
                continue
            
            if idle_time >= self.idle_seconds:
                # File has been idle long enough
                files_to_process.append(file_path_str)
                
                # Remove from tracking
                del self.file_timestamps[file_path_str]
        
        # Process files
        for file_path_str in files_to_process:
            file_path = Path(file_path_str)
            if file_path.exists():
                processed = process_entry(file_path, self.index, self.docstore, self.llm, self.fusion_llm)
                if processed:
                    self.processed_files.add(file_path_str)


# ============================================================================
# Main
# ============================================================================

def initialize_index_and_docstore(persist_dir: Path) -> tuple[VectorStoreIndex, SimpleDocumentStore]:
    """
    Initialize or load the vector index and BM25 docstore.
    
    Args:
        persist_dir: Directory for persistence
        
    Returns:
        Tuple of (index, docstore)
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    docstore_path = persist_dir / "bm25_docstore.json"
    
    # Try to load existing index
    try:
        if (persist_dir / "docstore.json").exists():
            logger.info(f"Loading existing index from {persist_dir}")
            # Load with explicit stores, no graph store
            storage_context = StorageContext.from_defaults(
                persist_dir=str(persist_dir),
                vector_store=SimpleVectorStore.from_persist_dir(str(persist_dir)),
                docstore=SimpleDocumentStore.from_persist_dir(str(persist_dir)),
                index_store=SimpleIndexStore.from_persist_dir(str(persist_dir))
            )
            index = load_index_from_storage(storage_context)
            logger.info("Index loaded successfully")
        else:
            raise FileNotFoundError("Index not found, creating new one")
    except Exception as e:
        logger.info(f"Creating new index: {e}")
        # Create with explicit stores, no graph store
        storage_context = StorageContext.from_defaults(
            vector_store=SimpleVectorStore(),
            docstore=SimpleDocumentStore(),
            index_store=SimpleIndexStore()
        )
        index = VectorStoreIndex([], storage_context=storage_context)
        index.storage_context.persist(persist_dir=str(persist_dir))
        logger.info("New index created and persisted")
    
    # Try to load existing BM25 docstore
    try:
        if docstore_path.exists():
            logger.info(f"Loading BM25 docstore from {docstore_path}")
            docstore = SimpleDocumentStore.from_persist_path(str(docstore_path))
            logger.info("BM25 docstore loaded successfully")
        else:
            raise FileNotFoundError("BM25 docstore not found, creating new one")
    except Exception as e:
        logger.info(f"Creating new BM25 docstore: {e}")
        docstore = SimpleDocumentStore()
        docstore.persist(persist_path=str(docstore_path))
        logger.info("New BM25 docstore created and persisted")
    
    return index, docstore


def main():
    """Main entry point for the journal watcher."""
    parser = argparse.ArgumentParser(description="AI Journal Analysis Watcher")
    parser.add_argument(
        '--vault',
        type=Path,
        default=VAULT_PATH,
        help=f"Path to Obsidian vault (default: {VAULT_PATH})"
    )
    args = parser.parse_args()
    
    # Update paths based on vault argument
    vault_path = args.vault
    daily_notes_path = vault_path / DAILY_DIR
    index_persist_path = vault_path / PERSIST_DIR
    log_file_path = vault_path / LOG_PATH
    
    # Setup logging
    setup_logging(log_file_path)
    
    logger.info("=" * 60)
    logger.info("AI Journal Analysis Watcher Starting")
    logger.info("=" * 60)
    logger.info(f"Vault Path: {vault_path}")
    logger.info(f"Daily Notes Path: {daily_notes_path}")
    logger.info(f"Index Path: {index_persist_path}")
    logger.info(f"LLM Model: {LLM_MODEL}")
    logger.info(f"Embedding Model: {EMBED_MODEL}")
    logger.info(f"Idle Seconds: {IDLE_SECONDS}")
    logger.info(f"Minimum Length: {MIN_LENGTH} chars")
    logger.info(f"Top K Retrieval: {TOP_K}")
    
    # Check if daily notes directory exists
    if not daily_notes_path.exists():
        logger.error(f"Daily notes directory not found: {daily_notes_path}")
        print(f"Error: Daily notes directory not found: {daily_notes_path}")
        return
    
    # Initialize Ollama models
    logger.info("Initializing Ollama models...")
    logger.info(f"Mode: {OLLAMA_MODE}")
    try:
        # Embedding model always uses local Ollama
        embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url="http://localhost:11434"  # Always local for embeddings
        )
        Settings.embed_model = embed_model
        logger.info(f"Embedding model initialized: {EMBED_MODEL} (local)")
        
        # Main LLM - conditional client for cloud mode
        if OLLAMA_MODE == "cloud":
            logger.info("Creating cloud client with authentication...")
            ollama_client = Client(
                host=OLLAMA_URL,
                headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'}
            )
            llm = Ollama(
                model=LLM_MODEL,
                base_url=OLLAMA_URL,
                request_timeout=600.0,
                additional_kwargs={"think": "medium"},
                client=ollama_client
            )
            logger.info(f"LLM initialized: {LLM_MODEL} (cloud, thinking mode: medium)")
        else:
            llm = Ollama(
                model=LLM_MODEL,
                base_url=OLLAMA_URL,
                request_timeout=600.0,
                additional_kwargs={"think": "medium"}
            )
            logger.info(f"LLM initialized: {LLM_MODEL} (local, thinking mode: medium)")
        
        # Fusion LLM (lightweight for retrieval) - always local for speed
        fusion_llm = Ollama(
            model=LLM_MODEL if OLLAMA_MODE == "local" else "gpt-oss:20b",
            base_url="http://localhost:11434",
            request_timeout=60.0
        )
        logger.info("Fusion LLM initialized (local)")
        
    except Exception as e:
        logger.error(f"Error initializing Ollama models: {e}")
        print(f"Error: Could not connect to Ollama. Make sure it's running at {OLLAMA_URL}")
        print(f"Details: {e}")
        return
    
    # Initialize index and docstore
    logger.info("Initializing index and docstore...")
    try:
        index, docstore = initialize_index_and_docstore(index_persist_path)
        logger.info("Index and docstore ready")
    except Exception as e:
        logger.error(f"Error initializing index/docstore: {e}")
        print(f"Error initializing index/docstore: {e}")
        return
    
    # Create event handler
    event_handler = JournalWatcherHandler(
        index=index,
        docstore=docstore,
        llm=llm,
        fusion_llm=fusion_llm,
        idle_seconds=IDLE_SECONDS
    )
    
    # Create and start observer
    observer = Observer()
    observer.schedule(event_handler, str(daily_notes_path), recursive=False)
    observer.start()
    
    logger.info(f"Watching directory: {daily_notes_path}")
    logger.info("Watcher is running. Press Ctrl+C to stop.")
    print(f"\nWatching: {daily_notes_path}")
    print(f"Log file: {log_file_path}")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            time.sleep(5)  # Check every 5 seconds
            event_handler.check_idle_files()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
        observer.stop()
        print("\nStopping watcher...")
    
    observer.join()
    logger.info("Watcher stopped successfully")
    print("Watcher stopped.")


if __name__ == "__main__":
    main()

