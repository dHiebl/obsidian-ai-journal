"""
Context File Ingestion Script
Loads markdown files from Journal/Context/ and adds them to the RAG index.
"""
import logging
from pathlib import Path
from typing import List

from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

from watch_vault import initialize_index_and_docstore, setup_logging
from config import (
    CONTEXT_PATH,
    INDEX_PERSIST_PATH,
    LOG_FILE_PATH,
    EMBED_MODEL,
    OLLAMA_URL,
)

logger = logging.getLogger(__name__)


def ingest_context_files(
    context_dir: Path,
    index,
    docstore
) -> None:
    """
    Ingest all markdown files from the context directory into the index.
    
    Args:
        context_dir: Path to the context files directory
        index: Vector store index
        docstore: BM25 document store
    """
    if not context_dir.exists():
        logger.warning(f"Context directory not found: {context_dir}")
        logger.info(f"Creating directory: {context_dir}")
        context_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Add your context markdown files to this directory and run again.")
        return
    
    # Find all markdown files
    md_files = list(context_dir.glob("*.md"))
    
    if not md_files:
        logger.warning(f"No markdown files found in {context_dir}")
        logger.info("Add your context markdown files (.md) to this directory and run again.")
        return
    
    logger.info(f"Found {len(md_files)} markdown file(s) to ingest")
    
    ingested_count = 0
    updated_count = 0
    
    for file_path in md_files:
        try:
            logger.info(f"Processing: {file_path.name}")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Skipping empty file: {file_path.name}")
                continue
            
            # Create document with absolute path as ID
            doc_id = str(file_path.absolute())
            filename = file_path.name
            title = filename.replace('.md', '').replace('-', ' ').replace('_', ' ').title()
            
            document = Document(
                text=content,
                doc_id=doc_id,
                metadata={
                    'doc_type': 'context',
                    'filename': filename,
                    'file_path': str(file_path),
                    'title': title,
                }
            )
            
            # Check if document already exists
            existing_doc_ids = list(docstore.docs.keys()) if hasattr(docstore, 'docs') else []
            
            if doc_id in existing_doc_ids:
                logger.info(f"Updating existing context file: {filename}")
                index.refresh_ref_docs([document])
                updated_count += 1
            else:
                logger.info(f"Inserting new context file: {filename}")
                index.insert(document)
                ingested_count += 1
            
            # Update BM25 docstore
            docstore.add_documents([document])
            
            logger.info(f"Successfully processed: {filename}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)
            continue
    
    # Persist both stores
    try:
        index.storage_context.persist(persist_dir=str(INDEX_PERSIST_PATH))
        docstore.persist(persist_path=str(INDEX_PERSIST_PATH / "bm25_docstore.json"))
        logger.info("Index and docstore persisted successfully")
    except Exception as e:
        logger.error(f"Error persisting stores: {e}")
        raise
    
    logger.info("=" * 60)
    logger.info(f"Context ingestion complete!")
    logger.info(f"  New files ingested: {ingested_count}")
    logger.info(f"  Existing files updated: {updated_count}")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("Context Ingestion Complete!")
    print(f"  New files ingested: {ingested_count}")
    print(f"  Existing files updated: {updated_count}")
    print(f"  Context directory: {context_dir}")
    print("=" * 60 + "\n")


def main():
    """Main entry point for context ingestion."""
    # Setup logging
    setup_logging(LOG_FILE_PATH)
    
    logger.info("=" * 60)
    logger.info("Context File Ingestion Starting")
    logger.info("=" * 60)
    logger.info(f"Context Path: {CONTEXT_PATH}")
    logger.info(f"Index Path: {INDEX_PERSIST_PATH}")
    
    # Initialize embedding model
    logger.info("Initializing embedding model...")
    try:
        embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=OLLAMA_URL
        )
        Settings.embed_model = embed_model
        logger.info(f"Embedding model initialized: {EMBED_MODEL}")
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        print(f"Error: Could not connect to Ollama. Make sure it's running at {OLLAMA_URL}")
        print(f"Details: {e}")
        return
    
    # Initialize index and docstore
    logger.info("Loading index and docstore...")
    try:
        index, docstore = initialize_index_and_docstore(INDEX_PERSIST_PATH)
        logger.info("Index and docstore loaded successfully")
    except Exception as e:
        logger.error(f"Error loading index/docstore: {e}")
        print(f"Error loading index/docstore: {e}")
        return
    
    # Ingest context files
    try:
        ingest_context_files(CONTEXT_PATH, index, docstore)
    except Exception as e:
        logger.error(f"Error during context ingestion: {e}", exc_info=True)
        print(f"\nError during context ingestion: {e}")
        return


if __name__ == "__main__":
    main()

