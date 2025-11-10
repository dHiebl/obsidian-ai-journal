"""
7-Day Summary Generation Script
Generates AI-powered summaries from the last 7 days of journal entries.
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

from llama_index.core.llms import ChatMessage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from ollama import Client

from watch_vault import initialize_index_and_docstore, setup_logging, split_user_text
from config import (
    DAILY_NOTES_PATH,
    WEEKLY_PATH,
    INDEX_PERSIST_PATH,
    LOG_FILE_PATH,
    LLM_MODEL,
    EMBED_MODEL,
    OLLAMA_URL,
    OLLAMA_MODE,
    OLLAMA_API_KEY,
    WEEKLY_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def get_last_seven_days() -> tuple[datetime, datetime]:
    """
    Calculate the date range for the last 7 days from today.
    
    Returns:
        Tuple of (start_date, end_date) where end_date is today
    """
    today = datetime.now()
    start_date = today - timedelta(days=6)  # 6 days ago + today = 7 days total
    
    return start_date, today


def load_week_entries(
    daily_notes_path: Path,
    start_date: datetime,
    end_date: datetime
) -> List[tuple[datetime, str, str]]:
    """
    Load 7 consecutive daily journal entries.
    
    Args:
        daily_notes_path: Path to daily notes directory
        start_date: Start date (oldest day)
        end_date: End date (most recent day, typically today)
        
    Returns:
        List of tuples: (date, filename, content)
    """
    entries = []
    current_date = start_date
    
    while current_date <= end_date:
        filename = f"{current_date.strftime('%Y-%m-%d')}.md"
        file_path = daily_notes_path / filename
        
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract user text only (remove AI Analysis if present)
                user_text = split_user_text(content)
                
                entries.append((current_date, filename, user_text))
                logger.info(f"Loaded: {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                entries.append((current_date, filename, f"[Could not load entry: {e}]"))
        else:
            logger.warning(f"Missing entry: {filename}")
            entries.append((current_date, filename, "[No entry for this day]"))
        
        current_date += timedelta(days=1)
    
    return entries


def retrieve_relevant_context(
    combined_week_text: str,
    index,
    llm,
    top_k: int = 3
) -> List[str]:
    """
    Retrieve relevant context files for the 7-day period's content.
    
    Args:
        combined_week_text: Combined text from all 7 entries
        index: Vector store index
        llm: LLM instance for retrieval
        top_k: Number of context files to retrieve
        
    Returns:
        List of formatted context snippets
    """
    try:
        logger.info(f"Retrieving top {top_k} relevant context files...")
        
        # Use vector retriever only (simpler for context files)
        vector_retriever = index.as_retriever(similarity_top_k=top_k * 2)
        nodes = vector_retriever.retrieve(combined_week_text)
        
        # Filter to only context files
        context_nodes = [n for n in nodes if n.metadata.get('doc_type') == 'context'][:top_k]
        
        # Format context files
        context_snippets = []
        for node in context_nodes:
            filename = node.metadata.get('filename', 'Unknown')
            file_basename = filename.replace('.md', '')
            title = file_basename.replace('-', ' ').replace('_', ' ').title()
            text = node.get_content(metadata_mode='none')
            snippet = f"[Background Context: {title} - link as [[{file_basename}]]]\n{text}"
            context_snippets.append(snippet)
        
        logger.info(f"Retrieved {len(context_snippets)} context file(s)")
        return context_snippets
        
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return []


def generate_weekly_summary(
    entries: List[tuple[datetime, str, str]],
    context_snippets: List[str],
    llm: Ollama
) -> str:
    """
    Generate 7-day summary using LLM.
    
    Args:
        entries: List of (date, filename, content) tuples
        context_snippets: Retrieved context files
        llm: LLM instance
        
    Returns:
        Generated summary
    """
    try:
        logger.info("Generating 7-day summary...")
        
        # Build the daily entries section
        daily_section = []
        for date, filename, content in entries:
            day_name = date.strftime('%A')
            date_str = date.strftime('%Y-%m-%d')
            daily_section.append(f"### {day_name} [[{date_str}]]\n\n{content}\n")
        
        daily_entries_text = "\n---\n\n".join(daily_section)
        
        # Build context section
        if context_snippets:
            context_section = "\n\n".join(context_snippets)
            context_intro = f"Background context about the person:\n\n{context_section}\n\n{'=' * 60}\n\n"
        else:
            context_intro = ""
        
        # Build the prompt
        user_prompt = f"""{context_intro}Here are the journal entries from the last 7 days:

{daily_entries_text}

Please provide your analysis following the exact structure specified in your system prompt."""
        
        # Generate response
        messages = [
            ChatMessage(role="system", content=WEEKLY_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt)
        ]
        
        response = llm.chat(messages)
        summary = response.message.content
        
        # Log thinking activity if present
        thinking_blocks = [block for block in response.message.blocks if hasattr(block, '__class__') and block.__class__.__name__ == 'ThinkingBlock']
        if thinking_blocks:
            thinking_length = sum(len(block.content) for block in thinking_blocks)
            logger.info(f"AI generated {thinking_length} chars of internal reasoning")
        
        logger.info("7-day summary generated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating 7-day summary: {e}")
        raise


def save_weekly_summary(
    weekly_path: Path,
    start_date: datetime,
    end_date: datetime,
    summary: str
) -> Path:
    """
    Save weekly summary to file.
    
    Args:
        weekly_path: Path to weekly summaries directory
        start_date: Start date of period
        end_date: End date of period
        summary: Generated summary text
        
    Returns:
        Path to saved file
    """
    # Create directory if needed
    weekly_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename using date range
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    filename = f"{start_str}_to_{end_str}.md"
    file_path = weekly_path / filename
    
    # Build file content
    content = f"""---
date_range: {start_str} to {end_str}
generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
---

# 7-Day Summary

**Period:** [[{start_str}]] to [[{end_str}]]

---

{summary}
"""
    
    # Write file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"7-day summary saved to: {file_path}")
    return file_path


def main():
    """Main entry point for 7-day summary generation."""
    # Setup logging
    setup_logging(LOG_FILE_PATH)
    
    logger.info("=" * 60)
    logger.info("7-Day Summary Generation Starting")
    logger.info("=" * 60)
    
    # Calculate last 7 days from today
    start_date, end_date = get_last_seven_days()
    logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Initialize embedding model (always local)
    logger.info("Initializing embedding model...")
    logger.info(f"Mode: {OLLAMA_MODE}")
    try:
        embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url="http://localhost:11434"  # Always local for embeddings
        )
        Settings.embed_model = embed_model
        logger.info(f"Embedding model initialized: {EMBED_MODEL} (local)")
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        print(f"Error: Could not connect to local Ollama for embeddings. Make sure it's running at http://localhost:11434")
        print(f"Details: {e}")
        return
    
    # Initialize LLM - conditional client for cloud mode
    logger.info("Initializing LLM...")
    try:
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
                additional_kwargs={"think": "high"},
                client=ollama_client
            )
            logger.info(f"LLM initialized: {LLM_MODEL} (cloud, thinking mode: high)")
        else:
            llm = Ollama(
                model=LLM_MODEL,
                base_url=OLLAMA_URL,
                request_timeout=600.0,
                additional_kwargs={"think": "high"}
            )
            logger.info(f"LLM initialized: {LLM_MODEL} (local, thinking mode: high)")
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        print(f"Error initializing LLM: {e}")
        return
    
    # Initialize index
    logger.info("Loading index...")
    try:
        index, _ = initialize_index_and_docstore(INDEX_PERSIST_PATH)
        logger.info("Index loaded successfully")
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        print(f"Error loading index: {e}")
        return
    
    # Load last 7 days of entries
    logger.info("Loading entries from last 7 days...")
    try:
        entries = load_week_entries(DAILY_NOTES_PATH, start_date, end_date)
        
        # Check if we have enough entries
        actual_entries = [e for e in entries if "[No entry for this day]" not in e[2] and "[Could not load entry]" not in e[2]]
        if len(actual_entries) < 3:
            logger.warning(f"Only {len(actual_entries)} valid entries found for this period")
            print(f"\nWarning: Only {len(actual_entries)} valid entries found for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print("Generate summary anyway? [y/N]: ", end='')
            response = input().strip().lower()
            if response != 'y':
                print("Aborted.")
                return
        
        logger.info(f"Loaded {len(entries)} entries ({len(actual_entries)} valid)")
    except Exception as e:
        logger.error(f"Error loading entries: {e}")
        print(f"Error loading entries: {e}")
        return
    
    # Combine all entry texts for context retrieval
    combined_text = "\n\n".join([content for _, _, content in entries])
    
    # Retrieve relevant context files
    context_snippets = retrieve_relevant_context(combined_text, index, llm)
    
    # Generate weekly summary
    try:
        summary = generate_weekly_summary(entries, context_snippets, llm)
    except Exception as e:
        logger.error(f"Error generating summary: {e}", exc_info=True)
        print(f"\nError generating summary: {e}")
        return
    
    # Save summary
    try:
        file_path = save_weekly_summary(WEEKLY_PATH, start_date, end_date, summary)
        
        logger.info("=" * 60)
        logger.info("7-day summary generation complete!")
        logger.info(f"Saved to: {file_path}")
        logger.info("=" * 60)
        
        print("\n" + "=" * 60)
        print("7-Day Summary Generated Successfully!")
        print(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"  Saved to: {file_path}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"Error saving summary: {e}", exc_info=True)
        print(f"\nError saving summary: {e}")
        return


if __name__ == "__main__":
    main()

