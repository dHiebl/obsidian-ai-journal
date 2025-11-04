#!/usr/bin/env python3
"""
Test script to verify the AI Journal Analysis system components.
Run this before starting the watcher to ensure everything is set up correctly.

Make sure to run this from within your virtual environment:
    source .venv/bin/activate
    python test_components.py
"""
import sys
from pathlib import Path

# Check Python version
if sys.version_info < (3, 12):
    print(f"❌ Python 3.12+ required, but you have {sys.version_info.major}.{sys.version_info.minor}")
    print("Please upgrade Python and recreate your virtual environment.")
    sys.exit(1)

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    try:
        import watchdog
        print("✓ watchdog")
        import llama_index
        print("✓ llama-index")
        import yaml
        print("✓ PyYAML")
        from llama_index.llms.ollama import Ollama
        print("✓ llama-index-llms-ollama")
        from llama_index.embeddings.ollama import OllamaEmbedding
        print("✓ llama-index-embeddings-ollama")
        from llama_index.retrievers.bm25 import BM25Retriever
        print("✓ llama-index-retrievers-bm25")
        print("\n✅ All required packages are installed!\n")
        return True
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nPlease run: pip install -r requirements.txt\n")
        return False

def test_config():
    """Test that configuration is valid."""
    print("Testing configuration...")
    try:
        from config import (
            VAULT_PATH, DAILY_NOTES_PATH, INDEX_PERSIST_PATH, LOG_FILE_PATH,
            LLM_MODEL, EMBED_MODEL, OLLAMA_URL, READY_MARKERS, IDLE_SECONDS,
            MIN_LENGTH, TOP_K
        )
        
        print(f"✓ Vault Path: {VAULT_PATH}")
        print(f"✓ Daily Notes: {DAILY_NOTES_PATH}")
        print(f"✓ Index Path: {INDEX_PERSIST_PATH}")
        print(f"✓ LLM Model: {LLM_MODEL}")
        print(f"✓ Embed Model: {EMBED_MODEL}")
        print(f"✓ Ollama URL: {OLLAMA_URL}")
        print("\n✅ Configuration loaded successfully!\n")
        return True
    except Exception as e:
        print(f"\n❌ Configuration error: {e}\n")
        return False

def test_directories():
    """Test that required directories exist or can be created."""
    print("Testing directories...")
    from config import VAULT_PATH, DAILY_NOTES_PATH, INDEX_PERSIST_PATH
    
    if not VAULT_PATH.exists():
        print(f"❌ Vault path does not exist: {VAULT_PATH}")
        return False
    print(f"✓ Vault exists: {VAULT_PATH}")
    
    if not DAILY_NOTES_PATH.exists():
        print(f"⚠️  Daily notes directory not found: {DAILY_NOTES_PATH}")
        print("   The watcher will not work until this directory exists.")
        return False
    print(f"✓ Daily notes directory exists: {DAILY_NOTES_PATH}")
    
    # Index directory will be created automatically
    if not INDEX_PERSIST_PATH.exists():
        print(f"ℹ️  Index directory will be created: {INDEX_PERSIST_PATH}")
    else:
        print(f"✓ Index directory exists: {INDEX_PERSIST_PATH}")
    
    print("\n✅ Directories are ready!\n")
    return True

def test_ollama_connection():
    """Test connection to Ollama."""
    print("Testing Ollama connection...")
    try:
        from config import OLLAMA_URL, LLM_MODEL, EMBED_MODEL
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding
        
        # Try to initialize models
        print(f"Connecting to Ollama at {OLLAMA_URL}...")
        
        # Test embedding model
        print(f"Testing embedding model: {EMBED_MODEL}")
        embed = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_URL)
        print(f"✓ Embedding model connected: {EMBED_MODEL}")
        
        # Test LLM
        print(f"Testing LLM: {LLM_MODEL}")
        llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL, request_timeout=10.0)
        print(f"✓ LLM connected: {LLM_MODEL}")
        
        print("\n✅ Ollama connection successful!\n")
        return True
    except Exception as e:
        print(f"\n❌ Ollama connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Check if models are installed:")
        print(f"   ollama pull {EMBED_MODEL}")
        print(f"   ollama pull {LLM_MODEL}")
        print("3. Verify Ollama is accessible at:", OLLAMA_URL)
        print()
        return False

def test_helper_functions():
    """Test helper functions."""
    print("Testing helper functions...")
    try:
        from watch_vault import split_user_text, remove_frontmatter
        
        # Test split_user_text
        test_content = """---
date: 2025-11-04
---

## Entry
This is my entry.

---

## AI Analysis (2025-11-04 14:30)
This is AI analysis."""
        
        user_text = split_user_text(test_content)
        assert "AI Analysis" not in user_text
        print("✓ split_user_text() works correctly")
        
        # Test remove_frontmatter
        clean_text = remove_frontmatter(user_text)
        assert not clean_text.startswith("---")
        print("✓ remove_frontmatter() works correctly")
        
        print("\n✅ Helper functions work correctly!\n")
        return True
    except Exception as e:
        print(f"\n❌ Helper function test failed: {e}\n")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("AI Journal Analysis System - Component Tests")
    print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("=" * 60)
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("Configuration", test_config),
        ("Directories", test_directories),
        ("Helper Functions", test_helper_functions),
        ("Ollama Connection", test_ollama_connection),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test '{name}' crashed: {e}\n")
            results.append((name, False))
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print()
    if all_passed:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo start the watcher, run:")
        print("  python watch_vault.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above before running the watcher.")
        return 1
    
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())

