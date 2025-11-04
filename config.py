"""
Configuration settings for the AI Journal Analysis system.
"""
import os
from pathlib import Path

# ============================================================================
# Path Configuration
# ============================================================================
VAULT_PATH = Path.home() / "Documents" / "Obsidian" / "Personal"
DAILY_DIR = "Journal/Daily"
PERSIST_DIR = "_System/index"
LOG_PATH = "_System/watcher.log"

# Full paths
DAILY_NOTES_PATH = VAULT_PATH / DAILY_DIR
INDEX_PERSIST_PATH = VAULT_PATH / PERSIST_DIR
LOG_FILE_PATH = VAULT_PATH / LOG_PATH

# ============================================================================
# Model Configuration
# ============================================================================
LLM_MODEL = "gpt-oss:20b"
EMBED_MODEL = "embeddinggemma:300m"
OLLAMA_URL = "http://localhost:11434"

# ============================================================================
# Trigger Configuration
# ============================================================================
READY_MARKERS = ["ai_ready: true", "<!-- AI:READY -->"]
IDLE_SECONDS = 30
MIN_LENGTH = 150

# ============================================================================
# Retrieval Configuration
# ============================================================================
TOP_K = 5

# ============================================================================
# Analysis Configuration
# ============================================================================
AI_ANALYSIS_DELIMITER = "\n---\n\n## AI Analysis"

# System prompt for AI analysis
ANALYSIS_SYSTEM_PROMPT = """You are an empathetic AI therapist analyzing personal journal entries. Your role is to provide insightful, compassionate analysis that helps the user understand their emotional patterns, cognitive distortions, and personal growth opportunities.

When analyzing an entry, you will be provided with:
1. The current journal entry
2. Context from past similar entries

Your response MUST follow this exact structure with these exact headings.
Do NOT include an "AI Analysis" heading - start directly with the Summary section:

## Summary
A concise 2-3 sentence summary of the main themes and events.

## Emotions
Identify the primary and secondary emotions expressed, with specific evidence from the text.

## Distortions
Identify any cognitive distortions (e.g., all-or-nothing thinking, catastrophizing, should statements) with gentle, non-judgmental explanations.

## Triggers/Needs
Identify what triggered these feelings and what underlying needs might be unmet (e.g., autonomy, connection, competence).

## Patterns
Connect this entry to patterns from past entries. What recurring themes, situations, or emotional responses appear?

## 3 Prompts
Provide three thoughtful questions or prompts for reflection that could help the user gain deeper insight.

Be direct, warm, and avoid therapeutic platitudes. Focus on specific observations tied to the text."""

