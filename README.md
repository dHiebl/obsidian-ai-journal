# Fully Local Journal AI Analysis

An intelligent watcher that monitors your Obsidian daily journal entries and automatically appends AI-generated insights using RAG (Retrieval-Augmented Generation) with your past entries.

## ⚠️ Folder Structure

This system expects a specific folder structure by default:

```
~/Documents/Obsidian/Personal/          # Your vault
├── Journal/
│   ├── Daily/                          # Daily journal entries
│   ├── Context/                        # (Optional) Background context files
│   └── Weekly/                         # (Optional) Auto-generated weekly summaries
└── _System/                            # System files (index, logs)
    ├── ai/                             # This codebase
    └── index/                          # Vector store (auto-created)
```

**Using a different structure?** Edit `config.py` to customize:
- `DAILY_DIR` - Where your daily journal entries are (default: `"Journal/Daily"`)
- `CONTEXT_DIR` - Where to put background context files (default: `"Journal/Context"`)
- `WEEKLY_DIR` - Where weekly summaries are saved (default: `"Journal/Weekly"`)
- `PERSIST_DIR` - Where to store the vector index (default: `"_System/index"`)
- `LOG_PATH` - Where to write logs (default: `"_System/watcher.log"`)

## Features

- **Automatic Analysis**: Watches your daily journal folder and analyzes entries when you mark them ready
- **Context-Aware**: Uses hybrid retrieval (BM25 + vector search) to find relevant past entries for context
- **Background Context**: Optional support for static context files (personality tests, therapy notes, etc.)
- **Weekly Summaries**: Generate meta-level analysis of entire weeks
- **Smart Triggers**: Only processes entries that are marked ready, meet minimum length (150 chars), and have been idle for 30 seconds
- **Structured Insights**: Generates analysis with sections for Summary, Emotions, Distortions, Triggers/Needs, Patterns, and Reflection Prompts
- **Obsidian Links**: Creates clickable `[[YYYY-MM-DD]]` links to past journal entries when identifying patterns
- **Fully Local**: All processing happens on your machine using Ollama - your data never leaves your computer

## Prerequisites

- **Python 3.12+**
- **Ollama** running locally
- **Obsidian** with daily journal entries in `Journal/Daily/` folder

## Installation

### 1. Install Python Dependencies

Navigate to your project directory and install required packages.

**Using pip:**

```bash
cd ~/Obsidian/Personal/_System/ai
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Using uv (recommended):**

```bash
cd ~/Obsidian/Personal/_System/ai
uv sync --python 3.12
```

### 2. Setup Ollama

Make sure Ollama is installed and running, then pull the required models:

```bash
# Start Ollama server (if not already running)
ollama serve

# Pull the required models
ollama pull gpt-oss:20b
ollama pull embeddinggemma:300m
```

**Note**: The `gpt-oss:20b` model is quite large (~12GB) and runs in thinking mode for deeper analysis (takes longer but produces better insights). It's configured with `"medium"` thinking level for balanced speed and quality. If you prefer a smaller/faster model, you can edit `config.py` and change `LLM_MODEL` to something like `qwen3:8b` or `gemma2:9b`.

## Usage

Run the watcher from the directory where you installed the code:

```bash
cd /path/to/your/vault/_System/ai
python watch_vault.py
```

Or specify a custom vault path:

```bash
python watch_vault.py --vault /path/to/your/vault
```

Stop with `Ctrl+C`.

## Journal Template

Your daily journal entries need to signal when they're ready for analysis. See `daily.md` for a template example.

Add `ai_ready: true` to your frontmatter when ready:

```markdown
---
date: 2025-11-04
ai_ready: true
---

## Entry

Your journal content here...
```

Or use an HTML comment: `<!-- AI:READY -->`

## Configuration

Edit `config.py` to customize:

- **Models**: `LLM_MODEL` (default: `gpt-oss:20b`) or `EMBED_MODEL` (default: `embeddinggemma:300m`)
- **Triggers**: `IDLE_SECONDS` (default: 30), `MIN_LENGTH` (default: 150 characters)
- **Retrieval**: `TOP_K` (default: 5) - number of past entries to use as context
- **Prompt**: `ANALYSIS_SYSTEM_PROMPT` - customize the AI's analysis style

### Using Thinking Models

This project uses **GPT-OSS** with thinking mode enabled for deeper analysis.

#### GPT-OSS Thinking Levels

GPT-OSS supports three thinking levels that control the depth of internal reasoning:

- **`"low"`**: Faster generation, minimal reasoning (good for simple entries)
- **`"medium"`**: Balanced speed and quality (default, recommended)
- **`"high"`**: Most thorough analysis, slowest (for complex entries or deeper insights)

To change the thinking level, edit `watch_vault.py` and `generate_weekly.py`:

```python
llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    request_timeout=600.0,
    additional_kwargs={"think": "high"}  # Change "medium" to "low" or "high"
)
```

**Note**: GPT-OSS requires the `additional_kwargs` approach because it only accepts string values (`"low"`, `"medium"`, `"high"`), not booleans.

#### Other Thinking Models

If you switch to other thinking models like **Qwen3**, **DeepSeek-v3.1**, or **DeepSeek R1**, you can use the simpler `thinking=True` parameter:

```python
llm = Ollama(
    model="qwen3",  # or deepseek-v3.1, deepseek-r1
    base_url=OLLAMA_URL,
    request_timeout=600.0,
    thinking=True  # Simple boolean for most thinking models
)
```

These models accept boolean values directly via LlamaIndex's built-in `thinking` parameter.

## Troubleshooting

**Check logs first**: `_System/watcher.log` contains detailed information about processing.

### Common Issues

**Entry not processing:**
- Add `ai_ready: true` to frontmatter
- Entry must be at least 150 characters
- Wait 30 seconds after your last edit
- Each entry is only processed once per session

**Ollama connection errors:**
- Ensure Ollama is running: `ollama serve`
- Verify models are downloaded: `ollama list`
- Check `http://localhost:11434` is accessible

**Models not found:**
```bash
ollama pull gpt-oss:20b
ollama pull embeddinggemma:300m
```

**Index corrupted:**
Stop the watcher, delete `_System/index/`, restart. It will rebuild automatically.

## Tips

- Start with 3-5 journal entries to build up context before relying on pattern analysis
- The AI analysis supplements your reflection - review and edit as needed
- Try different models in `config.py` to balance quality and speed (e.g., `qwen3:8b`, `gemma2:9b`)
- Customize `ANALYSIS_SYSTEM_PROMPT` to match your preferred analysis style

## Optional Features

### Context Files

Add persistent background information that enriches every analysis (personality tests, family history, therapy notes, etc.).

**Setup:**

1. Create the context folder:
```bash
mkdir -p ~/Documents/Obsidian/Personal/Journal/Context
```

2. Add markdown files with your background information:
```bash
# Example files:
# - big5-personality.md
# - family-history.md
# - therapy-notes.md
```

3. Run the ingestion script:
```bash
cd ~/Documents/Obsidian/Personal/_System/ai
python ingest_context.py
```

**How it works:**
- Context files are indexed once and retrieved when relevant to your daily entries
- The AI will reference them naturally (e.g., "consistent with your high conscientiousness")
- Re-run `ingest_context.py` whenever you add or update context files
- Context files don't need date-based links - they're referenced by title

### Weekly Summaries

Generate meta-level analysis of the last 7 days of journal entries.

**Usage:**

```bash
cd ~/Documents/Obsidian/Personal/_System/ai
python generate_weekly.py
```

This will:
- Gather entries from the last 7 days (from today going back)
- Load all available daily entries from that period
- Retrieve relevant context files
- Generate a comprehensive 7-day analysis
- Save to `Journal/Weekly/YYYY-MM-DD_to_YYYY-MM-DD.md` (e.g., `2025-10-31_to_2025-11-06.md`)

**7-day analysis includes:**
- Week Overview: The arc and dominant themes
- Emotional Trajectory: How feelings evolved across the period
- Key Themes: Recurring patterns with links to specific days
- Progress & Wins: What worked well
- Stuck Points: What didn't shift
- Week-to-Week Patterns: Longer-term trends
- Recommendations: Actionable suggestions for the coming days

**Note:** These summaries are NOT added back to the index - they're standalone reflections.

## Running as a Background Service

**macOS:** See `INSTALL_LAUNCHAGENT.md` for auto-start on login setup.

**Linux:** Use `systemd` to create a user service.

## Privacy & Data

All processing is local. Nothing leaves your machine. No API keys required. Vector embeddings are stored locally in `_System/index/`.

## License

MIT - Feel free to modify for your needs.
