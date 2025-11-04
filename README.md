# AI Journal Analysis System

An intelligent watcher that monitors your Obsidian daily journal entries and automatically appends AI-generated insights using RAG (Retrieval-Augmented Generation) with your past entries.

## Features

- **Automatic Analysis**: Watches your daily journal folder and analyzes entries when you mark them ready
- **Context-Aware**: Uses hybrid retrieval (BM25 + vector search) to find relevant past entries for context
- **Incremental Indexing**: Efficiently updates the vector store without rebuilding from scratch
- **Smart Triggers**: Only processes entries that are explicitly marked ready and have been idle for 90 seconds
- **Structured Insights**: Generates consistent analysis with specific sections: Summary, Emotions, Distortions, Triggers/Needs, Patterns, and Reflection Prompts

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

**Note**: The `gpt-oss:20b` model is quite large (~12GB). If you prefer a smaller model, you can edit `config.py` and change `LLM_MODEL` to something like `qwen2.5:7b` or `gemma2:9b`.

## Usage

### Basic Usage

Simply run the watcher from your vault's `_System/ai` directory:

```bash
python watch_vault.py
```

The watcher will:
1. Monitor your `Journal/Daily/` folder for changes
2. Wait for entries to be marked as ready
3. Wait 30 seconds after the last edit
4. Generate AI analysis and append it to the file
5. Update the vector index with the new entry

### Custom Vault Path

If your vault is not at the default location (`~/Obsidian/Personal`), specify it:

```bash
python watch_vault.py --vault /path/to/your/vault
```

### Stopping the Watcher

Press `Ctrl+C` to stop the watcher gracefully.

## Obsidian Template

To use this system, your daily journal entries need to signal when they're ready for analysis. Add one of these markers to your template:

### Option 1: YAML Frontmatter (see daily.md)

```markdown
---
date: 2025-11-04
ai_ready: false
---

## Entry

Your journal content here...
```

### Option 2: HTML Comment

```markdown
---
date: 2025-11-04
---

## Entry

Your journal content here...

<!-- AI:READY -->
```

## Configuration

Edit `config.py` to customize:

- **Models**: Change `LLM_MODEL` or `EMBED_MODEL` to use different Ollama models
- **Paths**: Adjust vault and folder paths if your structure is different
- **Triggers**: Modify `IDLE_SECONDS` (default: 30) or `MIN_LENGTH` (default: 150 characters)
- **Retrieval**: Change `TOP_K` (default: 5) to retrieve more or fewer past entries

## Output Format

The AI analysis is appended to your journal entry with this structure:

```markdown
---

## AI Analysis

## Summary
[Concise summary of the entry]

## Emotions
[Identified emotions with evidence]

## Distortions
[Any cognitive distortions identified]

## Triggers/Needs
[What triggered these feelings and underlying needs]

## Patterns
[Connections to patterns from past entries]

## 3 Prompts
[Three reflection questions]
```

## How It Works

1. **File Watching**: Uses `watchdog` to monitor file modifications in `Journal/Daily/`
2. **Ready Check**: Verifies the entry has a ready marker, meets minimum length, and doesn't already have AI analysis
3. **Debouncing**: Waits 90 seconds after the last edit to ensure you're done writing
4. **Context Retrieval**: Searches the vector index for 5 similar past entries using hybrid BM25 + vector search
5. **AI Generation**: Sends your entry + past context to the LLM for analysis
6. **Appending**: Atomically writes the analysis to the end of your file
7. **Index Update**: Updates the vector store incrementally (no full rebuild)

## Troubleshooting

### Watcher Not Processing Entry

**Check the log file** at `_System/watcher.log` for detailed information.

Common issues:
- **No ready marker**: Add `ai_ready: true` to frontmatter or `<!-- AI:READY -->` to the body
- **Entry too short**: Must be at least 150 characters (configurable in `config.py`)
- **Not idle long enough**: Wait 30 seconds after your last edit
- **AI analysis already exists**: The system won't process the same entry twice

### Ollama Connection Errors

```
Error: Could not connect to Ollama
```

**Solutions**:
- Ensure Ollama is running: `ollama serve`
- Verify models are downloaded: `ollama list`
- Check if Ollama is listening on `http://localhost:11434`
- If using a different port, update `OLLAMA_URL` in `config.py`

### Models Not Found

```
Error: model 'gpt-oss:20b' not found
```

**Solution**: Pull the required models:
```bash
ollama pull gpt-oss:20b
ollama pull embeddinggemma:300m
```

### Permission Errors

If you get permission errors when writing to the vault:
- Ensure the watcher has write permissions to your vault folder
- Check that your vault isn't in a cloud-synced folder that's currently syncing

### Index Errors

If the index gets corrupted:
1. Stop the watcher
2. Delete the `_System/index/` directory
3. Restart the watcher (it will create a fresh index)

Note: You'll lose the index, but you can rebuild it by reprocessing your entries.

## File Structure

```
~/Obsidian/Personal/
├── Journal/
│   └── Daily/
│       ├── 2025-11-01.md
│       ├── 2025-11-02.md
│       └── 2025-11-04.md
└── _System/
    ├── ai/
    │   ├── watch_vault.py
    │   ├── config.py
    │   ├── requirements.txt
    │   └── README.md
    ├── index/                    # Auto-created
    │   ├── docstore.json
    │   ├── bm25_docstore.json
    │   └── [vector store files]
    └── watcher.log              # Auto-created
```

## Tips

1. **Start with a few entries**: Let the watcher process 3-5 entries first to build up context
2. **Review and edit**: The AI analysis is meant to supplement your reflection, not replace it
3. **Adjust the prompt**: Edit `ANALYSIS_SYSTEM_PROMPT` in `config.py` to customize the analysis style
4. **Use different models**: Try different Ollama models to find the right balance of quality and speed
5. **Monitor the log**: Keep an eye on `watcher.log` to understand what the system is doing

## Advanced Usage

### Running as a Background Service (Auto-start on Login)

**macOS (Recommended):**

See the detailed guide in `INSTALL_LAUNCHAGENT.md` for step-by-step instructions.

Quick setup:
```bash
# 1. Create your plist from template and customize paths
cp com.journal.watcher.plist.template com.journal.watcher.plist
# Edit com.journal.watcher.plist with your vault paths

# 2. Install
cp com.journal.watcher.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.journal.watcher.plist
```

The watcher will now start automatically on login and run in the background.

**Linux:**

Use `systemd` to create a user service.

### Rebuilding the Index

If you want to reindex all your entries:

1. Stop the watcher
2. Delete `_System/index/`
3. Remove all `## AI Analysis` sections from your entries (optional)
4. Add `ai_ready: true` to entries you want analyzed
5. Start the watcher

It will process each entry as it detects them.

## Privacy & Data

- **All processing is local**: Nothing leaves your machine
- **No API keys required**: Uses Ollama running on your computer
- **Your data stays private**: Vector embeddings are stored locally in `_System/index/`

## License

This is a personal tool. Feel free to modify it for your needs.

## Support

Check the logs at `_System/watcher.log` for debugging information. The log includes timestamps, processing stages, and any errors encountered.

