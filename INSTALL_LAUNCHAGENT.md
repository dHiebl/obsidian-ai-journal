# Installing the Journal Watcher LaunchAgent

This guide will help you set up the AI Journal Watcher to start automatically on login.

## Prerequisites

Make sure you've already:
1. Installed dependencies: `pip install -r requirements.txt` (or `uv pip install -r requirements.txt`)
2. Verified Ollama is running with required models
3. Tested the watcher manually at least once

## Installation Steps

### 1. Create your plist file from the template

Copy the template and customize it with your paths:

```bash
cp com.journal.watcher.plist.template com.journal.watcher.plist
```

Edit `com.journal.watcher.plist` and replace all instances of `/path/to/your/vault` with your actual vault path.

For example, if your vault is at `/Users/yourname/Documents/Obsidian/Personal`, update:
- `ProgramArguments` → `/Users/yourname/Documents/Obsidian/Personal/_system/ai/.venv/bin/python3`
- `ProgramArguments` → `/Users/yourname/Documents/Obsidian/Personal/_system/ai/watch_vault.py`
- `WorkingDirectory` → `/Users/yourname/Documents/Obsidian/Personal/_system/ai`
- `StandardOutPath` → `/Users/yourname/Documents/Obsidian/Personal/_System/watcher.log`
- `StandardErrorPath` → `/Users/yourname/Documents/Obsidian/Personal/_System/watcher_error.log`

**Tip:** The `_system` and `_System` capitalization matters! Make sure to match your actual directory structure.

### 2. Copy the plist file to LaunchAgents

```bash
cp com.journal.watcher.plist ~/Library/LaunchAgents/
```

### 3. Load the LaunchAgent

```bash
launchctl load ~/Library/LaunchAgents/com.journal.watcher.plist
```

This will start the watcher immediately and also configure it to start on every login.

### 3. Verify it's running

```bash
launchctl list | grep journal
```

You should see: `com.journal.watcher`

## Managing the Watcher

### Check if it's running
```bash
launchctl list | grep journal
```

### Stop the watcher
```bash
launchctl unload ~/Library/LaunchAgents/com.journal.watcher.plist
```

### Start the watcher
```bash
launchctl load ~/Library/LaunchAgents/com.journal.watcher.plist
```

### Restart the watcher (after making changes)
```bash
launchctl unload ~/Library/LaunchAgents/com.journal.watcher.plist
launchctl load ~/Library/LaunchAgents/com.journal.watcher.plist
```

### Remove the auto-start (uninstall)
```bash
launchctl unload ~/Library/LaunchAgents/com.journal.watcher.plist
rm ~/Library/LaunchAgents/com.journal.watcher.plist
```

## Logs

The watcher writes logs to two locations:

1. **Main application log**: `/path/to/your/vault/_System/watcher.log`
   - Contains all watcher activity (processing entries, etc.)

2. **LaunchAgent error log**: `/path/to/your/vault/_System/watcher_error.log`
   - Contains any startup errors or crashes

### View logs in real-time
```bash
tail -f /path/to/your/vault/_System/watcher.log
```

## Troubleshooting

### LaunchAgent won't start

1. **Check the error log**:
   ```bash
   cat /path/to/your/vault/_System/watcher_error.log
   ```

2. **Verify Python path**:
   ```bash
   ls -la /path/to/your/vault/_system/ai/.venv/bin/python3
   ```
   
   If this doesn't exist, you need to create a virtual environment first (requires Python 3.12+):
   ```bash
   cd /path/to/your/vault/_system/ai
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   
   Or if using uv:
   ```bash
   cd /path/to/your/vault/_system/ai
   uv sync --python 3.12
   ```

3. **Verify the plist file is valid**:
   ```bash
   plutil -lint ~/Library/LaunchAgents/com.journal.watcher.plist
   ```
   Should output: "OK"

### Watcher stops unexpectedly

The `KeepAlive` setting will automatically restart the watcher if it crashes.

Check the error log to see why it crashed:
```bash
cat /path/to/your/vault/_System/watcher_error.log
```

Common issues:
- Ollama not running
- Virtual environment not activated (plist handles this)
- Permissions issues with vault files

### Update after code changes

If you modify `watch_vault.py` or `config.py`:
```bash
launchctl unload ~/Library/LaunchAgents/com.journal.watcher.plist
launchctl load ~/Library/LaunchAgents/com.journal.watcher.plist
```

### Disable temporarily (don't uninstall)

```bash
launchctl unload ~/Library/LaunchAgents/com.journal.watcher.plist
```

Re-enable later with:
```bash
launchctl load ~/Library/LaunchAgents/com.journal.watcher.plist
```

## What the plist file does

- **Label**: Unique identifier for the service
- **ProgramArguments**: Path to Python and the watcher script
- **WorkingDirectory**: Where the script runs from
- **RunAtLoad**: Starts immediately when loaded (and on every login)
- **KeepAlive**: Automatically restarts if the watcher crashes
- **StandardOutPath/StandardErrorPath**: Where to write logs

## Notes

- The watcher runs in the background all the time (not just when Obsidian is open)
- It's very lightweight and only uses resources when processing entries
- Logs are appended to (not overwritten) so they'll grow over time
- You can manually delete old log entries if needed

