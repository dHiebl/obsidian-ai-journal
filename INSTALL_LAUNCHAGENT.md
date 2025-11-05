# Installing the Journal Watcher LaunchAgent (macOS)

This guide sets up the AI Journal Watcher to start automatically on login.

## Prerequisites

1. Installed dependencies: `pip install -r requirements.txt` (or `uv sync`)
2. Ollama is running with required models
3. Tested the watcher manually at least once

## Installation

### 1. Create your plist file

```bash
cp com.journal.watcher.plist.template com.journal.watcher.plist
```

Edit `com.journal.watcher.plist` and replace all paths with your actual vault location.

### 2. Install

```bash
cp com.journal.watcher.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.journal.watcher.plist
```

### 3. Verify

```bash
launchctl list | grep journal
```

Should show: `com.journal.watcher`

## Managing the Watcher

**Check status:**
```bash
launchctl list | grep journal
```

**Stop/Start:**
```bash
launchctl unload ~/Library/LaunchAgents/com.journal.watcher.plist
launchctl load ~/Library/LaunchAgents/com.journal.watcher.plist
```

**Uninstall:**
```bash
launchctl unload ~/Library/LaunchAgents/com.journal.watcher.plist
rm ~/Library/LaunchAgents/com.journal.watcher.plist
```

## Logs

- **Application log**: `_System/watcher.log` - All watcher activity
- **Error log**: `_System/watcher_error.log` - Startup errors and crashes

**View in real-time:**
```bash
tail -f /path/to/your/vault/_System/watcher.log
```

## Troubleshooting

**LaunchAgent won't start:**
1. Check error log: `cat _System/watcher_error.log`
2. Verify Python path exists: `ls -la .venv/bin/python3`
3. Validate plist: `plutil -lint ~/Library/LaunchAgents/com.journal.watcher.plist`

**After code changes:**
```bash
launchctl unload ~/Library/LaunchAgents/com.journal.watcher.plist
launchctl load ~/Library/LaunchAgents/com.journal.watcher.plist
```

**Common issues:**
- Ollama not running
- Python virtual environment not created
- Incorrect paths in plist file

## Notes

- Watcher runs continuously in the background (lightweight, only uses resources when processing)
- `KeepAlive` setting auto-restarts if it crashes
- Logs append over time - delete manually if they get too large
