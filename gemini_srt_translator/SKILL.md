---
name: subtitle-translator
description: Translate subtitle files (SRT, ASS) and video/media with embedded subtitles into any target language with high linguistic quality, sliding context window, timestamp alignment, and formatting preservation. Use when the user asks to translate a subtitle file (.srt, .ass) or video subtitles.
---

# Subtitle Translator Skill

## 🎯 Role & Agent Protocol (CRITICAL)

When this skill is invoked:

1. **YOU ARE THE TRANSLATOR:** You (the AI assistant) perform the translation directly in your context window using your own linguistic intelligence.
2. **NO API KEY REQUIRED:** Do **NOT** ask the user for an API key or external credentials. The `gst agent translate` protocol runs completely locally and does not require an API key.
3. **DO NOT RUN EXTERNAL TRANSLATION SCRIPTS:** Do **NOT** look for or execute external Python scripts. Use **ONLY** the `gst agent translate` CLI commands below.
4. **VIRTUAL ENVIRONMENT EXECUTION:** `gst` may be installed in a project virtual environment (`.venv` or `venv`) rather than globally. If `gst` is not recognized:
   - Use the virtual environment binary: `.venv/bin/gst` (Linux/macOS) or `.venv\Scripts\gst.exe` (Windows)
   - Or use environment package runners: `uv run gst ...` / `poetry run gst ...`

---

## When to Use

- When the user asks to translate a subtitle file (`.srt` or `.ass`) into another language.
- When translating multi-part dialogue requiring context awareness and consistent character tone/gender agreement.
- When working with video files (`.mp4`, `.mkv`, etc.) that contain extractable subtitle streams.

---

## Subtitle Translation Protocol

### Step 1: Start a Translation Session

Run the start command to initialize the session and receive the first batch of subtitle lines:

```bash
gst agent translate start <INPUT_FILE> -l "<TARGET_LANGUAGE>" [--batch-size N] [--context-size N] [--description "<OPTIONAL_CONTEXT>"] [--pretty]
```

`<INPUT_FILE>` can be a subtitle file (`.srt`, `.ass`) or a video file (`.mp4`, `.mkv`, `.avi`, etc. — embedded subtitles will be automatically extracted).

#### CLI Options Reference

| Flag | Description | Default |
| --- | --- | --- |
| `-l, --target-language` | Target language name (e.g. `"French"`, `"Spanish"`) | Required |
| `-b, --batch-size` | Number of subtitle lines per batch | `100` |
| `--context-size` | Preceding lines to include in context fields | `0` (agent retains chat history) |
| `-o, --output-file` | Custom output subtitle path | `<input>_translated.srt/.ass` |
| `-d, --description` | Background context notes (e.g. series name, character tone) | None |
| `--pretty` | Pretty-print JSON responses with indentation | `false` (compact JSON) |
| `--no-resume` | Start fresh without resuming previous `.progress` | `false` |

The CLI outputs a JSON response with `next_batch` containing:

- `batch`: Array of `[{"index": "0", "text": "Original text"}, ...]` to translate.
- `original_context`: Previous original source lines (empty by default unless `--context-size` > 0 is passed).
- `translated_context`: Previous translated lines for continuity (empty by default unless `--context-size` > 0 is passed).

> **Optimal Batch Size Guidance (`-b` / `--batch-size` is optional, defaults to 100):**
> As an agent, select the batch size you find most optimal for your model capabilities and the file length:
>
> - **Recommended Default (80–120 lines):** Optimal balance between narrative context, translation accuracy, and fast validation.
> - **High-Capacity Models:** Feel free to use **100–150 lines** to translate full scenes in fewer turns.
> - **Short Files / Anime Episodes (< 300 lines):** 60–80 lines provides 3–4 quick, responsive turns.
> - **Constrained Output / Local Models:** Use **40–60 lines** to guarantee the full JSON array fits comfortably within the model's output generation limits.

### Step 2: Translate In-Context & Commit Translated Batch

Translate each item in `batch` into the target language, preserving the exact item count and indices, then commit:

```bash
gst agent translate commit <INPUT_FILE> --data '<TRANSLATED_JSON>'
# or save to a file and commit:
gst agent translate commit <INPUT_FILE> --data-file batch_1_translated.json
```

**Commit Data Format:**

```json
[
  { "index": "0", "text": "Bonjour le monde !" },
  { "index": "1", "text": "Comment vas-tu aujourd'hui ?" }
]
```

### Step 3: Repeat Until Complete

Each successful `commit` automatically saves progress and returns the `next_batch`.
Repeat Step 2 until the response returns `"status": "completed"` or `"is_complete": true`.

### Helper Commands

```bash
gst agent translate status <INPUT_FILE> [--pretty]   # Check progress status
gst agent translate next <INPUT_FILE> -l "<TARGET_LANGUAGE>" [--pretty] # Re-fetch current pending batch
gst agent translate reset <INPUT_FILE> [--pretty]    # Reset translation progress
```

---

## Translation Rules

1. **Translation Item Parity:** The output JSON array must contain the exact same number of items with identical indices (`index`).
2. **Formatting Preservation:** Preserve all newlines (`\n`), italic tags (`<i>...</i>`), and ASS styling tags (`{\an8}`, `{\pos(...)}`, etc.).
3. **Punctuation & Tone:** Maintain dialogue flow, character voice, and natural target language phrasing without altering structural markers.
