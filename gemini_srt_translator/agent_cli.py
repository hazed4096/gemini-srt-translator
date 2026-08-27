"""
Agent CLI Module for Gemini SRT Translator
Provides structured JSON-based CLI subcommands for AI Agents to interact with the subtitle translation & transcription pipelines.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from .session import SubtitleSession


def _sanitize_for_json(data: Any) -> Any:
    """Recursively sanitize data structures for JSON serialization (stripping raw bytes)."""
    if isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items() if k != "audio_bytes"}
    elif isinstance(data, list):
        return [_sanitize_for_json(item) for item in data]
    elif isinstance(data, bytes):
        return None
    return data


def _print_json(data: Dict[str, Any], pretty: bool = False):
    """Output JSON to stdout for machine consumption."""
    clean_data = _sanitize_for_json(data)
    if pretty:
        print(json.dumps(clean_data, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(clean_data, ensure_ascii=False))


# ==============================================================================
# Subtitle Translation Commands
# ==============================================================================


def cmd_agent_start(args) -> int:
    """Initialize a subtitle translation session and return the first batch."""
    try:
        session = SubtitleSession(
            input_file=args.input_file,
            target_language=args.target_language,
            output_file=args.output_file,
            batch_size=getattr(args, "batch_size", None) or 100,
            context_size=getattr(args, "context_size", 0),
            description=getattr(args, "description", None),
            resume=getattr(args, "resume", True),
        )

        next_batch = session.get_next_batch(include_system_prompt=True)
        if not next_batch:
            _print_json(
                {
                    "status": "completed",
                    "session": session.get_status(),
                },
                pretty=args.pretty,
            )
            return 0

        system_prompt = next_batch.pop("system_prompt", None)
        _print_json(
            {
                "status": "ready",
                "session": session.get_status(),
                "system_prompt": system_prompt,
                "next_batch": next_batch,
            },
            pretty=args.pretty,
        )
        return 0
    except Exception as e:
        _print_json({"status": "error", "error": str(e)}, pretty=args.pretty)
        return 1


def cmd_agent_next(args) -> int:
    """Get the current pending batch for an in-progress subtitle file."""
    try:
        session = SubtitleSession(
            input_file=args.input_file,
            target_language=getattr(args, "target_language", None),
            output_file=getattr(args, "output_file", None),
            batch_size=getattr(args, "batch_size", None),
            resume=True,
        )

        next_batch = session.get_next_batch(include_system_prompt=True)
        if not next_batch:
            _print_json(
                {
                    "status": "completed",
                    "session": session.get_status(),
                },
                pretty=args.pretty,
            )
            return 0

        system_prompt = next_batch.pop("system_prompt", None)
        _print_json(
            {
                "status": "ok",
                "session": session.get_status(),
                "system_prompt": system_prompt,
                "next_batch": next_batch,
            },
            pretty=args.pretty,
        )
        return 0
    except Exception as e:
        _print_json({"status": "error", "error": str(e)}, pretty=args.pretty)
        return 1


def cmd_agent_commit(args) -> int:
    """Commit a translated batch from a JSON file, save to subtitle, and return the next batch."""
    try:
        data_file = getattr(args, "data_file", None)
        if not data_file:
            _print_json(
                {
                    "status": "error",
                    "error": "Missing translation file. Use '--data-file <path>'.",
                },
                pretty=args.pretty,
            )
            return 1

        if not os.path.exists(data_file):
            _print_json({"status": "error", "error": f"File not found: {data_file}"}, pretty=args.pretty)
            return 1

        with open(data_file, "r", encoding="utf-8") as f:
            data = f.read()

        if not data or not data.strip():
            _print_json(
                {
                    "status": "error",
                    "error": f"Translation file is empty: {data_file}",
                },
                pretty=args.pretty,
            )
            return 1

        session = SubtitleSession(
            input_file=args.input_file,
            target_language=getattr(args, "target_language", None),
            output_file=getattr(args, "output_file", None),
            batch_size=getattr(args, "batch_size", None),
            resume=True,
        )

        commit_result = session.commit_batch(data)
        if not commit_result.get("success"):
            _print_json(
                {
                    "status": "error",
                    "error": commit_result.get("error", "Validation failed"),
                    "session": session.get_status(),
                },
                pretty=args.pretty,
            )
            return 1

        status_dict = session.get_status()
        if session.is_complete():
            _print_json(
                {
                    "status": "completed",
                    "progress": {
                        "completed_lines": status_dict["completed_lines"],
                        "total_lines": status_dict["total_lines"],
                        "percent": status_dict["progress_percent"],
                    },
                    "output_file": session.output_file,
                },
                pretty=args.pretty,
            )
        else:
            next_batch = session.get_next_batch(include_system_prompt=False)
            _print_json(
                {
                    "status": "committed",
                    "progress": {
                        "batch": status_dict["batch_number"],
                        "total_batches": status_dict["total_batches"],
                        "completed_lines": status_dict["completed_lines"],
                        "total_lines": status_dict["total_lines"],
                        "percent": status_dict["progress_percent"],
                    },
                    "next_batch": next_batch,
                },
                pretty=args.pretty,
            )
        return 0
    except Exception as e:
        _print_json({"status": "error", "error": str(e)}, pretty=args.pretty)
        return 1


def cmd_agent_status(args) -> int:
    """Get translation session status."""
    try:
        session = SubtitleSession(
            input_file=args.input_file,
            output_file=getattr(args, "output_file", None),
            resume=True,
        )
        _print_json(
            {
                "status": "ok",
                "session": session.get_status(),
            },
            pretty=args.pretty,
        )
        return 0
    except Exception as e:
        _print_json({"status": "error", "error": str(e)}, pretty=args.pretty)
        return 1


def cmd_agent_reset(args) -> int:
    """Reset translation progress."""
    try:
        session = SubtitleSession(
            input_file=args.input_file,
            output_file=getattr(args, "output_file", None),
            resume=False,
        )
        session.reset_progress()
        _print_json(
            {
                "status": "reset",
                "session": session.get_status(),
            },
            pretty=args.pretty,
        )
        return 0
    except Exception as e:
        _print_json({"status": "error", "error": str(e)}, pretty=args.pretty)
        return 1


# ==============================================================================
# Argument Parsing Setup
# ==============================================================================


def add_agent_subparser(subparsers: argparse._SubParsersAction):
    """Add the agent subcommands to the main argument parser."""
    agent_parser = subparsers.add_parser(
        "agent",
        help="Commands for AI Agents (Antigravity, Codex, Cursor, etc.) to translate subtitles step-by-step",
    )
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command", help="Agent action")

    # Common translation arguments helper
    def add_common_translation_args(parser):
        parser.add_argument("input_file", help="Input subtitle (.srt, .ass) or video file")
        parser.add_argument("-o", "--output-file", help="Custom output file path")
        parser.add_argument("-b", "--batch-size", type=int, default=None, help="Batch size (number of subtitle lines)")
        parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    def setup_translation_subparsers(subparser_container):
        # Start
        start_p = subparser_container.add_parser("start", help="Start translation session and get the first batch")
        add_common_translation_args(start_p)
        start_p.add_argument("-l", "--target-language", required=True, help="Target translation language")
        start_p.add_argument("-d", "--description", help="Additional context/notes for translation")
        start_p.add_argument(
            "--context-size", type=int, default=0, help="Number of previous lines for context (default: 0)"
        )
        start_p.add_argument("--no-resume", dest="resume", action="store_false", default=True, help="Don't resume")

        # Next
        next_p = subparser_container.add_parser("next", help="Get the current pending batch")
        add_common_translation_args(next_p)
        next_p.add_argument("-l", "--target-language", help="Target translation language")

        # Commit
        commit_p = subparser_container.add_parser("commit", help="Commit a translated batch")
        add_common_translation_args(commit_p)
        commit_p.add_argument("-l", "--target-language", help="Target translation language")
        commit_p.add_argument("--data-file", "--file", "-f", help="Path to JSON file containing translated batch items")

        # Status
        status_p = subparser_container.add_parser("status", help="Get translation status")
        status_p.add_argument("input_file", help="Input subtitle file")
        status_p.add_argument("-o", "--output-file", help="Custom output file path")
        status_p.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

        # Reset
        reset_p = subparser_container.add_parser("reset", help="Reset translation progress")
        reset_p.add_argument("input_file", help="Input subtitle file")
        reset_p.add_argument("-o", "--output-file", help="Custom output file path")
        reset_p.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    # Direct subcommands: `gst agent <start|next|commit|status|reset>`
    setup_translation_subparsers(agent_subparsers)


def handle_agent_command(args) -> int:
    """Route agent subcommands."""
    cmd = getattr(args, "agent_command", None)
    if not cmd:
        print(
            "Please specify an agent subcommand: start, next, commit, status, reset",
            file=sys.stderr,
        )
        return 1

    if cmd == "start":
        return cmd_agent_start(args)
    elif cmd == "next":
        return cmd_agent_next(args)
    elif cmd == "commit":
        return cmd_agent_commit(args)
    elif cmd == "status":
        return cmd_agent_status(args)
    elif cmd == "reset":
        return cmd_agent_reset(args)

    return 1
