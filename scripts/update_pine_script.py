#!/usr/bin/env python3
"""
update_pine_script.py
─────────────────────
Reads data/new_events.json (produced by fetch_fb_events.py) and appends
the new events to the 8zz-indicator.pine file.

Steps
  1. Load new events from data/new_events.json.
  2. Deduplicate against timestamps already present in the .pine file.
  3. Append new array.push() blocks just before the closing blank line
     of the `if barstate.isfirst` block.
  4. Update the event-count comment in the file header
     (e.g. "// 事件: 63筆 | 期間: 2025/12 ~ 2026/04").

Usage:
  python scripts/update_pine_script.py
"""

import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PINE_FILE = ROOT / "8zz-indicator.pine"
EVENTS_FILE = ROOT / "data" / "new_events.json"

# Marker that ends the event-injection block inside `if barstate.isfirst`.
# Each event block ends with evt_ticker, so we insert after the last evt_ticker line.
# (Using evt_tips would insert inside the last event's block, displacing its ticker.)
LAST_PUSH_PATTERN = re.compile(r"^\s+array\.push\(evt_ticker,")

# Header comment pattern (matches the 事件: NNN筆 line)
HEADER_COUNT_PATTERN = re.compile(
    r"(// 事件: )(\d+)(筆 \| 期間: \d{4}/\d{2} ~ )(\d{4}/\d{2})"
)

EVENT_PATTERN = re.compile(
    r"array\.push\(evt_time,\s*(\d+)\)\s*"
    r"array\.push\(evt_dir,\s*(-?\d+)\)\s*"
    r"array\.push\(evt_str,\s*(\d+)\)\s*"
    r'array\.push\(evt_tips,\s*"((?:[^"\\]|\\.)*)"\)\s*'
    r'array\.push\(evt_ticker,\s*"([^"]*)"\)',
    re.DOTALL,
)


def load_new_events() -> list[dict]:
    if not EVENTS_FILE.exists():
        print(f"ℹ️  {EVENTS_FILE} not found – nothing to do.")
        return []
    try:
        events = json.loads(EVENTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: Could not parse {EVENTS_FILE}: {exc}")
        sys.exit(1)
    return events


def extract_existing_timestamps(pine_text: str) -> set[int]:
    """Return all unix-ms timestamps already embedded in the .pine file."""
    pattern = re.compile(r"array\.push\(evt_time,\s*(\d+)\)")
    return {int(m.group(1)) for m in pattern.finditer(pine_text)}


def extract_existing_events(pine_text: str) -> list[dict]:
    """Return the fields needed for timestamp and content-based deduplication."""
    return [
        {
            "unix_ms": int(match.group(1)),
            "tooltip": match.group(4).replace("\\n", "\n"),
        }
        for match in EVENT_PATTERN.finditer(pine_text)
    ]


def normalize_post_excerpt(tooltip: str) -> str:
    """Normalize the FB post excerpt embedded after the tooltip metadata."""
    lines = tooltip.replace("\\n", "\n").splitlines()
    post_text = " ".join(lines[2:] if len(lines) >= 3 else lines)
    post_text = re.sub(r"^FB\s+\d{2}/\d{2}\s+\d{2}:\d{2}\s+", "", post_text)
    post_text = post_text.removesuffix("...")
    post_text = unicodedata.normalize("NFKC", post_text)
    return re.sub(r"\s+", "", post_text).strip()


def is_duplicate_event(candidate: dict, existing_events: list[dict]) -> bool:
    """Detect the same post even when one source rounded away its seconds."""
    candidate_ts = int(candidate["unix_ms"])
    candidate_excerpt = normalize_post_excerpt(candidate.get("tooltip", ""))
    for existing in existing_events:
        if candidate_ts == int(existing["unix_ms"]):
            return True
        existing_excerpt = normalize_post_excerpt(existing.get("tooltip", ""))
        if min(len(candidate_excerpt), len(existing_excerpt)) < 12:
            continue
        if candidate_excerpt.startswith(existing_excerpt) or existing_excerpt.startswith(candidate_excerpt):
            return True
    return False


def escape_pine_string(text: str) -> str:
    """
    Escape a string for safe embedding inside a Pine Script double-quoted string.
    Pine Script string literals treat backslash as an escape character and do not
    support raw newlines inside a single-line string literal.
    """
    # Escape backslashes first (must be done before escaping other chars)
    text = text.replace("\\", "\\\\")
    # Escape double-quotes
    text = text.replace('"', '\\"')
    # Replace literal newlines with Pine's recognised escape sequence
    text = text.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")
    return text


def build_push_block(event: dict) -> str:
    """Return the five array.push lines for a single event."""
    ts     = event["unix_ms"]
    d      = event["direction"]
    s      = event["strength"]
    tip    = escape_pine_string(event["tooltip"])
    ticker = escape_pine_string(event.get("ticker", ""))
    return (
        f"    array.push(evt_time, {ts})\n"
        f"    array.push(evt_dir, {d})\n"
        f"    array.push(evt_str, {s})\n"
        f'    array.push(evt_tips, "{tip}")\n'
        f'    array.push(evt_ticker, "{ticker}")\n'
    )


def find_insertion_line(lines: list[str]) -> int:
    """
    Return the index (0-based) of the line *after* the last array.push(evt_tips,…)
    call inside the `if barstate.isfirst` block.
    This is where we will insert new event blocks.
    """
    last_push_line = -1
    for i, line in enumerate(lines):
        if LAST_PUSH_PATTERN.match(line):
            last_push_line = i
    if last_push_line == -1:
        print("ERROR: Could not locate array.push(evt_tips, …) in .pine file.")
        sys.exit(1)
    # Insertion point is the line immediately after the last push block
    return last_push_line + 1


def update_header_count(lines: list[str], total_events: int) -> list[str]:
    """Update the '// 事件: NNN筆 | 期間: YYYY/MM ~ YYYY/MM' header line."""
    now_utc = datetime.now(timezone.utc)
    new_end_period = now_utc.strftime("%Y/%m")

    updated = []
    for line in lines:
        m = HEADER_COUNT_PATTERN.search(line)
        if m:
            new_line = HEADER_COUNT_PATTERN.sub(
                lambda _: f"{m.group(1)}{total_events}{m.group(3)}{new_end_period}",
                line,
            )
            updated.append(new_line)
        else:
            updated.append(line)
    return updated


def main() -> None:
    new_events = load_new_events()
    if not new_events:
        print("ℹ️  No new events to inject.")
        return

    pine_text = PINE_FILE.read_text(encoding="utf-8")
    existing_events = extract_existing_events(pine_text)

    # Manual backfills have minute precision while Facebook carries seconds, so
    # timestamp-only deduplication can insert the same post twice.
    to_insert = []
    for event in new_events:
        if is_duplicate_event(event, existing_events):
            continue
        to_insert.append(event)
        existing_events.append(event)
    if not to_insert:
        print("ℹ️  All events already present in the .pine file. Nothing to update.")
        return

    lines = pine_text.splitlines(keepends=True)
    insertion_index = find_insertion_line(lines)

    # Build the new push blocks, sorted ascending by time
    to_insert.sort(key=lambda e: e["unix_ms"])
    new_blocks = "".join(build_push_block(e) for e in to_insert)

    lines.insert(insertion_index, new_blocks)

    # Re-count total events
    new_text = "".join(lines)
    total_events = len(extract_existing_timestamps(new_text))

    updated_lines = update_header_count(new_text.splitlines(keepends=True), total_events)
    final_text = "".join(updated_lines)

    PINE_FILE.write_text(final_text, encoding="utf-8")
    print(
        f"✅ Injected {len(to_insert)} new event(s) into {PINE_FILE.name}. "
        f"Total events: {total_events}."
    )

    # Clear the processed events file so it doesn't re-insert on the next run
    EVENTS_FILE.write_text("[]", encoding="utf-8")


if __name__ == "__main__":
    main()
