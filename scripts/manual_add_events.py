#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "google-generativeai>=0.8.0",
#   "yfinance==0.2.54",
#   "requests==2.32.3",
# ]
# ///
"""
manual_add_events.py
────────────────────
Manually add FB posts to the indicator, replacing the Apify auto-scraper
(disabled 2026-07-07: quota exhaustion + unreliable coverage).

Reads a plain-text file of posts, classifies each with the same
Gemini / keyword pipeline as fetch_fb_events.py, writes the results to
data/new_events.json, then runs update_pine_script.py and
build_site_data.py to inject them.

Input file format (UTF-8) — one block per post, separated by `---` lines.
First line of each block is the post time in **Taipei time (UTC+8)**,
format YYYY-MM-DD HH:MM. Everything after it is the post text (multiline OK):

    2026-07-04 12:30
    今天又停損了，好痛苦
    [委託成功截圖]
    ---
    2026-07-05 09:00
    漲停追進去了！這支一定飆！

Usage (uv auto-installs the deps declared above):
  GOOGLE_API_KEY=xxx uv run scripts/manual_add_events.py posts.txt
  uv run scripts/manual_add_events.py posts.txt --dry-run   # classify only

Without GOOGLE_API_KEY it falls back to keyword rules (less accurate —
prefer running with the key). Duplicate timestamps are deduped against the
.pine file by update_pine_script.py, so re-running the same file is safe.
The last_event_timestamp.json state file is ignored in manual mode.
"""

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_fb_events import (  # noqa: E402
    GOOGLE_API_KEY,
    boost_strength_if_order_proof,
    build_tooltip,
    classify_with_ai,
    classify_with_keywords,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT / "data" / "new_events.json"
TAIPEI = timezone(timedelta(hours=8))

TIME_PATTERN = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2})$")


def parse_posts_file(path: Path) -> list[tuple[datetime, str]]:
    """Return [(utc_datetime, post_text), …] from the manual posts file."""
    blocks = re.split(r"^-{3,}\s*$", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    posts: list[tuple[datetime, str]] = []
    for i, block in enumerate(blocks, 1):
        block = block.strip()
        if not block:
            continue
        first_line, _, rest = block.partition("\n")
        m = TIME_PATTERN.match(first_line.strip())
        if not m:
            print(f"ERROR: block {i} first line is not 'YYYY-MM-DD HH:MM': {first_line[:40]!r}")
            sys.exit(1)
        y, mo, d, h, mi = (int(g) for g in m.groups())
        dt_utc = datetime(y, mo, d, h, mi, tzinfo=TAIPEI).astimezone(timezone.utc)
        text = rest.strip()
        if not text:
            print(f"ERROR: block {i} ({first_line.strip()}) has no post text.")
            sys.exit(1)
        posts.append((dt_utc, text))
    return posts


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = "--dry-run" in sys.argv[1:]
    if len(args) != 1:
        print(__doc__)
        sys.exit(1)

    posts_file = Path(args[0])
    if not posts_file.exists():
        print(f"ERROR: {posts_file} not found.")
        sys.exit(1)

    print(f"ℹ️  Classifier: {'Gemini Flash (AI)' if GOOGLE_API_KEY else 'keyword rules (set GOOGLE_API_KEY for better accuracy)'}")
    posts = parse_posts_file(posts_file)
    print(f"ℹ️  Parsed {len(posts)} post(s) from {posts_file}")

    new_events: list[dict] = []
    for post_time, text in posts:
        direction, strength, action, ticker = 0, 1, "", ""
        ai = classify_with_ai(text)
        if ai is None and GOOGLE_API_KEY:
            # Free-tier Gemini has a per-minute quota; one retry usually clears it
            print("  … Gemini failed, retrying in 60s")
            time.sleep(60)
            ai = classify_with_ai(text)
        if ai is not None:
            direction, strength = ai["direction"], ai["strength"]
            action, ticker = ai["action"], ai["ticker"]
            strength = boost_strength_if_order_proof(text, direction, strength)
            print(f"  [Gemini] dir={direction} str={strength} action='{action}' ticker='{ticker or '(0050)'}' | {text[:40]}")
        else:
            kw = classify_with_keywords(text)
            if kw is None:
                print(f"  [keyword] no match – skipping | {text[:40]}")
                continue
            direction, strength, action = kw
            strength = boost_strength_if_order_proof(text, direction, strength)
            print(f"  [keyword] dir={direction} str={strength} action='{action}' | {text[:40]}")

        if direction == 0:
            print("  → direction=0, not investment-related, skipping")
            continue

        new_events.append({
            "unix_ms":   int(post_time.timestamp() * 1000),
            "direction": direction,
            "strength":  strength,
            "ticker":    ticker,
            "tooltip":   build_tooltip(text, direction, strength, action, post_time),
        })

    if dry_run:
        print(f"ℹ️  Dry run: {len(new_events)} classifiable event(s), nothing written.")
        return

    if not new_events:
        print("ℹ️  No classifiable events – nothing to inject.")
        return

    new_events.sort(key=lambda e: e["unix_ms"])
    OUTPUT_FILE.write_text(json.dumps(new_events, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ {len(new_events)} event(s) written to {OUTPUT_FILE}")

    for script in ("update_pine_script.py", "build_site_data.py"):
        print(f"ℹ️  Running {script} …")
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)])
        if result.returncode != 0:
            print(f"ERROR: {script} failed (exit {result.returncode}). Fix and re-run it manually.")
            sys.exit(result.returncode)

    print(
        "\n✅ Done. Review the diff, then commit & push:\n"
        "   git diff 8zz-indicator.pine docs/events.json\n"
        "   git add 8zz-indicator.pine data/new_events.json docs/events.json\n"
        '   git commit -m "feat: add manual FB events" && git push'
    )


if __name__ == "__main__":
    main()
