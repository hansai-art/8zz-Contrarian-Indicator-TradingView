#!/usr/bin/env python3
"""Regression tests for Facebook event deduplication, time zones, and sources."""

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_site_data import _parse_chart_proxy_payload, load_price_cache, parse_events, save_price_cache  # noqa: E402
from fetch_fb_events import build_tooltip  # noqa: E402
from update_pine_script import is_duplicate_event  # noqa: E402


class DataPipelineTests(unittest.TestCase):
    def test_price_cache_round_trip_filters_invalid_values(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "prices.json"
            save_price_cache({"0050.TW": {"2026-07-01": 100.5}}, path)
            unchanged = path.read_text(encoding="utf-8")
            save_price_cache({"0050.TW": {"2026-07-01": 100.5}}, path)
            self.assertEqual(path.read_text(encoding="utf-8"), unchanged)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["prices"]["0050.TW"]["bad"] = None
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                load_price_cache(["0050.TW", "MU"], path),
                {"0050.TW": {"2026-07-01": 100.5}, "MU": {}},
            )

    def test_chart_proxy_payload_parser(self) -> None:
        payload = (
            "Title: \n\nURL Source: http://example.test\n\nMarkdown Content:\n"
            '{"chart":{"result":[{"timestamp":[100,200],"indicators":{"quote":[{"close":[10.5,null]}]}}],"error":null}}'
        )
        self.assertEqual(_parse_chart_proxy_payload(payload), [(100, 10.5)])

    def test_tooltip_uses_taipei_time(self) -> None:
        tooltip = build_tooltip(
            "測試貼文",
            direction=1,
            strength=2,
            action="測試",
            dt=datetime(2026, 7, 1, 4, 21, tzinfo=timezone.utc),
        )
        self.assertIn("FB 07/01 12:21", tooltip)

    def test_same_post_with_rounded_seconds_is_duplicate(self) -> None:
        existing = [{
            "unix_ms": 1_788_270_066_000,
            "tooltip": "怎麼辦\n指標: 偏多 ▲ | 強度: ★★☆\nFB 07/01 12:21 同一篇 Facebook 貼文...",
        }]
        candidate = {
            "unix_ms": 1_788_270_060_000,
            "tooltip": "怎麼辦\n指標: 偏多 ▲ | 強度: ★★☆\nFB 07/01 12:21 同一篇 Facebook 貼文完整內容",
        }
        self.assertTrue(is_duplicate_event(candidate, existing))

    def test_distinct_posts_in_same_minute_are_not_duplicates(self) -> None:
        existing = [{
            "unix_ms": 1_788_270_066_000,
            "tooltip": "貼文\n指標: 偏多 ▲ | 強度: ★★☆\nFB 07/01 12:21 第一篇不同內容",
        }]
        candidate = {
            "unix_ms": 1_788_270_060_000,
            "tooltip": "貼文\n指標: 偏多 ▲ | 強度: ★★☆\nFB 07/01 12:21 第二篇完全不同內容",
        }
        self.assertFalse(is_duplicate_event(candidate, existing))

    def test_dashboard_event_uses_taipei_date_and_source_url(self) -> None:
        archive = json.loads(
            (ROOT / "data" / "facebook_posts_2026-05-01_2026-07-20.json").read_text(encoding="utf-8")
        )
        post = next(item for item in archive["posts"] if item["investment_related"])
        pine = (
            f"array.push(evt_time, {post['unix_ms']})\n"
            "array.push(evt_dir, 1)\n"
            "array.push(evt_str, 2)\n"
            'array.push(evt_tips, "測試\\n指標: 偏多 ▲ | 強度: ★★☆\\nFB 05/01 12:00 測試")\n'
            'array.push(evt_ticker, "")\n'
        )
        event = parse_events(pine)[0]
        self.assertEqual(event["date"], post["published_at"][:10])
        self.assertEqual(event["source_url"], post["permalink_url"])


if __name__ == "__main__":
    unittest.main()
