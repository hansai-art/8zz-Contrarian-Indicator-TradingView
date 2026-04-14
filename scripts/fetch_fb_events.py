#!/usr/bin/env python3
"""
fetch_fb_events.py
──────────────────
Fetches recent posts from the tracked FB page via Apify API, classifies
each post's sentiment using Google Gemini AI (with keyword-rule fallback),
then writes new events to data/new_events.json for update_pine_script.py.

State tracking:
  data/last_event_timestamp.json  – stores last successfully processed
                                    post timestamp (unix ms) so re-runs
                                    are idempotent.

Environment variables (set as GitHub Actions secrets):
  APIFY_TOKEN       – Apify API token (Settings → Integrations → API tokens)
  GOOGLE_API_KEY    – Google Gemini API key for AI classification.
                      If absent, falls back to the keyword rule table.

Usage:
  python scripts/fetch_fb_events.py
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import urllib.request
import urllib.error

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "data" / "last_event_timestamp.json"
OUTPUT_FILE = ROOT / "data" / "new_events.json"

# ── Configuration ─────────────────────────────────────────────────────────────
APIFY_TOKEN: str    = os.environ.get("APIFY_TOKEN", "")
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")

# Apify actor for Facebook Posts Scraper
APIFY_ACTOR_ID   = "apify~facebook-posts-scraper"
APIFY_FB_URL     = "https://www.facebook.com/DieWithoutBang"
APIFY_MAX_POSTS  = 10   # posts per Apify run

# ── Apify FB scraping ─────────────────────────────────────────────────────────

def fetch_posts_via_apify() -> list[dict]:
    """
    Trigger an Apify run of facebook-posts-scraper synchronously,
    then return the resulting dataset items.
    Falls back gracefully if APIFY_TOKEN is missing or API call fails.
    """
    if not APIFY_TOKEN:
        print("WARNING: APIFY_TOKEN not set. Skipping FB scrape.")
        return []

    # ── 1. Trigger a synchronous run (waits until finished) ───────────────────
    run_url = (
        f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"
        f"?token={APIFY_TOKEN}&timeout=120&memory=256"
    )
    payload = json.dumps({
        "startUrls": [{"url": APIFY_FB_URL}],
        "resultsLimit": APIFY_MAX_POSTS,
    }).encode("utf-8")

    print(f"ℹ️  Triggering Apify run for {APIFY_FB_URL} …")
    try:
        req = urllib.request.Request(
            run_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=150) as resp:
            items = json.loads(resp.read().decode("utf-8"))
        print(f"ℹ️  Apify returned {len(items)} item(s).")
        return items
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: Apify HTTP {e.code}: {body[:300]}")
    except Exception as exc:
        print(f"ERROR: Apify call failed: {exc}")

    return []


def parse_apify_post(item: dict) -> tuple[str, datetime] | None:
    """
    Extract (text, datetime) from an Apify facebook-posts-scraper item.
    Returns None if the item lacks usable text or timestamp.
    """
    text: str = (
        item.get("text") or
        item.get("postText") or
        item.get("message") or
        ""
    ).strip()
    if not text:
        return None

    # Try several timestamp field names used by different actor versions
    raw_time = (
        item.get("time") or
        item.get("timestamp") or
        item.get("created_time") or
        item.get("date") or
        ""
    )
    post_time: datetime | None = None
    if raw_time:
        try:
            # ISO 8601 string  (e.g. "2026-04-13T05:30:00.000Z")
            post_time = datetime.fromisoformat(
                str(raw_time).replace("Z", "+00:00")
            )
        except ValueError:
            pass
    if post_time is None:
        # Unix seconds fallback
        unix = item.get("unixTimestamp") or item.get("unix_timestamp")
        if unix:
            post_time = datetime.fromtimestamp(int(unix), tz=timezone.utc)

    if post_time is None:
        print(f"  WARNING: could not parse timestamp for item, skipping.")
        return None

    if post_time.tzinfo is None:
        post_time = post_time.replace(tzinfo=timezone.utc)

    return text, post_time


# ── Google Gemini AI classification ──────────────────────────────────────────

_SYSTEM = """你是「8zz 反指標」系統的情緒分析器，專門分析台灣散戶的 Facebook 投資貼文。

【核心原則：情緒 > 動作】
就算他「買進」，但情緒是痛苦/被套/停損 → direction: 1（偏多▲）
就算他「賣出」，但情緒是得意/歡呼/獲利了結 → direction: -1（偏空▼）
關鍵是發文者當下的心理狀態，不是他做了什麼動作。

【必須跳過的貼文 → direction: 0】
以下類型與投資決策無關，一律輸出 direction: 0：
- 抱怨「被人當工具」、「被複製」、「被跟單」等關於個人聲譽的牢騷
- 宣告直播、錄影、開會、上課等活動通知
- 聊天氣、運動、飲食、旅遊等生活日常
- 轉貼新聞、評論時事，但沒有明確提到自己的持倉或交易動作
- 情緒模糊、無法判斷是否與投資有關的貼文
- 發文者明確說「完全沒想法」、「不知道要買什麼」、「隨便」等，代表本人無投資立場 → direction: 0
- 被動觀察持倉狀態，如「還沒到停損範圍」、「繼續持有看看」，沒有伴隨強烈情緒 → direction: 0
  （注意：「還沒停損」和「已停損」是完全不同的！否定句不算停損訊號）
- 粉絲/追蹤者「問他要對帳單」、「問他為什麼沒貼對帳單」→ 這是關於粉絲行為的抱怨，發文者沒有自己曬單，一律 direction: 0
  （判斷方式：誰在問？誰在曬？「別人問我要對帳單」≠「我自己貼出對帳單」）

direction:
  1  = 偏多 ▲（恐慌/痛苦/被套 → 市場可能近底部）
 -1  = 偏空 ▼（亢奮/追漲/自大 → 市場可能近頂部）
  0  = 跳過（與投資無關，或情緒完全中性）

strength（情緒強度）:
  1 = 輕微  2 = 明顯  3 = 極端（爆倉/漲停追買/歡天喜地）
  ★ 強制規則：如果貼文出現「委託成功」「全部成交」「部分成交」「委託狀態」「掛單成功」等
    委托單截圖的文字，代表已真金白銀下單，強度一律設為 3（不論其他情緒強弱）

ticker:
- 貼文明確提到特定標的 → 輸出 yfinance ticker（如 5274.TWO、TSM、0050.TW、GLD、BTCUSDT）
- 不確定或未提及 → 輸出空字串 ""

輸出純 JSON，不加 markdown：
{"direction": 1, "strength": 2, "action": "停損", "ticker": "", "reasoning": "一句話說明"}"""

_EXAMPLES = [
    {"role": "user",      "content": "我今天停損出場了，損失超過10萬，心情很差"},
    {"role": "assistant", "content": '{"direction": 1, "strength": 3, "action": "停損", "ticker": "", "reasoning": "大額停損出場，情緒極度痛苦，強烈偏多訊號"}'},
    {"role": "user",      "content": "停損渣男鈦昇後買信驊，情緒衝動了"},
    {"role": "assistant", "content": '{"direction": 1, "strength": 3, "action": "衝動(復仇)", "ticker": "5274.TWO", "reasoning": "停損後衝動買進，情緒是痛苦+衝動，即使買進動作仍是偏多訊號"}'},
    {"role": "user",      "content": "今天漲停板追進去了！感覺這支會繼續飆！"},
    {"role": "assistant", "content": '{"direction": -1, "strength": 3, "action": "漲停追買", "ticker": "", "reasoning": "漲停追買加上極度亢奮的FOMO情緒，強烈偏空訊號"}'},
    {"role": "user",      "content": "加碼台積電，長期看好半導體"},
    {"role": "assistant", "content": '{"direction": -1, "strength": 2, "action": "加碼", "ticker": "TSM", "reasoning": "主動加碼且語氣自信看好，屬於偏空訊號"}'},
    {"role": "user",      "content": "今天天氣真好，出去走走"},
    {"role": "assistant", "content": '{"direction": 0, "strength": 1, "action": "", "ticker": "", "reasoning": "與投資無關的生活貼文"}'},
    {"role": "user",      "content": "我已經變成工具了嗎？大家都在跟單，好無奈"},
    {"role": "assistant", "content": '{"direction": 0, "strength": 1, "action": "", "ticker": "", "reasoning": "抱怨被人跟單當工具，無具體投資動作，跳過"}'},
    {"role": "user",      "content": "yeah~ 來錄影了，如果大家都把我寫成工具…"},
    {"role": "assistant", "content": '{"direction": 0, "strength": 1, "action": "", "ticker": "", "reasoning": "活動通知兼抱怨被跟單，非投資貼文，跳過"}'},
    {"role": "user",      "content": "花了我250元買的錄面，說什麼都要炫富一下 不要問我明天買啥了… 我完全沒想法😅! 跌的也還沒到停損範圍，就放著"},
    {"role": "assistant", "content": '{"direction": 0, "strength": 1, "action": "", "ticker": "", "reasoning": "發文者明確說完全沒想法，且提到的是還沒到停損範圍（否定句，代表被動持有非停損動作），屬於閒聊非訊號，跳過"}'},
    {"role": "user",      "content": "今天衝啊啊啊啊！！！今天大回血了❤️ 一定會更好的！！ 突然想到，被動元件好像炒的沒有pcb/abf跟記憶體兇，感覺可以補一下 日月光只是看ADR很猛，掛著買看看好了\n[委托單截圖]\n日月光投控 委託成功\n尖點 全部成交\n尖點 全部取消"},
    {"role": "assistant", "content": '{"direction": -1, "strength": 3, "action": "追買(委托單)", "ticker": "3711.TW", "reasoning": "極度亢奮FOMO追買，更有委托單截圖證明已真實下單，強度強制升為3，強烈偏空訊號"}'},
    {"role": "user",      "content": "這篇沒有名牌 只是因為私訊問我是不是有開群組啊 今天怎麼沒開示啊 為什麼還沒有對帳單啊 的人實在太多了 我也不..."},
    {"role": "assistant", "content": '{"direction": 0, "strength": 1, "action": "", "ticker": "", "reasoning": "粉絲在問發文者要對帳單，是關於粉絲行為的抱怨，發文者沒有自己曬單或做任何投資動作，跳過"}'},
]


def classify_with_ai(text: str) -> dict | None:
    """
    Call Google Gemini Flash to classify a post.
    Returns dict(direction, strength, action, ticker, reasoning), or None on failure.
    """
    if not GOOGLE_API_KEY:
        return None

    try:
        import google.generativeai as genai
    except ImportError:
        print("WARNING: google-generativeai not installed. Falling back to keyword rules.")
        return None

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-lite",
            system_instruction=_SYSTEM,
        )

        few_shot = ""
        for i in range(0, len(_EXAMPLES), 2):
            u = _EXAMPLES[i]["content"]
            a = _EXAMPLES[i + 1]["content"]
            few_shot += f"貼文：{u}\n回答：{a}\n\n"
        prompt = few_shot + f"貼文：{text}\n回答："

        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        result = json.loads(raw)

        return {
            "direction": int(result.get("direction", 0)),
            "strength":  max(1, min(3, int(result.get("strength", 1)))),
            "action":    str(result.get("action", "")).strip(),
            "ticker":    str(result.get("ticker", "")).strip(),
            "reasoning": str(result.get("reasoning", "")).strip(),
        }
    except Exception as exc:
        print(f"WARNING: Gemini API error ({type(exc).__name__}: {exc}). Falling back to keyword rules.")
        return None


# ── Keyword-rule fallback ─────────────────────────────────────────────────────
SENTIMENT_RULES: list[tuple[list[str], int, int]] = [
    (["停損", "認賠", "虧損", "損失", "全賠", "出清停損", "畢業", "爆倉"], 1, 3),
    (["被套", "套牢", "跌停", "住套房", "房貸", "公園", "淨值歸零"],       1, 3),
    (["賣出", "停利", "獲利了結", "出場", "認損"],    1, 2),
    (["心碎", "白做工", "不懂", "怎麼辦", "救我"],    1, 2),
    (["觀望", "等", "修正", "怕", "謹慎"],            1, 1),
    (["漲停買", "漲停追", "市價掛", "追漲"],          -1, 3),
    (["無敵", "一定漲", "必漲"],                      -1, 3),
    (["買進", "加碼", "補倉", "買了", "入手", "佈局", "再買"], -1, 2),
    (["看多", "多頭", "應該漲", "會漲", "繼續持有"],  -1, 2),
    (["持有", "觀察", "等待", "慢慢漲", "長期"],       -1, 1),
]


# Negation prefixes: if any of these appear immediately before a keyword,
# the keyword match is suppressed (e.g. "還沒停損" should NOT trigger "停損").
_NEGATION_PREFIXES = ["還沒", "沒有", "沒到", "未到", "不用", "不會", "不必"]


def classify_with_keywords(text: str) -> tuple[int, int, str] | None:
    # Skip posts where the author explicitly states they have no investment idea.
    no_signal_phrases = ["完全沒想法", "沒有想法", "不知道要買", "不知道買什麼", "完全不知道"]
    if any(phrase in text for phrase in no_signal_phrases):
        return None

    for keywords, direction, strength in SENTIMENT_RULES:
        for kw in keywords:
            idx = text.find(kw)
            if idx == -1:
                continue
            # Check if the keyword is preceded by a negation prefix
            prefix_window = text[max(0, idx - 4):idx]
            if any(neg in prefix_window for neg in _NEGATION_PREFIXES):
                continue
            return direction, strength, kw
    return None


# ── Order-proof strength booster ─────────────────────────────────────────────
# When a post contains actual order confirmation text (委托單截圖內文), the person
# has provably committed real money → strongest contrarian signal regardless of
# how the AI or keyword rules scored the emotion intensity.
_ORDER_CONFIRMATION_PHRASES = [
    "委託成功", "委托成功",
    "全部成交", "部分成交",
    "委託狀態", "委托狀態",
    "掛單成功", "下單成功",
    "委託單", "委托單",
]


def boost_strength_if_order_proof(text: str, direction: int, strength: int) -> int:
    """Return strength=3 if the post contains order confirmation phrases and is a signal."""
    if direction != 0 and any(phrase in text for phrase in _ORDER_CONFIRMATION_PHRASES):
        return 3
    return strength


# ── Tooltip builder ───────────────────────────────────────────────────────────

def build_tooltip(post_text: str, direction: int, strength: int, action: str, dt: datetime) -> str:
    dir_label = "偏多 ▲" if direction == 1 else "偏空 ▼"
    stars = {1: "★☆☆", 2: "★★☆", 3: "★★★"}.get(strength, "★☆☆")
    snippet = post_text.replace("\n", " ").strip()
    if len(snippet) > 60:
        snippet = snippet[:57] + "..."
    date_str = dt.astimezone(timezone.utc).strftime("FB %m/%d %H:%M")
    return (
        f"{action or '貼文'}\n"
        f"指標: {dir_label} | 強度: {stars}\n"
        f"{date_str} {snippet}"
    )


# ── State persistence ─────────────────────────────────────────────────────────

def load_state() -> int:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return int(data.get("last_fetched_unix_ms", 0))
        except (json.JSONDecodeError, ValueError):
            pass
    return 0


def save_state(last_unix_ms: int) -> None:
    now_utc = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(
        json.dumps(
            {"last_fetched_unix_ms": last_unix_ms, "last_run_utc": now_utc},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    using_ai = bool(GOOGLE_API_KEY)
    print(f"ℹ️  Classifier: {'Gemini Flash (AI)' if using_ai else 'keyword rules'}")

    last_unix_ms = load_state()
    raw_items = fetch_posts_via_apify()

    new_events: list[dict] = []
    latest_unix_ms = last_unix_ms

    for item in raw_items:
        parsed = parse_apify_post(item)
        if parsed is None:
            continue
        text, post_time = parsed

        unix_ms = int(post_time.timestamp() * 1000)
        if unix_ms <= last_unix_ms:
            print(f"  Skipping already-processed post ({post_time.strftime('%m/%d %H:%M')})")
            continue

        # ── Classify ──────────────────────────────────────────────────────────
        direction, strength, action, ticker = 0, 1, "", ""
        ai = classify_with_ai(text)

        if ai is not None:
            direction = ai["direction"]
            strength  = ai["strength"]
            action    = ai["action"]
            ticker    = ai["ticker"]
            strength  = boost_strength_if_order_proof(text, direction, strength)
            print(f"  [Gemini] dir={direction} str={strength} action='{action}' ticker='{ticker or '(0050)'}' | {text[:40]}")
        else:
            kw = classify_with_keywords(text)
            if kw is None:
                print(f"  [keyword] no match – skipping | {text[:40]}")
                continue
            direction, strength, action = kw
            strength = boost_strength_if_order_proof(text, direction, strength)
            print(f"  [keyword] dir={direction} str={strength} action='{action}' | {text[:40]}")

        # Always advance the state pointer — even direction=0 posts must not be
        # re-processed on the next run (otherwise skipped posts loop forever).
        if unix_ms > latest_unix_ms:
            latest_unix_ms = unix_ms

        if direction == 0:
            print(f"  → direction=0, not investment-related, skipping")
            continue

        tooltip = build_tooltip(text, direction, strength, action, post_time)
        new_events.append({
            "unix_ms":   unix_ms,
            "direction": direction,
            "strength":  strength,
            "ticker":    ticker,
            "tooltip":   tooltip,
        })

    new_events.sort(key=lambda e: e["unix_ms"])
    OUTPUT_FILE.write_text(
        json.dumps(new_events, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Always save the furthest-seen timestamp so skipped (direction=0) posts
    # are never re-processed on the next run.
    save_state(latest_unix_ms)
    if new_events:
        print(f"✅ {len(new_events)} new event(s) written to {OUTPUT_FILE}")
    else:
        print("ℹ️  No new classifiable events found this run.")


if __name__ == "__main__":
    main()
