"""Tokyo Insight — configuration & the robots/legal guardrails.

Design (decided 2026-06-07):
  - SHIP THE ENGINE, NOT THE CONTENT. This package contains only MOBIUS's own
    work (parsers, schema, prompts) — never bundled minutes text.
  - LOCAL-ONLY. Each user fetches records on their own machine; there is no
    MOBIUS-operated server, proxy, or shared index (avoids カラオケ法理 /
    ロクラクII making the distributor the 複製主体).
  - ROBOTS-RESPECTING. Only the English-slug committee paths and the plenary
    `proceedings/` path are fetched — these are NOT in the 都議会 robots.txt
    Disallow list. The romaji-slug legacy paths and the DB-Search search tree
    (record.gikai...) are Disallow:/ and are HARD-REFUSED here.
  - NON-COMMERCIAL civic research / personal verification. Use is grounded in
    著作権法40条 (公開された政治上の陳述の利用) + 私的使用30条 + 引用32条;
    output cites + links back to the official source and never redistributes
    full text.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_URL = "https://www.gikai.metro.tokyo.lg.jp/record"
USER_AGENT = ("Tokyo-Insight/0.1 (MOBIUS civic-research engine; non-commercial; "
              "robots-respecting; local-only)")
REQUEST_DELAY_SEC = 1.5

# robots-PERMITTED standing-committee slugs (English) + the plenary path.
COMMITTEES = {
    "general-affairs": "総務委員会",
    "financial": "財政委員会",
    "educational": "文教委員会",
    "welfare": "厚生委員会",
    "urban-development": "都市整備委員会",
    "economic-port-and-harbor": "経済・港湾委員会",
    "environmental-construction": "環境・建設委員会",
}
# Anything outside this set is refused by the fetcher (legacy romaji slugs such
# as /record/bunkyo/ are robots-Disallow and must never be auto-fetched).
ALLOWED_SLUGS = frozenset(COMMITTEES) | {"proceedings"}

# Embedding / retrieval.
E5_MODEL_ID = "intfloat/multilingual-e5-large"
TOP_K_DENSE = 12
TOP_K_BM25 = 12
HYBRID_TOPN = 12
DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4
MAX_CHUNK_CHARS = 1500

# LLM (OpenAI-compatible; BYO endpoint+key — 120B-first, provider-agnostic).
LLM_API_BASE = os.environ.get("TOKYO_INSIGHT_API_BASE",
                              os.environ.get("GROQ_API_BASE",
                                             "https://api.groq.com/openai/v1"))
LLM_MODEL = os.environ.get("TOKYO_INSIGHT_MODEL",
                           os.environ.get("REASONING_MODEL", "openai/gpt-oss-120b"))
# Key resolved at call time from TOKYO_INSIGHT_API_KEY or GROQ_API_KEY.

# Local data layout (NOT distributed — created on the user's machine).
DATA_DIR = Path(os.environ.get("TOKYO_INSIGHT_DATA", "tokyo_insight_data")).resolve()
RAW_DIR = DATA_DIR / "raw"          # on-demand fetch cache (session-local)
INDEX_DIR = DATA_DIR / "index"      # Pro-only full library (NOT shipped)

# Routing pack (the ONLY data artifact shipped publicly): facts + MOBIUS-derived
# record vectors, no minutes text. Defaults to the dev location; the public
# package ships it inside the package dir.
ROUTING_DIR = Path(os.environ.get("TOKYO_INSIGHT_ROUTING",
                                  str(Path(__file__).resolve().parent / "routing_pack"))).resolve()

# On-demand (live) retrieval caps — the politeness / anti-bulk-crawl guard.
LIVE_TOP_K = int(os.environ.get("TOKYO_INSIGHT_LIVE_K", 3))   # records to fetch
LIVE_MAX_FETCH = int(os.environ.get("TOKYO_INSIGHT_LIVE_MAX", 6))  # hard cap/query

# Routing-pack freshness. `refresh` is INCREMENTAL (new records only) and
# user-initiated; ask-live just *notices* when the pack is older than this and
# suggests `refresh`. No silent background fetching (consent / anti-bulk-crawl).
ROUTING_STALE_DAYS = int(os.environ.get("TOKYO_INSIGHT_STALE_DAYS", 90))


def committee_name(slug: str) -> str:
    return COMMITTEES.get(slug, slug)
