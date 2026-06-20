# Tokyo Insight

A **local-first, citation-grounded** civic-RAG engine for **東京都議会**
(Tokyo Metropolitan Assembly) committee deliberation records (委員会速記録).

Ask a question; Tokyo Insight finds the relevant record(s) on the official
assembly site, fetches them **on your own machine**, and answers — distinguishing
議員 (questions) from 執行側 (answers), anchored to meeting / agenda / speaker, and
cited back to the official record.

## What ships — and what does NOT

Tokyo Insight **ships the engine, not the content.** This repository contains
MOBIUS's own work only:

- the engine (parsers, schema, retrieval, prompts), and
- a small **routing pack** — per-record *facts* (committee, date, session,
  speaker roster) plus MOBIUS-derived record vectors. **No minutes text.**

It does **not** bundle or redistribute any assembly minutes. When you ask a
question, your machine fetches just the few relevant records from the official
site, uses them to answer, and does not build or share any corpus.

- **Local-only.** Routing, fetching, retrieval, and reasoning all run on your
  machine. There is no Tokyo-Insight-operated server, proxy, or shared index.
- **Robots-respecting.** Only the robots-permitted English-slug committee paths
  are fetched, with a polite delay and a hard per-query fetch cap. The legacy
  romaji paths and the DB-Search tree are `Disallow:` and are **hard-refused**.
- **Non-commercial** civic research / personal verification. Output summarizes +
  briefly quotes + links to the official record; it never reproduces full text.

You are responsible for your own use. This is a tool, like a browser or feed
reader; it grants no rights beyond what the law already allows.

## Install

```bash
pip install -r requirements.txt        # or: pip install .
export GROQ_API_KEY=...                 # or TOKYO_INSIGHT_API_KEY (any OpenAI-compatible endpoint)
```

The embedding model (`intfloat/multilingual-e5-large`) is pulled on first run.
A CUDA GPU is used automatically if present (`TOKYO_INSIGHT_DEVICE=cpu` to force CPU).

## Use

```bash
python -m tokyo_insight committees                      # list committees
python -m tokyo_insight ask-live "教員採用について議員の指摘と執行側の答弁を整理して"
python -m tokyo_insight ask-live "都債の発行" --committee financial   # scope the routing
python -m tokyo_insight ask-live "..." --dry-run        # route+fetch+retrieve, no LLM (no key needed)
python -m tokyo_insight refresh                         # incrementally add newly-published records
python -m tokyo_insight refresh --dry-run               # list new records without fetching
```

### Committees covered (11)

Seven standing committees — 総務 (`general-affairs`), 財政 (`financial`),
文教 (`educational`), 厚生 (`welfare`), 都市整備 (`urban-development`),
経済・港湾 (`economic-port-and-harbor`), 環境・建設 (`environmental-construction`) —
plus 警察・消防 (`police-fire-fighting`), 公営企業 (`public-enterprise`), the
各会計決算特別委員会 (`special-accountiong`), and the **予算特別委員会**
(`budget`). Records run **平成12年(2000)–present** (~5,300 records).

The 予算特別委員会 uses a year-directory layout
(`/record/budget/<year>/<n-mm>.html`, one 総括質疑 speaker-segment per page) —
fetched the same robots-respecting way. The 決算/予算 special committees range
across every bureau, so scoping with `--committee special-accountiong` /
`--committee budget` sharpens routing when you specifically want them.

### Fail-safe

For out-of-scope or no-matching-record questions the engine **abstains** rather
than fabricate — first via a retrieval-confidence gate, then via a content-aware
check in the answer prompt. It will tell you it found nothing rather than guess.

`ask-live` notices when the routing pack is older than 90 days and suggests
`refresh`. `refresh` is incremental and user-initiated — it fetches only records
not already known, never the whole site, and never on a silent timer.

## How it works

```
question
  → route via the routing pack (facts + record vectors; no minutes fetched)
  → fetch only the top-K candidate records on demand (robots-OK, polite, capped)
  → chunk-level hybrid (e5 dense + BM25) retrieval over just those records
  → governance prompt (role discipline, cite [n], no full-text) → LLM answer
  → fetched HTML is a session-local cache; no corpus is built or shared
```

## Legal basis (non-commercial civic use)

Use rests on Japanese copyright law: **40条** (public political statements may be
used by any means — the most on-point basis for assembly deliberation), **30条**
(private use), **30条の4** (analysis), and **32条** (quotation). The routing pack
contains only facts + derived vectors, not expression. `robots`-permitted ≠
reuse-permitted: the engine cites and links, and never redistributes minutes text.

## License

Code & schema: **AGPL-3.0-or-later** (see `LICENSE`). MOBIUS brand reserved.
