"""Tokyo Insight CLI.

  python -m tokyo_insight list <committee>
  python -m tokyo_insight fetch <committee> [<rec> ...] [--latest N]
  python -m tokyo_insight build
  python -m tokyo_insight ask "<question>"
  python -m tokyo_insight committees
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config


def _cmd_committees(_):
    for slug, name in config.COMMITTEES.items():
        print(f"  {slug:28} {name}")


def _cmd_list(a):
    from .fetch import list_records
    for rec, label in list_records(a.committee):
        print(f"  {rec:10} {label}")


def _cmd_fetch(a):
    from .fetch import list_records, fetch_many, _guard
    _guard(a.committee)   # refuse non-allowlisted slugs before any network call
    recs = a.recs
    if a.latest:
        recs = [r for r, _ in list_records(a.committee)[:a.latest]]
    if not recs:
        print("nothing to fetch (give record ids or --latest N)", file=sys.stderr)
        return 2
    print(f"fetching {len(recs)} record(s) from {a.committee} "
          f"({config.committee_name(a.committee)}) …")
    fetch_many(a.committee, recs)
    print(f"saved under {config.RAW_DIR}")


def _cmd_build(_):
    from .index import build
    print("building local index (parse → e5 embed → FAISS + BM25) …")
    meta = build()
    print(f"indexed {meta['chunks']} chunks from {meta['records']} record(s)")
    print(f"index at {config.INDEX_DIR}")


def _cmd_ask(a):
    from .ask import ask
    r = ask(a.question)
    print("\n" + "=" * 70)
    print(r["answer"])
    print("=" * 70)
    print(f"\n[{r['latency_ms']}ms | tokens "
          f"in={r['usage'].get('prompt_tokens', 0)} "
          f"out={r['usage'].get('completion_tokens', 0)} | "
          f"retrieved {len(r['retrieved'])} chunks]")
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))


def _cmd_refresh(a):
    from .refresh import refresh
    committees = [a.committee] if a.committee else None
    r = refresh(committees=committees, dry_run=a.dry_run)
    if a.dry_run:
        print(f"{r['new']} new record(s) not yet in the routing pack"
              + (":" if r["records"] else " (pack is current)"))
        for slug, rec in r["records"][:60]:
            print(f"  {slug}/{rec}")
        return 0
    if not r.get("applied"):
        print("routing pack already current (0 new records)")
        return 0
    print(f"refreshed: +{r['added']} record(s) embedded (total {r['total']})")


def _stale_notice():
    from .refresh import pack_age_days
    age = pack_age_days()
    if age is not None and age > config.ROUTING_STALE_DAYS:
        print(f"note: routing pack is {age} days old (> {config.ROUTING_STALE_DAYS}). "
              f"run `refresh` to include recent meetings.\n")


def _cmd_ask_live(a):
    from .live import ask_live
    _stale_notice()
    r = ask_live(a.question, committee=a.committee, answer=not a.dry_run)
    print("routed records (fetched on demand):")
    for rec in r["routed_records"]:
        print(f"  {rec['score']:.3f}  {rec['committee']:<26} "
              f"{rec['record']} {rec['session']}")
    if a.dry_run:
        print(f"\n[dry-run: no LLM] retrieved {len(r['retrieved'])} chunks "
              f"from the fetched records:")
        for c in r["retrieved"][:8]:
            print(f"  [{c['n']}] {c['meeting']}  議題={c['agenda'] or '—'}  "
                  f"発言者={c['speaker']} ({c['role']})")
        return 0
    print("\n" + "=" * 70)
    print(r["answer"])
    print("=" * 70)
    print(f"\n[{r['latency_ms']}ms | retrieved {len(r['retrieved'])} chunks]")
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tokyo_insight")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("committees").set_defaults(fn=_cmd_committees)
    s = sub.add_parser("list"); s.add_argument("committee"); s.set_defaults(fn=_cmd_list)
    s = sub.add_parser("fetch")
    s.add_argument("committee"); s.add_argument("recs", nargs="*")
    s.add_argument("--latest", type=int, default=0); s.set_defaults(fn=_cmd_fetch)
    sub.add_parser("build").set_defaults(fn=_cmd_build)
    s = sub.add_parser("ask")
    s.add_argument("question"); s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_cmd_ask)
    s = sub.add_parser("ask-live")
    s.add_argument("question")
    s.add_argument("--committee", default=None, help="restrict routing to one slug")
    s.add_argument("--dry-run", action="store_true",
                   help="route + fetch + retrieve only, no LLM (no API key needed)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_cmd_ask_live)
    s = sub.add_parser("refresh")
    s.add_argument("--committee", default=None, help="refresh one slug only")
    s.add_argument("--dry-run", action="store_true",
                   help="list new records without fetching/embedding")
    s.set_defaults(fn=_cmd_refresh)
    a = p.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
