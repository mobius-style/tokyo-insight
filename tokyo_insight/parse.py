"""Parse a 都議会 committee 速記録 HTML into 4-axis structured utterances.

4 axes: meeting_kind / meeting_date / speaker(name+role) / agenda_item.
Validated in the spike at speaker_resolved=1.00, agenda≈0.95-0.98, meta=100%
across 文教/総務, 6 meetings / 1,225 utterances. Gotchas handled:
  - speaker glyph varies ○(U+25CB) vs 〇(U+3007)
  - speaker token + speech share one line (split on 全角space)
  - inconsistent roster surname boundary (surname-prefix-or-LCP>=2)
  - footer keyword collision (footer-ONLY sentinels)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_K = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
      "六": 6, "七": 7, "八": 8, "九": 9}


def kansuji(s: str) -> int:
    total = cur = 0
    for ch in s.strip():
        if ch in _K:
            cur = _K[ch]
        elif ch == "十":
            cur = (cur or 1) * 10
            total += cur
            cur = 0
        elif ch == "百":
            cur = (cur or 1) * 100
            total += cur
            cur = 0
    return total + cur


def strip_ws(s: str) -> str:
    return re.sub(r"[\s　]", "", s or "")


def _iso(y: int, m: int, d: int) -> str:
    return f"{y + 2018:04d}-{m:02d}-{d:02d}" if y > 0 else ""


_RE_TITLE = re.compile(r"^(.+?委員会)速記録第([一二三四五六七八九十〇百]+)号$")
_RE_DATE = re.compile(r"^令和([一二三四五六七八九十〇]+)年([一二三四五六七八九十〇]+)月"
                      r"([一二三四五六七八九十〇]+)日")
# bureau/agency section header — the token before 関係 must be an org suffix
# (avoids matching "…八丈島の関係" inside a sentence).
_RE_SECTION = re.compile(r"^(.{1,18}(?:局|庁|本部|部|院|委員会|会議))関係$")
_RE_SPEAKER = re.compile(r"^([○◎◆〇])\s*([^\s　].*)$")
_RE_STAGE = re.compile(r"^[〔（(].*[〕）)]$")
_FOOTER = ("このサイトについて", "ご意見・ご要望", "東京都議会議会局管理部",
           "Copyright", "All Rights Reserved")
_ROLE_KW = ("副委員長", "委員長", "副議長", "議長", "理事", "委員", "議員",
            "教育長", "本部長", "局長", "次長", "担当部長", "部長", "課長",
            "館長", "所長", "技監", "監", "参事", "管理者")
_EXEC_KW = ("教育長", "本部長", "局長", "次長", "担当部長", "部長", "課長",
            "館長", "所長", "技監", "監", "参事", "管理者")
_OFFICER_KW = ("委員長", "副委員長", "議長", "副議長")


@dataclass
class Utterance:
    speaker_label: str
    speaker_name: str
    speaker_role: str          # officer | assembly_member | executive | unknown
    resolved: bool
    agenda_item: Optional[str]
    text: str


@dataclass
class Meeting:
    committee_slug: str
    record: str
    url: str
    meeting_kind: Optional[str]
    meeting_date: Optional[str]
    session_no: Optional[int]
    members: List[Dict] = field(default_factory=list)
    execs: List[Dict] = field(default_factory=list)
    agenda_items: List[str] = field(default_factory=list)
    utterances: List[Utterance] = field(default_factory=list)


def _lines(html: str) -> List[str]:
    body = html.split("</head>", 1)[-1]
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", body, flags=re.S)
    return [ln.strip() for ln in re.sub(r"<[^>]+>", "\n", body).splitlines() if ln.strip()]


def _meta(lines: List[str]) -> Tuple[Dict, int]:
    meta = {"meeting_kind": None, "session_no": None, "meeting_date": None}
    start = 0
    for i, ln in enumerate(lines):
        m = _RE_TITLE.match(strip_ws(ln))
        if m:
            meta["meeting_kind"], meta["session_no"], start = m.group(1), kansuji(m.group(2)), i
            break
    for ln in lines[start:start + 8]:
        md = _RE_DATE.match(strip_ws(ln))
        if md:
            meta["meeting_date"] = _iso(kansuji(md.group(1)), kansuji(md.group(2)),
                                        kansuji(md.group(3)))
            break
    return meta, start


def _rosters(lines: List[str]) -> Tuple[List[Dict], List[Dict]]:
    members, execs = [], []

    def is_name(s: str) -> bool:
        return s.endswith("君") and 2 <= len(strip_ws(s)) <= 12

    i = next((k for k, l in enumerate(lines) if strip_ws(l).startswith("出席委員")), None)
    if i is not None:
        role = None
        for ln in lines[i + 1:]:
            raw, s = ln.strip(), strip_ws(ln)
            if s.startswith(("欠席委員", "出席説明員", "本日の会議")):
                break
            if s in ("委員長", "副委員長", "理事"):
                role = s
            elif is_name(s):
                members.append({"name": raw[:-1], "role": role or "委員"})
                role = None
    i = next((k for k, l in enumerate(lines) if strip_ws(l).startswith("出席説明員")), None)
    if i is not None:
        role = None
        for ln in lines[i + 1:]:
            raw, s = ln.strip(), strip_ws(ln)
            if s.startswith(("本日の会議", "○", "〇")):
                break
            if any(s.endswith(kw) for kw in _EXEC_KW) and not s.endswith("君"):
                role = s
            elif is_name(s):
                execs.append({"name": raw[:-1], "role": role or "説明員"})
                role = None
    return members, execs


def _agenda_block(lines: List[str], start: int) -> Tuple[List[str], int]:
    i = next((k for k in range(start, len(lines))
              if strip_ws(lines[k]).startswith("本日の会議")), None)
    if i is None:
        b = next((k for k in range(start, len(lines)) if _RE_SPEAKER.match(lines[k])), start)
        return [], b
    items, j = [], i + 1
    while j < len(lines) and not _RE_SPEAKER.match(lines[j]):
        s = lines[j].strip()
        if s:
            items.append(re.sub(r"^[・･]\s*", "", s))
        j += 1
    return items, j


def _lcp(a: str, b: str) -> int:
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def _surname(raw: str) -> Optional[str]:
    parts = re.split(r"[\s　]+", raw.strip())
    return parts[0] if len(parts) >= 2 and parts[0] else None


def _resolve(label: str, members: List[Dict], execs: List[Dict]) -> Tuple[str, str, bool]:
    lab = strip_ws(label)
    best, kind, score = None, None, 0
    for k, roster in (("member", members), ("exec", execs)):
        for ent in roster:
            n = strip_ws(ent["name"])
            sur = _surname(ent["name"])
            if sur and lab.startswith(strip_ws(sur)):
                s = 100 + len(sur)
            else:
                s = _lcp(lab, n)
                if s < 2:
                    continue
            if s > score:
                score, best, kind = s, ent, k
    if best is not None:
        if kind == "member":
            rc = "officer" if best["role"] in ("委員長", "副委員長") else "assembly_member"
            return best["name"], rc, True
        return best["name"], "executive", True
    if any(k in lab for k in _EXEC_KW):
        return lab, "executive", True
    if any(k in lab for k in _OFFICER_KW):
        return lab, "officer", False
    return lab, "unknown", False


def _agenda_cue(text: str, items: List[Tuple[str, str]]) -> Optional[str]:
    t = strip_ws(text)
    if "議題といたします" in t or "を議題と" in t:
        for ci, raw in items:
            if ci in t:
                return raw
        return "付託議案"
    m = re.search(r"次に[、，]?(.{2,30}?)(?:の質疑を行います|について質疑|"
                  r"の質疑に入ります|を行います)", text)
    if m:
        return strip_ws(m.group(1))
    m2 = re.search(r"(.{1,16}(?:局|庁|本部|部|院|委員会)関係)(?:に入り|について|の事務事業|の審査)", text)
    if m2:
        return strip_ws(m2.group(1))
    if "事務事業に対する質疑" in t or "事務事業について" in t:
        return "事務事業質疑"
    if "請願陳情" in t or "特定事件" in t:
        return "請願陳情・特定事件"
    return None


def _body(lines: List[str], start: int, members, execs,
          items: List[str]) -> List[Utterance]:
    utts: List[Utterance] = []
    cur: Optional[Utterance] = None
    buf: List[str] = []
    agenda: Optional[str] = None
    compact = [(strip_ws(a), a) for a in items if len(strip_ws(a)) >= 6]

    def flush():
        nonlocal cur, buf
        if cur is not None:
            cur.text = "\n".join(buf).strip()
            cur.agenda_item = cur.agenda_item or agenda
            utts.append(cur)
        cur, buf = None, []

    for ln in lines[start:]:
        if any(f in ln for f in _FOOTER):
            break
        if _RE_STAGE.match(ln.strip()):
            if cur is not None:
                buf.append(ln.strip())
            continue
        sec = _RE_SECTION.match(strip_ws(ln))
        if sec and not _RE_SPEAKER.match(ln):
            if not strip_ws(ln).startswith("以上で"):
                agenda = sec.group(1)
            continue
        m = _RE_SPEAKER.match(ln)
        if m:
            flush()
            label_part, sep, first = m.group(2).partition("　")
            if not sep:
                mm = re.match(r"^(\S+?)\s{1,}(.*)$", m.group(2))
                label_part, first = (mm.group(1), mm.group(2)) if mm else (m.group(2), "")
            name, role, ok = _resolve(label_part, members, execs)
            cur = Utterance(strip_ws(label_part), name, role, ok, agenda, "")
            if first.strip():
                buf.append(first.strip())
                if role == "officer":
                    cue = _agenda_cue(first, compact)
                    if cue:
                        agenda = cue
            cur.agenda_item = agenda
            continue
        if cur is not None and cur.speaker_role == "officer":
            cue = _agenda_cue(ln, compact)
            if cue:
                agenda = cue
                cur.agenda_item = agenda
        if cur is not None:
            buf.append(ln.strip())
    flush()
    return utts


def parse_html(html: str, *, slug: str, rec: str, url: str) -> Meeting:
    lines = _lines(html)
    meta, start = _meta(lines)
    members, execs = _rosters(lines)
    items, body_start = _agenda_block(lines, start)
    utts = _body(lines, body_start, members, execs, items)
    return Meeting(
        committee_slug=slug, record=rec, url=url,
        meeting_kind=meta["meeting_kind"], meeting_date=meta["meeting_date"],
        session_no=meta["session_no"], members=members, execs=execs,
        agenda_items=items, utterances=utts,
    )
