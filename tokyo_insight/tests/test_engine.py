"""Unit/integration tests for the Tokyo Insight engine.

Run:  PYTHONPATH=<repo> python -m pytest tokyo_insight/tests/ -q
Network-free except where marked; the role test uses a cached record if present.
"""
import os
import pytest

from tokyo_insight import config
from tokyo_insight.parse import kansuji, parse_html
from tokyo_insight.fetch import _guard
from tokyo_insight.refresh import _jp_date, _session


# ---- pure helpers ---------------------------------------------------------
def test_kansuji():
    assert kansuji("十二") == 12
    assert kansuji("二十三") == 23
    assert kansuji("百七十九") == 179
    assert kansuji("一〇") == 10  # 〇 used as numeral zero


def test_jp_date():
    assert _jp_date("第14号（令和7年11月18日）") == "2025-11-18"
    assert _jp_date("第1号（平成12年2月17日）") == "2000-02-17"
    assert _jp_date("第1号（令和元年5月1日）") == "2019-05-01"
    assert _jp_date("（昭和六十年三月）") is None  # only arabic month/day parsed
    assert _jp_date("no date here") is None


def test_session():
    assert _session("2025-14") == "第14号"
    assert _session("2000-01") == "第1号"


# ---- robots guard (the gray/black line) ----------------------------------
def test_robots_guard_allows_english_committees():
    for slug in ("educational", "financial", "special-accountiong",
                 "police-fire-fighting", "public-enterprise"):
        _guard(slug)  # must not raise


@pytest.mark.parametrize("slug", ["bunkyo", "soumu", "zaisei", "yotoku",
                                  "kakketsu", "kouketsu", "tosei", "gikai"])
def test_robots_guard_refuses_disallowed_romaji(slug):
    with pytest.raises(PermissionError):
        _guard(slug)


# ---- role classification (the fix: 副委員長/理事 are 議員) ----------------
def _cached(slug, rec):
    p = config.RAW_DIR / slug / f"{rec}.html"
    return p if p.exists() else None


def test_role_classification_real_record():
    p = _cached("educational", "2015-08")
    if p is None:
        pytest.skip("cached record not present")
    m = parse_html(p.read_text(encoding="utf-8"), slug="educational",
                   rec="2015-08", url="")
    roles = {u.speaker_name: u.speaker_role for u in m.utterances}
    # 里吉ゆみ was a 副委員長 in this meeting -> must be assembly_member, NOT officer
    sato = [r for n, r in roles.items() if "里吉" in n]
    assert sato and all(r == "assembly_member" for r in sato), sato
    # the meeting must contain both questioners and answerers
    rset = set(roles.values())
    assert "assembly_member" in rset and "executive" in rset


def test_no_vicechair_tagged_officer():
    """Across cached records, every 副委員長/理事 must resolve to assembly_member."""
    checked = 0
    for slug, rec in [("educational", "2015-08"), ("financial", "2020-15"),
                      ("welfare", "2011-03")]:
        p = _cached(slug, rec)
        if p is None:
            continue
        m = parse_html(p.read_text(encoding="utf-8"), slug=slug, rec=rec, url="")
        vice = {e["name"] for e in m.members if e["role"] in ("副委員長", "理事")}
        for u in m.utterances:
            if u.speaker_name in vice:
                assert u.speaker_role == "assembly_member", (slug, u.speaker_name)
        checked += 1
    if checked == 0:
        pytest.skip("no cached records")


# ---- router caps + committee filter --------------------------------------
@pytest.mark.skipif(not (config.ROUTING_DIR / "routing_vectors.npy").exists(),
                    reason="routing pack not built")
def test_router_caps_and_filter():
    from tokyo_insight.index import _embedder
    from tokyo_insight.router import route
    model = _embedder()
    cands = route("教員採用について", model, k=99)
    assert len(cands) <= config.LIVE_MAX_FETCH
    only = route("予算について", model, k=5, committee="financial")
    assert all(c.committee == "financial" for c in only)
