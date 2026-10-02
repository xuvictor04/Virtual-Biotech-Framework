#!/usr/bin/env python3
"""PubMed MCP server (NCBI E-utilities).

The paper's scientist agents "query biomedical literature through PubMed" and
the trial-annotation cascade searches PubMed "with NCT ID confirmation". The
upstream repository routes this through web tools; this small FastMCP server
gives every provider a structured, citable PubMed interface.

Set NCBI_API_KEY (optional) for higher rate limits.

Literature date ceiling (leakage control): when VBT_LITERATURE_MAXDATE is set
(``YYYY/MM/DD``; the no-web profile sets it through ``tool_env``), every search
adds ``datetype=pdat&mindate=1800&maxdate=<ceiling>`` and ``fetch_abstracts``
withholds records whose earliest publication date (print or electronic) may
fall after the ceiling, reporting them under ``withheld``. Dates with missing
month/day are read as the latest possible day, so ambiguous records are
withheld. An unparseable ceiling makes every tool fail rather than run undated.
"""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from fastmcp import FastMCP  # same framework as the upstream servers

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAXDATE_ENV = "VBT_LITERATURE_MAXDATE"
mcp = FastMCP("pubmed")
_last = [0.0]

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_SEASONS = {"spring": 6, "summer": 9, "fall": 12, "autumn": 12, "winter": 12}
_DATE_TOKEN = re.compile(r"(\d{4})(?:[\s/-]+([A-Za-z]+|\d{1,2}(?!\d)))?(?:[\s/-]+(\d{1,2}(?!\d)))?")


def literature_ceiling() -> tuple[int, int, int] | None:
    """The configured ceiling as (year, month, day), or None when unset.

    Raises ValueError for a malformed value so the server fails closed.
    """
    raw = (os.environ.get(MAXDATE_ENV) or "").strip()
    if not raw:
        return None
    m = re.fullmatch(r"(\d{4})(?:[/-](\d{1,2}))?(?:[/-](\d{1,2}))?", raw)
    if not m:
        raise ValueError(f"{MAXDATE_ENV}={raw!r} is not a YYYY/MM/DD date")
    y, mo, d = int(m.group(1)), int(m.group(2) or 12), int(m.group(3) or 31)
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        raise ValueError(f"{MAXDATE_ENV}={raw!r} is not a valid date")
    return (y, mo, d)


def _fmt(date: tuple[int, int, int]) -> str:
    return f"{date[0]:04d}/{date[1]:02d}/{date[2]:02d}"


def _date_params() -> dict[str, str]:
    ceiling = literature_ceiling()
    if ceiling is None:
        return {}
    return {"datetype": "pdat", "mindate": "1800", "maxdate": _fmt(ceiling)}


def latest_possible(text: str | None) -> tuple[int, int, int] | None:
    """Latest calendar day a (possibly partial) PubMed date string can denote.

    '2024 Mar 5' -> (2024, 3, 5); '2024 Mar' -> (2024, 3, 31); '2024' -> (2024, 12, 31);
    '2024 Dec-2025 Jan' -> (2025, 1, 31). None when no year is present.
    """
    if not text:
        return None
    best = None
    for y, mo, d in _DATE_TOKEN.findall(str(text)):
        month = 12
        if mo:
            if mo.isdigit():
                month = min(max(int(mo), 1), 12)
            else:
                month = _MONTHS.get(mo[:3].lower()) or _SEASONS.get(mo.lower(), 12)
        day = int(d) if d and month and mo else 31
        cand = (int(y), month, min(max(day, 1), 31))
        if best is None or cand > best:
            best = cand
    return best


def _element_date(node) -> tuple[int, int, int] | None:
    if node is None:
        return None
    medline = node.findtext("MedlineDate")
    if medline:
        return latest_possible(medline)
    year = node.findtext("Year")
    if not year:
        return None
    parts = [year, node.findtext("Month") or "", node.findtext("Day") or ""]
    return latest_possible(" ".join(p for p in parts if p))


def availability_date(article) -> tuple[int, int, int] | None:
    """Earliest publication date (print or electronic) of a PubmedArticle element."""
    dates = [_element_date(article.find(".//Journal/JournalIssue/PubDate"))]
    dates += [_element_date(a) for a in article.findall(".//ArticleDate")]
    dates = [d for d in dates if d]
    return min(dates) if dates else None


def summary_date(doc: dict) -> tuple[int, int, int] | None:
    dates = [latest_possible(doc.get("pubdate")), latest_possible(doc.get("epubdate"))]
    dates = [d for d in dates if d]
    return min(dates) if dates else None


def _get(endpoint: str, params: dict[str, Any]) -> httpx.Response:
    params = {**params, "tool": "vbt-harness", "email": os.environ.get("NCBI_EMAIL", "")}
    if os.environ.get("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    wait = 0.11 if "api_key" in params else 0.35  # stay under NCBI rate limits
    delay = _last[0] + wait - time.time()
    if delay > 0:
        time.sleep(delay)
    for attempt in range(4):
        r = httpx.get(f"{EUTILS}/{endpoint}", params=params, timeout=60)
        _last[0] = time.time()
        if r.status_code != 429:
            r.raise_for_status()
            return r
        time.sleep(2 ** attempt)
    r.raise_for_status()
    return r


def search(query: str, max_results: int = 20, sort: str = "relevance") -> dict:
    ceiling = literature_ceiling()
    r = _get("esearch.fcgi", {"db": "pubmed", "term": query, "retmax": min(max_results, 200),
                              "retmode": "json", "sort": sort, **_date_params()})
    data = r.json()["esearchresult"]
    ids = data.get("idlist", [])
    summaries = _summaries(ids) if ids else []
    withheld = []
    if ceiling is not None:
        kept = []
        for doc in summaries:
            when = doc.pop("_date", None)
            if when is None or when > ceiling:
                withheld.append(doc["pmid"])
            else:
                kept.append(doc)
        summaries = kept
    else:
        for doc in summaries:
            doc.pop("_date", None)
    out = {"query": query, "count": int(data.get("count", 0)), "results": summaries,
           "summary": f"{data.get('count', 0)} PubMed records match; showing {len(summaries)}."}
    if ceiling is not None:
        out["literature_max_date"] = _fmt(ceiling)
        out["summary"] += f" Restricted to publications up to {_fmt(ceiling)}."
        if withheld:
            out["withheld"] = withheld
            out["summary"] += f" {len(withheld)} result(s) withheld (date after the ceiling or unknown)."
    return out


@mcp.tool()
def search_pubmed(query: str, max_results: int = 20, sort: str = "relevance") -> dict:
    """Search PubMed. Supports full PubMed syntax, e.g. 'NCT01234567[si]' to find
    publications that list a ClinicalTrials.gov ID in their secondary-source
    field, or '"B7-H3"[tiab] AND lung'. Returns PMIDs with titles, journal, year.
    In leakage-controlled runs results are limited to a publication-date ceiling.

    Args:
        query: PubMed query string.
        max_results: maximum PMIDs to return (<= 200).
        sort: 'relevance' or 'pub_date'.
    """
    return search(query, max_results=max_results, sort=sort)


def _summaries(ids: list[str]) -> list[dict]:
    r = _get("esummary.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
    res = r.json().get("result", {})
    out = []
    for pmid in ids:
        d = res.get(pmid, {})
        out.append({"pmid": pmid, "title": d.get("title"), "journal": d.get("fulljournalname"),
                    "pubdate": d.get("pubdate"), "epubdate": d.get("epubdate") or None,
                    "_date": summary_date(d),
                    "authors": [a.get("name") for a in d.get("authors", [])[:6]],
                    "doi": next((a["value"] for a in d.get("articleids", []) if a.get("idtype") == "doi"), None)})
    return out


def parse_articles(xml_text: str, ceiling: tuple[int, int, int] | None = None) -> tuple[list[dict], list[dict]]:
    """Parse efetch XML into (articles, withheld) honouring the date ceiling."""
    root = ET.fromstring(xml_text)
    out, withheld = [], []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID")
        if ceiling is not None:
            when = availability_date(art)
            if when is None or when > ceiling:
                withheld.append({"pmid": pmid, "reason": (
                    f"publication date unknown; withheld under the {_fmt(ceiling)} literature ceiling"
                    if when is None else
                    f"published {when[0]:04d}-{when[1]:02d} or later, after the {_fmt(ceiling)} literature ceiling")})
                continue
        abstract = " ".join(
            (f"{t.get('Label')}: " if t.get("Label") else "") + "".join(t.itertext())
            for t in art.findall(".//Abstract/AbstractText"))
        registry = [a.text for a in art.findall(".//DataBank/AccessionNumberList/AccessionNumber") if a.text]
        registry += re.findall(r"NCT\d{8}", abstract)
        out.append({
            "pmid": pmid,
            "title": "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else "",
            "journal": art.findtext(".//Journal/Title"),
            "year": art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate"),
            "publication_types": [p.text for p in art.findall(".//PublicationType")],
            "registry_ids": sorted(set(registry)),
            "abstract": abstract,
        })
    return out, withheld


def fetch(pmids: list[str]) -> dict:
    ceiling = literature_ceiling()
    pmids = [str(p) for p in pmids][:50]
    r = _get("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
    out, withheld = parse_articles(r.text, ceiling)
    result = {"articles": out, "summary": f"Fetched {len(out)} of {len(pmids)} requested abstracts."}
    if ceiling is not None:
        result["literature_max_date"] = _fmt(ceiling)
        result["withheld"] = withheld
        if withheld:
            result["summary"] += (f" {len(withheld)} withheld: published after the {_fmt(ceiling)} "
                                  f"literature ceiling (or date unknown).")
    return result


@mcp.tool()
def fetch_abstracts(pmids: list[str]) -> dict:
    """Fetch title, abstract, publication types and registry IDs (e.g. NCT numbers
    in the DataBank list) for up to 50 PMIDs. Use to confirm a publication
    reports a specific trial before extracting its results. In leakage-controlled
    runs, records published after the date ceiling are withheld and listed.

    Args:
        pmids: list of PubMed IDs.
    """
    return fetch(pmids)


if __name__ == "__main__":
    mcp.run()
