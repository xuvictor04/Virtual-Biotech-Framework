#!/usr/bin/env python3
"""PubMed MCP server (NCBI E-utilities).

The paper's scientist agents "query biomedical literature through PubMed" and
the trial-annotation cascade searches PubMed "with NCT ID confirmation". The
upstream repository routes this through web tools; this small FastMCP server
gives every provider a structured, citable PubMed interface.

Set NCBI_API_KEY (optional) for higher rate limits.
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
mcp = FastMCP("pubmed")
_last = [0.0]


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


@mcp.tool()
def search_pubmed(query: str, max_results: int = 20, sort: str = "relevance") -> dict:
    """Search PubMed. Supports full PubMed syntax, e.g. 'NCT01234567[si]' to find
    publications that list a ClinicalTrials.gov ID in their secondary-source
    field, or '"B7-H3"[tiab] AND lung'. Returns PMIDs with titles, journal, year.

    Args:
        query: PubMed query string.
        max_results: maximum PMIDs to return (<= 200).
        sort: 'relevance' or 'pub_date'.
    """
    r = _get("esearch.fcgi", {"db": "pubmed", "term": query, "retmax": min(max_results, 200),
                              "retmode": "json", "sort": sort})
    data = r.json()["esearchresult"]
    ids = data.get("idlist", [])
    summaries = _summaries(ids) if ids else []
    return {"query": query, "count": int(data.get("count", 0)), "results": summaries,
            "summary": f"{data.get('count', 0)} PubMed records match; showing {len(ids)}."}


def _summaries(ids: list[str]) -> list[dict]:
    r = _get("esummary.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
    res = r.json().get("result", {})
    out = []
    for pmid in ids:
        d = res.get(pmid, {})
        out.append({"pmid": pmid, "title": d.get("title"), "journal": d.get("fulljournalname"),
                    "pubdate": d.get("pubdate"), "authors": [a.get("name") for a in d.get("authors", [])[:6]],
                    "doi": next((a["value"] for a in d.get("articleids", []) if a.get("idtype") == "doi"), None)})
    return out


@mcp.tool()
def fetch_abstracts(pmids: list[str]) -> dict:
    """Fetch title, abstract, publication types and registry IDs (e.g. NCT numbers
    in the DataBank list) for up to 50 PMIDs. Use to confirm a publication
    reports a specific trial before extracting its results.

    Args:
        pmids: list of PubMed IDs.
    """
    pmids = [str(p) for p in pmids][:50]
    r = _get("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
    root = ET.fromstring(r.text)
    out = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID")
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
    return {"articles": out, "summary": f"Fetched {len(out)} of {len(pmids)} requested abstracts."}


if __name__ == "__main__":
    mcp.run()
