"""
topic_deep_analyzer.py
Automated arXiv paper search, PDF download, and deep analysis via LLM API.

Workflow:
  1. Search arXiv exhaustively by topic (paginate all results)
  2. Download full PDFs locally
  3. Extract text from PDFs
  4. Send to LLM (gemini-2.5-pro via OpenAI-compatible API) for structured analysis
  5. Write per-paper analysis + cross-paper synthesis to Markdown (optionally PDF)

Requirements:
    pip install requests pymupdf
    Optional PDF rendering: install pandoc (https://pandoc.org/installing.html)

Usage:
    python topic_deep_analyzer.py --topic "diffusion model" --start 2024-01-01 --max 20
    python topic_deep_analyzer.py --topic "NeRF" --start 2024-01-01 --lang zh --pdf
"""

import hashlib
import requests
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import re
import sys
import argparse
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# =============================================================================
#  API key is loaded from api_key.txt (one line, plain text).
#  Place api_key.txt in the same directory as this script.
# =============================================================================
def _load_api_config() -> tuple:
    """Load from api_key.txt:
       Line 1: API key
       Line 2: API URL
       Line 3: model name (optional, falls back to gemini-2.5-pro)
    """
    key_file = Path(__file__).parent / "api_key.txt"
    try:
        lines = [l.strip() for l in key_file.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        if len(lines) < 2:
            raise ValueError("api_key.txt must have at least 2 lines: API key, then API URL")
        key   = lines[0]
        url   = lines[1]
        model = lines[2] if len(lines) >= 3 else "gemini-2.5-pro"
        return key, url, model
    except FileNotFoundError:
        print(f"[ERROR] api_key.txt not found at: {key_file}")
        print("        Line 1: API key")
        print("        Line 2: API URL")
        print("        Line 3: model name (optional, default: gemini-2.5-pro)")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to read api_key.txt: {e}")
        sys.exit(1)

LLM_API_KEY, LLM_API_URL, DEFAULT_MODEL = _load_api_config()
# =============================================================================

# ── arXiv / download settings ─────────────────────────────────────────────────
ARXIV_API        = "http://export.arxiv.org/api/query"
BATCH_SIZE       = 25      # small batches avoid IncompleteRead on large queries
DEFAULT_MAX      = 20
DOWNLOAD_WORKERS = 4
PDF_TEXT_LIMIT   = 15000   # chars extracted per PDF
REQUEST_DELAY    = 1.5     # seconds between arXiv API pages
MAX_RETRIES      = 3

OUTPUT_DIR = Path("analysis_output")
PAPERS_DIR = Path("downloaded_papers")

# =============================================================================
# PDF extraction
# =============================================================================

def _detect_pdf_backend() -> Optional[str]:
    try:
        import fitz
        fitz.TOOLS.mupdf_display_errors(False)   # suppress C-level stderr noise
        return "pymupdf"
    except ImportError:
        pass
    try:
        import pdfplumber  # noqa: F401
        return "pdfplumber"
    except ImportError:
        pass
    return None

PDF_BACKEND = _detect_pdf_backend()


def extract_text_from_pdf(pdf_path: Path, char_limit: int = PDF_TEXT_LIMIT) -> str:
    if PDF_BACKEND == "pymupdf":
        import fitz
        try:
            doc = fitz.open(str(pdf_path))
            parts, total = [], 0
            for page_num in range(len(doc)):
                try:
                    t = doc[page_num].get_text()
                except Exception:
                    continue   # skip corrupt/unreadable pages silently
                parts.append(t)
                total += len(t)
                if total >= char_limit:
                    break
            doc.close()
            return "\n".join(parts)[:char_limit]
        except Exception as e:
            return f"[PDF error: {e}]"
    elif PDF_BACKEND == "pdfplumber":
        import pdfplumber
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                parts, total = [], 0
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    parts.append(t)
                    total += len(t)
                    if total >= char_limit:
                        break
                return "\n".join(parts)[:char_limit]
        except Exception as e:
            return f"[PDF error: {e}]"
    return "[No PDF library. Run: pip install pymupdf]"

# =============================================================================
# arXiv search
# =============================================================================

def _fetch_page(params: dict, attempt: int = 0) -> Optional[bytes]:
    """Fetch one arXiv API page via urllib with exponential-backoff retry."""
    raw_query = params["search_query"].replace(" ", "+")
    base      = {k: v for k, v in params.items() if k != "search_query"}
    url       = f"{ARXIV_API}?search_query={raw_query}&{urllib.parse.urlencode(base)}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()
    except Exception as e:
        if attempt < MAX_RETRIES - 1:
            wait = 2 ** attempt
            print(f"  [!] Retry {attempt+1}/{MAX_RETRIES}: {e}  (wait {wait}s)")
            time.sleep(wait)
            return _fetch_page(params, attempt + 1)
        print(f"  [!] arXiv API failed: {e}")
        return None


def search_arxiv(query: str, start_date: str, end_date: str,
                 max_papers: Optional[int]) -> list:
    sd = start_date.replace("-", "") + "000000"
    ed = end_date.replace("-", "")   + "235959"
    quoted     = urllib.parse.quote(f'"{query}"')
    full_query = f"(ti:{quoted}+OR+abs:{quoted})+AND+submittedDate:[{sd}+TO+{ed}]"

    papers, seen, offset = [], set(), 0
    total_str = "?"
    ns = "{http://www.w3.org/2005/Atom}"

    print(f"\n[Search] topic={query!r}  {start_date} -> {end_date}"
          f"  cap={max_papers or 'unlimited'}")

    while True:
        raw = _fetch_page({
            "search_query": full_query,
            "start":        offset,
            "max_results":  BATCH_SIZE,
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",
        })
        if raw is None:
            break

        try:
            root = ET.fromstring(raw)
        except ET.ParseError as e:
            print(f"  [!] XML parse error at offset {offset}: {e}")
            break

        total_tag = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
        if total_tag is not None:
            total_str = total_tag.text

        entries = root.findall(f"{ns}entry")
        if not entries:
            break

        for entry in entries:
            id_tag = entry.find(f"{ns}id")
            if id_tag is None:
                continue
            arxiv_id = id_tag.text.strip().split("/")[-1]
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)

            def _t(tag):  # safe text: leaf elements are falsy in ElementTree!
                el = entry.find(f"{ns}{tag}")
                return el.text.strip() if el is not None and el.text else ""

            title   = _t("title").replace("\n", " ")
            summary = _t("summary").replace("\n", " ")
            pub     = _t("published")
            authors = [a.find(f"{ns}name").text.strip()
                       for a in entry.findall(f"{ns}author")
                       if a.find(f"{ns}name") is not None]
            pdf_url = next(
                (lnk.attrib["href"] for lnk in entry.findall(f"{ns}link")
                 if lnk.attrib.get("title") == "pdf"),
                f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            )
            papers.append({
                "arxiv_id":  arxiv_id,
                "title":     title,
                "abstract":  summary,
                "pdf_url":   pdf_url,
                "authors":   authors,
                "published": pub[:10],
            })

        fetched = len(entries)
        print(f"  page offset={offset}  batch={fetched}"
              f"  total_avail={total_str}  collected={len(papers)}", end="\r")

        if max_papers and len(papers) >= max_papers:
            papers = papers[:max_papers]
            break
        if fetched < BATCH_SIZE:
            break

        offset += fetched
        time.sleep(REQUEST_DELAY)

    print(f"\n  Found {len(papers)} papers.")
    return papers

# =============================================================================
# Google Scholar search
# =============================================================================

def _scholar_available() -> bool:
    try:
        import scholarly  # noqa: F401
        return True
    except ImportError:
        return False


def _make_scholar_id(title: str) -> str:
    """Stable short ID from paper title."""
    h = hashlib.md5(title.lower().encode()).hexdigest()[:10]
    return f"scholar_{h}"


def search_scholar(query: str, start_date: str, end_date: str,
                   max_papers: Optional[int]) -> list:
    """
    Search Google Scholar via the `scholarly` library.
    Returns the same paper dict format as search_arxiv().
    Note: Scholar may rate-limit or block; retries with backoff are applied.
    """
    if not _scholar_available():
        print("[Scholar] 'scholarly' not installed. Run: pip install scholarly")
        return []

    from scholarly import scholarly as sc

    year_low  = int(start_date[:4])
    year_high = int(end_date[:4])
    cap       = max_papers or 9999

    print(f"\n[Scholar] query={query!r}  {year_low}-{year_high}"
          f"  cap={max_papers or 'unlimited'}")
    print("  Note: Google Scholar may throttle requests; pauses are normal.")

    papers = []
    seen_titles: set = set()

    try:
        gen = sc.search_pubs(query, year_low=year_low, year_high=year_high)
        consecutive_errors = 0
        collected = 0

        while collected < cap:
            try:
                pub = next(gen)
                consecutive_errors = 0
            except StopIteration:
                break
            except Exception as e:
                consecutive_errors += 1
                wait = min(2 ** consecutive_errors, 60)
                print(f"\n  [!] Scholar fetch error (retry in {wait}s): {e}")
                time.sleep(wait)
                if consecutive_errors >= MAX_RETRIES:
                    print("  [!] Too many Scholar errors, stopping.")
                    break
                continue

            bib = pub.get("bib", {})
            title = (bib.get("title") or "").strip()
            if not title:
                continue

            norm = title.lower()
            if norm in seen_titles:
                continue
            seen_titles.add(norm)

            # Filter by year (scholarly year_low/high is advisory only)
            pub_year = str(bib.get("pub_year") or "")
            if pub_year.isdigit():
                y = int(pub_year)
                if y < year_low or y > year_high:
                    continue
            published = f"{pub_year}-01-01" if pub_year else ""

            abstract = (bib.get("abstract") or "").strip()
            raw_authors = bib.get("author") or []
            if isinstance(raw_authors, str):
                raw_authors = [a.strip() for a in raw_authors.split(" and ")]
            authors = [str(a) for a in raw_authors]

            # Prefer eprint (often a direct PDF), then pub_url
            pdf_url = (pub.get("eprint_url") or pub.get("pub_url") or "").strip()

            papers.append({
                "arxiv_id":  _make_scholar_id(title),
                "title":     title,
                "abstract":  abstract,
                "pdf_url":   pdf_url,
                "authors":   authors,
                "published": published,
                "source":    "scholar",
            })
            collected += 1
            print(f"  collected={collected}", end="\r")
            time.sleep(0.5)   # gentle pacing

    except Exception as e:
        print(f"\n  [!] Scholar search failed: {e}")

    print(f"\n  Found {len(papers)} Scholar papers.")
    return papers


# =============================================================================
# PubMed search (NCBI E-utilities)
# =============================================================================

PUBMED_ESEARCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_OA_API   = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
PUBMED_DELAY    = 0.35   # NCBI allows ~3 req/s without API key


def _resolve_pmc_pdf_url(pmc_id: str) -> str:
    """
    Use the PMC Open Access API to get a direct HTTPS PDF link.
    Falls back to the standard PMC article URL if the paper is not in OA subset.
    """
    try:
        resp = requests.get(PUBMED_OA_API,
                            params={"id": f"PMC{pmc_id}", "format": "pdf"},
                            timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for link in root.findall(".//link"):
            fmt  = link.get("format", "")
            href = link.get("href", "")
            if fmt == "pdf" and href:
                # Convert ftp:// → https://
                return href.replace("ftp://ftp.ncbi.nlm.nih.gov",
                                    "https://ftp.ncbi.nlm.nih.gov")
    except Exception:
        pass
    # Fallback: new PMC domain (avoids the old redirect/HHS page)
    return f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_id}/pdf/"


def _pubmed_fetch_xml(pmids: list) -> bytes:
    """Batch-fetch PubMed XML for a list of PMIDs."""
    params = {
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    try:
        resp = requests.get(PUBMED_EFETCH, params=params, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"  [!] PubMed efetch error: {e}")
        return b""


def _parse_pubmed_xml(xml_bytes: bytes) -> list:
    """Parse PubMed XML into paper dicts."""
    papers = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return papers

    for article in root.findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        if citation is None:
            continue

        pmid_tag = citation.find("PMID")
        pmid = pmid_tag.text.strip() if pmid_tag is not None else ""

        art = citation.find("Article")
        if art is None:
            continue

        # Title
        title_tag = art.find("ArticleTitle")
        title = (title_tag.text or "").strip() if title_tag is not None else ""
        title = re.sub(r"<[^>]+>", "", title)  # strip inline XML tags

        # Abstract (may have multiple AbstractText elements)
        abstract_parts = []
        for at in art.findall(".//AbstractText"):
            label = at.get("Label", "")
            text  = (at.text or "").strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            elif text:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors = []
        for au in art.findall(".//Author"):
            ln = au.findtext("LastName", "")
            fn = au.findtext("ForeName", "")
            col = au.findtext("CollectiveName", "")
            name = f"{fn} {ln}".strip() if (fn or ln) else col
            if name:
                authors.append(name)

        # Publication date
        pub_date = ""
        for date_path in ("Journal/JournalIssue/PubDate",
                          "ArticleDate"):
            d = art.find(date_path)
            if d is not None:
                year  = d.findtext("Year",  "")
                month = d.findtext("Month", "01")
                day   = d.findtext("Day",   "01")
                # Month may be abbreviated (Jan, Feb…)
                month_map = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04",
                             "May":"05","Jun":"06","Jul":"07","Aug":"08",
                             "Sep":"09","Oct":"10","Nov":"11","Dec":"12"}
                month = month_map.get(month, month).zfill(2)
                day   = day.zfill(2)
                if year:
                    pub_date = f"{year}-{month}-{day}"
                    break

        # PMC ID → free full-text PDF
        pmc_id  = ""
        doi_val = ""
        for aid in article.findall(".//ArticleId"):
            if aid.get("IdType") == "pmc":
                pmc_id = (aid.text or "").strip().lstrip("PMC")
            if aid.get("IdType") == "doi":
                doi_val = (aid.text or "").strip()

        if pmc_id:
            pdf_url = _resolve_pmc_pdf_url(pmc_id)
        elif doi_val:
            pdf_url = f"https://doi.org/{doi_val}"
        else:
            pdf_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        if not title:
            continue

        papers.append({
            "arxiv_id":  f"pubmed_{pmid}",
            "title":      title,
            "abstract":   abstract,
            "pdf_url":    pdf_url,
            "authors":    authors,
            "published":  pub_date,
            "source":     "pubmed",
            "pmc_id":     pmc_id,
        })
    return papers


def search_pubmed(query: str, start_date: str, end_date: str,
                  max_papers: Optional[int]) -> list:
    """Search PubMed via NCBI E-utilities (esearch → efetch XML)."""
    cap = max_papers or 500

    # Build date-range term (YYYY/MM/DD format for PubMed)
    sd = start_date.replace("-", "/")
    ed = end_date.replace("-", "/")
    term = f'({query})[Title/Abstract] AND ("{sd}"[Date - Publication] : "{ed}"[Date - Publication])'

    print(f"\n[PubMed] query={query!r}  {start_date} -> {end_date}"
          f"  cap={max_papers or 'unlimited'}")

    # 1. esearch – get all matching PMIDs
    all_pmids: list = []
    retstart  = 0
    batch     = 200

    while len(all_pmids) < cap:
        try:
            resp = requests.get(PUBMED_ESEARCH, params={
                "db":       "pubmed",
                "term":     term,
                "retmax":   min(batch, cap - len(all_pmids)),
                "retstart": retstart,
                "retmode":  "json",
                "sort":     "date",
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()["esearchresult"]
        except Exception as e:
            print(f"  [!] PubMed esearch error: {e}")
            break

        ids = data.get("idlist", [])
        if not ids:
            break
        all_pmids.extend(ids)
        total = int(data.get("count", 0))
        print(f"  fetched {len(all_pmids)}/{min(total, cap)} PMIDs", end="\r")

        if len(all_pmids) >= total or len(ids) < batch:
            break
        retstart += len(ids)
        time.sleep(PUBMED_DELAY)

    if not all_pmids:
        print(f"\n  Found 0 PubMed papers.")
        return []

    # 2. efetch – get full XML in batches of 50
    papers: list = []
    efetch_batch = 50
    for i in range(0, len(all_pmids), efetch_batch):
        chunk = all_pmids[i : i + efetch_batch]
        xml   = _pubmed_fetch_xml(chunk)
        papers.extend(_parse_pubmed_xml(xml))
        time.sleep(PUBMED_DELAY)

    print(f"\n  Found {len(papers)} PubMed papers.")
    return papers


# =============================================================================
# bioRxiv search via Europe PMC (indexes all bioRxiv preprints, keyword-searchable)
# =============================================================================

EUROPEPMC_API   = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_BATCH = 100


def search_biorxiv(query: str, start_date: str, end_date: str,
                   max_papers: Optional[int]) -> list:
    """
    Search bioRxiv preprints via the Europe PMC REST API.
    Europe PMC indexes all bioRxiv papers with full keyword search support.
    No scraping — official JSON API, no authentication required.
    """
    cap = max_papers or 200

    # Europe PMC query: keyword + journal filter + date range
    epmc_query = (f'({query}) AND JOURNAL:"bioRxiv" AND '
                  f'FIRST_PDATE:[{start_date} TO {end_date}]')

    print(f"\n[bioRxiv/EuropePMC] query={query!r}  {start_date} -> {end_date}"
          f"  cap={max_papers or 'unlimited'}")

    papers: list = []
    seen:   set  = set()
    cursor  = "*"

    while len(papers) < cap:
        try:
            resp = requests.get(EUROPEPMC_API, params={
                "query":      epmc_query,
                "format":     "json",
                "resultType": "core",
                "pageSize":   min(EUROPEPMC_BATCH, cap - len(papers)),
                "cursorMark": cursor,
            }, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [!] Europe PMC error: {e}")
            break

        items = data.get("resultList", {}).get("result", [])
        if not items:
            break

        for item in items:
            doi    = (item.get("doi") or "").strip()
            pmcid  = (item.get("pmcid") or "").strip().lstrip("PMC")
            title  = (item.get("title") or "").strip()
            # Deduplicate by DOI if present, else by title hash
            dedup_key = doi or hashlib.md5(title.lower().encode()).hexdigest()
            if not title or dedup_key in seen:
                continue
            seen.add(dedup_key)

            abstract = (item.get("abstractText") or "").strip()
            pub_date = (item.get("firstPublicationDate") or "")[:10]

            # Authors
            authors = []
            for au in (item.get("authorList") or {}).get("author", []):
                name = (au.get("fullName") or
                        f"{au.get('firstName','')} {au.get('lastName','')}".strip())
                if name:
                    authors.append(name)

            # PDF URL priority:
            # 1. Direct bioRxiv URL when DOI is a 10.1101/ preprint
            # 2. PMC OA API direct link when PMC ID is available
            # 3. Any PDF link from fullTextUrlList
            # 4. Fallback to DOI-based URL
            pdf_url = ""
            if doi.startswith("10.1101/"):
                pdf_url = f"https://www.biorxiv.org/content/{doi}.full.pdf"
            elif pmcid:
                pdf_url = _resolve_pmc_pdf_url(pmcid)
            if not pdf_url:
                for link in (item.get("fullTextUrlList") or {}).get("fullTextUrl", []):
                    if link.get("documentStyle") == "pdf":
                        candidate = link.get("url", "")
                        if candidate:
                            pdf_url = candidate
                            break
            if not pdf_url and doi:
                pdf_url = f"https://doi.org/{doi}"

            # Stable ID: bioRxiv DOI > PMC ID > EPMC item ID > MD5
            if doi.startswith("10.1101/"):
                short_id = re.sub(r"[^\w.]", "_", doi.replace("10.1101/", ""))
            elif pmcid:
                short_id = f"PMC{pmcid}"
            elif item.get("id"):
                short_id = re.sub(r"[^\w.]", "_", item["id"])
            else:
                short_id = hashlib.md5(title.lower().encode()).hexdigest()[:12]

            papers.append({
                "arxiv_id":  f"biorxiv_{short_id}",
                "title":      title,
                "abstract":   abstract,
                "pdf_url":    pdf_url,
                "authors":    authors,
                "published":  pub_date,
                "source":     "biorxiv",
                "doi":        doi,
            })
            if len(papers) >= cap:
                break

        total = data.get("hitCount", "?")
        print(f"  cursor={cursor[:8]}…  batch={len(items)}"
              f"  matched={len(papers)}  total={total}", end="\r")

        next_cursor = data.get("nextCursorMark", "")
        if not next_cursor or next_cursor == cursor or len(items) < EUROPEPMC_BATCH:
            break
        cursor = next_cursor
        time.sleep(0.5)

    print(f"\n  Found {len(papers)} bioRxiv papers.")
    return papers


# =============================================================================
# Semantic Scholar search
# =============================================================================

S2_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS     = ("paperId,title,abstract,authors,year,publicationDate,"
                 "externalIds,openAccessPdf,venue")
S2_DELAY      = 1.1   # free tier: ~1 req/s


def search_semantic_scholar(query: str, start_date: str, end_date: str,
                             max_papers: Optional[int]) -> list:
    """Search Semantic Scholar via the public paper-search API."""
    cap      = max_papers or 500
    year_low = int(start_date[:4])
    year_hi  = int(end_date[:4])

    print(f"\n[S2] query={query!r}  {start_date} -> {end_date}"
          f"  cap={max_papers or 'unlimited'}")

    papers: list = []
    seen:   set  = set()
    offset  = 0
    limit   = min(100, cap)

    while len(papers) < cap:
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(S2_SEARCH_API, params={
                    "query":  query,
                    "fields": S2_FIELDS,
                    "limit":  limit,
                    "offset": offset,
                }, timeout=60)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 10)) + 2
                    print(f"\n  [S2] Rate limited, waiting {wait}s ...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.HTTPError:
                print(f"  [!] Semantic Scholar API error: {resp.status_code}")
                data = {}
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"  [!] Semantic Scholar API error: {e}")
                data = {}
                break
        else:
            data = {}

        if not data:
            break

        items = data.get("data", [])
        if not items:
            break

        for item in items:
            pid = item.get("paperId", "")
            if not pid or pid in seen:
                continue

            title = (item.get("title") or "").strip()
            if not title:
                continue

            # Year filter
            year = item.get("year") or 0
            if year and (year < year_low or year > year_hi):
                continue

            seen.add(pid)
            abstract = (item.get("abstract") or "").strip()

            # Date
            pub_date = (item.get("publicationDate") or
                        (f"{year}-01-01" if year else ""))

            # Authors
            authors = [a.get("name", "") for a in (item.get("authors") or [])]

            # PDF: prefer openAccessPdf, then arXiv, then DOI
            oap     = item.get("openAccessPdf") or {}
            pdf_url = oap.get("url", "")
            ext_ids = item.get("externalIds") or {}
            if not pdf_url:
                arxiv_id_s2 = ext_ids.get("ArXiv", "")
                if arxiv_id_s2:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id_s2}.pdf"
            if not pdf_url:
                doi_s2 = ext_ids.get("DOI", "")
                if doi_s2:
                    pdf_url = f"https://doi.org/{doi_s2}"

            # Stable ID: prefer arXiv ID, else S2 ID
            arxiv_s2 = ext_ids.get("ArXiv", "")
            stable_id = arxiv_s2 if arxiv_s2 else f"s2_{pid[:12]}"

            papers.append({
                "arxiv_id":  stable_id,
                "title":      title,
                "abstract":   abstract,
                "pdf_url":    pdf_url,
                "authors":    authors,
                "published":  pub_date[:10] if pub_date else "",
                "source":     "semantic_scholar",
                "venue":      (item.get("venue") or ""),
            })
            if len(papers) >= cap:
                break

        total = data.get("total", "?")
        print(f"  offset={offset}  batch={len(items)}"
              f"  matched={len(papers)}  total={total}", end="\r")

        if len(items) < limit:
            break
        offset += len(items)
        time.sleep(S2_DELAY)

    print(f"\n  Found {len(papers)} Semantic Scholar papers.")
    return papers


# =============================================================================
# Multi-source search + deduplication
# =============================================================================

def _norm_title(t: str) -> str:
    """Lowercase, strip punctuation/spaces for fuzzy dedup."""
    return re.sub(r"[^a-z0-9]", "", t.lower())


VALID_SOURCES = {"arxiv", "scholar", "pubmed", "biorxiv", "semantic_scholar"}
SOURCE_ALIASES = {
    "bio": ["pubmed", "biorxiv", "arxiv"],
    "all": ["arxiv", "scholar", "pubmed", "biorxiv", "semantic_scholar"],
}
SOURCE_FN = {
    "arxiv":            search_arxiv,
    "scholar":          search_scholar,
    "pubmed":           search_pubmed,
    "biorxiv":          search_biorxiv,
    "semantic_scholar": search_semantic_scholar,
}


def _resolve_sources(raw: list) -> list:
    """Expand aliases, deduplicate, preserve order."""
    seen, result = set(), []
    for s in raw:
        for atomic in SOURCE_ALIASES.get(s, [s]):
            if atomic not in seen:
                seen.add(atomic)
                result.append(atomic)
    return result


def search_papers(sources, query: str, start_date: str,
                  end_date: str, max_papers: Optional[int]) -> list:
    """
    Dispatch to one or more sources, then deduplicate by normalized title.
    sources: str or list – any combination of
        arxiv | scholar | pubmed | biorxiv | semantic_scholar | bio | all
    """
    if isinstance(sources, str):
        sources = sources.split()   # also handles single-word strings
    sources = _resolve_sources(sources)

    per_source = max_papers
    combined   = []
    for src in sources:
        fn = SOURCE_FN.get(src)
        if fn is None:
            print(f"  [!] Unknown source '{src}', skipping.")
            continue
        results = fn(query, start_date, end_date, per_source)
        for p in results:
            p.setdefault("source", src)
        combined.extend(results)

    # Deduplicate by normalized title (first occurrence wins)
    seen: set = set()
    deduped   = []
    for p in combined:
        key = _norm_title(p["title"])
        if key and key not in seen:
            seen.add(key)
            deduped.append(p)

    if max_papers:
        deduped = deduped[:max_papers]

    if len(sources) > 1:
        src_counts: dict = {}
        for p in deduped:
            s = p.get("source", "?")
            src_counts[s] = src_counts.get(s, 0) + 1
        print(f"\n[Merge] {len(deduped)} unique papers after dedup  {src_counts}")

    return deduped

# =============================================================================
# PDF download
# =============================================================================

def _is_valid_pdf(path: Path) -> bool:
    """Check that the file starts with the PDF magic bytes %PDF."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


def _download_one(paper: dict, dest: Path, attempt: int = 0) -> Optional[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-.]", "_", paper["arxiv_id"])
    out  = dest / f"{safe}.pdf"
    if out.exists() and out.stat().st_size > 1024 and _is_valid_pdf(out):
        return out
    elif out.exists():
        out.unlink()   # delete stale/corrupt cached file
    try:
        req = urllib.request.Request(
            paper["pdf_url"],
            headers={"User-Agent": "Mozilla/5.0 arxiv-downloader/1.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp, open(out, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
        if not _is_valid_pdf(out):
            raise ValueError("response is not a PDF (got HTML or redirect page)")
        return out
    except Exception as e:
        if out.exists():
            out.unlink()
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
            return _download_one(paper, dest, attempt + 1)
        print(f"\n  [!] Download failed {paper['arxiv_id']}: {e}")
        return None


def download_all(papers: list, dest: Path) -> dict:
    print(f"\n[Download] {len(papers)} PDFs -> '{dest}/'")
    result = {}
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futures = {ex.submit(_download_one, p, dest): p for p in papers}
        done = 0
        for fut in as_completed(futures):
            p    = futures[fut]
            path = fut.result()
            done += 1
            print(f"  [{done}/{len(papers)}] {p['arxiv_id']}  {'OK' if path else 'FAIL'}",
                  end="\r")
            if path:
                result[p["arxiv_id"]] = path
    print(f"\n  Downloaded {len(result)}/{len(papers)}.")
    return result

# =============================================================================
# LLM call
# =============================================================================

def call_llm(prompt: str, model: str = DEFAULT_MODEL,
             temperature: float = 0.3, max_tokens: int = 4000) -> str:
    if not LLM_API_KEY:
        return "[ERROR] LLM_API_KEY is empty. Check api_key.txt."
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "messages":    [{"role": "user", "content": prompt}],
    }
    try:
        resp = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError:
        return f"[ERROR] HTTP {resp.status_code}: {resp.text[:300]}"
    except Exception as e:
        return f"[ERROR] LLM call failed: {e}"

# =============================================================================
# Prompts (bilingual)
# =============================================================================

DEEP_ANALYSIS_EN = """\
You are an expert research analyst. Below is the full text of an academic paper.
Provide a PROFOUND and STRUCTURED analysis covering ALL sections below:

## 1. Core Contribution & Novelty
- The single most important new idea or result.
- How it differs from prior work.

## 2. Problem Formulation
- The precise problem addressed.
- Why it is important and hard.

## 3. Methodology Deep-Dive
- Technical approach in detail (pipeline, architecture, algorithm steps).
- Key design choices and their justifications.
- Noteworthy mathematical formulations or model structures.

## 4. Experimental Evaluation
- Datasets / benchmarks used.
- Key quantitative results (cite actual numbers).
- Ablation or sensitivity analysis findings.

## 5. Strengths & Limitations
- What the paper does particularly well.
- Weaknesses, open questions, or threats to validity.

## 6. Connections to the Broader Field
- Relation to landmark prior work.
- Research directions this work opens.

## 7. Practical Implications
- Industry / deployment applicability.
- Difficulty to reproduce or productionize.

## 8. One-Sentence Verdict
- Summarize the paper's value in one crisp sentence.

---
PAPER TEXT (may be truncated):
{paper_text}
---

Be specific, use technical language, and reference actual content from the paper.
"""

DEEP_ANALYSIS_ZH = """\
你是一位资深科研分析专家。以下是一篇学术论文的全文（可能有截断）。
请用中文对该论文进行深度、结构化分析，必须覆盖以下全部章节：

## 一、核心贡献与创新点
- 本文最重要的新思想或新结果。
- 与已有工作相比，创新点体现在哪里。

## 二、问题定义
- 论文精确解决的是什么问题。
- 该问题为何重要、为何困难。

## 三、方法详解
- 详细描述技术方案（流程、模型架构、算法步骤）。
- 关键设计选择及其背后的理由。
- 值得关注的数学公式、算法或网络结构。

## 四、实验评估
- 使用了哪些数据集 / 基准测试。
- 核心定量结果（请引用具体数字）。
- 消融实验或敏感性分析的结论。

## 五、优势与局限
- 论文做得特别好的地方。
- 存在的不足、开放问题或有效性威胁。

## 六、与领域的关联
- 与哪些重要前期工作相关或有所延伸。
- 开辟了哪些新研究方向。

## 七、实际应用价值
- 能否应用于工业界 / 实际系统，如何落地。
- 复现或工程化部署的难度。

## 八、一句话总评
- 用一句话精炼概括本文的价值。

---
论文全文（可能被截断）：
{paper_text}
---

请使用专业技术语言，结合论文实际内容作答，不要泛泛而谈。
"""

SYNTHESIS_EN = """\
You are an expert research analyst who has just read {n} papers on the topic: "{topic}".
Below are the individual deep analyses of each paper.

Write a COMPREHENSIVE SYNTHESIS covering:

## 1. Landscape Overview
- Current state of the field; dominant paradigms and approaches.

## 2. Common Themes & Patterns
- Methodological ideas that recur; shared assumptions.

## 3. Key Disagreements & Debates
- Where papers disagree on approach, evaluation, or interpretation.

## 4. Evolution of Ideas
- How approaches evolved from the earliest to most recent papers.

## 5. Critical Open Problems
- Most important unsolved problems; evaluation or methodology gaps.

## 6. Recommended Reading Order
- For a newcomer: which papers to read first and why.

## 7. Forward-Looking Assessment
- The most promising direction going forward.
- What you would investigate next and why.

---
INDIVIDUAL ANALYSES:
{analyses}
---

Be specific, reference paper titles, and provide a graduate-level synthesis.
"""

SYNTHESIS_ZH = """\
你是一位资深科研分析专家，刚刚精读了 {n} 篇关于"{topic}"的学术论文。
以下是各篇论文的逐篇深度分析。

请用中文撰写一份综合研究报告，涵盖以下全部章节：

## 一、领域全景
- 该领域目前处于什么发展阶段，主流范式与技术路线是什么。

## 二、共同主题与规律
- 哪些方法论思路反复出现，论文之间共享哪些假设或设计原则。

## 三、主要分歧与争议
- 在方法、评估或结论上，论文之间存在哪些分歧。

## 四、思想演进脉络
- 从最早到最近的论文，研究方法与关注点如何演变。

## 五、关键开放问题
- 目前最重要的未解决问题，评估方法或研究方法上的空白。

## 六、推荐阅读顺序
- 对初学者而言，应按什么顺序阅读这些论文，请说明理由。

## 七、前瞻性评估
- 最值得深入探索的方向，以及你会聚焦的下一个研究问题。

---
各篇论文分析如下：
{analyses}
---

请引用具体论文标题，提供研究生水平的深度综合分析。
"""


def get_prompts(lang: str) -> tuple:
    if lang == "zh":
        return DEEP_ANALYSIS_ZH, SYNTHESIS_ZH
    return DEEP_ANALYSIS_EN, SYNTHESIS_EN

# =============================================================================
# Reference extraction
# =============================================================================

# Matches bare arXiv IDs like 2301.12345 or 2301.12345v2, with optional
# "arXiv:" or "arxiv.org/abs/" prefix.
_ARXIV_ID_RE = re.compile(
    r'(?:arxiv\s*[:/]\s*|arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)',
    re.IGNORECASE
)


def extract_arxiv_ids_from_text(text: str) -> list:
    """Return unique arXiv IDs (version-stripped) found in *text*."""
    seen, ids = set(), []
    for m in _ARXIV_ID_RE.finditer(text):
        base = re.sub(r'v\d+$', '', m.group(1))
        if base not in seen:
            seen.add(base)
            ids.append(base)
    return ids


def fetch_papers_by_ids(arxiv_ids: list) -> list:
    """Fetch paper metadata for a list of arXiv IDs via the arXiv API."""
    if not arxiv_ids:
        return []
    papers = []
    ns = "{http://www.w3.org/2005/Atom}"
    batch_size = 50
    for i in range(0, len(arxiv_ids), batch_size):
        batch = arxiv_ids[i : i + batch_size]
        id_list = ",".join(batch)
        url = f"{ARXIV_API}?id_list={urllib.parse.quote(id_list)}&max_results={len(batch)}"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                raw = resp.read()
        except Exception as e:
            print(f"  [!] Failed to fetch ID batch: {e}")
            continue
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            continue
        for entry in root.findall(f"{ns}entry"):
            id_tag = entry.find(f"{ns}id")
            if id_tag is None:
                continue
            arxiv_id = re.sub(r'v\d+$', '', id_tag.text.strip().split("/")[-1])

            def _t(tag):
                el = entry.find(f"{ns}{tag}")
                return el.text.strip() if el is not None and el.text else ""

            title   = _t("title").replace("\n", " ")
            summary = _t("summary").replace("\n", " ")
            pub     = _t("published")
            authors = [a.find(f"{ns}name").text.strip()
                       for a in entry.findall(f"{ns}author")
                       if a.find(f"{ns}name") is not None]
            pdf_url = next(
                (lnk.attrib["href"] for lnk in entry.findall(f"{ns}link")
                 if lnk.attrib.get("title") == "pdf"),
                f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            )
            if title:
                papers.append({
                    "arxiv_id":  arxiv_id,
                    "title":     title,
                    "abstract":  summary,
                    "pdf_url":   pdf_url,
                    "authors":   authors,
                    "published": pub[:10],
                    "source":    "arxiv",
                })
        time.sleep(REQUEST_DELAY)
    return papers


def crawl_references(pdf_map: dict, known_ids: set,
                     papers_dir: Path, depth: int) -> tuple:
    """
    Recursively follow arXiv references up to *depth* hops.
    Returns (new_papers, updated_pdf_map).
    """
    if depth <= 0:
        return [], {}

    print(f"\n[Refs] Scanning {len(pdf_map)} PDFs for arXiv references (depth={depth}) ...")
    ref_ids = set()
    for arxiv_id, pdf_path in pdf_map.items():
        # Use a larger char limit to reach reference sections at end of papers
        text = extract_text_from_pdf(pdf_path, char_limit=80000)
        for rid in extract_arxiv_ids_from_text(text):
            if rid not in known_ids:
                ref_ids.add(rid)

    if not ref_ids:
        print("  No new referenced arXiv IDs found.")
        return [], {}

    print(f"  Found {len(ref_ids)} new referenced arXiv IDs — fetching metadata ...")
    ref_papers = fetch_papers_by_ids(list(ref_ids))
    # Deduplicate against known set
    ref_papers = [p for p in ref_papers if p["arxiv_id"] not in known_ids]
    print(f"  Downloading {len(ref_papers)} referenced papers ...")
    new_pdf_map = download_all(ref_papers, papers_dir)

    # Update known set for next hop
    updated_known = known_ids | {p["arxiv_id"] for p in ref_papers}
    # Recurse
    deeper_papers, deeper_map = crawl_references(
        new_pdf_map, updated_known, papers_dir, depth - 1
    )
    all_papers  = ref_papers  + deeper_papers
    all_pdf_map = {**new_pdf_map, **deeper_map}
    return all_papers, all_pdf_map



def analyze_paper(paper: dict, pdf_path: Optional[Path],
                  model: str, lang: str) -> Optional[dict]:
    print(f"  Analyzing: {paper['title'][:72]}...")
    deep_prompt, _ = get_prompts(lang)

    if not (pdf_path and PDF_BACKEND):
        print(f"    [Skip] No PDF available for '{paper['arxiv_id']}'")
        return None

    text = extract_text_from_pdf(pdf_path)
    if not text.strip() or text.startswith("[PDF"):
        print(f"    [Skip] Full-text extraction failed for '{paper['arxiv_id']}'")
        return None

    analysis = call_llm(deep_prompt.format(paper_text=text), model=model, max_tokens=4000)

    return {
        "arxiv_id":     paper["arxiv_id"],
        "title":        paper["title"],
        "authors":      paper["authors"],
        "published":    paper["published"],
        "pdf_url":      paper.get("pdf_url", ""),
        "abstract":     paper["abstract"],
        "paper_source": paper.get("source", "arxiv"),
        "source":       "full PDF",
        "analysis":     analysis,
    }

# =============================================================================
# Output writers
# =============================================================================

def _safe(s: str) -> str:
    return re.sub(r"[^\w\-.]", "_", s)


def write_paper_md(result: dict, out_dir: Path, lang: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_safe(result['arxiv_id'])}.md"
    authors_str = ", ".join(result["authors"][:5])
    if len(result["authors"]) > 5:
        authors_str += " et al."

    paper_source = result.get("paper_source", "arxiv")
    pid          = result["arxiv_id"]
    if paper_source == "arxiv":
        id_line = f"- **arXiv**: [{pid}](https://arxiv.org/abs/{pid})\n"
    else:
        id_line = f"- **ID**: {pid}\n- **Source**: Google Scholar\n"

    pdf_line = f"- **PDF**: {result['pdf_url']}\n" if result.get("pdf_url") else ""

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {result['title']}\n\n")
        f.write(id_line)
        f.write(f"- **Authors**: {authors_str}\n")
        f.write(f"- **Published**: {result['published']}\n")
        f.write(pdf_line)
        f.write(f"- **Text source**: {result['source']}\n\n")
        f.write("## Abstract\n\n" + result["abstract"] + "\n\n")
        f.write("## Deep Analysis\n\n" + result["analysis"] + "\n")
    return path


def write_synthesis_md(topic: str, results: list, synthesis: str,
                       out_dir: Path, lang: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"synthesis_{ts}.md"
    with open(path, "w", encoding="utf-8") as f:
        if lang == "zh":
            f.write(f"# Synthesis Report: {topic}\n\n")
        else:
            f.write(f"# Synthesis Report: {topic}\n\n")
        f.write(f"*Generated: {datetime.now():%Y-%m-%d %H:%M:%S}"
                f"  |  Papers analyzed: {len(results)}*\n\n---\n\n")
        f.write("## Paper List\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"{i}. [{r['title']}](https://arxiv.org/abs/{r['arxiv_id']})"
                    f" ({r['published']})\n")
        f.write("\n---\n\n## Analysis\n\n" + synthesis + "\n")
    return path


OVERVIEW_PROMPT_EN = """\
You are an expert research analyst. Write a concise but informative overview of the research topic: "{topic}".
Cover:
1. **Historical Background** - when and how this field emerged, key milestones.
2. **Current State** - dominant methods, architectures, and paradigms as of today.
3. **Key Baselines & Benchmarks** - the most important baseline models/methods and standard evaluation benchmarks.
4. **Major Research Groups & Venues** - leading labs, conferences, and journals.
5. **Open Challenges** - the most pressing unsolved problems.

Be concise (800-1200 words), factual, and use technical language suitable for a graduate researcher.
"""

OVERVIEW_PROMPT_ZH = """\
你是一位资深科研分析专家。请为以下研究主题撰写一份简洁但信息量丰富的综述引言："{topic}"。
请覆盖：
1. **历史背景** — 该领域的起源、发展历程和关键里程碑。
2. **现状** — 当前主流方法、模型架构和技术范式。
3. **核心基准与Baseline** — 最重要的基线方法和标准评测基准。
4. **主要研究团队与发表渠道** — 领域内顶尖实验室、重要会议和期刊。
5. **开放挑战** — 目前最迫切的未解决问题。

字数控制在800-1200字，语言专业，适合研究生阅读。
"""


def write_index_md(topic: str, results: list, out_dir: Path,
                   model: str, lang: str) -> Path:
    path = out_dir / "index.md"

    print(f"\n[Index] Generating topic overview for '{topic}' ...")
    prompt   = (OVERVIEW_PROMPT_ZH if lang == "zh" else OVERVIEW_PROMPT_EN).format(topic=topic)
    overview = call_llm(prompt, model=model, max_tokens=4000)

    src_label = {"arxiv": "arXiv", "scholar": "Google Scholar"}
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n")
        f.write(f"*{len(results)} papers | Generated: {datetime.now():%Y-%m-%d}*\n\n")
        f.write("---\n\n")
        f.write("## Topic Overview\n\n")
        f.write(overview + "\n\n")
        f.write("---\n\n")
        f.write("## Paper List\n\n")
        for r in results:
            src  = src_label.get(r.get("paper_source", "arxiv"), r.get("paper_source", ""))
            date = r["published"] or "n/d"
            f.write(f"- **[{r['title']}]({_safe(r['arxiv_id'])}.md)**  \n")
            f.write(f"  {date} | {src} | {r['arxiv_id']}\n\n")
    return path

# =============================================================================
# PDF rendering via pandoc
# =============================================================================

def render_to_pdf(md_files: list, out_dir: Path, lang: str):
    """Convert all markdown files to PDF using pandoc."""
    if not shutil.which("pandoc"):
        print("\n[PDF] pandoc not found. Install from https://pandoc.org/installing.html")
        print("      Skipping PDF rendering.")
        return

    # Choose a CJK-compatible PDF engine when language is Chinese
    if lang == "zh":
        extra = ["--pdf-engine=xelatex",
                 "-V", "CJKmainfont=SimSun",
                 "-V", "geometry:margin=2.5cm"]
    else:
        extra = ["--pdf-engine=xelatex",
                 "-V", "geometry:margin=2.5cm"]

    base_cmd = ["pandoc", "--standalone",
                "-V", "fontsize=11pt",
                "-V", "colorlinks=true"] + extra

    pdf_dir = out_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for md in md_files:
        out_pdf = pdf_dir / (md.stem + ".pdf")
        cmd = base_cmd + [str(md), "-o", str(out_pdf)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"\n  [!] pandoc failed for {md.name}: {e.stderr.decode()[:200]}")
            fail += 1

    print(f"\n[PDF] Rendered {ok} files to '{pdf_dir}/'  ({fail} failed)")

# =============================================================================
# CLI + main
# =============================================================================

def parse_args():
    today = datetime.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(
        description="arXiv Topic Deep Analyzer — search, download, analyze with gemini-2.5-pro",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--topic",  "-t", required=True,
                   help="Search topic / keywords, e.g. 'diffusion model image generation'")
    one_year_ago = (datetime.now().replace(year=datetime.now().year - 1)).strftime("%Y-%m-%d")
    p.add_argument("--start",  "-s", default=one_year_ago,
                   help="Start date YYYY-MM-DD (default: one year before today)")
    p.add_argument("--end",    "-e", default=today,
                   help="End date YYYY-MM-DD")
    p.add_argument("--max",    "-n", type=int, default=DEFAULT_MAX,
                   help="Max papers to analyze (0 = unlimited)")
    p.add_argument("--model",  "-m", default=DEFAULT_MODEL,
                   help="LLM model name")
    p.add_argument("--lang",   "-l", default="en", choices=["en", "zh"],
                   help="Report language: en=English, zh=Chinese")
    p.add_argument("--pdf",    action="store_true",
                   help="Render all Markdown output files to PDF via pandoc")
    p.add_argument("--source", "-S", nargs="+", default=["arxiv"],
                   metavar="SRC",
                   help=("One or more paper sources (space-separated): "
                         "arxiv scholar pubmed biorxiv semantic_scholar "
                         "bio(=pubmed+biorxiv+arxiv) all"))
    p.add_argument("--refs",      action="store_true",
                   help="Crawl arXiv references from downloaded PDFs and include them")
    p.add_argument("--ref-depth", type=int, default=1,
                   help="How many hops of reference crawling (default: 1, requires --refs)")
    p.add_argument("--no-download",  action="store_true",
                   help="Skip PDF download; analyze abstract text only")
    p.add_argument("--no-synthesis", action="store_true",
                   help="Skip the cross-paper synthesis step")
    return p.parse_args()


def main():
    args       = parse_args()
    topic      = args.topic
    start_date = args.start
    end_date   = args.end
    max_papers = None if args.max == 0 else args.max
    model      = args.model
    lang       = args.lang

    # Validate sources
    all_known = VALID_SOURCES | set(SOURCE_ALIASES)
    bad = [s for s in args.source if s not in all_known]
    if bad:
        print(f"[ERROR] Unknown source(s): {bad}")
        print(f"        Valid: {sorted(all_known)}")
        sys.exit(1)
    sources_display = " ".join(args.source)

    print("=" * 65)
    print("   arXiv Topic Deep Analyzer  --  " + model)
    print("=" * 65)
    print(f"  Topic    : {topic}")
    print(f"  Source   : {sources_display}")
    print(f"  Dates    : {start_date} -> {end_date}")
    print(f"  Cap      : {max_papers or 'unlimited'}")
    print(f"  Model    : {model}")
    print(f"  Language : {'Chinese' if lang == 'zh' else 'English'}")
    print(f"  Render   : {'PDF (pandoc)' if args.pdf else 'Markdown only'}")
    print(f"  API URL  : {LLM_API_URL}")
    print("=" * 65)

    safe_topic  = _safe(topic.lower().replace(" ", "_"))[:40]
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_DIR / f"{safe_topic}_{ts}"
    papers_dir  = PAPERS_DIR / safe_topic

    # 1. Search
    papers = search_papers(args.source, topic, start_date, end_date, max_papers)
    if not papers:
        print("No papers found. Try different keywords or a wider date range.")
        sys.exit(0)

    # 2. Download
    pdf_map = {}
    if not args.no_download:
        pdf_map = download_all(papers, papers_dir)
    else:
        print("\n[Download] Skipped (--no-download).")

    # 2b. Reference crawling
    if args.refs and not args.no_download:
        known_ids   = {p["arxiv_id"] for p in papers}
        ref_papers, ref_pdf_map = crawl_references(
            pdf_map, known_ids, papers_dir, args.ref_depth
        )
        if ref_papers:
            print(f"  Adding {len(ref_papers)} referenced papers to analysis set.")
            papers.extend(ref_papers)
            pdf_map.update(ref_pdf_map)
    elif args.refs and args.no_download:
        print("\n[Refs] Skipped (--no-download disables reference crawling).")

    # 3. Per-paper analysis
    print(f"\n[Analysis] Analyzing {len(papers)} papers with '{model}' ...")
    md_files = []
    results  = []
    for i, paper in enumerate(papers, 1):
        print(f"\n  [{i}/{len(papers)}] ", end="")
        # Skip papers that failed to download (only when download is active)
        if not args.no_download and paper["arxiv_id"] not in pdf_map:
            print(f"  [Skip] Download failed — {paper['title'][:60]}")
            continue
        result = analyze_paper(paper, pdf_map.get(paper["arxiv_id"]), model, lang)
        if result is None:
            continue
        md     = write_paper_md(result, session_dir, lang)
        md_files.append(md)
        results.append(result)

    # 4. Synthesis
    syn_path = None
    if not args.no_synthesis:
        _, synth_prompt = get_prompts(lang)
        print(f"\n[Synthesis] Building cross-paper synthesis ({len(results)} papers) ...")
        combined = "\n\n---\n\n".join(
            f"### Paper {i}: {r['title']} ({r['published']})\n{r['analysis']}"
            for i, r in enumerate(results, 1)
        )[:60000]
        synthesis = call_llm(
            synth_prompt.format(n=len(results), topic=topic, analyses=combined),
            model=model, max_tokens=5000,
        )
        syn_path = write_synthesis_md(topic, results, synthesis, session_dir, lang)
        md_files.append(syn_path)

    idx_path = write_index_md(topic, results, session_dir, model, lang)
    md_files.append(idx_path)

    # 5. PDF rendering
    if args.pdf:
        render_to_pdf(md_files, session_dir, lang)

    print("\n" + "=" * 65)
    print("  Done!")
    print(f"  Papers   : {len(results)}")
    print(f"  Output   : {session_dir}/")
    print(f"  Index    : {idx_path}")
    if syn_path:
        print(f"  Synthesis: {syn_path}")
    if args.pdf:
        print(f"  PDFs     : {session_dir}/pdf/")
    print("=" * 65)


if __name__ == "__main__":
    main()
