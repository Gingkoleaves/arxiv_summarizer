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
        import fitz  # noqa: F401
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
            for page in doc:
                t = page.get_text()
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
# PDF download
# =============================================================================

def _download_one(paper: dict, dest: Path, attempt: int = 0) -> Optional[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-.]", "_", paper["arxiv_id"])
    out  = dest / f"{safe}.pdf"
    if out.exists() and out.stat().st_size > 1024:
        return out
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
        if out.stat().st_size < 1024:
            raise ValueError("file too small")
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
             temperature: float = 0.3, max_tokens: int = 2500) -> str:
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
# Analysis
# =============================================================================

def analyze_paper(paper: dict, pdf_path: Optional[Path],
                  model: str, lang: str) -> dict:
    print(f"  Analyzing: {paper['title'][:72]}...")
    deep_prompt, _ = get_prompts(lang)

    if pdf_path and PDF_BACKEND:
        text   = extract_text_from_pdf(pdf_path)
        source = "full PDF"
    else:
        text   = f"Title: {paper['title']}\n\nAbstract:\n{paper['abstract']}"
        source = "abstract only"

    if not text.strip() or text.startswith("[PDF"):
        text   = f"Title: {paper['title']}\n\nAbstract:\n{paper['abstract']}"
        source = "abstract only (PDF parse failed)"

    analysis = call_llm(deep_prompt.format(paper_text=text), model=model)

    return {
        "arxiv_id":  paper["arxiv_id"],
        "title":     paper["title"],
        "authors":   paper["authors"],
        "published": paper["published"],
        "pdf_url":   paper["pdf_url"],
        "abstract":  paper["abstract"],
        "source":    source,
        "analysis":  analysis,
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

    if lang == "zh":
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {result['title']}\n\n")
            f.write(f"- **arXiv**: [{result['arxiv_id']}](https://arxiv.org/abs/{result['arxiv_id']})\n")
            f.write(f"- **Authors**: {authors_str}\n")
            f.write(f"- **Published**: {result['published']}\n")
            f.write(f"- **PDF**: {result['pdf_url']}\n")
            f.write(f"- **Analysis source**: {result['source']}\n\n")
            f.write("## Abstract\n\n" + result["abstract"] + "\n\n")
            f.write("## Deep Analysis\n\n" + result["analysis"] + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {result['title']}\n\n")
            f.write(f"- **arXiv**: [{result['arxiv_id']}](https://arxiv.org/abs/{result['arxiv_id']})\n")
            f.write(f"- **Authors**: {authors_str}\n")
            f.write(f"- **Published**: {result['published']}\n")
            f.write(f"- **PDF**: {result['pdf_url']}\n")
            f.write(f"- **Analysis source**: {result['source']}\n\n")
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


def write_index_md(topic: str, results: list, out_dir: Path) -> Path:
    path = out_dir / "index.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Paper Index: {topic}\n\n")
        f.write(f"*{len(results)} papers | Generated: {datetime.now():%Y-%m-%d}*\n\n")
        for r in results:
            f.write(f"- **[{r['title']}]({_safe(r['arxiv_id'])}.md)**  \n")
            f.write(f"  {r['published']} | {r['arxiv_id']}\n\n")
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

    print("=" * 65)
    print("   arXiv Topic Deep Analyzer  --  " + model)
    print("=" * 65)
    print(f"  Topic    : {topic}")
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
    papers = search_arxiv(topic, start_date, end_date, max_papers)
    if not papers:
        print("No papers found. Try different keywords or a wider date range.")
        sys.exit(0)

    # 2. Download
    pdf_map = {}
    if not args.no_download:
        pdf_map = download_all(papers, papers_dir)
    else:
        print("\n[Download] Skipped (--no-download).")

    # 3. Per-paper analysis
    print(f"\n[Analysis] Analyzing {len(papers)} papers with '{model}' ...")
    md_files = []
    results  = []
    for i, paper in enumerate(papers, 1):
        print(f"\n  [{i}/{len(papers)}] ", end="")
        result = analyze_paper(paper, pdf_map.get(paper["arxiv_id"]), model, lang)
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
        )[:30000]
        synthesis = call_llm(
            synth_prompt.format(n=len(results), topic=topic, analyses=combined),
            model=model, max_tokens=3000,
        )
        syn_path = write_synthesis_md(topic, results, synthesis, session_dir, lang)
        md_files.append(syn_path)

    idx_path = write_index_md(topic, results, session_dir)
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
