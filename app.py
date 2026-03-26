from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import requests
import re

app = FastAPI(title="Zotero Bridge v3", version="3.0.0")

ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")  # optional, improves PubMed throughput
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "researcher@example.com")

# ---------- Helpers ----------

def require_zotero_credentials():
    if not ZOTERO_USER_ID or not ZOTERO_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Zotero credentials")

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()

def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def extract_abstract_from_openalex(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    if not inverted_index:
        return ""
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions.keys()))

def normalize_doi(raw: str) -> str:
    raw = (raw or "").strip()
    return raw.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "").strip()

def split_author_name(display_name: str) -> Dict[str, str]:
    parts = (display_name or "").strip().split()
    if not parts:
        return {"firstName": "", "lastName": ""}
    if len(parts) == 1:
        return {"firstName": "", "lastName": parts[0]}
    return {"firstName": " ".join(parts[:-1]), "lastName": parts[-1]}

def openalex_to_internal(item: Dict[str, Any]) -> Dict[str, Any]:
    authors = []
    for a in item.get("authorships", []) or []:
        display_name = safe_get(a, "author", "display_name", default="") or ""
        if display_name:
            authors.append(split_author_name(display_name))
    abstract = extract_abstract_from_openalex(item.get("abstract_inverted_index"))
    concepts = item.get("concepts", []) or []
    concept_tags = [c.get("display_name", "") for c in concepts[:8] if c.get("display_name")]
    return {
        "title": item.get("title", ""),
        "abstract": abstract,
        "authors": authors,
        "journal": safe_get(item, "primary_location", "source", "display_name", default="") or "",
        "year": str(item.get("publication_year", "") or ""),
        "doi": normalize_doi(item.get("doi", "") or ""),
        "url": item.get("id", "") or "",
        "tags": concept_tags,
        "type": item.get("type", ""),
        "cited_by_count": item.get("cited_by_count", 0) or 0,
        "source": "OpenAlex"
    }

def crossref_to_internal(item: Dict[str, Any]) -> Dict[str, Any]:
    authors = []
    for a in item.get("author", []) or []:
        authors.append({
            "firstName": a.get("given", "") or "",
            "lastName": a.get("family", "") or ""
        })
    title = " ".join(item.get("title", []) or [])
    journal = " ".join(item.get("container-title", []) or [])
    year = ""
    issued = safe_get(item, "issued", "date-parts", default=[])
    if issued and isinstance(issued, list) and issued[0]:
        year = str(issued[0][0])
    abstract = item.get("abstract", "") or ""
    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "year": year,
        "doi": normalize_doi(item.get("DOI", "") or ""),
        "url": item.get("URL", "") or "",
        "tags": [],
        "type": item.get("type", "") or "",
        "cited_by_count": item.get("is-referenced-by-count", 0) or 0,
        "source": "Crossref"
    }

def pubmed_summary_to_internal(summary: Dict[str, Any]) -> Dict[str, Any]:
    authors = []
    for a in summary.get("authors", []) or []:
        name = a.get("name", "") or ""
        if name:
            authors.append(split_author_name(name))
    article_ids = summary.get("articleids", []) or []
    doi = ""
    for aid in article_ids:
        if aid.get("idtype") == "doi":
            doi = aid.get("value", "")
            break
    return {
        "title": summary.get("title", "") or "",
        "abstract": "",
        "authors": authors,
        "journal": summary.get("fulljournalname", "") or summary.get("source", "") or "",
        "year": str(summary.get("pubdate", "")[:4]) if summary.get("pubdate") else "",
        "doi": normalize_doi(doi),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{summary.get('uid', '')}/" if summary.get("uid") else "",
        "tags": ["PubMed"],
        "type": summary.get("pubtype", [""])[0] if summary.get("pubtype") else "",
        "cited_by_count": 0,
        "source": "PubMed",
        "pmid": str(summary.get("uid", "") or "")
    }

def dedupe_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for p in papers:
        key = normalize_doi(p.get("doi", "")) or normalize_text(p.get("title", ""))
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

def build_zotero_item(paper: Dict[str, Any], extra_tags: Optional[List[str]] = None) -> Dict[str, Any]:
    creators = []
    for a in paper.get("authors", []) or []:
        creators.append({
            "creatorType": "author",
            "firstName": a.get("firstName", ""),
            "lastName": a.get("lastName", "")
        })
    tags = [{"tag": t} for t in (paper.get("tags", []) or []) if t]
    for t in (extra_tags or []):
        if t:
            tags.append({"tag": t})
    return {
        "itemType": "journalArticle",
        "title": paper.get("title", ""),
        "creators": creators,
        "abstractNote": paper.get("abstract", "") or "",
        "publicationTitle": paper.get("journal", "") or "",
        "date": paper.get("year", "") or "",
        "DOI": paper.get("doi", "") or "",
        "url": paper.get("url", "") or "",
        "tags": tags
    }

def zotero_headers() -> Dict[str, str]:
    return {
        "Zotero-API-Key": ZOTERO_API_KEY,
        "Content-Type": "application/json"
    }

def zotero_item_exists_by_title(title: str) -> bool:
    require_zotero_credentials()
    q = requests.utils.quote(title)
    url = f"https://api.zotero.org/users/{ZOTERO_USER_ID}/items?q={q}&qmode=title"
    r = requests.get(url, headers={"Zotero-API-Key": ZOTERO_API_KEY}, timeout=60)
    if r.status_code != 200:
        return False
    try:
        data = r.json()
    except Exception:
        return False
    target = normalize_text(title)
    for item in data:
        existing = normalize_text(safe_get(item, "data", "title", default="") or "")
        if existing == target:
            return True
    return False

# ---------- Models ----------

class Author(BaseModel):
    firstName: str
    lastName: str

class ImportRequest(BaseModel):
    title: str
    abstract: Optional[str] = ""
    authors: List[Author] = []
    journal: Optional[str] = ""
    year: Optional[str] = ""
    doi: Optional[str] = ""
    url: Optional[str] = ""
    tags: List[str] = []

class SearchRequest(BaseModel):
    topic: str = Field(..., description="Research topic in English")
    years: int = Field(5, ge=1, le=20)
    max_results_per_source: int = Field(10, ge=1, le=30)
    english_only: bool = True
    article_types: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=lambda: ["openalex", "pubmed", "crossref"])

class ScoreWeights(BaseModel):
    relevance: int = 35
    method_match: int = 20
    system_match: int = 15
    quality: int = 15
    recency: int = 15

class ScoreRequest(BaseModel):
    topic: str
    papers: List[Dict[str, Any]]
    threshold: int = 85
    weights: ScoreWeights = ScoreWeights()

class SearchScoreImportRequest(BaseModel):
    topic: str
    years: int = 5
    max_results_per_source: int = 10
    english_only: bool = True
    article_types: List[str] = []
    sources: List[str] = Field(default_factory=lambda: ["openalex", "pubmed", "crossref"])
    threshold: int = 85
    import_tags: List[str] = []
    dry_run: bool = False

# ---------- Search functions ----------

def search_openalex(topic: str, years: int, max_results: int, english_only: bool, article_types: List[str]) -> List[Dict[str, Any]]:
    current_year = 2026
    from_year = current_year - years + 1
    params = {
        "search": topic,
        "filter": f"from_publication_date:{from_year}-01-01",
        "per-page": max_results,
        "sort": "relevance_score:desc"
    }
    r = requests.get("https://api.openalex.org/works", params=params, timeout=60)
    if r.status_code != 200:
        return []
    results = r.json().get("results", []) or []
    papers = []
    for item in results:
        if english_only and item.get("language") not in [None, "en"]:
            continue
        mapped = openalex_to_internal(item)
        if article_types:
            typ = (mapped.get("type") or "").lower()
            if typ not in [t.lower() for t in article_types]:
                continue
        papers.append(mapped)
    return papers

def search_pubmed(topic: str, years: int, max_results: int) -> List[Dict[str, Any]]:
    current_year = 2026
    from_year = current_year - years + 1
    term = f"({topic}) AND ({from_year}:3000[pdat])"
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance",
        "tool": "zotero_bridge_v3",
        "email": CONTACT_EMAIL
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params, timeout=60)
    if r.status_code != 200:
        return []
    idlist = safe_get(r.json(), "esearchresult", "idlist", default=[]) or []
    if not idlist:
        return []

    sum_params = {
        "db": "pubmed",
        "id": ",".join(idlist),
        "retmode": "json",
        "tool": "zotero_bridge_v3",
        "email": CONTACT_EMAIL
    }
    if NCBI_API_KEY:
        sum_params["api_key"] = NCBI_API_KEY

    s = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params=sum_params, timeout=60)
    if s.status_code != 200:
        return []

    data = s.json().get("result", {}) or {}
    out = []
    for pmid in idlist:
        if pmid in data:
            out.append(pubmed_summary_to_internal(data[pmid]))
    return out

def search_crossref(topic: str, years: int, max_results: int, article_types: List[str]) -> List[Dict[str, Any]]:
    current_year = 2026
    from_year = current_year - years + 1
    params = {
        "query.bibliographic": topic,
        "rows": max_results,
        "filter": f"from-pub-date:{from_year}-01-01",
        "mailto": CONTACT_EMAIL
    }
    r = requests.get("https://api.crossref.org/works", params=params, timeout=60)
    if r.status_code != 200:
        return []
    items = safe_get(r.json(), "message", "items", default=[]) or []
    papers = []
    for item in items:
        mapped = crossref_to_internal(item)
        if article_types:
            typ = (mapped.get("type") or "").lower()
            if typ not in [t.lower() for t in article_types]:
                continue
        papers.append(mapped)
    return papers

# ---------- Scoring ----------

KEYWORD_GROUPS = {
    "selenium": ["selenium", "selenate", "selenite", "nano-selenium", "nanoselenium", "selenium nanoparticle", "se "],
    "paddy": ["paddy", "rice", "rice soil", "rice root", "rhizosphere", "flooded soil"],
    "microbe": ["microbial", "microbiome", "microorganism", "bacteria", "bacterial", "fungal", "community", "communities"],
    "function": ["functional", "gene", "genes", "metagenome", "denitrification", "nitrification", "phosphorus", "nitrogen", "carbon cycle", "nutrient cycling"]
}

def score_relevance(text: str) -> float:
    txt = normalize_text(text)
    total = 0
    for _, kws in KEYWORD_GROUPS.items():
        if any(k in txt for k in kws):
            total += 1
    return total / 4.0

def score_method_match(text: str) -> float:
    txt = normalize_text(text)
    strong = ["metagenome", "metagenomic", "functional gene", "genes", "network", "sequencing", "16s", "shotgun", "denitrification", "nitrification"]
    hits = sum(1 for k in strong if k in txt)
    return min(hits / 4.0, 1.0)

def score_system_match(text: str) -> float:
    txt = normalize_text(text)
    hits = 0
    if "rice" in txt or "paddy" in txt:
        hits += 1
    if "soil" in txt or "rhizosphere" in txt or "root" in txt:
        hits += 1
    if "long-term" in txt or "long term" in txt:
        hits += 1
    return min(hits / 3.0, 1.0)

def score_quality(paper: Dict[str, Any]) -> float:
    cites = paper.get("cited_by_count", 0) or 0
    journal = normalize_text(paper.get("journal", ""))
    source = (paper.get("source", "") or "").lower()
    base = 0.35
    if cites >= 50:
        base += 0.3
    elif cites >= 20:
        base += 0.2
    elif cites >= 5:
        base += 0.1
    if source == "pubmed":
        base += 0.1
    strong_journals = ["nature", "science", "environment", "microbiology", "plant and soil", "journal of agricultural and food chemistry", "frontiers", "soil biology"]
    if any(j in journal for j in strong_journals):
        base += 0.2
    return min(base, 1.0)

def score_recency(year_str: str) -> float:
    try:
        year = int(year_str)
    except Exception:
        return 0.3
    current = 2026
    age = current - year
    if age <= 1:
        return 1.0
    if age == 2:
        return 0.85
    if age == 3:
        return 0.7
    if age == 4:
        return 0.55
    if age == 5:
        return 0.4
    return 0.2

def paper_score(paper: Dict[str, Any], weights: ScoreWeights) -> Dict[str, Any]:
    text = " ".join([
        paper.get("title", "") or "",
        paper.get("abstract", "") or "",
        " ".join(paper.get("tags", []) or []),
        paper.get("journal", "") or "",
        paper.get("type", "") or ""
    ])
    r = score_relevance(text)
    m = score_method_match(text)
    s = score_system_match(text)
    q = score_quality(paper)
    rc = score_recency(paper.get("year", ""))
    score = round(
        r * weights.relevance +
        m * weights.method_match +
        s * weights.system_match +
        q * weights.quality +
        rc * weights.recency
    )
    reasons = []
    if r >= 0.75:
        reasons.append("high topical relevance")
    if m >= 0.5:
        reasons.append("strong method/function match")
    if s >= 0.67:
        reasons.append("good rice/paddy system match")
    if q >= 0.6:
        reasons.append("solid source quality")
    if rc >= 0.7:
        reasons.append("recent publication")
    out = dict(paper)
    out["score"] = score
    out["score_breakdown"] = {
        "relevance": round(r * weights.relevance, 1),
        "method_match": round(m * weights.method_match, 1),
        "system_match": round(s * weights.system_match, 1),
        "quality": round(q * weights.quality, 1),
        "recency": round(rc * weights.recency, 1),
    }
    out["reason"] = ", ".join(reasons) if reasons else "moderate match"
    return out

# ---------- Routes ----------

@app.get("/")
def root():
    return {"message": "Zotero bridge v3 is running"}

@app.post("/import_to_zotero")
def import_to_zotero(req: ImportRequest):
    require_zotero_credentials()
    paper = {
        "title": req.title,
        "abstract": req.abstract,
        "authors": [a.dict() for a in req.authors],
        "journal": req.journal,
        "year": req.year,
        "doi": req.doi,
        "url": req.url,
        "tags": req.tags
    }
    item = build_zotero_item(paper)
    zotero_url = f"https://api.zotero.org/users/{ZOTERO_USER_ID}/items"
    r = requests.post(zotero_url, headers=zotero_headers(), json=[item], timeout=60)
    if r.status_code not in [200, 201]:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return {"success": True, "message": "Imported to Zotero", "zotero_response": r.json()}

@app.post("/search_papers")
def search_papers(req: SearchRequest):
    sources = [s.lower() for s in req.sources]
    papers = []
    if "openalex" in sources:
        papers.extend(search_openalex(req.topic, req.years, req.max_results_per_source, req.english_only, req.article_types))
    if "pubmed" in sources:
        papers.extend(search_pubmed(req.topic, req.years, req.max_results_per_source))
    if "crossref" in sources:
        papers.extend(search_crossref(req.topic, req.years, req.max_results_per_source, req.article_types))
    papers = dedupe_papers(papers)
    return {
        "count": len(papers),
        "sources_used": sources,
        "papers": papers
    }

@app.post("/score_papers")
def score_papers(req: ScoreRequest):
    scored = [paper_score(p, req.weights) for p in req.papers]
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    selected = [p for p in scored if p.get("score", 0) >= req.threshold]
    return {
        "total": len(scored),
        "threshold": req.threshold,
        "selected_count": len(selected),
        "selected": selected,
        "all_scored": scored
    }

@app.post("/search_score_import")
def search_score_import(req: SearchScoreImportRequest):
    search_resp = search_papers(SearchRequest(
        topic=req.topic,
        years=req.years,
        max_results_per_source=req.max_results_per_source,
        english_only=req.english_only,
        article_types=req.article_types,
        sources=req.sources
    ))
    papers = search_resp["papers"]
    scored = [paper_score(p, ScoreWeights()) for p in papers]
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    selected = [p for p in scored if p.get("score", 0) >= req.threshold]

    if req.dry_run:
        return {
            "mode": "dry_run",
            "threshold": req.threshold,
            "searched": len(papers),
            "selected_count": len(selected),
            "selected": selected
        }

    require_zotero_credentials()
    zotero_url = f"https://api.zotero.org/users/{ZOTERO_USER_ID}/items"

    imported = []
    skipped = []
    failed = []

    for p in selected:
        try:
            if zotero_item_exists_by_title(p["title"]):
                skipped.append({"title": p["title"], "reason": "duplicate title"})
                continue
            extra_tags = list(req.import_tags) + [
                f"score:{p['score']}",
                f"source:{(p.get('source', '') or '').lower()}",
                "bridge:v3"
            ]
            item = build_zotero_item(p, extra_tags=extra_tags)
            r = requests.post(zotero_url, headers=zotero_headers(), json=[item], timeout=60)
            if r.status_code in [200, 201]:
                imported.append({
                    "title": p["title"],
                    "score": p["score"],
                    "journal": p.get("journal", ""),
                    "year": p.get("year", ""),
                    "source": p.get("source", "")
                })
            else:
                failed.append({
                    "title": p["title"],
                    "status_code": r.status_code,
                    "detail": r.text[:300]
                })
        except Exception as e:
            failed.append({"title": p.get("title", ""), "detail": str(e)})

    return {
        "searched": len(papers),
        "selected_count": len(selected),
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "imported": imported,
        "skipped": skipped,
        "failed": failed,
        "selected_preview": selected[:10]
    }

