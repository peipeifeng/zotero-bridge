from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import requests
import re

app = FastAPI(title="Zotero Bridge v2", version="2.0.0")

ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")

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

def openalex_to_internal(item: Dict[str, Any]) -> Dict[str, Any]:
    authorships = item.get("authorships", []) or []
    authors = []
    for a in authorships:
        display_name = safe_get(a, "author", "display_name", default="") or ""
        if display_name:
            parts = display_name.strip().split()
            if len(parts) == 1:
                authors.append({"firstName": "", "lastName": parts[0]})
            else:
                authors.append({"firstName": " ".join(parts[:-1]), "lastName": parts[-1]})
    abstract = extract_abstract_from_openalex(item.get("abstract_inverted_index"))
    concepts = item.get("concepts", []) or []
    concept_tags = [c.get("display_name", "") for c in concepts[:8] if c.get("display_name")]
    return {
        "title": item.get("title", ""),
        "abstract": abstract,
        "authors": authors,
        "journal": safe_get(item, "primary_location", "source", "display_name", default="") or "",
        "year": str(item.get("publication_year", "") or ""),
        "doi": (item.get("doi", "") or "").replace("https://doi.org/", ""),
        "url": item.get("id", "") or "",
        "tags": concept_tags,
        "type": item.get("type", ""),
        "cited_by_count": item.get("cited_by_count", 0) or 0,
        "source": "OpenAlex"
    }

def build_zotero_item(paper: Dict[str, Any], extra_tags: Optional[List[str]] = None) -> Dict[str, Any]:
    creators = []
    for a in paper.get("authors", []) or []:
        creators.append({
            "creatorType": "author",
            "firstName": a.get("firstName", ""),
            "lastName": a.get("lastName", "")
        })
    tags = [{"tag": t} for t in (paper.get("tags", []) or [])]
    for t in (extra_tags or []):
        tags.append({"tag": t})
    item = {
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
    return item

def zotero_headers() -> Dict[str, str]:
    return {
        "Zotero-API-Key": ZOTERO_API_KEY,
        "Content-Type": "application/json"
    }

def zotero_item_exists_by_title(title: str) -> bool:
    require_zotero_credentials()
    q = requests.utils.quote(title)
    url = f"https://api.zotero.org/users/{ZOTERO_USER_ID}/items?q={q}&qmode=title"
    r = requests.get(url, headers={"Zotero-API-Key": ZOTERO_API_KEY})
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
    max_results: int = Field(20, ge=1, le=50)
    article_types: List[str] = Field(default_factory=list, description="Optional OpenAlex types, e.g. article, review")
    english_only: bool = True

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
    max_results: int = 20
    article_types: List[str] = []
    english_only: bool = True
    threshold: int = 85
    import_tags: List[str] = []
    dry_run: bool = False

# ---------- Scoring ----------

KEYWORD_GROUPS = {
    "selenium": ["selenium", "se ", "selenate", "selenite", "nano-selenium", "nanoselenium", "selenium nanoparticle"],
    "paddy": ["paddy", "rice", "rice soil", "rice root", "rhizosphere", "flooded soil"],
    "microbe": ["microbial", "microbiome", "microorganism", "bacteria", "bacterial", "fungal", "community", "communities"],
    "function": ["functional", "gene", "genes", "metagenome", "denitrification", "nitrification", "phosphorus", "nitrogen", "carbon cycle", "nutrient cycling"]
}

def score_relevance(text: str, topic: str) -> float:
    txt = normalize_text(text)
    total = 0
    for _, kws in KEYWORD_GROUPS.items():
        if any(k in txt for k in kws):
            total += 1
    # 0..4 -> 0..1
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
    base = 0.4
    if cites >= 50:
        base += 0.3
    elif cites >= 20:
        base += 0.2
    elif cites >= 5:
        base += 0.1
    strong_journals = ["nature", "science", "environment", "microbiology", "plant and soil", "journal of agricultural and food chemistry", "frontiers"]
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

def paper_score(paper: Dict[str, Any], topic: str, weights: ScoreWeights) -> Dict[str, Any]:
    text = " ".join([
        paper.get("title", "") or "",
        paper.get("abstract", "") or "",
        " ".join(paper.get("tags", []) or []),
        paper.get("journal", "") or ""
    ])
    r = score_relevance(text, topic)
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
    return {"message": "Zotero bridge v2 is running"}

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
    r = requests.post(zotero_url, headers=zotero_headers(), json=[item])
    if r.status_code not in [200, 201]:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    return {"success": True, "message": "Imported to Zotero", "zotero_response": r.json()}

@app.post("/search_papers")
def search_papers(req: SearchRequest):
    current_year = 2026
    from_year = current_year - req.years + 1

    params = {
        "search": req.topic,
        "filter": f"from_publication_date:{from_year}-01-01",
        "per-page": req.max_results,
        "sort": "relevance_score:desc"
    }

    r = requests.get("https://api.openalex.org/works", params=params, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"OpenAlex error: {r.text}")

    data = r.json()
    results = data.get("results", []) or []
    papers = []

    for item in results:
        mapped = openalex_to_internal(item)

        if req.english_only:
            if (item.get("language") not in [None, "en"]):
                continue

        if req.article_types:
            paper_type = (mapped.get("type") or "").lower()
            if paper_type not in [t.lower() for t in req.article_types]:
                continue

        papers.append(mapped)

    return {
        "count": len(papers),
        "source": "OpenAlex",
        "papers": papers
    }

@app.post("/score_papers")
def score_papers(req: ScoreRequest):
    scored = [paper_score(p, req.topic, req.weights) for p in req.papers]
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
    # Step 1: Search
    search_resp = search_papers(SearchRequest(
        topic=req.topic,
        years=req.years,
        max_results=req.max_results,
        article_types=req.article_types,
        english_only=req.english_only
    ))
    papers = search_resp["papers"]

    # Step 2: Score
    scored = [paper_score(p, req.topic, ScoreWeights()) for p in papers]
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

    # Step 3: Import selected papers
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
            extra_tags = list(req.import_tags) + [f"score:{p['score']}", "source:openalex", "bridge:v2"]
            item = build_zotero_item(p, extra_tags=extra_tags)
            r = requests.post(zotero_url, headers=zotero_headers(), json=[item], timeout=60)
            if r.status_code in [200, 201]:
                imported.append({
                    "title": p["title"],
                    "score": p["score"],
                    "journal": p.get("journal", ""),
                    "year": p.get("year", "")
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

