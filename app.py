from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import requests

app = FastAPI()

ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")

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

@app.get("/")
def root():
    return {"message": "Zotero bridge is running"}

@app.post("/import_to_zotero")
def import_to_zotero(req: ImportRequest):
    if not ZOTERO_USER_ID or not ZOTERO_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Zotero credentials")

    creators = []
    for a in req.authors:
        creators.append({
            "creatorType": "author",
            "firstName": a.firstName,
            "lastName": a.lastName
        })

    item = {
        "itemType": "journalArticle",
        "title": req.title,
        "creators": creators,
        "abstractNote": req.abstract,
        "publicationTitle": req.journal,
        "date": req.year,
        "DOI": req.doi,
        "url": req.url,
        "tags": [{"tag": t} for t in req.tags]
    }

    zotero_url = f"https://api.zotero.org/users/{ZOTERO_USER_ID}/items"

    headers = {
        "Zotero-API-Key": ZOTERO_API_KEY,
        "Content-Type": "application/json"
    }

    r = requests.post(zotero_url, headers=headers, json=[item])

    if r.status_code not in [200, 201]:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    return {"success": True, "message": "Imported to Zotero", "zotero_response": r.json()}
