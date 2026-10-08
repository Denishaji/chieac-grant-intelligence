"""Neural retrieval, independent eligibility checks, and source-backed briefs."""
import json
import re
from datetime import date
from functools import lru_cache
from html import escape
from urllib.parse import urlparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .settings import DATA

PROFILES = {
    "ELEVATE": "Career development, internships, mentoring, workforce readiness and technical training for underserved young adults and international graduates in Chicago.",
    "IMPACT": "Emergency assistance, housing stability, transportation, school enrollment and family support for underserved Chicago students and migrant families.",
    "Data Science Alliance": "Applied data science, artificial intelligence, software engineering and digital skills training through community nonprofit technology projects."
}

def load_records():
    records = json.loads((DATA / "demo_grants.json").read_text(encoding="utf-8"))
    imported = DATA / "imported_grants.json"
    if imported.exists():
        records += json.loads(imported.read_text(encoding="utf-8"))
    local = DATA / "local_grants.json"
    if local.exists():
        records += json.loads(local.read_text(encoding="utf-8"))
    return records

def embed(texts, prefix):
    import requests
    response = requests.post("http://127.0.0.1:11434/api/embed", json={
        "model": "nomic-embed-text", "input": [prefix + t for t in texts],
        "truncate": False
    }, timeout=(5, 120))
    response.raise_for_status()
    vectors = np.asarray(response.json()["embeddings"], dtype=float)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

@lru_cache(maxsize=8)
def document_vectors(texts):
    # Chunk long announcements rather than silently truncating eligibility clauses.
    chunks, owners = [], []
    for i, text in enumerate(texts):
        words = text.split()
        for offset in range(0, max(1, len(words)), 240):
            chunks.append(" ".join(words[offset:offset+280]))
            owners.append(i)
    vectors = np.concatenate([embed(chunks[i:i+16], "search_document: ") for i in range(0,len(chunks),16)])
    return vectors, owners

def grant_text(record):
    return record["title"] + ". " + record["text"]

def rank(query, records, method="Neural embeddings"):
    if not query.strip() or not records:
        return []
    texts = tuple(grant_text(r) for r in records)
    if method == "Keyword baseline":
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return []
        scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
    else:
        vectors, owners = document_vectors(texts)
        vector = embed([query], "search_query: ")[0]
        chunk_scores = vectors @ vector
        scores = np.array([max(float(s) for s, owner in zip(chunk_scores, owners) if owner == i) for i in range(len(texts))])
    return [dict(records[i], similarity=round(float(scores[i]), 4))
            for i in np.argsort(-scores, kind="stable")]

def deadline_status(record, today=None):
    today = today or date.today()
    status = record.get("status", "").lower()
    if status == "conflicting":
        return "Conflicting sources — verify status"
    if status == "invitation":
        return "Prior stage closed — invitation required"
    if status == "unclear":
        return "Deadline unclear"
    if status == "forecasted":
        return "Forecast — not yet open"
    if record.get("opens_on") and date.fromisoformat(record["opens_on"]) > today:
        return "Forecast — not yet open"
    if status == "rolling":
        return "Rolling inquiry — verify current intake"
    raw = record.get("deadline")
    if raw:
        try:
            close = date.fromisoformat(raw)
            if close < today:
                return "Closed"
            if close == today:
                return "Due today — verify cutoff"
        except (ValueError, TypeError):
            return "Deadline unclear"
    status = record.get("status", "").lower()
    if status in ("closed", "archived"):
        return "Closed"
    if status == "forecasted":
        return "Forecast — not yet open"
    if not raw:
        return "Deadline unknown"
    return "Future deadline — verify source"

def available_for_research(record, today=None):
    return deadline_status(record, today) in (
        "Future deadline — verify source", "Due today — verify cutoff",
        "Rolling inquiry — verify current intake"
    )

def select_collection(records, collection):
    kinds = {"Chicago-area research": "Curated local source", "Fictional demo": "Fictional demo"}
    if collection == "All records":
        return records
    if collection == "Public source imports":
        return [r for r in records if r["kind"] not in kinds.values()]
    return [r for r in records if r["kind"] == kinds[collection]]

def checks(record, nonprofit="Unknown", location="Illinois", budget=None):
    """Known structured requirements only; never infer full eligibility."""
    rows = []
    for rule in record.get("requirements", []):
        field, expected = rule["field"], rule["value"]
        result = "Needs review"
        if field == "nonprofit":
            if nonprofit != "Unknown":
                result = "Matches declared profile" if (nonprofit == "Yes") == expected else "Does not match"
        elif field == "state":
            if location:
                result = "Matches declared profile" if location.casefold() in [x.casefold() for x in expected] else "Does not match"
        elif field == "max_budget" and budget is not None:
            result = "Matches declared profile" if budget <= expected else "Does not match"
        rows.append({"Requirement": rule["label"], "Result": result, "Evidence": rule["evidence"]})
    rows.append({"Requirement": "Other conditions in the full announcement",
                 "Result": "Needs review",
                 "Evidence": "Automated checks cover only the structured requirements shown above."})
    return rows

def evidence_passages(record, query, count=3):
    passages = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", record["text"]) if p.strip()]
    # Deterministic lexical passage selection; not described as neural evidence verification.
    terms = set(re.findall(r"[a-z]{3,}", query.lower()))
    passages.sort(key=lambda p: -len(terms.intersection(re.findall(r"[a-z]{3,}", p.lower()))))
    return passages[:count]

def safe_url(value):
    return value if urlparse(value).scheme in ("https", "http") else ""

def brief_html(record, query, rows, method):
    esc = lambda v: escape(str(v))
    evidence = "".join("<li>" + esc(p) + "</li>" for p in evidence_passages(record, query))
    checks_html = "".join("<tr><td>" + esc(r["Requirement"]) + "</td><td>" + esc(r["Result"]) + "</td><td>" + esc(r["Evidence"]) + "</td></tr>" for r in rows)
    source = safe_url(record.get("url", ""))
    link = '<a href="' + esc(source) + '">Original source</a>' if source else "Fictional demonstration record"
    return """<!doctype html><html><head><meta charset="utf-8"><title>Funding research brief</title>
<style>body{font:16px/1.6 system-ui;max-width:920px;margin:48px auto;padding:24px;color:#183232}h1{line-height:1.15}small{color:#526565}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cad9d3;padding:12px;text-align:left}header{border-top:8px solid #087f6d} @media print{body{margin:0}}</style></head><body><header>
<small>CHIEAC GRANT INTELLIGENCE · RESEARCH PROTOTYPE</small><h1>""" + esc(record["title"]) + "</h1></header><p><b>Data type:</b> " + esc(record["kind"]) + " · <b>Retrieved:</b> " + esc(record.get("retrieved_at", "N/A")) + "</p><p>" + link + "</p><h2>Program being researched</h2><p>" + esc(query) + "</p><p><b>Method:</b> " + esc(method) + " · <b>Similarity:</b> " + esc(record.get("similarity", "N/A")) + " (not a funding probability)</p><p><b>Deadline:</b> " + esc(record.get("deadline") or "Unknown") + " · " + esc(deadline_status(record)) + "</p><h2>Research passages</h2><p>" + esc(record.get("text_basis", "Stored announcement text")) + "</p><p>" + esc(record.get("review_notes", "")) + "</p><ul>" + evidence + "</ul><h2>Requirement review</h2><table><tr><th>Requirement</th><th>Result</th><th>Evidence</th></tr>" + checks_html + "</table><p>Profile facts are user-declared. Research support only; review the full funder announcement before deciding to apply. Public-source snapshots may change.</p></body></html>"
