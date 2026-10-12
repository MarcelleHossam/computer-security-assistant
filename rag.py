import math
import re
from collections import Counter

SOURCE_TITLE = "Wikipedia — Computer security"
SOURCE_URL = "https://en.wikipedia.org/wiki/Computer_security"


def tokenize(text):
    """Small, dependency-free tokenizer suitable for this demo."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_chunks(text, source_url=SOURCE_URL, chunk_size=1000, overlap=200):
    """Split source text into overlapping chunks with stable source metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer a word boundary near the end without creating tiny chunks.
        if end < len(text):
            boundary = text.rfind(" ", start + int(chunk_size * 0.65), end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append({
                "text": piece,
                "title": SOURCE_TITLE,
                "url": source_url,
                "tokens": tokenize(piece),
            })
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def retrieve_chunks(question, chunks, top_k=3):
    """
    BM25-style ranking over the in-memory chunks.

    Scores are useful for ranking, not calibrated probabilities. This implementation
    is intentionally lightweight and should be evaluated on a project-specific test set.
    """
    query_terms = tokenize(question)
    if not query_terms or not chunks:
        return []

    documents = [item.get("tokens") or tokenize(item.get("text", "")) for item in chunks]
    lengths = [len(doc) for doc in documents]
    avgdl = sum(lengths) / len(lengths) if lengths else 0
    if avgdl == 0:
        return []

    doc_frequency = Counter()
    for doc in documents:
        doc_frequency.update(set(doc))
    n_docs = len(documents)
    k1, b = 1.5, 0.75
    query_counts = Counter(query_terms)
    scored = []

    for item, doc, dl in zip(chunks, documents, lengths):
        frequencies = Counter(doc)
        score = 0.0
        for term, q_count in query_counts.items():
            tf = frequencies.get(term, 0)
            if not tf:
                continue
            df = doc_frequency[term]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf * (k1 + 1) / denom) * min(q_count, 2)
        scored.append({**item, "score": score})

    scored.sort(key=lambda row: row["score"], reverse=True)
    return [item for item in scored[:max(1, top_k)] if item["score"] > 0]
