import os
import math
import json
import re
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "destinations")
_INDEX_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".rag_index.json")

_chunks: list[dict] = []  # [{"id", "city", "content", "tokens", "section_name", "chunk_index"}]
_idf: dict[str, float] = {}
_doc_len: dict[str, int] = {}  # chunk id -> token count (for BM25)
_avg_dl: float = 0.0
_indexed = False

# BM25 parameters
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    # Split on non-alphanumeric for CJK + Latin
    tokens = re.findall(r'[a-z]+|[一-鿿]|[0-9]+', text)
    # For CJK, also split into bigrams for better matching
    cjk = re.findall(r'[一-鿿]+', text)
    for seg in cjk:
        for i in range(len(seg) - 1):
            tokens.append(seg[i:i+2])
    return tokens


def _tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def _tfidf_vec(tokens: list[str]) -> dict[str, float]:
    tf_vals = _tf(tokens)
    return {t: v * _idf.get(t, 0) for t, v in tf_vals.items()}


def _cosine(v1: dict[str, float], v2: dict[str, float]) -> float:
    keys = set(v1) & set(v2)
    if not keys:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in keys)
    n1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    n2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def _bm25_score(query_tokens: list[str], chunk_tokens: list[str], chunk_id: str) -> float:
    """Compute BM25 score for a single chunk against query tokens."""
    dl = _doc_len.get(chunk_id, len(chunk_tokens))
    tf_counts = Counter(chunk_tokens)
    n_docs = len(_chunks) or 1
    score = 0.0
    for qt in query_tokens:
        df = sum(1 for c in _chunks if qt in c.get("tokens", []))
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        freq = tf_counts.get(qt, 0)
        tf_norm = (freq * (_BM25_K1 + 1)) / (freq + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / _avg_dl))
        score += idf * tf_norm
    return score


def _split_by_paragraphs(text: str, size_min: int = 200, size_max: int = 500) -> list[dict]:
    """Split text by paragraphs first, then by size.
    Each chunk dict has 'content' and 'section_name' keys.
    """
    # Extract sections from ## headers
    section_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
    sections = []
    last_end = 0
    current_section = ""

    for m in section_pattern.finditer(text):
        # Content before this header belongs to previous section
        if m.start() > last_end:
            sections.append({"section_name": current_section, "content": text[last_end:m.start()]})
        current_section = m.group(2).strip()
        last_end = m.end()

    # Remaining content after last header
    if last_end < len(text):
        sections.append({"section_name": current_section, "content": text[last_end:]})

    if not sections:
        sections = [{"section_name": "", "content": text}]

    # Now split each section's content into chunks of size_min..size_max
    chunks = []
    for sec in sections:
        content = sec["content"].strip()
        if not content:
            continue

        # Split on paragraph boundaries (double newlines)
        paragraphs = re.split(r'\n\s*\n', content)
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if not current_chunk:
                current_chunk = para
            elif len(current_chunk) + len(para) + 1 <= size_max:
                current_chunk += "\n\n" + para
            else:
                # Flush current_chunk if it meets minimum size
                if len(current_chunk.strip()) >= size_min:
                    chunks.append({"content": current_chunk.strip(), "section_name": sec["section_name"]})
                    current_chunk = para
                else:
                    # Too small, keep accumulating
                    current_chunk += "\n\n" + para

        # Flush remaining
        if current_chunk.strip():
            remaining = current_chunk.strip()
            # If remaining is too large, split by size
            if len(remaining) > size_max:
                start = 0
                while start < len(remaining):
                    end = min(start + size_max, len(remaining))
                    chunk_text = remaining[start:end].strip()
                    if len(chunk_text) >= size_min or end == len(remaining):
                        chunks.append({"content": chunk_text, "section_name": sec["section_name"]})
                    start += size_max
            else:
                chunks.append({"content": remaining, "section_name": sec["section_name"]})

    # Merge tiny trailing chunks into the previous one
    merged = []
    for chunk in chunks:
        if merged and len(chunk["content"]) < size_min:
            merged[-1]["content"] += "\n\n" + chunk["content"]
        else:
            merged.append(chunk)

    return merged if merged else chunks


def _build_index():
    global _chunks, _idf, _doc_len, _avg_dl, _indexed

    # Try loading cached index
    if os.path.exists(_INDEX_FILE):
        try:
            with open(_INDEX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            _chunks = data["chunks"]
            _idf = data["idf"]
            _doc_len = data.get("doc_len", {})
            _avg_dl = data.get("avg_dl", 0.0)
            if not _avg_dl and _chunks:
                _avg_dl = sum(len(c.get("tokens", [])) for c in _chunks) / len(_chunks)
            _indexed = True
            return len(_chunks)
        except Exception:
            pass

    doc_freq = Counter()
    raw_docs = []
    total_len = 0

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(DATA_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        city = fname.replace(".md", "")

        for i, chunk_dict in enumerate(_split_by_paragraphs(text)):
            content = chunk_dict["content"]
            section_name = chunk_dict["section_name"]
            tokens = _tokenize(content)
            chunk_id = f"{city}_{i}"
            raw_docs.append({
                "id": chunk_id,
                "city": city,
                "content": content,
                "tokens": tokens,
                "section_name": section_name,
                "chunk_index": i,
            })
            _doc_len[chunk_id] = len(tokens)
            total_len += len(tokens)
            for t in set(tokens):
                doc_freq[t] += 1

    n_docs = len(raw_docs) or 1
    _idf = {t: math.log((n_docs + 1) / (df + 1)) + 1 for t, df in doc_freq.items()}
    _avg_dl = total_len / n_docs if n_docs else 1.0
    _chunks = raw_docs

    # Cache to disk
    try:
        with open(_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "chunks": _chunks,
                "idf": _idf,
                "doc_len": _doc_len,
                "avg_dl": _avg_dl,
            }, f, ensure_ascii=False)
    except Exception:
        pass

    _indexed = True
    return len(_chunks)


def index_destinations() -> int:
    if _indexed:
        return len(_chunks)
    return _build_index()


def search(query: str, top_k: int = 5) -> list[dict]:
    """Search the RAG index using combined BM25 + TF-IDF scoring.

    Returns list of dicts with keys:
        content, city, section, distance, score
    Score = 0.4 * BM25_normalized + 0.6 * TF-IDF cosine
    """
    if not _indexed:
        _build_index()
    if not _chunks:
        return []

    q_tokens = _tokenize(query)
    q_vec = _tfidf_vec(q_tokens)

    scored = []
    bm25_scores = []
    tfidf_scores = []

    for chunk in _chunks:
        c_vec = _tfidf_vec(chunk["tokens"])
        tfidf_sim = _cosine(q_vec, c_vec)
        bm25 = _bm25_score(q_tokens, chunk["tokens"], chunk["id"])
        bm25_scores.append(bm25)
        tfidf_scores.append(tfidf_sim)

    # Normalize BM25 scores to [0, 1]
    bm25_max = max(bm25_scores) if bm25_scores else 1.0
    bm25_min = min(bm25_scores) if bm25_scores else 0.0
    bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0

    for i, chunk in enumerate(_chunks):
        bm25_norm = (bm25_scores[i] - bm25_min) / bm25_range
        tfidf_norm = tfidf_scores[i]
        combined = 0.4 * bm25_norm + 0.6 * tfidf_norm
        scored.append({
            "content": chunk["content"],
            "city": chunk["city"],
            "section": chunk.get("section_name", ""),
            "distance": round(1 - combined, 3),
            "score": round(combined, 4),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def get_context(query: str, max_len: int = 1500) -> str:
    """Return formatted context string with section names for LLM consumption."""
    results = search(query, 5)
    if not results:
        return ""
    ctx = "【知识库参考】\n"
    total = 0
    for r in results:
        if total + len(r["content"]) > max_len:
            break
        section_label = f" [{r['section']}]" if r["section"] else ""
        ctx += f"- ({r['city']}{section_label}) {r['content']}\n"
        total += len(r["content"])
    return ctx


def search_by_city(city: str, top_k: int = 5) -> list[dict]:
    """Search the RAG index restricted to a specific city.

    Returns list of dicts with keys:
        content, city, section, distance, score
    """
    if not _indexed:
        _build_index()

    city_lower = city.lower()
    # Normalize: remove spaces, match against stored city names
    city_chunks = [c for c in _chunks if c["city"].lower().replace(" ", "") == city_lower.replace(" ", "")]
    if not city_chunks:
        # Fallback: partial match
        city_chunks = [c for c in _chunks if city_lower in c["city"].lower() or c["city"].lower() in city_lower]

    if not city_chunks:
        return []

    q_tokens = _tokenize(city)
    q_vec = _tfidf_vec(q_tokens)

    bm25_scores = []
    tfidf_scores = []

    for chunk in city_chunks:
        c_vec = _tfidf_vec(chunk["tokens"])
        tfidf_sim = _cosine(q_vec, c_vec)
        bm25 = _bm25_score(q_tokens, chunk["tokens"], chunk["id"])
        bm25_scores.append(bm25)
        tfidf_scores.append(tfidf_sim)

    # Normalize BM25 scores to [0, 1]
    bm25_max = max(bm25_scores) if bm25_scores else 1.0
    bm25_min = min(bm25_scores) if bm25_scores else 0.0
    bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0

    scored = []
    for i, chunk in enumerate(city_chunks):
        bm25_norm = (bm25_scores[i] - bm25_min) / bm25_range
        tfidf_norm = tfidf_scores[i]
        combined = 0.4 * bm25_norm + 0.6 * tfidf_norm
        scored.append({
            "content": chunk["content"],
            "city": chunk["city"],
            "section": chunk.get("section_name", ""),
            "distance": round(1 - combined, 3),
            "score": round(combined, 4),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
