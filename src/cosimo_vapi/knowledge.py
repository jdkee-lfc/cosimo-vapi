"""Collection knowledge for Cosimo (UPGRADE_PLAN Phase B).

- catalog_index(): one line per item, `id | title | artist | room | period`, written to
  data/catalog_index.txt and placed in the system prompt so Cosimo knows what is in the building.
- get_artwork(id): the full cleaned record for one item.
- search(query): hybrid retrieval — BM25 (rank_bm25) fused with local embeddings
  (nomic-embed-text via Ollama) using reciprocal rank fusion. Embeddings are computed once and
  cached to data/embeddings.npy.

Everything runs locally; the embedding model is served by the local Ollama daemon.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

EMBED_MODEL_DEFAULT = "nomic-embed-text"
RRF_K = 60  # standard reciprocal-rank-fusion constant

# Fields returned by get_artwork, in a readable order. Empty values are dropped.
_RECORD_FIELDS = [
    "id", "title", "artistName", "objectType", "category", "period", "yearStart", "yearEnd",
    "culture", "school", "medium", "material", "technique", "description", "dimensions",
    "inscription", "signedName", "signedLocation", "building", "room", "position",
    "provenance", "source", "receivedAs", "frameDescription",
]

_STOPWORDS = {
    "a", "an", "and", "any", "are", "about", "at", "by", "can", "do", "does", "for", "from",
    "have", "has", "here", "i", "in", "is", "it", "its", "me", "of", "on", "or", "tell", "that",
    "the", "there", "this", "to", "was", "what", "whats", "which", "who", "with", "you", "your",
    "we", "s",
}


# Visitors say "France", the catalog says "French": fold country and demonym to one token.
_PLACE_SYNONYMS = {
    "french": "france", "italian": "italy", "spanish": "spain", "russian": "russia",
    "greek": "greece", "chinese": "china", "burmese": "burma", "myanmar": "burma",
    "english": "england", "british": "britain", "american": "america", "usa": "america",
    "german": "germany", "germanic": "germany", "flemish": "flanders", "cretan": "crete",
    "dutch": "netherlands", "holland": "netherlands", "japanese": "japan", "indian": "india",
}


def _tokenize(text: str) -> list[str]:
    tokens = []
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOPWORDS:
            continue
        tok = _PLACE_SYNONYMS.get(tok, tok)
        # crude plural folding so "paintings" matches "painting"
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


def _inscription_text(item: dict) -> str:
    ins = item.get("inscription")
    if isinstance(ins, dict):
        return " ".join(str(ins.get(k, "")) for k in ("text", "translation"))
    return str(ins or "")


class Collection:
    def __init__(
        self,
        collection_path: str | Path = "data/collection_optimized.json",
        embed_model: str = EMBED_MODEL_DEFAULT,
    ):
        self.path = Path(collection_path)
        self.data_dir = self.path.parent
        self.embed_model = embed_model

        raw = json.loads(self.path.read_text())["items"]
        # The source file repeats 17 ids verbatim (377 rows, 359 unique items); keep the first.
        self.items: list[dict] = []
        self._by_id: dict[str, dict] = {}
        for item in raw:
            key = str(item["id"])
            if key not in self._by_id:
                self._by_id[key] = item
                self.items.append(item)

        self._bm25 = None
        self._embeddings: np.ndarray | None = None
        self._embed_client = None
        self.embeddings_available = False

    # ------------------------------------------------------------------ B1
    def catalog_index(self, write: bool = True) -> str:
        lines = [
            " | ".join([
                str(i["id"]),
                i.get("title") or "",
                i.get("artistName") or "",
                i.get("room") or "",
                i.get("period") or "",
            ])
            # grouped by room so "what's in the X" is a contiguous block for the model
            for i in sorted(self.items, key=lambda i: (i.get("room") or "~", str(i["id"])))
        ]
        text = "\n".join(lines)
        if write:
            (self.data_dir / "catalog_index.txt").write_text(text + "\n")
        return text

    # ------------------------------------------------------------------ B2
    def get_artwork(self, artwork_id: str) -> dict | None:
        item = self._by_id.get(str(artwork_id).strip())
        if item is None:
            return None
        record = {}
        for field in _RECORD_FIELDS:
            value = item.get(field)
            if field == "dimensions" and isinstance(value, dict):
                # drop internal cataloguing notes; keep the measurements
                value = {k: v for k, v in value.items() if k != "notes" and v not in (None, "")}
            if value in (None, "", {}, []):
                continue
            record[field] = value
        return record

    # ------------------------------------------------------------------ B3
    def _bm25_text(self, item: dict) -> str:
        title = item.get("title") or ""
        artist = item.get("artistName") or ""
        parts = [
            title, title, title,          # weight title
            artist, artist,               # weight artist
            item.get("objectType"), item.get("category"), item.get("culture"),
            item.get("school"), item.get("period"), item.get("building"), item.get("room"),
            item.get("medium"), item.get("material"), item.get("technique"),
            item.get("description"), _inscription_text(item),
        ]
        return " ".join(str(p) for p in parts if p)

    def _embed_text(self, item: dict) -> str:
        desc = (item.get("description") or "")[:2000]
        return (
            f"{item.get('title', '')}. {item.get('category', '')} by {item.get('artistName') or 'unknown artist'}. "
            f"{item.get('culture', '')} {item.get('school', '')}, {item.get('period', '')}. "
            f"Medium: {item.get('medium', '')}, {item.get('material', '')}. "
            f"Location: {item.get('building', '')}, {item.get('room', '')}. {desc}"
        )

    def build(self) -> None:
        """Build the BM25 index and load or compute the embedding cache."""
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([_tokenize(self._bm25_text(i)) for i in self.items])
        try:
            self._load_or_compute_embeddings()
            self.embeddings_available = True
        except Exception as exc:  # Ollama down or model missing: keep working on BM25 alone
            self.embeddings_available = False
            self.embedding_error = str(exc)

    def _client(self):
        if self._embed_client is None:
            import ollama
            self._embed_client = ollama
        return self._embed_client

    def _embed(self, texts: list[str], prefix: str) -> np.ndarray:
        # nomic-embed-text expects task prefixes
        out = []
        for start in range(0, len(texts), 32):
            batch = [prefix + t for t in texts[start:start + 32]]
            resp = self._client().embed(model=self.embed_model, input=batch)
            out.extend(resp["embeddings"])
        vecs = np.asarray(out, dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs

    def _load_or_compute_embeddings(self) -> None:
        texts = [self._embed_text(i) for i in self.items]
        digest = hashlib.sha256(
            (self.embed_model + "\n" + "\n".join(texts)).encode()
        ).hexdigest()
        npy = self.data_dir / "embeddings.npy"
        meta = self.data_dir / "embeddings.meta.json"
        if npy.exists() and meta.exists():
            try:
                if json.loads(meta.read_text()).get("sha256") == digest:
                    self._embeddings = np.load(npy)
                    return
            except (ValueError, OSError):
                pass
        self._embeddings = self._embed(texts, "search_document: ")
        np.save(npy, self._embeddings)
        meta.write_text(json.dumps({"model": self.embed_model, "sha256": digest, "count": len(texts)}))

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        if self._bm25 is None:
            self.build()

        ranked_lists = []
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_order = [i for i in np.argsort(-bm25_scores) if bm25_scores[i] > 0][:50]
        ranked_lists.append(bm25_order)

        if self.embeddings_available:
            try:
                q = self._embed([query], "search_query: ")[0]
                sims = self._embeddings @ q
                ranked_lists.append(list(np.argsort(-sims)[:50]))
            except Exception:
                pass

        fused: dict[int, float] = {}
        for ranking in ranked_lists:
            for rank, idx in enumerate(ranking):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

        top = sorted(fused, key=fused.get, reverse=True)[:max_results]
        return [self.summary(self.items[i]) for i in top]

    @staticmethod
    def summary(item: dict) -> dict:
        desc = item.get("description") or ""
        return {
            "id": item["id"],
            "title": item.get("title"),
            "artist": item.get("artistName"),
            "category": item.get("category"),
            "culture": item.get("culture"),
            "period": item.get("period"),
            "room": item.get("room"),
            "snippet": desc[:240] + ("…" if len(desc) > 240 else ""),
        }


# Tool schemas (Ollama / OpenAI function-calling format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_collection",
            "description": (
                "Search the Kreb's Center collection by meaning or keywords — subject, appearance, "
                "artist, culture, material, room. Returns the best matching items with their ids."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What the visitor is asking about"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_artwork",
            "description": (
                "Get the full record for one item by its id (from the catalog or a search result): "
                "description, medium, culture, provenance, inscription, dimensions, location."
            ),
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Item id, e.g. KR23.1.2"}},
                "required": ["id"],
            },
        },
    },
]


def run_tool(collection: Collection, name: str, arguments: dict) -> str:
    """Execute a tool call and return a JSON string for the model."""
    if name == "search_collection":
        return json.dumps(collection.search(str(arguments.get("query", ""))), ensure_ascii=False)
    if name == "get_artwork":
        record = collection.get_artwork(str(arguments.get("id", "")))
        if record is None:
            return json.dumps({"error": "No item with that id is in the collection."})
        return json.dumps(record, ensure_ascii=False)
    return json.dumps({"error": f"Unknown tool {name}"})
