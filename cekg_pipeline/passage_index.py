"""
Passage Index — RAG context retrieval for causal assessment.

Segments the novel into overlapping text passages, embeds them once,
and provides fast top-K retrieval. Used to inject grounding narrative
context into the LLM causal assessment prompt.
"""

from typing import List, Tuple, Optional, Any

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _EMBED_AVAILABLE = True
except ImportError:
    _EMBED_AVAILABLE = False

PASSAGE_SIZE = 400    # target chars per passage
PASSAGE_OVERLAP = 80  # overlap between adjacent passages
MIN_PASSAGE_LEN = 60  # discard shorter fragments


def _segment_text(text: str, passage_size: int = PASSAGE_SIZE, overlap: int = PASSAGE_OVERLAP) -> List[str]:
    """
    Split text into overlapping character-level passages, breaking at
    sentence boundaries where possible to avoid mid-sentence cuts.
    """
    passages = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + passage_size, length)
        # Try to extend to the next sentence boundary (within 120 chars)
        if end < length:
            for punct in ('.', '?', '!'):
                boundary = text.find(punct, end)
                if 0 < boundary - end < 120:
                    end = boundary + 1
                    break
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_PASSAGE_LEN:
            passages.append(chunk)
        next_start = end - overlap
        if next_start <= start:   # guarantee forward progress
            next_start = start + passage_size
        start = next_start

    return passages


class PassageIndex:
    """
    Semantic index over all novel passages.

    Build once from chapter texts, then call retrieve() for any query.
    If sentence-transformers is unavailable, retrieve() returns [] silently.
    """

    def __init__(self) -> None:
        self.passages: List[str] = []
        self.embeddings: Optional[Any] = None   # np.ndarray (N, D)
        self._model: Optional[Any] = None

    def build(self, chapters: List[Tuple[int, str]]) -> None:
        """
        Segment and embed all chapter texts.
        chapters: list of (chapter_id, text) tuples from text_processor.split_chapters()
        """
        if not _EMBED_AVAILABLE:
            print("[passage_index] sentence-transformers not installed — RAG context disabled.")
            return

        print(f"[passage_index] Segmenting {len(chapters)} chapters into passages...")
        for _, chapter_text in chapters:
            self.passages.extend(_segment_text(chapter_text))

        if not self.passages:
            return

        print(f"[passage_index] Embedding {len(self.passages):,} passages...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self._model.encode(
            [p[:300] for p in self.passages],
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=256,
        )
        # Pre-normalise for fast dot-product cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.embeddings = self.embeddings / norms
        print(f"[passage_index] Ready — {len(self.passages):,} passages indexed.")

    def retrieve(self, query: str, top_k: int = 2, min_score: float = 0.30) -> List[str]:
        """
        Return up to top_k passages most relevant to query.
        Returns [] if index was not built or no passage clears min_score.
        """
        if self.embeddings is None or self._model is None or not self.passages:
            return []

        q_emb = self._model.encode([query[:200]], convert_to_numpy=True)
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            return []
        q_emb = q_emb[0] / q_norm

        sims = self.embeddings.dot(q_emb)          # (N,) — fast because embeddings are pre-normalised
        top_idx = np.argsort(sims)[-top_k:][::-1]

        return [
            self.passages[i][:250]
            for i in top_idx
            if sims[i] >= min_score
        ]

    @property
    def is_ready(self) -> bool:
        return self.embeddings is not None and len(self.passages) > 0
