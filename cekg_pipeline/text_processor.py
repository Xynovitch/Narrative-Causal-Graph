import re
from typing import List

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*?\*\*\*",
    re.IGNORECASE,
)
_GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*?\*\*\*",
    re.IGNORECASE,
)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header/footer.

    PG eBooks bracket the actual work with markers like
    `*** START OF THE PROJECT GUTENBERG EBOOK ... ***` and
    `*** END OF THE PROJECT GUTENBERG EBOOK ... ***`.
    Without trimming, the multi-thousand-char license/donation
    boilerplate after the END marker leaks into the last chapter
    and causes silent extraction failures + hallucinated events.
    """
    start_match = _GUTENBERG_START_RE.search(text)
    if start_match:
        text = text[start_match.end():]
        print(f"[gutenberg] Stripped header before *** START OF *** "
              f"({start_match.start()} chars removed)")
    end_match = _GUTENBERG_END_RE.search(text)
    if end_match:
        removed = len(text) - end_match.start()
        text = text[:end_match.start()]
        print(f"[gutenberg] Stripped trailing license from *** END OF *** "
              f"onwards ({removed} chars removed)")
    return text


def split_chapters(text: str) -> List[tuple[int, str]]:
    text = strip_gutenberg_boilerplate(text)

    patterns = [
        r"(?m)^Chapter\s+[IVXLCDM]+\.\s*$",
        r"(?m)^CHAPTER\s+[IVXLCDM]+\.?\s*$",
        r"(?m)^Chapter\s+\d+\.?\s*$",
        r"(?m)^CHAPTER\s+\d+\.?\s*$",
        r"(?im)^chapter\s+[IVXLCDM\d]+\.?\s*$",
    ]
    
    parts = None
    
    for i, pattern in enumerate(patterns):
        parts = re.split(pattern, text)
        if len(parts) > 1:
            print(f"[chapter split] Matched pattern {i+1}: found {len(parts)-1} chapters")
            break
    
    if parts is None or len(parts) <= 1:
        print("[chapter split] No chapter markers found, using paragraph splitting")
        paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        return list(enumerate(paras, start=1))
    
    chapters = []
    
    # --- FIX IS HERE ---
    # parts[0] is the content *before* the first chapter (e.g., title page).
    # We must iterate from parts[1:], which is the content of Chapter 1.
    # We use enumerate(..., start=1) to get the correct chapter number (1, 2, 3...).
    
    for idx, part in enumerate(parts[1:], start=1):
        cleaned = part.strip()
        if cleaned:
            chapters.append((idx, cleaned)) # 'idx' is now the correct chapter number
    # --- END FIX ---
    
    print(f"[chapter split] Successfully split into {len(chapters)} chapters")
    return chapters

def split_into_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 50]
    
    chunks = []
    for para in paragraphs:
        if len(para) > 800:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = []
            current_length = 0
            
            for sent in sentences:
                if current_length + len(sent) > 600 and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sent]
                    current_length = len(sent)
                else:
                    current_chunk.append(sent)
                    current_length += len(sent)
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        else:
            chunks.append(para)
    
    return chunks if chunks else paragraphs