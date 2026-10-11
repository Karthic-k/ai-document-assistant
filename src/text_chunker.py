import re
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Stop when the end of the text is reached
        if end == len(text):
            break

        start = end - overlap

    return chunks




def sentence_aware_chunk_text(
    text,
    chunk_size=500,
    overlap_sentences=1
):
    """Create sentence-aware chunks with safe sentence overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative")

    text = text.strip()

    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_sentences = []

    for sentence in sentences:

        # Handle a sentence longer than the chunk limit.
        if len(sentence) > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []

            # Split oversized sentences into smaller pieces.
            for start in range(0, len(sentence), chunk_size):
                piece = sentence[start:start + chunk_size].strip()
                if piece:
                    chunks.append(piece)

            continue

        candidate = " ".join(current_sentences + [sentence])

        if len(candidate) <= chunk_size:
            current_sentences.append(sentence)

        else:
            if current_sentences:
                chunks.append(" ".join(current_sentences))

            # Retain overlap, but drop old sentences if needed
            # to make room for the next sentence.
            current_sentences = (
                current_sentences[-overlap_sentences:]
                if overlap_sentences > 0
                else []
            )

            while current_sentences and len(
                " ".join(current_sentences + [sentence])
            ) > chunk_size:
                current_sentences.pop(0)

            current_sentences.append(sentence)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def create_page_aware_chunks(
    pages,
    chunk_size=500,
    overlap_sentences=1
):
    """
    Create sentence-aware chunks while preserving
    the original PDF page number.
    """

    all_chunks = []
    chunk_index = 0

    for page in pages:

        page_number = page["page_number"]
        page_text = page["text"]

        page_chunks = sentence_aware_chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap_sentences=overlap_sentences
        )

        for chunk in page_chunks:

            all_chunks.append({
                "text": chunk,
                "page_number": page_number,
                "chunk_index": chunk_index
            })

            chunk_index += 1

    return all_chunks