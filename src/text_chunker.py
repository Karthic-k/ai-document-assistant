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
    """
    Create sentence-aware chunks with sentence overlap.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive"
        )

    if overlap_sentences < 0:
        raise ValueError(
            "overlap_sentences cannot be negative"
        )

    text = text.strip()

    if not text:
        return []

    # Split text into sentences.
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    chunks = []

    current_sentences = []
    current_length = 0

    i = 0

    while i < len(sentences):

        sentence = sentences[i]

        additional_length = (
            len(sentence)
            + (1 if current_sentences else 0)
        )

        # Add sentence if it fits.
        if (
            current_length
            + additional_length
            <= chunk_size
        ):

            current_sentences.append(sentence)

            current_length += additional_length

            i += 1

        else:

            # Save current chunk.
            if current_sentences:

                chunks.append(
                    " ".join(current_sentences)
                )

            # Keep the last N sentences
            # for overlap.
            if overlap_sentences > 0:

                current_sentences = (
                    current_sentences[
                        -overlap_sentences:
                    ]
                )

            else:

                current_sentences = []

            current_length = len(
                " ".join(current_sentences)
            )

            # If a single sentence is larger
            # than chunk_size, keep it as its
            # own chunk rather than losing it.
            if (
                len(sentence) > chunk_size
                and not current_sentences
            ):

                chunks.append(sentence)

                i += 1

                current_sentences = []

                current_length = 0

    # Add final chunk.
    if current_sentences:

        chunks.append(
            " ".join(current_sentences)
        )

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