
import os

from dotenv import load_dotenv
from groq import Groq
from groq import APIConnectionError, APIStatusError, RateLimitError


# Load configuration from the project-root .env file.
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


def generate_answer(question, retrieved_chunks):
    """Generate an answer grounded in retrieved PDF chunks."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return (
            "I couldn't find enough relevant information "
            "in the uploaded document to answer this question. "
            "Please try asking about something that is discussed "
            "in the document."
        )

    if not API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add it to your project-root .env file."
        )

    # Build context from the retrieved passages.
    context = "\n\n".join(
        f"[Passage {i}]\n{result['chunk']}"
        for i, result in enumerate(
            retrieved_chunks, start=1
        )
    )

    client = Groq(api_key=API_KEY)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document question-answering assistant. "
                        "Answer the user's question using ONLY the supplied "
                        "document passages. "
                        "Do not use outside knowledge. "
                        "Do not invent or assume facts that are not present "
                        "in the passages. "
                        "If the passages do not contain enough information "
                        "to answer the question, clearly say that the "
                        "document does not provide enough information. "
                        "For every factual claim, cite the supporting passage "
                        "using [Passage 1], [Passage 2], etc."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question:\n{question}"
                    )
                }
            ],
            temperature=0.2,
            max_tokens=600
        )

    except RateLimitError as e:
        raise RuntimeError(
            "Groq API rate limit reached. Please try again later."
        ) from e

    except APIConnectionError as e:
        raise RuntimeError(
            "Could not connect to Groq. Check your internet connection."
        ) from e

    except APIStatusError as e:
        raise RuntimeError(
            f"Groq API returned an error (HTTP {e.status_code})."
        ) from e


    return response.choices[0].message.content
