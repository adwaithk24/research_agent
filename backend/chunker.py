from pathlib import Path
from chunking_evaluation.chunking import RecursiveTokenChunker
from chunking_evaluation.utils import openai_token_count


CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50


def recursive_chunker(
    markdown_text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[str]:
    text_splitter = RecursiveTokenChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    docs = text_splitter.split_text(markdown_text)
    return docs


if __name__ == "__main__":
    text = ""
    input_file = Path(
        "backend/temp_processing/output/0e34658f-6083-4447-ba10-ea5ed9d9084c.md"
    )
    with open(input_file, "r") as f:
        text = f.read()

    chunks = recursive_chunker(text)
    print(len(chunks))
    print(chunks[0])
