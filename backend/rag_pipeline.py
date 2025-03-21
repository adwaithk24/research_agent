from typing import Literal

import chromadb

from chunker import recursive_chunker
from vector_store import add_chunks_to_collection, retrieve_relevant_chunks


class RAGPipeline:
    def __init__(
        self,
        pdf_id: str,
        text: str,
        vector_store: Literal["chroma", "pinecone", "naive"],
        chunking_strategy: Literal["recursive", "semantic", "fixed"],
    ):
        self.pdf_id = pdf_id
        self.vector_store = vector_store
        self.chunking_strategy = chunking_strategy
        self.text = text

    def process(self):
        chunks = []
        match self.chunking_strategy:
            case "recursive":
                chunks = recursive_chunker(self.text)
            case _:
                raise ValueError(
                    f"Unsupported chunking strategy: {self.chunking_strategy}"
                )

        match self.vector_store:
            case "chroma":
                client = chromadb.PersistentClient(path="./backend/chroma_db")
                collection = client.get_or_create_collection(self.pdf_id)
                add_chunks_to_collection(collection, chunks)
            case "pinecone":
                pass
            case "naive":
                pass

    def get_relevant_chunks(self, query: str, k: int = 5):
        match self.vector_store:
            case "chroma":
                client = chromadb.PersistentClient(path=".chroma_db")
                collection = client.get_or_create_collection(self.pdf_id)
                return retrieve_relevant_chunks(collection, query, k)
            case _:
                raise ValueError(f"Unsupported vector store: {self.vector_store}")
