from abc import ABC, abstractmethod
import uuid
import chromadb
import chromadb.types
from chromadb.utils import embedding_functions
from chromadb.api.types import IncludeEnum
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_store")


class VectorStore(ABC):
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    TOP_K = 5

    @abstractmethod
    def get_or_create_collection(self, collection_name: str):
        pass

    @abstractmethod
    def add_chunks_to_collection(self, chunks: list[str]) -> int:
        pass

    @abstractmethod
    def retrieve_relevant_chunks(self, query: str, top_k: int = TOP_K) -> list[str]:
        pass


# class ChromaVectorStore(VectorStore):
#     PERSIST_DIRECTORY = "./chroma_db"
#     TOP_K = 5

#     def __init__(self, collection_name: str):
#         self.collection_name = collection_name
#         self.client, self.collection = self.get_or_create_collection(
#             self.collection_name
#         )

# def get_collection(self) -> chromadb.Collection:
#     return self.collection


def add_chunks_to_collection(collection: chromadb.Collection, chunks: list[str]) -> int:
    """
    Add document chunks to the ChromaDB collection.

    Args:
        chunks (list): List of text chunks
        url (str): Source URL of the document

    Returns:
        int: Number of chunks added
    """
    if not chunks:
        return 0
    ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(documents=chunks, ids=ids)
    return len(chunks)


def retrieve_relevant_chunks(
    collection: chromadb.Collection, query: str, top_k: int = 5
):
    """
    Retrieve the most relevant document chunks for a query.

    Args:
        query (str): The query text

    Returns:
        list: List of relevant document chunks
        list: List of source URLs for each chunk
    """

    # Query the collection for similar chunks
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=[
            IncludeEnum.documents,
            IncludeEnum.metadatas,
            IncludeEnum.distances,
        ],
    )

    chunks = results["documents"][0]  # First list is for the first query
    distances = results["distances"][0]

    # Print retrieval information for debugging
    # print(f"Retrieved {len(chunks)} chunks for query: '{query}'")
    # for i, (chunk, distance) in enumerate(zip(chunks, distances)):
    #     print("-" * 40)
    #     print(f"\nChunk {i+1} (Distance: {distance:.4f}):")
    #     preview = chunk
    #     print(preview)

    return chunks

    # def get_or_create_collection(self, collection_name: str):
    #     """
    #     Get or create a ChromaDB collection.

    #     Args:
    #         collection_name (str): The name of the collection to get or create.

    #     Returns:
    #         chromadb.Collection: The collection object.
    #     """
    #     client = chromadb.PersistentClient(path=self.PERSIST_DIRECTORY)
    #     embedding_func = embedding_functions.DefaultEmbeddingFunction()

    #     collection = client.get_or_create_collection(
    #         name=collection_name,
    #         embedding_function=embedding_func,
    #         metadata={"hnsw:space": "cosine"},
    #     )
    #     return client, collection

    # def delete_collection(self):
    #     self.client.delete_collection(self.collection_name)
