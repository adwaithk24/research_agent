import os
from sentence_transformers import SentenceTransformer
from chunking_evaluation.chunking import RecursiveTokenChunker
from chunking_evaluation.utils import openai_token_count
from pinecone import Pinecone, ServerlessSpec

def read_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def chunk_document(text, chunk_size=400, chunk_overlap=50):
    chunker = RecursiveTokenChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=openai_token_count,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    return chunker.split_text(text)

def embed_texts(texts, model_name="all-MiniLM-L6-v2"):
    embedding_model = SentenceTransformer(model_name)
    return embedding_model.encode(texts)

# Load and process file
file_path = "nvidia_2024.md"
content = read_markdown_file(file_path)
chunks = chunk_document(content)

# Create embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_texts(chunks)

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "document-embeddings"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(index_name)

# Prepare and upsert vectors
vectors = [
    {"id": f"chunk-{i}", "values": embedding.tolist(), "metadata": {"text": chunk}}
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
]

index.upsert(vectors=vectors)

print(f"Loaded {len(chunks)} chunks into Pinecone index '{index_name}'.")
