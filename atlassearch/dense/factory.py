from __future__ import annotations


def create_dense_retriever(backend: str, model_name: str = "intfloat/multilingual-e5-base"):
    backend = backend.strip().lower()
    if backend == "e5":
        from .e5_faiss import DenseRetriever
        return DenseRetriever(model_name)
    if backend == "hashing":
        from .hashing import HashingDenseRetriever
        return HashingDenseRetriever()
    raise ValueError("dense backend must be 'e5' or 'hashing'")
