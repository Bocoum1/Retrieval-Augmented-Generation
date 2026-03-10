from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "phi3"

    chunk_size: int = 900
    chunk_overlap: int = 150

    top_k_vector: int = 10
    top_k_bm25: int = 10
    top_k_fused: int = 10
    top_k_rerank: int = 4

    data_dir: Path = Path("data")
    index_dir: Path = Path("data/indexes")
    upload_dir: Path = Path("data/uploads")

    max_context_chars: int = 7000

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()