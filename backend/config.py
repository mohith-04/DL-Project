from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class Settings(BaseSettings):
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_model_primary: str = "openai/gpt-oss-120b"
    llm_model_fallback: str = "qwen/qwen3.8-27b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_per_source: int = 3
    max_sources: int = 6  # supports hybrid URL + discovered sources + local files
    faiss_index_dir: str = str(BASE_DIR / "faiss_indices")

    class Config:
        env_file = str(BASE_DIR / ".env")

@lru_cache()
def get_settings() -> Settings:
    return Settings()
