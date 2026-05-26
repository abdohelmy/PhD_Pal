"""Configuration for the Python PhD course recommendation agent.

Change defaults here, or override them with environment variables when running
`python3 python_agent.py`.
"""

from __future__ import annotations

import os
from pathlib import Path


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_text(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def env_mode(name: str, default: str = "on") -> str:
    return os.getenv(name, default).lower()


ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
DATA_DIR = ROOT / "data"
COURSE_CACHE = DATA_DIR / "courses.json"
SUBSCRIBERS_FILE = DATA_DIR / "subscribers.json"
EMBEDDING_CACHE = DATA_DIR / "embeddings.json"
LLM_INSIGHTS_CACHE = DATA_DIR / "llm-insights.json"
OUTBOX_DIR = DATA_DIR / "outbox"

SOURCE_BASE = env_text("SOURCE_BASE", "https://phdcourses.dk")
USER_AGENT = env_text("USER_AGENT", "Danish PhD Course Recommendation Agent Python/0.1")
PORT = env_int("PORT", 3000)

HF_EMBEDDING_MODEL = env_text("HF_EMBEDDING_MODEL", "BAAI/bge-m3")
HF_FEATURE_ENDPOINT = (
    f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBEDDING_MODEL}"
)
EMBEDDING_ENDPOINT = env_text("EMBEDDING_ENDPOINT", HF_FEATURE_ENDPOINT)
EMBEDDING_API_KEY = env_text("EMBEDDING_API_KEY")

HF_REASONING_MODEL = env_text("HF_REASONING_MODEL", "Qwen/Qwen3-14B")
HF_CHAT_ENDPOINT = env_text("LLM_CHAT_ENDPOINT", "https://router.huggingface.co/v1/chat/completions")
LLM_API_KEY = env_text("LLM_API_KEY")

HF_TOKEN = env_text("HF_TOKEN") or env_text("HUGGINGFACE_HUB_TOKEN")

REASONING_REFRESH_DAYS = env_int("REASONING_REFRESH_DAYS", 21)
MAX_SCRAPE_PAGES = env_int("MAX_SCRAPE_PAGES", 120)
SCRAPE_FULL_DETAILS = env_mode("SCRAPE_FULL_DETAILS", "true")
COURSE_DETAIL_LIMIT = env_text("COURSE_DETAIL_LIMIT")

SEMANTIC_CANDIDATE_LIMIT = env_int("SEMANTIC_CANDIDATE_LIMIT", 60)
SEMANTIC_CANDIDATE_THRESHOLD = env_float("SEMANTIC_CANDIDATE_THRESHOLD", 0.25)
LLM_RERANK_CHUNK_SIZE = max(5, min(10, env_int("LLM_RERANK_CHUNK_SIZE", 8)))
LLM_RERANK_FINALIST_LIMIT = env_int("LLM_RERANK_FINALIST_LIMIT", 20)
LLM_RERANK_FINAL_PASS = env_mode("LLM_RERANK_FINAL_PASS", "true")
LLM_RERANK_MODE = env_mode("LLM_RERANK_MODE", "on")

PROFILE_INSIGHTS_MAX_TOKENS = env_int("PROFILE_INSIGHTS_MAX_TOKENS", 1200)
COURSE_SUMMARY_MAX_TOKENS = env_int("COURSE_SUMMARY_MAX_TOKENS", 1400)
LLM_RERANK_MAX_TOKENS = env_int("LLM_RERANK_MAX_TOKENS", 2200)
CLUSTER_INSIGHTS_MAX_TOKENS = env_int("CLUSTER_INSIGHTS_MAX_TOKENS", 1800)

LLM_JSON_THINKING_MODE = env_mode("LLM_JSON_THINKING_MODE", "off")
REASONING_MODE = env_mode("REASONING_MODE", "on")
RECOMMENDER_MODE = env_mode("RECOMMENDER_MODE", "semantic")
CLUSTER_REASONING_MODE = env_mode("CLUSTER_REASONING_MODE", "on")
