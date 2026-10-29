"""
Resilient LLM Client with Multi-Key Pool, Smart Rate-Limit Backoff, and Context Optimization.

Features:
  1. Multi-Key Rotation: Automatically switches Groq keys when daily quota is reached.
  2. Smart 429 Backoff: Automatically extracts Groq's wait-time and sleeps before retrying.
  3. Token Trimming: Reduces prompt tokens by ~50-60% to preserve quota.
"""
import json
import re
import time
from typing import Dict, List, Optional, Any
from groq import Groq, RateLimitError
from config import get_settings

settings = get_settings()


class GroqPoolManager:
    """Manages rotation and state for multiple Groq API keys."""

    def __init__(self):
        self.keys = settings.get_groq_keys()
        if not self.keys:
            self.keys = [settings.groq_api_key]
        self.current_idx = 0
        self.exhausted_keys = set()
        print(f"[llm_client] Initialized Groq pool with {len(self.keys)} key(s).")

    @property
    def current_key(self) -> str:
        return self.keys[self.current_idx]

    def get_client(self) -> Groq:
        return Groq(api_key=self.current_key)

    def rotate_key(self, reason: str = "rate_limit") -> bool:
        """Rotate to next available key in the pool. Returns True if rotated, False if all exhausted."""
        if len(self.keys) <= 1:
            return False

        old_key_masked = self.current_key[:8] + "..." + self.current_key[-4:]
        self.exhausted_keys.add(self.current_idx)

        # Find next non-exhausted key
        for step in range(1, len(self.keys)):
            next_idx = (self.current_idx + step) % len(self.keys)
            if next_idx not in self.exhausted_keys:
                self.current_idx = next_idx
                new_key_masked = self.current_key[:8] + "..." + self.current_key[-4:]
                print(f"[llm_client] 🔄 Rotated API key ({reason}): {old_key_masked} -> {new_key_masked}")
                return True

        # If all keys were marked exhausted, reset cycle
        print("[llm_client] ⚠️ All Groq keys have encountered limits. Resetting rotation cycle.")
        self.exhausted_keys.clear()
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        return True


# Global pool manager
_groq_pool = GroqPoolManager()


def parse_wait_time(error_str: str, default_wait: float = 60.0) -> float:
    """Extract wait seconds from Groq rate limit message (e.g. 'try again in 48.2s')."""
    match = re.search(r"try again in ([\d\.]+)s", error_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + 2.0  # +2s safety buffer
        except Exception:
            pass

    match_ms = re.search(r"try again in ([\d\.]+)ms", error_str, re.IGNORECASE)
    if match_ms:
        try:
            return (float(match_ms.group(1)) / 1000.0) + 2.0
        except Exception:
            pass

    return default_wait


def is_daily_limit(error_str: str) -> bool:
    """Check if the error is a Daily Quota Limit (TPD / RPD)."""
    lower = error_str.lower()
    return "day" in lower or "daily" in lower or "tpd" in lower or "rpd" in lower


def safe_chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1500,
    response_format: Optional[Dict[str, str]] = None,
    max_retries: int = 6,
) -> Any:
    """
    Execute chat completion with automatic 429 rate limit backoff and multi-key rotation.
    """
    target_model = model or settings.llm_model_primary

    for attempt in range(max_retries):
        client = _groq_pool.get_client()
        try:
            kwargs = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format

            resp = client.chat.completions.create(**kwargs)
            return resp

        except RateLimitError as e:
            err_msg = str(e)
            print(f"[llm_client] ⚠️ Rate limit on {target_model} (attempt {attempt + 1}/{max_retries}): {err_msg}")

            # 1. If daily limit and multiple keys exist, rotate immediately
            if is_daily_limit(err_msg):
                rotated = _groq_pool.rotate_key(reason="daily_quota_limit")
                if rotated:
                    time.sleep(2.0)
                    continue

            # 2. Extract required cooldown duration
            wait_sec = parse_wait_time(err_msg, default_wait=45.0)
            # Cap wait time to max 75 seconds per attempt
            wait_sec = min(wait_sec, 75.0)
            print(f"[llm_client] ⏸ Pausing for {wait_sec:.1f}s to reset token rate limit...")
            time.sleep(wait_sec)

        except Exception as e:
            err_msg = str(e)
            # If 429 caught as general exception
            if "429" in err_msg or "rate limit" in err_msg.lower():
                wait_sec = parse_wait_time(err_msg, default_wait=45.0)
                if is_daily_limit(err_msg):
                    _groq_pool.rotate_key(reason="daily_limit")
                print(f"[llm_client] ⏸ 429 encountered. Pausing for {wait_sec:.1f}s...")
                time.sleep(wait_sec)
            else:
                # If model is unavailable, attempt fallback model
                if target_model == settings.llm_model_primary and settings.llm_model_fallback:
                    print(f"[llm_client] Falling back from {target_model} to {settings.llm_model_fallback}: {err_msg}")
                    target_model = settings.llm_model_fallback
                    time.sleep(1.0)
                    continue
                raise e

    raise RuntimeError(f"Failed chat completion after {max_retries} retries due to persistent rate limits.")


def build_compact_context(
    per_source_chunks: Dict[str, List[Dict[str, Any]]],
    max_chars_per_source: int = 1200,
) -> str:
    """
    Build context with token optimization: keeps top chunks per source and
    caps total characters so token budget is not exceeded.
    """
    context_parts = []
    for source_id, chunks in per_source_chunks.items():
        if not chunks:
            continue

        # Sort chunks by score descending if score available
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)

        combined_text = ""
        for c in sorted_chunks:
            text = c.get("text", "").strip()
            if len(combined_text) + len(text) <= max_chars_per_source:
                combined_text += ("\n" + text) if combined_text else text
            else:
                # Fill remaining budget
                remaining = max_chars_per_source - len(combined_text)
                if remaining > 100:
                    combined_text += ("\n" + text[:remaining])
                break

        if combined_text:
            context_parts.append(f"=== {source_id} ===\n{combined_text.strip()}")

    return "\n\n".join(context_parts)
