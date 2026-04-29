from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from openai import APIStatusError


def load_env() -> None:
    """
    Load environment variables from the repo root `.env` if present.

    This keeps the open-source code runnable without shell-side exports.
    """
    repo_root = Path(__file__).resolve().parents[3]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if (not s) or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def _strip_code_fences(txt: str) -> str:
    t = (txt or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t


def _invalid_prompt_log_path() -> Path:
    raw = (os.environ.get("CONSTRUM_INVALID_PROMPT_LOG") or "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[1] / "output" / "logs" / "invalid_prompt.jsonl"


def _is_invalid_prompt_error(exc: Exception) -> bool:
    if "invalid_prompt" in str(exc).lower():
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and str(err.get("code") or "") == "invalid_prompt":
            return True
    return False


def _log_invalid_prompt_event(
    exc: Exception,
    *,
    model: str,
    system: str,
    user: str,
    attempt: int,
) -> None:
    rid = getattr(exc, "request_id", None) if isinstance(exc, APIStatusError) else None
    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "attempt": int(attempt),
        "request_id": rid,
        "message": str(exc),
        "system_chars": len(system or ""),
        "user_chars": len(user or ""),
        "user_sha256_24": hashlib.sha256((user or "").encode("utf-8", "replace")).hexdigest()[:24],
    }
    path = _invalid_prompt_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _sanitize_text(s: str) -> str:
    """
    Best-effort ensure the string is valid UTF-8 without lone surrogates.
    This avoids rare request JSON encoding/decoding issues when upstream text
    contains invalid Unicode.
    """
    return (s or "").encode("utf-8", "replace").decode("utf-8")


class LLM:
    def __init__(self, *, api_key: Optional[str] = None) -> None:
        load_env()
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set (expected in repo root .env or env var).")

        from openai import OpenAI  # imported lazily to keep import errors clearer

        self._client = OpenAI(api_key=api_key)

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        timeout_s: Optional[float] = None,
        response_format_json_object: bool = True,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = 0.0,
    ) -> Any:
        """
        Call OpenAI chat completions and parse JSON.

        Notes:
        - Determinism is best-effort only (temperature=0).
        - Some models may not support response_format; we fall back to plain parsing.
        """
        params: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": _sanitize_text(system)},
                {"role": "user", "content": _sanitize_text(user)},
            ],
        }
        # Prefer structured outputs when available.
        if response_format_json_object:
            params["response_format"] = {"type": "json_object"}
        # Some models (e.g., gpt-5) may reject non-default temperature values.
        # We default to NOT sending temperature for gpt-5 unless explicitly enabled.
        send_temperature = True
        if str(model).startswith("gpt-5"):
            enable_temp = os.environ.get("CONSTRUM_ENABLE_TEMPERATURE_FOR_GPT5", "0").strip() in ("1", "true", "True")
            send_temperature = bool(enable_temp)
        if send_temperature and (temperature is not None):
            params["temperature"] = float(temperature)
        if max_output_tokens is not None:
            # Newer OpenAI APIs use max_completion_tokens; older may accept max_tokens.
            params["max_completion_tokens"] = int(max_output_tokens)

        if timeout_s is None:
            try:
                timeout_s = float(os.environ.get("CONSTRUM_TIMEOUT_S", "180"))
            except Exception:
                timeout_s = 180.0

        t0 = time.perf_counter()
        resp = None
        inv_try = 0
        try:
            inv_max = max(1, int(os.environ.get("CONSTRUM_INVALID_PROMPT_RETRIES", "3")))
        except Exception:
            inv_max = 3
        while resp is None:
            try:
                resp = self._client.chat.completions.create(timeout=timeout_s, **params)
            except Exception as e:
                if _is_invalid_prompt_error(e):
                    _log_invalid_prompt_event(
                        e, model=model, system=system, user=user, attempt=inv_try
                    )
                    inv_try += 1
                    if inv_try < inv_max:
                        time.sleep(min(30.0, 2.0 ** inv_try))
                        continue
                    raise
                msg = str(e)
                if "We could not parse the JSON body of your request" in msg:
                    resp = self._client.chat.completions.create(timeout=timeout_s, **params)
                elif (
                    "response_format" in msg
                    or "Unknown parameter" in msg
                    or "unrecognized" in msg
                ):
                    params_rf = {**params}
                    params_rf.pop("response_format", None)
                    resp = self._client.chat.completions.create(timeout=timeout_s, **params_rf)
                else:
                    raise
        _ = time.perf_counter() - t0

        txt = _strip_code_fences((resp.choices[0].message.content or "").strip())
        try:
            return json.loads(txt)
        except Exception:
            preview = (txt or "<EMPTY>")[:1200]
            raise ValueError(f"LLM JSON parse failed. Preview:\n{preview}")

