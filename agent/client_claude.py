from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class ClaudeNewAPIClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "claude-opus-4-6",
        timeout_sec: int = 90,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/messages"
        self.model = model
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2025-06-01",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        reraise=True,
    )
    def create_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 1200,
        effort: str = "medium",  # low/medium/high/max
        stream: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "stream": stream,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
        if tools:
            payload["tools"] = tools

        resp = self.session.post(
            self.endpoint,
            headers=self.headers,
            json=payload,
            timeout=self.timeout_sec,
        )
        try:
            data = resp.json()
        except Exception:
            resp.raise_for_status()
            raise RuntimeError(f"Non-JSON response: {resp.text[:1000]}")

        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {data}")
        return data