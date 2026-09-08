"""Model access for Step 5: one interface, three backends, full instrumentation.

Backends
--------
`anthropic`  direct API, needs ANTHROPIC_API_KEY
`bedrock`    AWS Bedrock, needs standard AWS credentials and a region
`replay`     serves previously-cached responses and refuses to make network
             calls; this is what makes the eval harness runnable in CI and by
             anyone cloning the repo without credentials

Every call is content-addressed and cached on disk. That is not just a cost
saving: it makes the whole Step 5 pipeline reproducible, so a reviewer re-runs
the evaluation and gets exactly the outputs that were scored, rather than
whatever the model happens to say today.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path


def _load_dotenv() -> None:
    """Read `.env` from the project root, without overriding a real env var.

    Credentials go in a gitignored file rather than the shell, so a key never
    has to be pasted into a terminal, a command history, or a transcript.
    Parsed by hand to keep the import graph free of a dependency that would
    otherwise be needed just to read seven lines.
    """
    root = Path(__file__).resolve().parents[3]
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

# Prices are OPERATOR-SUPPLIED, in USD per million tokens, and are only as
# current as whoever last edited this table. They are kept here rather than
# inferred so that a wrong cost number is a visible config error rather than a
# silent fiction in the report. Verify against the vendor pricing page before
# quoting these figures anywhere.
PRICE_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-5":            {"input": 5.00, "output": 25.00},
    "claude-sonnet-5":          {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}
DEFAULT_MODEL = "claude-sonnet-5"


@dataclass
class Call:
    """One model call, with everything needed to cost and audit it."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool = False
    stop_reason: str = ""
    attempts: int = 1
    error: str | None = None

    @property
    def cost_usd(self) -> float:
        p = PRICE_PER_MTOK.get(self.model)
        if p is None:
            return float("nan")
        return (self.input_tokens * p["input"] + self.output_tokens * p["output"]) / 1e6

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cost_usd"] = self.cost_usd
        return d


class ProviderError(RuntimeError):
    pass


@dataclass
class Provider:
    """Cached, instrumented text generation.

    `temperature` defaults to 0. Generation is a production artefact here, not
    a creative exercise, and a non-deterministic generator would make the
    judge scores unreproducible for no benefit.
    """

    backend: str = "anthropic"
    model: str = DEFAULT_MODEL
    cache_dir: Path = Path("data/genai_cache")
    temperature: float = 0.0
    max_tokens: int = 1200
    max_retries: int = 4
    region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    _client: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- cache ------------------------------------------------------------
    def _key(self, system: str, prompt: str) -> str:
        blob = json.dumps(
            {"b": "any", "m": self.model, "s": system, "p": prompt,
             "t": self.temperature, "mt": self.max_tokens},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def _cached(self, key: str) -> Call | None:
        f = self.cache_dir / f"{key}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text(encoding="utf-8"))
        d.pop("cost_usd", None)
        return Call(**{**d, "cached": True})

    def _store(self, key: str, call: Call) -> None:
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(call.as_dict(), indent=2), encoding="utf-8"
        )

    # -- clients ----------------------------------------------------------
    def _anthropic(self):
        if self._client is None:
            import anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ProviderError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic()
        return self._client

    def _bedrock(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    # -- generation -------------------------------------------------------
    def complete(self, prompt: str, system: str = "") -> Call:
        key = self._key(system, prompt)
        hit = self._cached(key)
        if hit is not None:
            return hit
        if self.backend == "replay":
            raise ProviderError(
                f"replay backend has no cached response for {key}. "
                "Run once with --backend anthropic|bedrock to populate data/genai_cache/."
            )

        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                call = self._dispatch(prompt, system, attempt)
                call.latency_s = time.perf_counter() - t0
                self._store(key, call)
                return call
            except Exception as exc:  # noqa: BLE001 - retried and surfaced below
                last = exc
                if attempt == self.max_retries or not _retryable(exc):
                    break
                # Exponential backoff. Overload and rate-limit responses are
                # the common case and are transient.
                time.sleep(min(2.0**attempt, 30.0))
        raise ProviderError(f"generation failed after {self.max_retries} attempts: {last}")

    def _dispatch(self, prompt: str, system: str, attempt: int) -> Call:
        if self.backend == "anthropic":
            client = self._anthropic()
            kw = dict(model=self.model, max_tokens=self.max_tokens,
                      temperature=self.temperature,
                      messages=[{"role": "user", "content": prompt}])
            if system:
                kw["system"] = system
            r = client.messages.create(**kw)
            return Call(
                text="".join(b.text for b in r.content if b.type == "text"),
                model=self.model, input_tokens=r.usage.input_tokens,
                output_tokens=r.usage.output_tokens, latency_s=0.0,
                stop_reason=r.stop_reason or "", attempts=attempt,
            )

        if self.backend == "bedrock":
            client = self._bedrock()
            body = {"anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens, "temperature": self.temperature,
                    "messages": [{"role": "user", "content": prompt}]}
            if system:
                body["system"] = system
            r = client.invoke_model(modelId=self.model, body=json.dumps(body))
            payload = json.loads(r["body"].read())
            u = payload.get("usage", {})
            return Call(
                text="".join(b.get("text", "") for b in payload.get("content", [])),
                model=self.model, input_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0), latency_s=0.0,
                stop_reason=payload.get("stop_reason", ""), attempts=attempt,
            )

        raise ProviderError(f"unknown backend {self.backend!r}")


def _retryable(exc: Exception) -> bool:
    """Overload, rate limit, timeout and 5xx are transient; 4xx is not."""
    name = type(exc).__name__.lower()
    if any(k in name for k in ("overloaded", "ratelimit", "timeout", "connection",
                               "apistatus", "internalserver", "serviceunavailable",
                               "throttling")):
        return True
    status = getattr(exc, "status_code", None)
    return status is not None and (status == 429 or status >= 500)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response.

    Models wrap JSON in prose or fences often enough that parsing the whole
    string is not viable, and a silent parse failure in an eval harness turns
    into a fabricated score. This raises instead.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    return json.loads(s[start : end + 1])
