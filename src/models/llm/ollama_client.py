"""Phase 3.2: deterministic local-LLM sentiment client (Ollama via LangChain).

Determinism: temperature 0.0 (enforced by config validation) + fixed sampler
seed + JSON output format. Any failure — connection, malformed JSON, values
out of range — returns None; callers map that to neutral 0.0. Never raises
into the batch loop.

``transport`` is injectable (callable: messages -> raw string) so the parsing
and failure paths are unit-testable without a running Ollama server.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from pydantic import BaseModel, Field

from src.common.config import Config, load_config
from src.models.llm.prompts import build_messages

Transport = Callable[[list[dict]], str]


class SentimentResult(BaseModel):
    sentiment: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


def _ollama_transport(cfg: Config) -> Transport:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,   # 0.0, config-enforced
        seed=cfg.llm.seed,
        num_predict=cfg.llm.max_tokens,
        format="json",
    )
    role_map = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}

    def call(messages: list[dict]) -> str:
        lc_messages = [role_map[m["role"]](content=m["content"]) for m in messages]
        return llm.invoke(lc_messages).content

    return call


class SentimentClient:
    def __init__(self, cfg: Config | None = None, transport: Transport | None = None):
        self.cfg = cfg or load_config()
        self._transport = transport or _ollama_transport(self.cfg)

    def score(self, text: str) -> SentimentResult | None:
        """One announcement -> validated SentimentResult, or None on ANY failure."""
        try:
            raw = self._transport(build_messages(text))
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            return SentimentResult.model_validate(json.loads(match.group(0)))
        except Exception:
            return None
