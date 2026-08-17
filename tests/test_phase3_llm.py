"""Phase 3 gate: sentiment client parsing/failure paths and prompt hygiene.

All tests use an injected fake transport — no Ollama server required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config
from src.models.llm.ollama_client import SentimentClient
from src.models.llm.prompts import build_messages

cfg = load_config()


def client(reply: str) -> SentimentClient:
    return SentimentClient(cfg, transport=lambda messages: reply)


def test_valid_json_parses():
    r = client('{"sentiment": -0.6, "confidence": 0.8}').score("plant fire")
    assert r is not None and r.sentiment == -0.6 and r.confidence == 0.8


def test_json_embedded_in_chatter_parses():
    r = client('Sure! Here is the score: {"sentiment": 0.5, "confidence": 0.9} hope that helps').score("order win")
    assert r is not None and r.sentiment == 0.5


def test_garbage_returns_none():
    assert client("I cannot score this announcement.").score("x") is None


def test_out_of_range_sentiment_rejected():
    assert client('{"sentiment": 3.5, "confidence": 0.9}').score("x") is None


def test_missing_field_rejected():
    assert client('{"sentiment": 0.2}').score("x") is None


def test_transport_exception_returns_none():
    def boom(messages):
        raise ConnectionError("ollama not running")
    assert SentimentClient(cfg, transport=boom).score("x") is None


def test_prompt_truncates_long_text():
    msgs = build_messages("A" * 50_000)
    assert len(msgs[-1]["content"]) <= 2000
    assert msgs[0]["role"] == "system"


def test_determinism_config_enforced():
    assert cfg.llm.temperature == 0.0
