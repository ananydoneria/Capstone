"""Prompt for deterministic sentiment normalization to [-1.0, +1.0]."""

SYSTEM = """You are a financial analyst scoring Indian stock-exchange corporate \
announcements for their likely near-term impact on the announcing company's stock.

Respond with ONLY a JSON object, no other text:
{"sentiment": <float -1.0 to 1.0>, "confidence": <float 0.0 to 1.0>}

Calibration:
 -1.0  severe negative (fraud, default, plant shutdown, major recall)
 -0.5  clearly negative (profit warning, lost contract, downgrade, strike)
  0.0  routine/neutral (AGM notice, record date, trading-window closure,
       investor-meet intimation, clarification with no new information)
 +0.5  clearly positive (strong results, large order win, capacity expansion)
 +1.0  transformational positive (huge contract, breakthrough approval, buyout premium)

Most exchange filings are routine: prefer 0.0 unless the text carries real
economic news. "confidence" reflects how unambiguous the text is."""

FEW_SHOT: list[tuple[str, str]] = [
    (
        "Intimation of closure of trading window pursuant to SEBI (Prohibition of "
        "Insider Trading) Regulations, 2015.",
        '{"sentiment": 0.0, "confidence": 0.95}',
    ),
    (
        "The Company has received a large order worth Rs 1,200 crore from a leading "
        "European OEM for supply of EV differential assemblies over 5 years.",
        '{"sentiment": 0.7, "confidence": 0.85}',
    ),
    (
        "Operations at the Company's Pune forging plant have been suspended following "
        "a fire incident. Assessment of damage is underway; insurance claim initiated.",
        '{"sentiment": -0.6, "confidence": 0.8}',
    ),
]


def build_messages(text: str, max_chars: int = 2000) -> list[dict]:
    """Chat messages: system + few-shot pairs + the announcement to score."""
    msgs: list[dict] = [{"role": "system", "content": SYSTEM}]
    for user, assistant in FEW_SHOT:
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": assistant})
    msgs.append({"role": "user", "content": text[:max_chars]})
    return msgs
