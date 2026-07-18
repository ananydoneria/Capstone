"""Phase 8a: daily paper-trading loop — evening, post-close. NO REAL ORDERS.

Each run (once per trading day, after ~18:30 IST when EOD data settles):
  1. Executes YESTERDAY's pending recommendation at TODAY's open (the fill the
     agent was trained to expect), updating the hypothetical portfolio.
  2. Fetches fresh bars (yfinance) + today's NSE filings, computes today's
     features, one GNN forward, LLM-scores the filings (neutral if Ollama is
     unreachable or --no-llm).
  3. Feeds the trained PPO policy today's state -> target weights for
     TOMORROW's open, printed as a human-readable order list.
  4. Appends equity + weights to the forward-test ledger.

Artifacts (git-ignored) in src/rl_agent/logs/paper/:
  paper_state.json   hypothetical cash/shares + the pending recommendation
  ledger.csv         one row per run: date, equity, weights, order summary

    python scripts/paper_trade.py [--ablation full] [--no-llm]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from src.common.config import Config, load_config
from src.common.seeding import set_global_seed
from src.data_pipeline.features import build_features
from src.data_pipeline.ohlcv_ingest import OHLCV_COLS, build_panel
from src.data_pipeline.relationships import build_graph, to_edge_index
from src.env.exchange_env import ACTION_LOGIT_SCALE
from src.env.portfolio import Portfolio
from src.models.gnn.graph_dataset import features_tensor, supervision_pairs
from src.models.gnn.model import PropagationGNN
from src.models.gnn.train import WEIGHTS_DIR
from src.rl_agent.train_ppo import LOGS_DIR

PAPER_DIR = LOGS_DIR / "paper"
LOOKBACK_DAYS = 400  # calendar days of bars: covers 63d warmup with margin


# --------------------------------------------------------------- data (today)
def fetch_recent_panel(cfg: Config) -> pd.DataFrame:
    frames = {}
    start = (pd.Timestamp.now() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    for ticker in cfg.universe.tickers:
        df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"no recent data for {ticker}")
        df.index = df.index.tz_localize(None).normalize()
        df.index.name = "date"
        frames[ticker] = df[OHLCV_COLS].copy()
    return build_panel(frames, cfg)


def todays_sentiment(cfg: Config, today: pd.Timestamp, no_llm: bool) -> np.ndarray:
    neutral = np.full(len(cfg.universe.tickers), cfg.llm.neutral_sentiment, dtype=np.float32)
    if no_llm:
        return neutral
    try:
        from src.models.llm.announcements_ingest import API, HEADERS, parse_announcement_date
        from src.models.llm.ollama_client import SentimentClient
        import requests

        sess = requests.Session()
        sess.headers.update(HEADERS)
        client = SentimentClient(cfg)
        out = neutral.copy()
        for i, ticker in enumerate(cfg.universe.tickers):
            resp = sess.get(API, params={
                "index": "equities", "symbol": ticker.replace(".NS", ""),
                "from_date": f"{today:%d-%m-%Y}", "to_date": f"{today:%d-%m-%Y}",
            }, timeout=20)
            items = resp.json() if resp.status_code == 200 else []
            scores = []
            for item in items if isinstance(items, list) else []:
                text = " | ".join(str(item.get(k, "")) for k in ("desc", "attchmntText") if item.get(k))
                if text and parse_announcement_date(item.get("an_dt", "")) == today:
                    r = client.score(text)
                    if r is not None:
                        scores.append(r.sentiment)
            if scores:
                out[i] = float(np.mean(scores))
        return out
    except Exception as err:
        print(f"WARN sentiment unavailable ({type(err).__name__}) -> neutral")
        return neutral


def todays_gnn(cfg: Config, features: pd.DataFrame) -> np.ndarray:
    tickers = cfg.universe.tickers
    _, x = features_tensor(features, tickers)
    norm = torch.load(WEIGHTS_DIR / "feature_norm.pt", weights_only=True)
    x = (x[-1] - norm["mean"]) / norm["std"]
    ei, _ = to_edge_index(build_graph(cfg), tickers, symmetrize=True)
    pos = {t: i for i, t in enumerate(tickers)}
    pairs = torch.tensor([[pos[u], pos[v]] for u, v in supervision_pairs(cfg)])
    model = PropagationGNN(in_dim=x.shape[-1], cfg=cfg)
    model.load_state_dict(torch.load(WEIGHTS_DIR / "propagation_gnn.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, torch.tensor(ei), pairs)).numpy().astype(np.float32)


# ------------------------------------------------------------------- policy
def load_policy(ablation: str):
    from stable_baselines3 import PPO

    run_dir = LOGS_DIR / f"ppo_{ablation}"
    if not (run_dir / "model.zip").exists():
        sys.exit(f"No trained model at {run_dir} — train first "
                 f"(scripts/train_ppo.py --ablation {ablation}), see HOWTORUN.txt")
    with open(run_dir / "vecnormalize.pkl", "rb") as fh:
        vecnorm = pickle.load(fh)
    return PPO.load(run_dir / "model.zip"), vecnorm


def normalize_obs(obs: np.ndarray, vecnorm) -> np.ndarray:
    rms = vecnorm.obs_rms
    return np.clip((obs - rms.mean) / np.sqrt(rms.var + vecnorm.epsilon),
                   -vecnorm.clip_obs, vecnorm.clip_obs).astype(np.float32)


def action_to_weights(action: np.ndarray) -> np.ndarray:
    logits = np.clip(action, -1, 1) * ACTION_LOGIT_SCALE
    w = np.exp(logits - logits.max())
    w /= w.sum()
    return w[:-1]  # last bucket = cash


# -------------------------------------------------------------------- state
def load_paper_state(cfg: Config) -> dict:
    f = PAPER_DIR / "paper_state.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"cash": cfg.env.initial_cash_inr,
            "shares": [0.0] * len(cfg.universe.tickers),
            "pending": None, "last_run": None}


def save_paper_state(state: dict) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_DIR / "paper_state.json").write_text(json.dumps(state, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip filing sentiment (neutral) — e.g. no Ollama on this box")
    args = parser.parse_args()

    cfg = load_config()
    set_global_seed(cfg.project.seed)
    model, vecnorm = load_policy(args.ablation)   # fail fast before network work

    panel = fetch_recent_panel(cfg)
    features = build_features(panel, cfg)
    today = features.index.get_level_values("date").max()
    tickers = cfg.universe.tickers
    open_px = panel["Open"].unstack("ticker").loc[today, tickers].to_numpy()
    close_px = panel["Close"].unstack("ticker").loc[today, tickers].to_numpy()

    state = load_paper_state(cfg)
    if state["last_run"] == str(today.date()):
        sys.exit(f"already ran for {today.date()} — one run per trading day")

    portfolio = Portfolio(cfg.env.initial_cash_inr, len(tickers))
    portfolio.cash = state["cash"]
    portfolio.shares = np.array(state["shares"], dtype=np.float64)

    # 1. fill yesterday's pending recommendation at today's open
    if state["pending"] is not None:
        pending = np.array(state["pending"]["weights"])
        turnover = portfolio.rebalance(pending, open_px, cfg.env.costs)
        print(f"filled pending orders from {state['pending']['decision_date']} "
              f"at today's open (turnover INR {turnover:,.0f})")

    equity = portfolio.equity(close_px)

    # 2-3. today's state -> policy -> tomorrow's target weights
    gnn = todays_gnn(cfg, features)
    if args.ablation in ("no-gnn", "neither"):
        gnn = np.zeros_like(gnn)
    sent = todays_sentiment(cfg, today, args.no_llm or args.ablation in ("no-sentiment", "neither"))
    obs = np.concatenate([
        features.loc[today].loc[tickers].to_numpy().reshape(-1),
        gnn, sent, portfolio.weights(close_px),
        [portfolio.cash / equity if equity > 0 else 1.0],
    ]).astype(np.float32)
    action, _ = model.predict(normalize_obs(obs, vecnorm), deterministic=True)
    target = action_to_weights(np.asarray(action).ravel())

    # 4. report + persist
    print(f"\n=== PAPER TRADE {today.date()} (model: ppo_{args.ablation}) — NOT REAL ORDERS ===")
    print(f"equity: INR {equity:,.0f}   cash: {portfolio.cash / equity:.1%}")
    print(f"{'ticker':14s} {'now':>7s} {'target':>7s} {'delta INR':>12s}")
    current_w = portfolio.weights(close_px)
    for i, t in enumerate(tickers):
        delta = (target[i] - current_w[i]) * equity
        if abs(delta) > equity * 0.002:
            print(f"{t:14s} {current_w[i]:6.1%} {target[i]:6.1%} {delta:+12,.0f}")
    print(f"{'CASH':14s} {portfolio.cash / equity:6.1%} {1 - target.sum():6.1%}")

    state.update({
        "cash": portfolio.cash,
        "shares": portfolio.shares.tolist(),
        "pending": {"decision_date": str(today.date()), "weights": target.tolist()},
        "last_run": str(today.date()),
    })
    save_paper_state(state)

    ledger = PAPER_DIR / "ledger.csv"
    header = not ledger.exists()
    with open(ledger, "a", encoding="utf-8") as fh:
        if header:
            fh.write("run_ts,decision_date,equity_inr,cash_frac,target_weights\n")
        fh.write(f"{datetime.now():%Y-%m-%d %H:%M},{today.date()},{equity:.2f},"
                 f"{portfolio.cash / equity:.4f},\"{[round(w, 4) for w in target]}\"\n")
    print(f"\nledger updated -> {ledger}")


if __name__ == "__main__":
    main()
