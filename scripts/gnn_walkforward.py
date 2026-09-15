"""Walk-forward evaluation of GNN variants (legacy vs extended data and inputs).

Every variant is judged on the SAME out-of-sample test windows, with labels
from the corrected (split/demerger-adjusted) price history and the shock
definition in config.yaml (gnn.shock_basis / shock_vol_window / shock_z /
shock_min_move). Two no-learning rules are scored on the identical samples:
the pair's trailing 2y return correlation, and "lower previous-day 21d
volatility of the destination" — the shortcut that beat every GNN under the
old raw-return / 21d-sigma label. A volatility-stratified AUC (mean AUC
inside quintiles of that volatility) shows what each model adds beyond it.

    folds (test windows): 2023H2, 2024H1, 2024H2, 2025H1, 2025H2
    for each fold:  train on everything before it (expanding window; the last
                    k days dropped so no label window crosses into the test),
                    early-stop on the last 15% of that training period,
                    score the fold. 2026 (the PPO test window) is never used.

Variants differ only in config overrides (see VARIANTS). Seeds are averaged
two ways: per-seed pooled AUC, and a seed-ensemble (mean logit) AUC.
Confidence intervals come from a day-block bootstrap (all pairs scored on the
same shock day are resampled together — they are not independent).

    python scripts/gnn_walkforward.py                   # all variants, seeds 42 7
    python scripts/gnn_walkforward.py --variants V0 V4 --seeds 42 --jobs 4

Each (variant, seed, fold) run is checkpointed to reports/gnn_walkforward/runs/
the moment it finishes; rerunning the same command skips finished runs, so a
crash or kill loses at most the runs in flight. --fresh discards checkpoints
(do that whenever the input data changes).

Output: reports/gnn_walkforward/{predictions.parquet, results.json, WALKFORWARD.md}
config.yaml and the shipped weights are never modified.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "reports" / "gnn_walkforward"
RUNS_DIR = OUT_DIR / "runs"
FOLDS = [("2023H2", "2023-07-01", "2023-12-31"), ("2024H1", "2024-01-01", "2024-06-30"),
         ("2024H2", "2024-07-01", "2024-12-31"), ("2025H1", "2025-01-01", "2025-06-30"),
         ("2025H2", "2025-07-01", "2025-12-31")]
INNER_VAL_FRACTION = 0.15
HIST = {"history_start": "2011-10-01"}
PAIR = {"pair_features": ["rolling_corr", "edge_weight", "direction"]}
VARIANTS = {
    # V0 = the previous model's shape (2021+ data, GraphSAGE, no pair features).
    # It runs on the history data rather than features.parquet because the
    # excess-return label needs the Nifty Auto series; inputs are otherwise
    # identical apart from the constant 'present' channel.
    "V0": ("previous model shape (2021+ data, SAGE, no pair features)", {"history_start": "2021-07-01"}),
    "V1": ("long history 2011+ (adjusted, masked)", {**HIST}),
    "V2": ("V1 + edge weights (GraphConv)", {**HIST, "arch": "graphconv"}),
    "V3": ("V1 + pair features", {**HIST, **PAIR}),
    "V4": ("V1 + edge weights + pair features (idea 2)", {**HIST, "arch": "graphconv", **PAIR}),
    "V5": ("V1 + India context + breadth", {**HIST, "market_context": "india", "breadth": True}),
    "V6": ("V1 + full context + breadth, PCA-8", {**HIST, "market_context": "full", "breadth": True, "ctx_pca": 8}),
    "V7": ("V4 + India context + breadth", {**HIST, "arch": "graphconv", **PAIR, "market_context": "india", "breadth": True}),
    "V8": ("V4 + full context + breadth, PCA-8", {**HIST, "arch": "graphconv", **PAIR, "market_context": "full", "breadth": True, "ctx_pca": 8}),
    "V9": ("V4 + full context + breadth (raw 151)", {**HIST, "arch": "graphconv", **PAIR, "market_context": "full", "breadth": True}),
}


# Variant-controlled fields are reset to the legacy model's values before a
# variant's overrides are applied, so a variant means the same thing whatever
# config.yaml currently ships (V1 = "history only" even after V4 was adopted).
LEGACY = {"arch": "graphsage", "history_start": None, "market_context": "none", "breadth": False,
          "pair_features": [], "ctx_pca": 0}


def make_cfg(overrides: dict, seed: int):
    from src.common.config import Config, load_config
    raw = load_config().model_dump()
    raw["gnn"].update({**LEGACY, **overrides})
    raw["project"]["seed"] = seed
    return Config.model_validate(raw)


BASELINES = {
    "neg_dst_vol": "rule: lower previous-day 21d volatility of dst",
    "corr_2y": "rule: trailing 2y return correlation of the pair",
}


def reference_samples():
    """Evaluation samples (corrected-history labels, V1 data, fold windows) and
    the no-learning baseline scores for each of them."""
    from src.models.gnn.graph_dataset import all_samples, raw_inputs
    cfg = make_cfg(HIST, 42)
    raw = raw_inputs(cfg)
    samples = all_samples(cfg, raw)
    vol_prev = raw["vol"].shift(1)
    out, rows = {}, []
    for name, lo, hi in FOLDS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        out[name] = [s for s in samples if lo <= s.date <= hi]
        corr = raw["ret1"].loc[lo - pd.Timedelta(days=730): lo - pd.Timedelta(days=10)].corr()
        for s in out[name]:
            rows.append({"fold": name, "date": s.date, "src": s.src, "dst": s.dst,
                         "neg_dst_vol": -float(vol_prev.loc[s.date, s.dst]),
                         "corr_2y": float(corr.loc[s.src, s.dst])})
    return out, pd.DataFrame(rows)


def split_fold(samples, dates: list, test_start: pd.Timestamp, k: int):
    pos = {d: i for i, d in enumerate(dates)}
    first_test = next(i for i, d in enumerate(dates) if d >= test_start)
    pool = [s for s in samples if pos[s.date] < first_test - k]
    udates = sorted({s.date for s in pool})
    cut = int(len(udates) * (1 - INNER_VAL_FRACTION))
    tr_dates, va_dates = set(udates[: max(cut - k, 1)]), set(udates[cut:])
    return [s for s in pool if s.date in tr_dates], [s for s in pool if s.date in va_dates]


def run_path(vname: str, seed: int, fname: str) -> Path:
    return RUNS_DIR / f"{vname}_s{seed}_{fname}.parquet"


def run_job(job: tuple) -> str:
    import torch
    torch.set_num_threads(1)
    from src.models.gnn.graph_dataset import _graph_data, all_samples, apply_stats, compute_stats, raw_inputs
    from src.models.gnn.train import _batch_by_day, _epoch_scores, fit, make_model

    vname, seed, fname, lo, test = job
    cfg = make_cfg(VARIANTS[vname][1], seed)
    raw = raw_inputs(cfg)
    samples = all_samples(cfg, raw)
    k, W = cfg.gnn.cascade_window_days, cfg.gnn.temporal_window_days
    t0 = time.time()
    tr, va = split_fold(samples, raw["dates"], pd.Timestamp(lo), k)
    stats = compute_stats(raw, {s.date for s in tr})
    x, ctx = apply_stats(cfg, raw, stats)
    data = _graph_data(cfg, raw, x, ctx, stats, tr, va)
    best = fit(cfg, data, tr, va, verbose=False)
    model = make_model(cfg, data)
    model.load_state_dict(best["state"])
    model.eval()
    test = [s for s in test if s.date in data._date_pos]
    with torch.no_grad():
        logits, labels = _epoch_scores(model, data, _batch_by_day(data, test), W)
    ordered = sorted(test, key=lambda s: s.date)
    rows = []
    for s, lg, lb in zip(ordered, logits.tolist(), labels.tolist()):
        rows.append({"variant": vname, "seed": seed, "fold": fname, "date": s.date,
                     "src": s.src, "dst": s.dst, "label": int(lb), "logit": lg,
                     "best_epoch": best["epoch"], "inner_val_auc": best["val_auc"],
                     "train_auc": best["train_auc"], "n_train": len(tr)})
    path = run_path(vname, seed, fname)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    tmp.replace(path)                          # atomic: a killed run never leaves a half file
    return (f"{vname} seed {seed} {fname}: n_train {len(tr):5d}  inner-val {best['val_auc']:.3f}  "
            f"epoch {best['epoch']:3d}  ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ analysis

def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties (vectorised)."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    n_pos = int((y == 1).sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def block_bootstrap(df: pd.DataFrame, cols: list[str], n: int = 2000, seed: int = 0) -> dict:
    """Day-block bootstrap of AUC for each score column and pairwise diffs vs cols[0]."""
    rng = np.random.default_rng(seed)
    days = df["date"].to_numpy()
    uniq = np.unique(days)
    idx = {d: np.where(days == d)[0] for d in uniq}
    y = df["label"].to_numpy()
    S = {c: df[c].to_numpy() for c in cols}
    draws = {c: [] for c in cols}
    diffs = {c: [] for c in cols[1:]}
    for _ in range(n):
        ix = np.concatenate([idx[d] for d in rng.choice(uniq, size=len(uniq), replace=True)])
        if y[ix].min() == y[ix].max():
            continue
        a = {c: _auc(y[ix], S[c][ix]) for c in cols}
        for c in cols:
            draws[c].append(a[c])
        for c in cols[1:]:
            diffs[c].append(a[c] - a[cols[0]])
    out = {}
    for c in cols:
        d = np.array(draws[c])
        out[c] = {"lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
    for c in cols[1:]:
        d = np.array(diffs[c])
        out[c].update(diff_lo=float(np.percentile(d, 2.5)), diff_hi=float(np.percentile(d, 97.5)),
                      p_better=float((d > 0).mean()))
    return out


def analyse(pred: pd.DataFrame, baselines: pd.DataFrame, ref_variant: str = "V0") -> dict:
    key = ["fold", "date", "src", "dst"]
    variants = [v for v in VARIANTS if v in set(pred["variant"])]
    # common evaluation set: samples every variant scored
    counts = pred.groupby(key)["variant"].nunique()
    common = counts[counts == len(variants)].index
    p = pred.set_index(key).loc[common].reset_index()
    ens = p.groupby(["variant"] + key).agg(logit=("logit", "mean"), label=("label", "first")).reset_index()
    wide = ens.pivot_table(index=key, columns="variant", values="logit").reset_index()
    labels = ens.drop_duplicates(key).set_index(key)["label"]
    wide["label"] = labels.reindex(pd.MultiIndex.from_frame(wide[key])).to_numpy()
    wide = wide.merge(baselines, on=key, how="left")
    base_cols = [c for c in BASELINES if c in wide.columns and wide[c].notna().all()]
    others = [v for v in variants if v != ref_variant]

    # same bootstrap draws (seed) in both calls -> comparable differences
    boot = block_bootstrap(wide, [ref_variant] + others + base_cols)
    boot_vol = block_bootstrap(wide, ["neg_dst_vol"] + variants) if "neg_dst_vol" in base_cols else {}
    # volatility-stratified AUC: what a score adds once dst volatility is held fixed
    wide["_volq"] = pd.qcut(wide["neg_dst_vol"].rank(method="first"), 5, labels=False)

    def strat(col: str) -> float:
        return float(np.nanmean([_auc(g["label"].to_numpy(), g[col].to_numpy()) for _, g in wide.groupby("_volq")]))

    def vs_vol(col: str) -> dict:
        r = boot_vol.get(col, {})
        return {k: r[k] for k in ("diff_lo", "diff_hi", "p_better") if k in r}

    res = {"n_common": int(len(wide)), "n_days": int(wide["date"].nunique()),
           "pos_rate": float(wide["label"].mean()), "variants": {}, "baselines": {}}
    for v in variants:
        per_seed = [
            _auc(g["label"].to_numpy(), g["logit"].to_numpy())
            for _, g in p[p["variant"] == v].groupby("seed")
        ]
        per_fold = {f: _auc(g["label"].to_numpy(), g[v].to_numpy()) for f, g in wide.groupby("fold")}
        meta = pred[pred["variant"] == v].drop_duplicates(["seed", "fold"])
        res["variants"][v] = {
            "desc": VARIANTS[v][0],
            "ensemble_auc": _auc(wide["label"].to_numpy(), wide[v].to_numpy()),
            "vol_stratified_auc": strat(v),
            "seed_aucs": per_seed, "per_fold": per_fold,
            "mean_best_epoch": float(meta["best_epoch"].mean()),
            "mean_train_auc": float(meta["train_auc"].mean()),
            "mean_inner_val_auc": float(meta["inner_val_auc"].mean()),
            **boot[v], "vs_vol": vs_vol(v),
        }
    for c in base_cols:
        res["baselines"][c] = {
            "desc": BASELINES[c], "auc": _auc(wide["label"].to_numpy(), wide[c].to_numpy()),
            "vol_stratified_auc": strat(c),
            "per_fold": {f: _auc(g["label"].to_numpy(), g[c].to_numpy()) for f, g in wide.groupby("fold")},
            **boot[c],
        }
    return res


def write_report(res: dict, seeds: list[int], elapsed: float) -> Path:
    from src.common.config import load_config
    g = load_config().gnn
    label = (f"shock = |{'sector-excess' if g.shock_basis == 'excess' else 'raw'} 1d return| > "
             f"max({g.shock_z} × previous-day {g.shock_vol_window or load_config().features.volatility_window}d sigma, "
             f"{g.shock_min_move}); label = dst shocks within {g.cascade_window_days} days of a src shock")
    L = ["# GNN walk-forward evaluation", "",
         f"Out-of-sample test windows: {', '.join(f[0] for f in FOLDS)} (expanding-window training; "
         f"2026 = PPO test window, never touched). Seeds: {seeds} (ensemble = mean logit).", "",
         f"Common evaluation set: **{res['n_common']} samples over {res['n_days']} shock days** "
         f"({res['pos_rate']:.1%} positive). Labels from split/demerger-adjusted prices; {label}.", "",
         "Vol-strat AUC = mean AUC inside quintiles of the destination's previous-day volatility "
         "(what a score adds once that volatility is held fixed). P(>vol rule) = share of day-block "
         "bootstrap draws in which the model beats the volatility rule.", "",
         "| Variant | Description | Ensemble AUC | 95% CI | Vol-strat AUC | Δ vs V0 | Δ 95% CI | P(>V0) | P(>vol rule) | seed AUCs |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    ref = res["variants"]["V0"]["ensemble_auc"]
    for v, r in res["variants"].items():
        d = r["ensemble_auc"] - ref
        dci = f"{r['diff_lo']:+.3f} … {r['diff_hi']:+.3f}" if "diff_lo" in r else "—"
        pb = f"{r['p_better']:.0%}" if "p_better" in r else "—"
        pv = f"{r['vs_vol']['p_better']:.0%}" if r.get("vs_vol") else "—"
        seeds_s = ", ".join(f"{a:.3f}" for a in r["seed_aucs"])
        L.append(f"| {v} | {r['desc']} | **{r['ensemble_auc']:.4f}** | {r['lo']:.3f} … {r['hi']:.3f} | "
                 f"{r['vol_stratified_auc']:.3f} | {d:+.4f} | {dci} | {pb} | {pv} | {seeds_s} |")
    for c, r in res["baselines"].items():
        d = r["auc"] - ref
        L.append(f"| {c} | {r['desc']} | {r['auc']:.4f} | {r['lo']:.3f} … {r['hi']:.3f} | "
                 f"{r['vol_stratified_auc']:.3f} | {d:+.4f} | {r['diff_lo']:+.3f} … {r['diff_hi']:+.3f} | "
                 f"{r['p_better']:.0%} | — | — |")
    L += ["", "## Per-fold ensemble AUC", "",
          "| Variant | " + " | ".join(f[0] for f in FOLDS) + " |", "|---|" + "---|" * len(FOLDS)]
    for v, r in list(res["variants"].items()) + list(res["baselines"].items()):
        L.append(f"| {v} | " + " | ".join(f"{r['per_fold'].get(f[0], float('nan')):.3f}" for f in FOLDS) + " |")
    L += ["", "## Fit diagnostics (mean over folds and seeds)", "",
          "| Variant | best epoch | train AUC | inner-val AUC |", "|---|---|---|---|"]
    for v, r in res["variants"].items():
        L.append(f"| {v} | {r['mean_best_epoch']:.1f} | {r['mean_train_auc']:.3f} | {r['mean_inner_val_auc']:.3f} |")
    L += ["", f"_Generated by scripts/gnn_walkforward.py in {elapsed / 60:.1f} min._"]
    path = OUT_DIR / "WALKFORWARD.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 7])
    ap.add_argument("--jobs", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--folds", nargs="*", help="subset of fold names (smoke tests)")
    ap.add_argument("--fresh", action="store_true", help="discard checkpointed runs first")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUT_DIR / "predictions.parquet"
    base_path = OUT_DIR / "baselines.parquet"
    t0 = time.time()
    global FOLDS
    if args.folds:
        FOLDS = [f for f in FOLDS if f[0] in args.folds]
    if not args.analyse_only:
        if args.fresh:
            for f in RUNS_DIR.glob("*.parquet"):
                f.unlink()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        ref, baselines = reference_samples()
        baselines.to_parquet(base_path)
        print("reference test samples per fold:", {k: len(v) for k, v in ref.items()})
        wanted = [(v, s, f) for v in args.variants for s in args.seeds for f in FOLDS]
        todo = [(v, s, f[0], f[1], ref[f[0]]) for v, s, f in wanted if not run_path(v, s, f[0]).exists()]
        print(f"{len(wanted)} runs ({len(args.variants)} variants x {len(args.seeds)} seeds x "
              f"{len(FOLDS)} folds): {len(wanted) - len(todo)} checkpointed, {len(todo)} to do "
              f"on {args.jobs} processes", flush=True)
        if todo:
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.jobs, maxtasksperchild=6) as pool:
                for i, msg in enumerate(pool.imap_unordered(run_job, todo, chunksize=1), 1):
                    print(f"  [{i}/{len(todo)}] {msg}", flush=True)
        pred = pd.concat([pd.read_parquet(run_path(v, s, f[0])) for v, s, f in wanted], ignore_index=True)
        pred.to_parquet(pred_path)
    pred = pd.read_parquet(pred_path)
    res = analyse(pred, pd.read_parquet(base_path))
    (OUT_DIR / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    path = write_report(res, sorted(pred["seed"].unique().tolist()), time.time() - t0)
    print(f"\nreport -> {path}")
    for v, r in res["variants"].items():
        print(f"{v:3s} {r['ensemble_auc']:.4f}  [{r['lo']:.3f}, {r['hi']:.3f}]  strat {r['vol_stratified_auc']:.3f}  "
              f"{('P>V0 ' + format(r['p_better'], '.0%')) if 'p_better' in r else '        '}  "
              f"{('P>vol ' + format(r['vs_vol']['p_better'], '.0%')) if r.get('vs_vol') else ''}  {r['desc']}")
    for c, r in res["baselines"].items():
        print(f"{c:12s} {r['auc']:.4f}  [{r['lo']:.3f}, {r['hi']:.3f}]  strat {r['vol_stratified_auc']:.3f}  {r['desc']}")


if __name__ == "__main__":
    main()
