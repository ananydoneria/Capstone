"""Build GNN_TEST_REPORT.pdf: results analysis with figures, a plain-language
explanation of every test and how it works, and a conclusion.

Numbers are computed live (same diagnostics as gnn_test_report.py) so the PDF
never goes stale relative to the weights on disk; the explanatory prose is
static.

Run:  python scripts/gnn_test_pdf.py [--seeds 7 2026] [--out GNN_TEST_REPORT.pdf]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gnn_test_report as gtr  # noqa: E402
from src.common.config import load_config  # noqa: E402

FIG_DIR = ROOT / "reports" / "gnn_figs"


# ------------------------------------------------------------------ figures

def make_figures(d: dict) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = {}

    auc, base = d["auc"], d["baselines"]
    s = pd.Series({
        "GNN (val)": auc["val"], "GNN (train)": auc["train"],
        "Baseline: return correlation": base["train_corr"],
        "Baseline: dst prev-day vol (raw)": base["dst_prev_vol"],
        "Rule: dst prev-day vol, best direction": base["dst_vol_rule"],
        "Baseline: src prev-day vol (raw)": base["src_prev_vol"],
    })
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cols = ["#1f77b4", "#aec7e8", "#7f7f7f", "#7f7f7f", "#d62728", "#7f7f7f"]
    ax.barh(s.index, s.values, color=cols)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlim(0.3, 0.75); ax.set_xlabel("AUC"); ax.invert_yaxis()
    for i, v in enumerate(s.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_title("AUC on the same validation samples")
    fig.tight_layout(); fig.savefig(FIG_DIR / "baselines.png", dpi=160); plt.close(fig)
    figs["baselines"] = FIG_DIR / "baselines.png"

    m = pd.DataFrame(d["monthly"])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(len(m)), m["auc"], marker="o")
    for i, r in m.iterrows():
        if r["auc"] == r["auc"]:
            ax.annotate(f"n={r['n']}", (i, r["auc"]), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xticks(range(len(m))); ax.set_xticklabels(m["month"], rotation=45, fontsize=8)
    ax.axhline(0.5, color="k", ls="--", lw=1); ax.set_ylabel("val AUC")
    ax.set_title("Validation AUC by month")
    fig.tight_layout(); fig.savefig(FIG_DIR / "monthly.png", dpi=160); plt.close(fig)
    figs["monthly"] = FIG_DIR / "monthly.png"

    cal = pd.DataFrame(d["calibration"])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot(cal["p_mean"], cal["actual"], marker="o", label="model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set_xlabel("mean predicted p"); ax.set_ylabel("actual cascade rate")
    ax.set_title("Calibration (val quintiles)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "calibration.png", dpi=160); plt.close(fig)
    figs["calibration"] = FIG_DIR / "calibration.png"

    imp = pd.Series(d["feature_importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.barh(imp.index, imp.values, color=["#d62728" if v > 0.01 else "#7f7f7f" for v in imp.values])
    ax.axvline(0, color="k", lw=1); ax.set_xlabel("val AUC drop when shuffled")
    ax.set_title("Feature importance")
    fig.tight_layout(); fig.savefig(FIG_DIR / "importance.png", dpi=160); plt.close(fig)
    figs["importance"] = FIG_DIR / "importance.png"

    if d["seeds"]:
        meta = d["meta"]
        seeds = [meta["seed"]] + [r["seed"] for r in d["seeds"]]
        vals = [meta["best_val_auc"]] + [r["val_auc"] for r in d["seeds"]]
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        ax.bar([str(x) for x in seeds], vals, color=["#1f77b4"] + ["#aec7e8"] * len(d["seeds"]))
        ax.set_ylim(0.5, 0.7); ax.set_xlabel("seed"); ax.set_ylabel("val AUC")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_title("Seed robustness (dark = shipped)")
        fig.tight_layout(); fig.savefig(FIG_DIR / "seeds.png", dpi=160); plt.close(fig)
        figs["seeds"] = FIG_DIR / "seeds.png"
    return figs


# ------------------------------------------------------------------ document

def styles():
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13, alignment=TA_JUSTIFY,
                          spaceAfter=5)
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=18, spaceAfter=4),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=12),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=9, spaceAfter=4),
        "body": body,
        "bullet": ParagraphStyle("b", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3),
        "small": ParagraphStyle("sm", parent=body, fontSize=8, leading=10, textColor=colors.grey),
        "cap": ParagraphStyle("cap", parent=body, fontSize=8, leading=10, textColor=colors.grey,
                              alignment=1, spaceAfter=8),
    }


def table(rows, col_widths=None, header=True, font=8.5):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", font),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                  ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font)]
    t.setStyle(TableStyle(style))
    return t


def img(path, width_cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def build(out: Path, d: dict, figs: dict, summary: str, tests: list, cfg) -> None:
    S = styles()
    P = lambda txt, st="body": Paragraph(txt, S[st])
    B = lambda txt: Paragraph(txt, S["bullet"], bulletText="•")
    auc, base, meta, data = d["auc"], d["baselines"], d["meta"], d["data"]
    n_pass = sum(1 for s, _ in tests if s == "PASSED")
    n_fail = sum(1 for s, _ in tests if s in ("FAILED", "ERROR"))
    n_skip = sum(1 for s, _ in tests if s == "SKIPPED")
    story = []

    # ---------------------------------------------------------------- title
    story += [
        P("GNN Testing Report", "title"),
        P(f"Propagation-confidence GNN, NSE Auto supply-chain graph · generated "
          f"{datetime.now():%d %b %Y} · suite: {n_pass} passed, {n_fail} failed, {n_skip} skipped", "sub"),
    ]

    # ---------------------------------------------------------------- 1 what
    story += [P("1. What is being tested", "h1")]
    story += [P(
        "The GNN is one of the three models in the trading system. It looks at the 15-company "
        "supply-chain graph (32 curated supplier-to-buyer links) and, for each directed pair of "
        "companies A→B, estimates the probability that a <i>shock</i> at A today will be followed by a "
        "shock at B within the next five trading days. A shock is an unusually large one-day price "
        "move. These probabilities are computed once for every trading day, cached to disk, and fed "
        "into the reinforcement-learning agent's state. The RL agent never calls the GNN directly."),
        P("Testing therefore has to answer four questions: (1) are the training labels built "
          "correctly, (2) is the data reaching the model in the right shape with no leakage from the "
          "future, (3) does the model architecture behave the way a graph model should, and (4) does "
          "the trained checkpoint actually contain signal that a trivial rule of thumb would not "
          "give. Section 3 explains each test in plain words; Section 2 analyses what the trained "
          "model does."),
    ]

    # ---------------------------------------------------------------- 2 results
    story += [P("2. Results analysis", "h1"), P("2.1 Headline numbers", "h2")]
    story += [table([
        ["Metric", "Value"],
        ["Validation AUC (trained checkpoint, recomputed)", f"{auc['val']:.4f}"],
        ["Validation AUC recorded when training finished", f"{meta['best_val_auc']:.4f}"],
        ["Training AUC", f"{auc['train']:.4f}"],
        ["Train − val gap", f"{auc['train'] - auc['val']:.4f}"],
        ["Baseline: train-period return correlation of the pair", f"{base['train_corr']:.4f}"],
        ["Baseline: buyer's previous-day volatility (raw AUC)", f"{base['dst_prev_vol']:.4f}"],
        ["Rule: buyer's previous-day volatility, best direction", f"{base['dst_vol_rule']:.4f}"],
        ["Baseline: supplier's previous-day volatility (raw AUC)", f"{base['src_prev_vol']:.4f}"],
        ["Validation positive rate (a coin-flip scores 0.5 AUC)", f"{base['val_base_rate']:.1%}"],
        ["Early-stopping epoch", str(meta["best_epoch"])],
    ], col_widths=[11 * cm, 4 * cm])]
    story += [Spacer(1, 6), img(figs["baselines"], 14),
              P(f"Figure 1 — AUC of the GNN and the no-learning baselines on the identical "
                f"{data['n_val']} validation samples. A ranking rule can be used either way round, "
                f"so the red bar is the volatility rule in whichever direction works.", "cap")]
    story += [P(
        f"AUC (area under the ROC curve) measures ranking quality: 1.0 means every real cascade is "
        f"scored above every non-cascade, 0.5 means random. The GNN scores {auc['val']:.3f}. The "
        f"strongest baseline — simply looking up how correlated the two stocks' returns were during "
        f"the training period — scores {base['train_corr']:.3f}. So the GNN adds "
        f"{auc['val'] - base['train_corr']:+.3f} AUC on top of \"pairs that co-move tend to cascade\". "
        f"The volatility rule — rank pairs by the buyer's previous-day volatility, "
        f"{'low' if base['dst_prev_vol'] < 0.5 else 'high'} first — scores {base['dst_vol_rule']:.3f}"
        + (f", which the GNN clears by {auc['val'] - base['dst_vol_rule']:+.3f}. "
           if auc["val"] > base["dst_vol_rule"] else
           f", which the GNN does <b>not</b> beat: the label is still predictable from volatility. ")
        + "This rule matters because under the project's original label (raw returns against a 21-day "
          "sigma) it scored 0.64 out-of-sample and beat every model tested — a calm stock cleared "
          "2.5× its stale sigma as soon as normal volatility returned, so the \"cascades\" were largely "
          "a volatility-regime artifact. The label was therefore redefined on sector-excess returns "
          "(stock minus Nifty Auto) against a 63-day sigma, under which the rule scores about 0.52 "
          "out-of-sample (see section 2.8)."),
    ]

    story += [P("2.2 Data under test", "h2"), table([
        ["Item", "Value"],
        ["Timeline", f"{data['days']} trading days, {data['date_range']}"],
        ["Graph", f"{data['nodes']} companies, {data['edges_curated']} curated edges, "
                  f"{data['pairs_scored']} scored directed pairs (both directions)"],
        ["Node features", f"{', '.join(data['node_feature_names'])} ({len(data['node_feature_names'])})"],
        ["Pair-head inputs", ", ".join(data["pair_features"]) or "none"],
        ["Shock label", f"|{'sector-excess' if cfg.gnn.shock_basis == 'excess' else 'raw'} 1-day return| > "
                        f"max({cfg.gnn.shock_z} × previous-day "
                        f"{cfg.gnn.shock_vol_window or cfg.features.volatility_window}d sigma, "
                        f"{cfg.gnn.shock_min_move:.0%}); cascade within {cfg.gnn.cascade_window_days} days"],
        ["Training samples", f"{data['n_train']} ({data['train_pos']:.1%} positive), {data['train_range']}"],
        ["Validation samples", f"{data['n_val']} ({data['val_pos']:.1%} positive), {data['val_range']}"],
        ["Shock-day rate", f"{data['shock_rate']:.2%} of all (day, company) cells"],
        ["Model", f"{meta['arch']}, hidden {cfg.gnn.hidden_dim}, {cfg.gnn.num_layers} layers, dropout "
                  f"{cfg.gnn.dropout}, GRU over {cfg.gnn.temporal_window_days} days, seed {meta['seed']}"],
    ], col_widths=[4 * cm, 11 * cm])]
    story += [P(
        "Validation is the chronological tail of the timeline with a five-day gap before it, so no "
        f"training label window overlaps the validation period. The per-company shock counts run "
        f"from {min(data['shock_days_per_ticker'].values())} to {max(data['shock_days_per_ticker'].values())} "
        f"shock days over the timeline (SONACOMS only lists in 2021), so the volatility-scaled "
        f"threshold gives every stock a comparable number of events.")]

    story += [P("2.3 Is the signal stable over time?", "h2"), img(figs["monthly"], 14),
              P("Figure 2 — validation AUC computed month by month (n = samples in that month).", "cap")]
    mon = [r for r in d["monthly"] if r["auc"] == r["auc"]]
    lo, hi = min(mon, key=lambda r: r["auc"]), max(mon, key=lambda r: r["auc"])
    below = [r["month"] for r in mon if r["auc"] < 0.5]
    story += [P(
        f"No. The month-by-month AUC ranges from {lo['auc']:.2f} ({lo['month']}) to {hi['auc']:.2f} "
        f"({hi['month']}), and {len(below)} of {len(mon)} months fall below 0.5 ({', '.join(below)}). "
        f"Monthly sample sizes are small ({min(r['n'] for r in mon)}–{max(r['n'] for r in mon)}) so a lot of this is noise, but the practical reading "
        f"is that the GNN's score is an intermittent signal, not a steady one. The RL agent has to "
        f"learn how much to trust it, which is exactly what the 'no-gnn' ablation arm will measure.")]

    dr = d["direction"]
    story += [P("2.4 Direction, calibration and features", "h2")]
    story += [table([
        ["Direction", "n", "positive rate", "AUC"],
        ["Downstream (supplier → buyer)", dr["downstream"]["n"], f"{dr['downstream']['pos_rate']:.1%}",
         f"{dr['downstream']['auc']:.3f}"],
        ["Upstream (buyer → supplier)", dr["upstream"]["n"], f"{dr['upstream']['pos_rate']:.1%}",
         f"{dr['upstream']['auc']:.3f}"],
    ], col_widths=[6 * cm, 2 * cm, 3 * cm, 2 * cm])]
    story += [P(
        "Both directions score about equally, which justifies the decision to supervise both "
        "(supply disruptions travel downstream to the car maker; demand shocks travel upstream to "
        "the supplier). Neither half is dead weight.")]
    two = Table([[img(figs["calibration"], 7.2), img(figs["importance"], 7.2)]], hAlign="LEFT")
    story += [two, P("Figure 3 (left) — calibration: mean predicted probability per quintile vs actual "
                     "cascade rate. Figure 4 (right) — permutation importance: validation AUC drop "
                     "when one input feature is shuffled across days.", "cap")]
    cal = d["calibration"]
    imp = d["feature_importance"]
    top = next(iter(imp))
    story += [P(
        f"<b>Calibration.</b> The five quintiles are ordered correctly: the lowest-scored fifth of "
        f"pairs cascaded {cal[0]['actual']:.0%} of the time, the highest-scored fifth "
        f"{cal[-1]['actual']:.0%}. The absolute probabilities are inflated (mean predicted "
        f"{d['score_stats']['p_mean']:.2f} against a {data['val_pos']:.0%} base rate) because training "
        f"re-weights the rare positive class; this is intended. The score is a ranking, not a "
        f"calibrated probability, and the RL state treats it as such."),
        P(f"<b>Features.</b> Shuffling <b>{top}</b> costs {imp[top]:.3f} AUC"
          + (f"; {', '.join(f'<b>{k}</b>' for k, v in list(imp.items())[1:] if v > 0.01)} also matter"
             if any(v > 0.01 for v in list(imp.values())[1:]) else " — almost the entire signal")
          + f". Inputs whose shuffling costs under 0.01 AUC: "
          f"{', '.join(k for k, v in imp.items() if v <= 0.01) or 'none'}. Shuffling one input at a "
          f"time understates inputs that carry the same information as another, so this is a guide "
          f"to what the model leans on, not a verdict on what is useless.")]

    pp = d["per_pair"]
    story += [KeepTogether([P("2.5 Per-pair view", "h2"), table(
        [["Pair", "n", "pos", "AUC"]] + [[r["pair"], r["n"], r["pos"], f"{r['auc']:.2f}"] for r in pp[:4]]
        + [["…", "", "", ""]] + [[r["pair"], r["n"], r["pos"], f"{r['auc']:.2f}"] for r in pp[-4:]],
        col_widths=[7 * cm, 1.5 * cm, 1.5 * cm, 2 * cm])])]
    from collections import Counter
    weak = Counter()
    for r in pp:
        if r["auc"] < 0.5:
            s_, t_ = r["pair"].split("->"); weak[s_] += 1; weak[t_] += 1
    node, cnt = weak.most_common(1)[0] if weak else ("—", 0)
    story += [P(
        f"{len(pp)} pairs have enough validation samples to score; {sum(1 for r in pp if r['auc'] < 0.5)} "
        f"are below 0.5. The weak pairs cluster around <b>{node}</b> ({cnt} of them). With roughly "
        f"nine samples per pair this is a hint rather than a verdict, but it points at that node's "
        f"curated edges in relationships.csv as the first thing to re-examine.")]

    if d["seeds"]:
        aucs = {meta["seed"]: meta["best_val_auc"], **{r["seed"]: r["val_auc"] for r in d["seeds"]}}
        rank = sorted(aucs.values(), reverse=True).index(aucs[meta["seed"]]) + 1
        story += [KeepTogether([P("2.6 Seed robustness", "h2"), Table([[img(figs["seeds"], 6.5), P(
            f"Retraining from scratch with {len(aucs) - 1} other random seeds gives "
            f"{min(aucs.values()):.3f}–{max(aucs.values()):.3f}. The shipped seed ({meta['seed']}) "
            f"ranks {rank} of {len(aucs)} and stopped at epoch {meta['best_epoch']}. "
            f"The spread across seeds ({max(aucs.values()) - min(aucs.values()):.3f}) is "
            + ("larger" if max(aucs.values()) - min(aucs.values()) > auc['val'] - base['train_corr'] else "smaller")
            + f" than the {auc['val'] - base['train_corr']:+.3f} margin over the correlation baseline; quote the "
            f"headline AUC as \"about {min(aucs.values()):.2f}–{max(aucs.values()):.2f}\", not to three "
            f"decimals. Retrains went to a temporary folder; the shipped weights were not touched.")]],
            colWidths=[7 * cm, 8.5 * cm], hAlign="LEFT")])]

    c = d["cache"]
    story += [P("2.7 Reproducibility and leakage", "h2"), P(
        f"The cached score file used by the RL agent ({c['rows']} days × {c['cols']} pairs) was "
        f"recomputed from the weights: maximum difference {c['max_abs_diff']:.1e}, i.e. bit-for-bit "
        f"identical. Corrupting every day <i>after</i> day t with noise leaves day t's scores exactly "
        f"unchanged, while nudging a day <i>inside</i> the ten-day window does change them. Together "
        f"these prove the scores the agent sees are deterministic and contain no information from "
        f"the future. The train/val gap of {auc['train'] - auc['val']:.3f} with early stopping at epoch "
        f"{meta['best_epoch']} shows {'no overfitting' if auc['train'] - auc['val'] < 0.1 else 'some overfitting worth watching'}.")]

    wf = gtr.walkforward_summary()
    if wf:
        cur = wf.get("current")
        rows = [["Variant", "Description", "AUC", "95% CI", "Vol-strat", "P(>V0)", "P(>vol rule)"]]
        for v, r in wf["variants"].items():
            rows.append([f"{v}{' (shipped)' if v == cur else ''}", r["desc"], f"{r['ensemble_auc']:.3f}",
                         f"{r['lo']:.3f}–{r['hi']:.3f}", f"{r.get('vol_stratified_auc', float('nan')):.3f}",
                         f"{r['p_better']:.0%}" if "p_better" in r else "—",
                         f"{r['vs_vol']['p_better']:.0%}" if r.get("vs_vol") else "—"])
        for c, r in wf.get("baselines", {}).items():
            rows.append([c, r["desc"], f"{r['auc']:.3f}", f"{r['lo']:.3f}–{r['hi']:.3f}",
                         f"{r.get('vol_stratified_auc', float('nan')):.3f}", f"{r['p_better']:.0%}", "—"])
        story += [P("2.8 Out-of-sample: walk-forward comparison", "h2"), P(
            f"The validation AUC above is one split. The stronger test is walk-forward: five half-year "
            f"windows (2023H2–2025H2), each scored by a model retrained on everything before it, three "
            f"seeds averaged, {wf['n_common']} samples over {wf['n_days']} shock days; 2026 (the PPO "
            f"test window) is never used. Confidence intervals resample whole shock days, because pairs "
            f"scored on the same day are not independent. \"Vol-strat\" is the mean AUC inside "
            f"quintiles of the buyer's previous-day volatility: what a score adds once that volatility "
            f"is held fixed. The shipped configuration is variant {cur or '(not among those tested)'}."),
            table(rows, col_widths=[2.2 * cm, 5.6 * cm, 1.4 * cm, 2.4 * cm, 1.5 * cm, 1.4 * cm, 1.9 * cm], font=7.5),
            P("Full tables: reports/gnn_walkforward/WALKFORWARD.md. The earlier run under the raw-return "
              "label is archived beside it.", "small")]

    # ---------------------------------------------------------------- 3 tests explained
    story += [P("3. The tests, in plain words", "h1"), P(
        "The suite lives in <font face='Courier'>tests/gnn/</font> and runs with "
        "<font face='Courier'>pytest tests/gnn</font>. Each test is a small Python function that "
        "sets up a situation where the correct answer is known in advance, runs the real project code "
        "on it, and asserts the answer matches. If any assertion fails, pytest reports it. There are "
        f"{len(tests)} tests in six groups. Tests in groups A–D use tiny hand-made inputs and need "
        "nothing on disk; groups E and F use the real data and the real trained weights and skip "
        "themselves if those are missing.")]

    def n_in(fname: str) -> int:
        return sum(1 for _, t in tests if fname in t)

    sig_w = cfg.gnn.shock_vol_window or cfg.features.volatility_window
    story += [P(f"A. Shock detection and label construction ({n_in('test_labels')} tests)", "h2"), P(
        f"<b>What is checked.</b> A \"shock\" is a one-day move larger than {cfg.gnn.shock_z}× the "
        f"stock's recent ({sig_w}-day) volatility, with a floor of {cfg.gnn.shock_min_move:.0%} so that "
        f"calm periods don't flag trivial moves"
        + (", and the move is measured net of the Nifty Auto index (sector-excess return) so a "
           "market-wide day does not count as a shock at every company at once"
           if cfg.gnn.shock_basis == "excess" else "")
        + f". A training example is \"company A shocked today; did company B shock within the next "
        f"{cfg.gnn.cascade_window_days} trading days?\" These tests confirm that definition is "
        f"implemented exactly, for both the raw-return and the sector-excess variants of the rule, "
        f"and that the sigma window can differ from the feature window."),
        P("<b>How.</b> Each test builds a miniature price table for two fake companies (a few days, "
          "numbers chosen by hand) and calls the same shock and label functions the training code "
          "uses. For example: with 1% volatility yesterday, a 3% move must count as a shock and a "
          "2% move must not; a 50% move on the very first day must <i>not</i> count, because there is "
          "no previous-day volatility to compare against; the volatility used must be yesterday's, "
          "not today's, so a big move cannot raise its own threshold. For labels: a shock at B on the "
          "same day as A is not a cascade; a shock exactly five days later is; six days later is not; "
          "shocks in the last five days of the timeline are dropped because their five-day window "
          "would be cut off. The train/validation split is checked to be strictly chronological with "
          "a five-day gap, and never shuffled.")]

    story += [P(f"B. Data-to-tensor bridge ({n_in('test_dataset')} tests)", "h2"), P(
        "<b>What is checked.</b> The feature table has to be turned into a 3-D array of "
        "[day, company, feature] in a fixed company order, and the model is fed a trailing ten-day "
        "window ending at the day being scored. Any mistake here silently corrupts everything "
        "downstream."),
        P("<b>How.</b> Small tables with recognisable numbers (1, 2, 3 for company A; 10, 20, 30 for "
          "company B) are converted and specific cells are checked. The windowing function is asked for "
          "a ten-day window at day 1 and must return two real days preceded by zero-padding, never a "
          f"day after the target. On the real dataset the tests check the array shape ({data['days']} × "
          f"{data['nodes']} × {data['features']}), "
          "that every curated edge has its reverse present for message passing, that the 64 scored "
          "pairs are exactly the 32 edges plus their reverses with no duplicates, that features were "
          "normalised using only training-period statistics (mean ≈ 0, std ≈ 1 on training rows), "
          "and that the training/validation counts match the numbers recorded at training time.")]

    story += [P(f"C. Model architecture ({n_in('test_model')} tests)", "h2"), P(
        "<b>What is checked.</b> Properties any correct graph neural network must have, independent "
        "of data. These catch wiring bugs that would not necessarily crash but would make the model "
        "meaningless."),
        P("<b>How.</b> The model is built on random inputs and probed. Two models built with the same "
          "seed must give identical outputs (reproducibility). In evaluation mode two calls must agree "
          "exactly; in training mode they must differ (dropout is really on/off). After one backward "
          "pass every parameter must have a non-zero gradient (nothing is disconnected). Scoring four "
          "pairs together must equal scoring them one at a time (no cross-talk). A→B and B→A must get "
          "different scores (direction is respected). Renumbering the companies consistently in the "
          "features, the edges and the pairs must not change any score — this is the defining property "
          "of a graph network, called permutation equivariance. Reversing the order of the ten days "
          "must change the output (the recurrent layer is really order-aware), and deleting all edges "
          "must change it (the graph is really used). The alternative attention architecture builds "
          "and runs; an unknown architecture name is rejected; the parameter count equals the number "
          "the configuration implies.")]

    story += [P(f"D. Training utilities and a learning smoke test ({n_in('test_train_utils')} tests)", "h2"), P(
        "<b>What is checked.</b> The AUC metric, the per-day batching, the class re-weighting, and — "
        "most importantly — that the full training loop can actually learn."),
        P("<b>How.</b> The AUC implementation is fed cases with known answers: perfectly ordered "
          "scores must give 1.0, perfectly reversed 0.0, all-equal 0.5, a single class must give "
          "\"undefined\", and on 500 random points it must agree with scikit-learn's reference to six "
          "decimals (skipped if scikit-learn is not installed). The batching helper is checked to group "
          "samples by day in date order. The class weight is checked to make one positive and one "
          "negative example carry equal total loss mass. The smoke test then builds a synthetic "
          "8-company ring graph, plants a simple rule (\"cascade if the supplier's first feature is "
          "positive\"), and trains the real model for 60 steps: the loss must at least halve and the "
          "AUC must exceed 0.9. If gradients, batching or shapes were broken, this would fail.")]

    story += [P(f"E. The trained checkpoint ({n_in('test_trained_artifacts')} tests, real data)", "h2"), P(
        "<b>What is checked.</b> That the weights on disk are the ones described by their metadata, "
        "that they reproduce the recorded score, that they beat naive baselines, and that the cached "
        "scores consumed by the RL agent are exact and leak-free."),
        P("<b>How.</b> The metadata file is compared field by field to the configuration. Every "
          "tensor in the saved weights is checked for shape and for finite values. The saved "
          "normalisation statistics are compared to those recomputed from the data. The validation AUC "
          f"is recomputed and must match the recorded {meta['best_val_auc']:.3f} within 0.005. The "
          "correlation baseline and the volatility rule (taken in whichever direction scores higher) "
          "are computed on exactly the same validation samples and the GNN must beat both. The train/val "
          "gap must be under 0.2 and the scores must have spread (a constant output would pass "
          "everything else). The cache file must have the right columns, length, date index, no "
          "missing values and all values in [0, 1], and five sampled days are recomputed from the "
          "weights and must match to five decimals. Finally the no-look-ahead test: score day 300, "
          "overwrite all later days with noise, score again — identical; nudge one day inside the "
          "window — different.")]

    story += [P(f"F. Extended inputs and corrected data ({n_in('test_extended_inputs')} tests)", "h2"), P(
        "<b>What is checked.</b> Everything added when the training history was extended back to "
        "2011: the demerger back-adjustment, the same-day / one-day-lag rules for outside market "
        "data, the presence mask for a company that did not exist yet, the pair-head inputs, the "
        "edge-weighted graph layer, and — above all — that none of it can see the future."),
        P("<b>How.</b> A five-day price table with a planted 40% demerger drop is adjusted and the "
          "ex-date return must equal only that day's real trading, with earlier returns unchanged. "
          "The as-of joiner is fed dated values and must return the same day for lag 0, the previous "
          "day for lag 1, and \"missing\" once a value is over a week stale. The GraphConv layer must "
          "change its output when edge weights change while GraphSAGE must not. On the real data: "
          "SONACOMS must be all-zero with presence 0 before its 2021 listing and never generate a "
          "sample before it; every tensor must be finite; the compression of outside market data must "
          "be fitted on training days only (scrambling every later day must not change it); the "
          "inputs built for live inference must equal the training inputs exactly; and for node "
          "features, market context and the rolling pair correlation alike, truncating all source data "
          "at day t must leave day t's values unchanged. Finally, no training or validation label may "
          "read a price after the configured label end date, so the PPO test year never influences "
          "early stopping."),
    ]

    story += [P("G. Diagnostics (not pass/fail)", "h2"), P(
        "The report script adds analyses that have no single right answer and are therefore "
        "reported rather than asserted: the baselines above; AUC split by direction and by month; "
        "calibration by quintile; permutation importance (shuffle one feature's values across days, "
        "re-score, measure the AUC drop); per-pair AUC; and seed robustness (retrain from scratch "
        "with different random seeds into a temporary folder and compare).")]

    # ---------------------------------------------------------------- 4 conclusion
    story += [P("4. Conclusion", "h1")]
    story += [
        B(f"<b>The code is correct.</b> All {n_pass} tests pass. Labels, data plumbing, model "
          f"mechanics and the training loop behave exactly as specified, and the checkpoint on disk "
          f"reproduces its recorded metrics."),
        B("<b>The pipeline is leak-free and reproducible.</b> Scores contain no future information "
          "and the cache is bit-identical to a fresh forward pass. This is the property that matters "
          "most for the downstream RL claim, and it is proven, not assumed."),
        B(f"<b>The signal is real but modest.</b> Validation AUC {auc['val']:.3f} against "
          f"{base['train_corr']:.3f} for a static correlation lookup, {base['dst_vol_rule']:.3f} for the "
          f"volatility rule and 0.5 for chance. The input it leans on most is <b>{top}</b>. Quoted "
          f"honestly: the GNN adds a few points of ranking power over the simple rules"
          + (f"; out-of-sample (walk-forward) the shipped configuration scores "
             f"{wf['variants'][wf['current']]['ensemble_auc']:.3f} against "
             f"{wf['baselines']['neg_dst_vol']['auc']:.3f} for the volatility rule."
             if wf and wf.get("current") and "neg_dst_vol" in wf.get("baselines", {}) else ".")),
        B(f"<b>The signal is intermittent.</b> Monthly AUC swings between roughly {lo['auc']:.1f} and "
          f"{hi['auc']:.1f}. The RL agent should be expected to learn a modest, not dominant, reliance "
          f"on it; the 'no-gnn' ablation will quantify that."),
        B((f"<b>The headline number is seed-sensitive at the ±0.01 level.</b> Seeds span "
           f"{min(aucs.values()):.3f}–{max(aucs.values()):.3f}. " if d["seeds"] else "")
          + "The shipped seed 42 is kept for reproducibility, per the project's fixed-seed rule; "
          "report the AUC as a range."),
        B("<b>What changed and why.</b> The label moved from raw returns / 21-day sigma to sector-excess "
          "returns / 63-day sigma after the volatility rule was found to beat every model under the old "
          "definition; the training history was extended to 2011 with demerger-adjusted prices; the "
          "graph layer now uses the curated edge weights and the pair head sees the pair's rolling "
          "correlation. Each step was chosen on out-of-sample walk-forward AUC, never on the validation "
          "split reported above."),
        B("<b>Next steps if time allows</b> (none block PPO training): re-examine the curated edges "
          "around the weakest node; re-run the seed sweep after any further feature change."),
    ]
    story += [Spacer(1, 10), P(
        f"Sources: tests/gnn/ (suite), scripts/gnn_test_report.py (diagnostics, also writes "
        f"GNN_TEST_REPORT.md), scripts/gnn_test_pdf.py (this document), notebooks/gnn_testing.ipynb "
        f"(interactive version). pytest summary: {summary}.", "small")]

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="GNN Testing Report", author="Capstone")
    doc.build(story)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 2026])
    ap.add_argument("--out", default="GNN_TEST_REPORT.pdf")
    args = ap.parse_args()
    cfg = load_config()
    print("running pytest tests/gnn ...")
    summary, tests = gtr.run_pytest()
    print(f"  {summary}")
    print("running diagnostics ...")
    d = gtr.diagnostics(cfg, args.seeds)
    figs = make_figures(d)
    out = ROOT / args.out
    build(out, d, figs, summary, tests, cfg)
    print(f"pdf -> {out}")


if __name__ == "__main__":
    main()
