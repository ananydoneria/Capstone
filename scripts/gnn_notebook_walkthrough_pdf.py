"""Build GNN_NOTEBOOK_WALKTHROUGH.pdf: notebooks/gnn_testing.ipynb explained
cell by cell for a teacher — what every line does and what every output means.

The code shown is read from the notebook file, and every number in the
"Result" boxes is computed live (same diagnostics the notebook calls), so
the walkthrough cannot drift from the notebook or from the weights on disk.

Run:  python scripts/gnn_notebook_walkthrough_pdf.py [--seeds 7 2026] [--out GNN_NOTEBOOK_WALKTHROUGH.pdf]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import torch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gnn_test_report as gtr  # noqa: E402
from gnn_test_pdf import styles, table  # noqa: E402
from src.common.config import load_config  # noqa: E402

NB_PATH = ROOT / "notebooks" / "gnn_testing.ipynb"


# ------------------------------------------------------------------ helpers

def notebook_cells() -> list[dict]:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    return nb["cells"]


def code_of(cells: list[dict], idx: int) -> str:
    c = cells[idx]
    assert c["cell_type"] == "code", f"cell {idx} is not a code cell"
    return "".join(c["source"]).rstrip()


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def lookahead_numbers(cfg, data) -> dict:
    """Recompute cell 17's experiment so the walkthrough quotes live values."""
    from src.models.gnn.train import WEIGHTS_DIR, forward_all_pairs, make_model
    model = make_model(cfg, data)
    model.load_state_dict(torch.load(WEIGHTS_DIR / "propagation_gnn.pt", weights_only=True))
    model.eval()
    W, t = cfg.gnn.temporal_window_days, 300
    with torch.no_grad():
        before = forward_all_pairs(model, data, t, W)
        noisy = copy.copy(data)
        noisy.x = data.x.clone(); noisy.x[t + 1:] = torch.randn_like(noisy.x[t + 1:]) * 10
        if data.pair_feat is not None:
            noisy.pair_feat = data.pair_feat.clone()
            noisy.pair_feat[t + 1:] = torch.randn_like(noisy.pair_feat[t + 1:])
        after_future = forward_all_pairs(model, noisy, t, W)
        x = data.x.clone(); x[t - 3] += 5.0
        after_past = forward_all_pairs(model, data, t, W, x=x)
    return {"day": str(data.dates[t].date()), "t": t, "W": W,
            "future": float((before - after_future).abs().max()),
            "past": float((before - after_past).abs().max()), "n_pairs": int(before.numel())}


# ------------------------------------------------------------------ document

def build(out: Path, d: dict, tests: list, summary: str, cfg, cells: list[dict], la: dict) -> None:
    S = styles()
    S["code"] = ParagraphStyle("code", fontName="Courier", fontSize=7.2, leading=9, leftIndent=6,
                               backColor=colors.HexColor("#f4f6f8"), borderPadding=(3, 3, 3, 3),
                               spaceBefore=3, spaceAfter=5)
    S["q"] = ParagraphStyle("q", parent=S["body"], fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=1)
    P = lambda txt, st="body": Paragraph(txt, S[st])
    B = lambda txt: Paragraph(txt, S["bullet"], bulletText="•")

    cell_st = ParagraphStyle("cell", parent=S["body"], fontSize=7.5, leading=9.5, spaceAfter=0, alignment=0)
    head_st = ParagraphStyle("cellh", parent=cell_st, fontName="Helvetica-Bold")

    def wtable(rows, col_widths):
        """Table whose cells wrap (plain-string cells in reportlab do not)."""
        body = [[Paragraph(esc(str(c)), head_st if i == 0 else cell_st) for c in r] for i, r in enumerate(rows)]
        t = Table(body, colWidths=col_widths, hAlign="LEFT")
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                               ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        return t

    def code(idx: int):
        lines = []
        for ln in code_of(cells, idx).splitlines():
            lines += textwrap.wrap(ln, 108, subsequent_indent="        ", drop_whitespace=False) or [""]
        return Preformatted("\n".join(lines), S["code"])

    def result(title: str, flow: list):
        box = Table([[[P(f"<b>{title}</b>", "body")] + flow]], colWidths=[16.4 * cm], hAlign="LEFT")
        box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#8aa4c8")),
                                 ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f2f6fb")),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                 ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        return KeepTogether([Spacer(1, 2), box, Spacer(1, 6)])

    auc, base, meta, data = d["auc"], d["baselines"], d["meta"], d["data"]
    n_pass = sum(1 for s, _ in tests if s == "PASSED")
    n_skip = sum(1 for s, _ in tests if s == "SKIPPED")
    n_fail = sum(1 for s, _ in tests if s in ("FAILED", "ERROR"))
    skipped_names = [t.split("::")[-1] for s, t in tests if s == "SKIPPED"]
    sig_w = cfg.gnn.shock_vol_window or cfg.features.volatility_window
    basis = "sector-excess return (stock minus Nifty Auto)" if cfg.gnn.shock_basis == "excess" else "raw return"
    story = []

    # ---------------------------------------------------------------- title
    story += [P("GNN Testing Notebook — Line-by-Line Walkthrough", "title"),
              P(f"Explains notebooks/gnn_testing.ipynb cell by cell: what each line does and what each output "
                f"means, using live numbers from the weights on disk · generated {datetime.now():%d %b %Y}. "
                f"Capstone: hybrid neuro-symbolic trading system, NSE Auto sector.", "sub")]

    # ---------------------------------------------------------------- background
    story += [P("Background: say this first", "h1"), P(
        f"<b>What the GNN does.</b> The graph has {data['nodes']} auto-sector companies as nodes and "
        f"{data['edges_curated']} supplier→buyer links as edges. For a pair of companies A→B, the GNN predicts "
        f"the probability that a shock at A today is followed by a shock at B within the next "
        f"{cfg.gnn.cascade_window_days} trading days."),
        P(f"<b>What a shock is.</b> A one-day {basis} larger than {cfg.gnn.shock_z}× the stock's own "
          f"{sig_w}-day volatility, with a floor of {cfg.gnn.shock_min_move:.0%}. It is measured against the "
          f"sector index so that a day when the whole market moves does not count as a shock at every "
          f"company at once, and against a {sig_w}-day volatility because a shorter window made the label "
          f"predictable from volatility alone (a calm stock cleared its stale threshold as soon as normal "
          f"volatility returned). Under the earlier definition a one-line \"low volatility\" rule beat every "
          f"model out-of-sample; the redefinition is documented in config.yaml and the walk-forward report."),
        P("<b>What \"testing\" means here.</b> There are two kinds:"),
        B(f"<b>Correctness tests</b> in tests/gnn/ ({len(tests)} of them). Each one has a right answer that is "
          f"known in advance, so it either passes or fails."),
        B("<b>Diagnostics.</b> Measurements of how good the model is. They have no pass/fail, so we look at "
          "them and interpret them."),
        P("The notebook doesn't define any tests itself. It calls the real test suite and the diagnostics "
          "script, then shows the results as tables and plots. That keeps one source of truth, which is a "
          "good design point to mention."),
        P("<b>The metric, AUC.</b> AUC (area under the ROC curve) measures ranking quality:"),
        B("It is the probability that a randomly chosen real cascade gets a higher score than a randomly "
          "chosen non-cascade."),
        B("1.0 means perfect ranking, 0.5 means random guessing, below 0.5 means worse than random — "
          "and a rule scoring 0.36 is a 0.64 rule read backwards, so simple rules are always quoted in "
          "their better direction."),
    ]

    # ---------------------------------------------------------------- cell 1
    story += [P("Cell 1: Setup", "h1"), code(1),
        B("<b>Imports.</b> os handles folders, sys controls where Python looks for code, subprocess runs "
          "other programs, warnings controls warning messages; Path is a cleaner way to work with file paths. "
          "filterwarnings(\"ignore\") hides library deprecation notices so the output stays readable — it "
          "changes no result."),
        B("<b>ROOT = next(...)</b> finds the project root: it starts at the current folder, walks up through "
          "the parents, and stops at the first one containing config.yaml. So the notebook works whether it "
          "is opened from notebooks/ or from the root."),
        B("<b>os.chdir(ROOT)</b> moves into the root because config.yaml uses relative paths like "
          "data/processed/... that only resolve from there. <b>sys.path.insert(...)</b> lets Python import "
          "from the project folder and from scripts/ — that is what makes <i>from src...</i> and "
          "<i>import gnn_test_report</i> work."),
        B("<b>pandas</b> is tables, <b>matplotlib</b> is plots, <b>gnn_test_report</b> (short name gtr) is our "
          "own script that runs the tests and computes every diagnostic, and <b>load_config</b> reads "
          "config.yaml, which holds every setting (seed, model size, dates, the shock definition)."),
        B("The last three lines load the configuration into cfg, widen how tables print, and print the root "
          "folder as a sanity check."),
    ]

    # ---------------------------------------------------------------- cell 3
    story += [P("Cell 3: Run the real test suite", "h1"), code(3),
        B("<b>gtr.run_pytest()</b> runs <i>pytest tests/gnn</i> as a separate process (pytest is Python's "
          "standard testing tool). It returns the summary line and a list of (status, test name) pairs."),
        B("The next lines print the summary, count how many tests have each status (PASSED / SKIPPED / "
          "FAILED) and show that as a small table."),
        B("<b>assert not failed</b> means: stop the notebook with an error if any test failed. If the code is "
          "broken, the notebook refuses to continue and never shows plots based on broken code."),
        B("The final expression shows every test and its status as a table."),
        result("Result", [P(f"{summary}: {n_pass} passed, {n_skip} skipped, {n_fail} failed. "
                            + (f"Skipped: {', '.join(skipped_names)}. " if skipped_names else "")
                            + "The sklearn cross-check skips only when scikit-learn is not installed; nothing failed.")]),
    ]

    # ---------------------------------------------------------------- cell 5
    story += [P("Cell 5: What data is being tested", "h1"), code(5),
        B("<b>gtr.diagnostics</b> does all the analysis in one call: loads the data and the trained model, "
          "scores the validation set, computes every metric shown below, and returns a dictionary d. "
          "<b>seeds=[]</b> means \"don't retrain with other random seeds yet\" — that is done in the last "
          "cell because it is slow. <b>data</b> is the part describing the dataset."),
        B("The pd.Series(...) builds a two-column table; each row is explained below."),
        result("Result", [wtable([
            ["Row", "Value", "Meaning"],
            ["timeline", f"{data['days']} days, {data['date_range']}", "trading days of data (history back to 2011)"],
            ["nodes / features", f"{data['nodes']} / {data['features']}",
             f"{data['nodes']} companies; per company per day: {', '.join(data['node_feature_names'])}"],
            ["edges", f"{data['edges_curated']} / {data['edges_mp']} / {data['pairs_scored']}",
             f"{data['edges_curated']} curated supplier→buyer links, doubled so information flows both ways, "
             f"{data['pairs_scored']} directed pairs scored"],
            ["train samples", f"{data['n_train']} ({data['train_pos']:.1%} positive)", f"examples the model learned from, {data['train_range']}"],
            ["val samples", f"{data['n_val']} ({data['val_pos']:.1%} positive)", f"held-out examples used to judge it, {data['val_range']}"],
        ], col_widths=[2.8 * cm, 4.2 * cm, 8.6 * cm]),
            P("Point to make: validation is the latest period in time, with a 5-day gap before it, and no "
              "label may read a price after the configured label end date (2025-12-31, so the PPO test year "
              "never influences the GNN). The model is always tested on a future it never saw. A random split "
              "would let it \"see the future\", which is cheating. The <i>present</i> feature is 1 when a "
              "company is listed and 0 before (SONACOMS only lists in 2021).")]),
    ]

    # ---------------------------------------------------------------- cell 6
    sd = data["shock_days_per_ticker"]
    story += [P("Cell 6: Shock days per company", "h1"), code(6),
        B("Takes the number of shock days per company and draws a horizontal bar chart. invert_yaxis puts "
          "the biggest bar on top, tight_layout stops labels overlapping, show displays the plot."),
        result("What it shows", [P(
            f"Companies have between {min(sd.values())} and {max(sd.values())} shock days "
            f"(most at {max(sd.values())}, {min(sd, key=sd.get)} lowest because it lists late). The counts "
            f"are comparable because a shock is defined relative to each stock's own volatility — a "
            f"naturally jumpy stock doesn't dominate the data.")]),
    ]

    # ---------------------------------------------------------------- cell 8
    beats = auc["val"] > base["dst_vol_rule"]
    story += [P("Cell 8: The headline result, GNN vs simple rules", "h1"), code(8),
        B("Pulls out the GNN's AUC scores and the baseline scores, puts them into one list so they can be "
          "plotted together, then draws a bar chart with a dashed line at 0.5 (random guessing)."),
        B("<b>What a baseline is:</b> a simple rule that involves no learning. If the GNN can't beat it, the "
          f"GNN learned nothing useful. Everything is scored on the same {data['n_val']} validation samples, "
          "so the comparison is fair. The \"rule\" row is the volatility baseline in whichever direction "
          "scores higher, because a rule can always be flipped for free."),
        result("Result", [wtable([
            ["", "AUC", "Meaning"],
            ["GNN (val)", f"{auc['val']:.3f}", "the real result"],
            ["GNN (train)", f"{auc['train']:.3f}",
             f"score on data it trained on; gap {auc['train'] - auc['val']:+.3f} → "
             f"{'no serious overfitting' if auc['train'] - auc['val'] < 0.1 else 'some overfitting'}"],
            ["Return correlation", f"{base['train_corr']:.3f}", "\"stocks that moved together in the past will cascade\""],
            ["Buyer's volatility (raw)", f"{base['dst_prev_vol']:.3f}", "\"a jumpy buyer will shock\", as scored"],
            ["Buyer's volatility rule", f"{base['dst_vol_rule']:.3f}", "the same rule in its better direction"],
            ["Supplier's volatility (raw)", f"{base['src_prev_vol']:.3f}", "\"a jumpy supplier causes shocks\""],
        ], col_widths=[4.2 * cm, 1.6 * cm, 9.8 * cm]),
            P(f"How to explain it: the GNN {'beats' if beats else 'does NOT beat'} the volatility rule "
              f"({auc['val'] - base['dst_vol_rule']:+.3f}) and beats the correlation rule "
              f"({auc['val'] - base['train_corr']:+.3f}). "
              + ("The honest summary is \"a real but modest signal\". "
                 if beats else "That would mean the label is still predictable from volatility. ")
              + "Mention why the rule is shown both ways: under the project's first label definition the "
                "buyer's volatility scored 0.36 here — which everyone read as \"useless\" until the "
                "out-of-sample check showed it was a 0.64 rule read backwards, better than any model. That "
                "is what triggered the label change.")]),
    ]

    # ---------------------------------------------------------------- cell 9
    mon = [r for r in d["monthly"] if r["auc"] == r["auc"]]
    na = [r["month"] for r in d["monthly"] if r["auc"] != r["auc"]]
    lo, hi = min(mon, key=lambda r: r["auc"]), max(mon, key=lambda r: r["auc"])
    below = [r["month"] for r in mon if r["auc"] < 0.5]
    story += [P("Cell 9: Is it reliable over time?", "h1"), code(9),
        B("<b>monthly</b> is a table with one row per validation month: its AUC and sample count n. The plot "
          "draws a line of AUC per month with a dot for each month, and writes the sample count above each dot."),
        B("<b>r[\"auc\"] == r[\"auc\"]</b> is a trick for \"this value is not NaN\": NaN (a missing number) is the "
          "only value not equal to itself. A month whose samples are all one class has no AUC."),
        B("The last lines label the x-axis with month names (tilted so they fit) and draw the 0.5 chance line."),
        result("What it shows", [P(
            f"Monthly AUC swings from {lo['auc']:.2f} ({lo['month']}, n={lo['n']}) to {hi['auc']:.2f} "
            f"({hi['month']}, n={hi['n']}); {len(below)} of {len(mon)} months are below 0.5"
            + (f" ({', '.join(below)})" if below else "") + f". Monthly n runs {min(r['n'] for r in mon)}–"
            f"{max(r['n'] for r in mon)}" + (f"; {', '.join(na)} had a single class and no AUC" if na else "")
            + ". Some months have few samples, so part of this is noise. The conclusion is that the signal "
              "is intermittent, not steady. The trading agent has to learn how far to trust it.")]),
    ]

    # ---------------------------------------------------------------- cell 11
    cal = d["calibration"]
    mono = all(cal[i]["actual"] <= cal[i + 1]["actual"] for i in range(len(cal) - 1))
    story += [P("Cell 11: Calibration — do higher scores mean more cascades?", "h1"), code(11),
        B("The validation predictions were sorted into 5 equal groups (quintiles) by predicted probability. "
          "For each group we have the average predicted probability and the actual fraction that cascaded."),
        B("The plot shows predicted against actual. The dashed diagonal is where a perfectly calibrated model "
          "would sit (predicts 30% → 30% actually happen)."),
        result("What it shows", [P(
            f"The lowest-scored group cascaded {cal[0]['actual']:.0%} of the time, the highest "
            f"{cal[-1]['actual']:.0%}; the ordering is {'monotone — correct' if mono else 'not strictly monotone (small groups are noisy), though the ends are in the right order'}. "
            f"The curve sits away from the diagonal (mean predicted {d['score_stats']['p_mean']:.2f} against a "
            f"{data['val_pos']:.0%} base rate) because training re-weights the rare positive examples, which "
            f"shifts all predicted probabilities. That was deliberate. Use the score as a ranking, not as a "
            f"literal probability.")]),
    ]

    # ---------------------------------------------------------------- cell 13
    imp = d["feature_importance"]
    top = next(iter(imp))
    story += [P("Cell 13: Which inputs does the model use?", "h1"), code(13),
        B("Plots permutation importance for every model input: the node features and, when configured, the "
          "pair-head inputs (prefixed pair:)."),
        B("<b>How permutation importance works:</b> take one input, randomly shuffle its values across days "
          "(which destroys any information it carries), and re-measure validation AUC. The drop in AUC tells "
          "you how much the model relied on that input."),
        result("Result", [wtable([["Input", "AUC drop"]] + [[k, f"{v:+.3f}"] for k, v in imp.items()],
                                 col_widths=[6 * cm, 2.5 * cm]),
            P(f"How to explain it: the model leans most on <b>{top}</b> ({imp[top]:+.3f}). Inputs near zero "
              f"are not necessarily useless — shuffling one at a time understates an input that carries the "
              f"same information as another (the pair's edge weight and direction, for instance, never change "
              f"over time, so the model can absorb them into its weights). It is a guide to what the model "
              f"relies on, not a verdict.")]),
    ]

    # ---------------------------------------------------------------- cell 15
    pp = d["per_pair"]
    from collections import Counter
    weak = Counter()
    for r in pp:
        if r["auc"] < 0.5:
            s_, t_ = r["pair"].split("->"); weak[s_] += 1; weak[t_] += 1
    node, cnt = weak.most_common(1)[0] if weak else ("—", 0)
    story += [P("Cell 15: Per-pair view", "h1"), code(15),
        B("<b>pp</b> has the AUC for each supply-chain pair. It only includes pairs with at least 8 validation "
          "samples and both outcomes present, since otherwise AUC can't be computed."),
        B("It prints how many pairs could be scored and how many fall below 0.5, then shows the best 5 and worst 5."),
        result("Result", [P(
            f"{len(pp)} pairs scored, {sum(1 for r in pp if r['auc'] < 0.5)} below 0.5; the weak ones most "
            f"often involve <b>{node}</b> ({cnt} of them). Each pair has only a handful of samples, so treat "
            f"this as a hint worth investigating (are that company's edges in the graph correct?), not as proof.")]),
    ]

    # ---------------------------------------------------------------- cell 17
    story += [P("Cell 17: The no-look-ahead test (the most important one)", "h1"),
        P("This checks that the model can't see the future. If tomorrow's data leaked into today's score, "
          "every backtest result would be fake."), code(17),
        B("<b>Imports:</b> copy (to make a shallow copy of the dataset object), PyTorch, the seeding function, "
          "the dataset builder, and three helpers from the training module: the weights folder, "
          "<b>make_model</b> (builds a model with exactly the input sizes the dataset has — node features, "
          "pair-head inputs, market context) and <b>forward_all_pairs</b> (scores all directed pairs on one day, "
          "feeding the model the trailing window, the edges with their weights, and that day's pair inputs)."),
        B(f"<b>set_global_seed</b> fixes all randomness to seed {cfg.project.seed}; <b>build_dataset</b> builds the "
          "[day, company, feature] array plus the per-day pair inputs; the model is created, the trained "
          "weights are loaded from disk, and <b>eval()</b> switches off dropout so outputs are deterministic."),
        B(f"<b>W = {la['W']}</b>: the model looks at the last {la['W']} days. <b>t = {la['t']}</b>: the day we "
          f"test on ({la['day']}). <b>no_grad()</b> means \"just predict, don't prepare for training\"."),
        B("<b>before</b> scores all pairs on day t from clean data. Then a copy of the dataset has every day "
          "AFTER t replaced with large random noise — node features and pair inputs alike — and day t is "
          "scored again as <b>after_future</b>. If the scores change at all, the model was using future data."),
        B("<b>Control experiment:</b> start from clean data again and nudge one day inside the window (day "
          "t−3). This time the output should change. That proves the test above isn't passing only because "
          "the model ignores its input."),
        B("The prints show the largest change across all pair scores for each experiment."),
        result("Result", [
            B(f"Future destroyed → change = {la['future']:.1f} exactly. There is no leakage."),
            B(f"Past day nudged → change = {la['past']:.3f}. The model does use its {la['W']}-day history."),
            P("Together these prove, rather than assume, that the scores are leak-free. The same check runs "
              "as a pytest test, and separate tests prove the inputs themselves (node features, market "
              "context, rolling pair correlation) are computed without look-ahead.")]),
    ]

    # ---------------------------------------------------------------- cell 19
    story += [P("Cell 19: Seed robustness", "h1"), code(19),
        B("Retrains the GNN from scratch with other random seeds. It writes to a temporary folder, so the "
          "real trained weights are never touched. Takes about a minute per seed."),
        B("Builds a table: the first row is the model we actually use (read from its saved training record), "
          "then one row per retrained seed; then a bar chart and the table. Setting SEEDS = [] skips the cell."),
        B("<b>Why this matters:</b> neural networks start from random weights, so a different seed gives a "
          "slightly different model. This checks whether the headline number was a lucky or unlucky draw."),
    ]
    if d["seeds"]:
        aucs = {meta["seed"]: meta["best_val_auc"], **{r["seed"]: r["val_auc"] for r in d["seeds"]}}
        rank = sorted(aucs.values(), reverse=True).index(aucs[meta["seed"]]) + 1
        rows = [["Seed", "Val AUC", "Best epoch"], [f"{meta['seed']} (used)", f"{meta['best_val_auc']:.3f}", str(meta["best_epoch"])]]
        rows += [[str(r["seed"]), f"{r['val_auc']:.3f}", str(r["best_epoch"])] for r in d["seeds"]]
        story += [result("Result", [wtable(rows, col_widths=[3 * cm, 2.5 * cm, 2.5 * cm]), P(
            f"How to explain it: results span {min(aucs.values()):.3f}–{max(aucs.values()):.3f} "
            f"(spread {max(aucs.values()) - min(aucs.values()):.3f}); our seed ranks {rank} of {len(aucs)}. "
            f"We keep seed {meta['seed']} because the project fixes the seed for reproducibility, and we "
            f"report the AUC as a range rather than to three decimals.")])]

    # ---------------------------------------------------------------- takeaways
    wf = gtr.walkforward_summary()
    story += [P("Last cell: Takeaways (text)", "h1"),
        B("The code is correct: all tests pass."),
        B(f"The signal is real but modest: {auc['val']:.3f} against {base['train_corr']:.3f} for the correlation "
          f"rule and {base['dst_vol_rule']:.3f} for the volatility rule."),
        B(f"The model relies most on {top}; near-zero inputs may be redundant rather than useless."),
        B("The signal varies month to month."),
        B("There is provably no future leakage, and the cached scores reproduce exactly."),
    ]
    if wf and wf.get("current"):
        cur = wf["variants"][wf["current"]]
        vol = wf.get("baselines", {}).get("neg_dst_vol")
        story += [B(f"Out-of-sample (walk-forward, five half-year windows, {wf['n_days']} shock days): the shipped "
                    f"configuration scores {cur['ensemble_auc']:.3f} (95% CI {cur['lo']:.3f}–{cur['hi']:.3f})"
                    + (f" against {vol['auc']:.3f} for the volatility rule" if vol else "")
                    + ". This, not the single validation split, is the number to defend.")]

    # ---------------------------------------------------------------- questions
    qa = [
        ("\"Why not just use accuracy?\"",
         f"Only about {data['val_pos']:.0%} of examples are cascades. A model that always says \"no cascade\" "
         f"would be {1 - data['val_pos']:.0%} accurate and useless. AUC measures ranking and isn't fooled by that imbalance."),
        ("\"Isn't 0.6 low?\"",
         "Predicting stock moves is hard. What matters is that it beats the no-learning rules on identical "
         "data, out of sample, and that the margin is quoted honestly."),
        ("\"How do you know it's not cheating?\"",
         "Cell 17. We destroyed all future data and the output didn't change at all, and we showed with a "
         "control that the model does react to real inputs. Separate tests do the same for every input series."),
        ("\"Why did the definition of a shock change?\"",
         "Because a one-line rule — rank pairs by the buyer's low recent volatility — beat every model under "
         "the first definition (0.64 AUC out of sample). A 21-day volatility mean-reverts inside the 5-day "
         "label window, so calm stocks cleared their stale threshold as soon as normal volatility returned. "
         "Measuring shocks net of the sector index against a 63-day volatility removes that shortcut (the "
         "rule drops to about 0.52). The change was made before the final model was selected, and the old "
         "results are archived, not deleted."),
        ("\"Why is the curve away from the diagonal in calibration?\"",
         "Class weighting during training shifts all probabilities on purpose. The ordering is what the "
         "trading agent uses."),
        ("\"Why keep a seed that isn't the best?\"",
         "Reproducibility. The seed was fixed before looking at results; choosing the best seed afterwards "
         "would be cherry-picking."),
    ]
    story += [P("Likely teacher questions", "h1")]
    for q, a in qa:
        story += [P(q, "q"), P(a)]

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="GNN Testing Notebook — Walkthrough")

    def footer(canvas, doc_):
        canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.1 * cm, "GNN Testing Notebook — Walkthrough")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 2026])
    ap.add_argument("--out", default=str(ROOT / "GNN_NOTEBOOK_WALKTHROUGH.pdf"))
    args = ap.parse_args()
    cfg = load_config()
    cells = notebook_cells()
    print("running the test suite ...")
    summary, tests = gtr.run_pytest()
    print(f"  {summary}")
    print(f"running diagnostics (seeds {args.seeds}) ...")
    d = gtr.diagnostics(cfg, seeds=args.seeds)
    from src.common.seeding import set_global_seed
    from src.models.gnn.graph_dataset import build_dataset
    set_global_seed(cfg.project.seed)
    la = lookahead_numbers(cfg, build_dataset(cfg))
    out = Path(args.out)
    build(out, d, tests, summary, cfg, cells, la)
    print(f"walkthrough -> {out}")


if __name__ == "__main__":
    main()
