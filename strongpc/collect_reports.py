"""Gather every result of a training run into one folder + zip to take back.

    python strongpc\\collect_reports.py                 # latest run_* folder
    python strongpc\\collect_reports.py --run-dir reports/strongpc/run_20260905_090000

Copies into <run_dir>/artifacts/:
    REPORT.md, equity_curves.png, GNN_TEST_REPORT.md/.pdf, reports/gnn_figs/
    src/rl_agent/logs/baselines_test.json
    src/rl_agent/logs/ppo_<arm>/{train_meta.json,test_metrics.json,test_equity_curve.csv,events.*}
    data/processed/sentiment.parquet (+ sentiment_summary.json)
    state_h5_attrs.json, environment.json (versions, GPU)
    ppo_<arm>_scalars.csv (TensorBoard scalars exported) + ppo_training_curves.png
Then zips <run_dir> to reports/strongpc/<run_id>.zip.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "strongpc"
PPO_LOGS = ROOT / "src" / "rl_agent" / "logs"
ARMS = ["full", "no-gnn", "no-sentiment", "neither"]


def copy(src: Path, dst_dir: Path, note: list[str]) -> None:
    if src.is_file():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        note.append(f"  + {src.relative_to(ROOT)}")
    elif src.is_dir():
        shutil.copytree(src, dst_dir / src.name, dirs_exist_ok=True)
        note.append(f"  + {src.relative_to(ROOT)}/")


def export_tensorboard(art: Path, note: list[str]) -> dict[str, "object"]:
    """TensorBoard event files -> one CSV per arm; returns {arm: DataFrame}."""
    curves = {}
    try:
        from tensorboard.backend.event_processing import event_accumulator
        import pandas as pd
    except ImportError:
        note.append("  ! tensorboard/pandas not importable — scalar export skipped")
        return curves
    for arm in ARMS:
        run_dir = PPO_LOGS / f"ppo_{arm}"
        events = sorted(run_dir.glob("**/events.out.tfevents.*"))
        if not events:
            continue
        rows = {}
        for ev in events:
            ea = event_accumulator.EventAccumulator(str(ev), size_guidance={"scalars": 0})
            try:
                ea.Reload()
            except Exception:
                continue
            for tag in ea.Tags().get("scalars", []):
                for s in ea.Scalars(tag):
                    rows.setdefault(s.step, {})[tag] = s.value
        if not rows:
            continue
        df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
        df.index.name = "timesteps"
        out = art / f"ppo_{arm}_scalars.csv"
        df.to_csv(out)
        curves[arm] = df
        note.append(f"  + {out.name} ({len(df)} rows, {df.shape[1]} tags)")
    return curves


def plot_curves(curves: dict, art: Path, note: list[str]) -> None:
    if not curves:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tags = [("rollout/ep_rew_mean", "episode reward (mean)"),
            ("train/loss", "loss"), ("train/entropy_loss", "entropy loss"),
            ("train/explained_variance", "explained variance")]
    tags = [(t, l) for t, l in tags if any(t in df.columns for df in curves.values())]
    if not tags:
        return
    fig, axes = plt.subplots(len(tags), 1, figsize=(9, 3 * len(tags)), sharex=True)
    axes = [axes] if len(tags) == 1 else list(axes)
    for ax, (tag, label) in zip(axes, tags):
        for arm, df in curves.items():
            if tag in df.columns:
                s = df[tag].dropna()
                ax.plot(s.index, s.values, label=arm, lw=1.2)
        ax.set_ylabel(label); ax.grid(alpha=0.3)
    axes[0].legend(title="PPO arm"); axes[-1].set_xlabel("timesteps")
    fig.suptitle("PPO training curves (from TensorBoard logs)")
    fig.tight_layout()
    fig.savefig(art / "ppo_training_curves.png", dpi=140)
    plt.close(fig)
    note.append("  + ppo_training_curves.png")


def environment_json() -> dict:
    env = {"host": platform.node(), "os": f"{platform.system()} {platform.release()}",
           "python": sys.version.split()[0], "collected": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
    except Exception as e:
        env["torch"] = f"n/a ({e})"
    for mod in ("stable_baselines3", "gymnasium", "torch_geometric", "pandas", "numpy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = "n/a"
    try:
        env["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        pass
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                           text=True, timeout=15, cwd=ROOT).stdout.strip()
        env["git_branch"] = subprocess.run(["git", "branch", "--show-current"], capture_output=True,
                                           text=True, timeout=15, cwd=ROOT).stdout.strip()
    except Exception:
        pass
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", help="run folder to collect into (default: latest run_*)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
    else:
        runs = sorted(REPORT_ROOT.glob("run_*"))
        if not runs:
            run_dir = REPORT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S_collect")
        else:
            run_dir = runs[-1]
    art = run_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    note: list[str] = [f"collecting into {art}"]

    for f in ("REPORT.md", "equity_curves.png", "GNN_TEST_REPORT.md", "GNN_TEST_REPORT.pdf"):
        p = ROOT / f
        if p.exists():
            copy(p, art, note)
    if (ROOT / "reports" / "gnn_figs").exists():
        copy(ROOT / "reports" / "gnn_figs", art, note)
    if (PPO_LOGS / "baselines_test.json").exists():
        copy(PPO_LOGS / "baselines_test.json", art, note)
    for arm in ARMS:
        d = PPO_LOGS / f"ppo_{arm}"
        if not d.exists():
            continue
        dst = art / f"ppo_{arm}"
        for f in ("train_meta.json", "test_metrics.json", "test_equity_curve.csv"):
            if (d / f).exists():
                copy(d / f, dst, note)
        for ev in d.glob("**/events.out.tfevents.*"):
            copy(ev, dst / "tensorboard", note)
    sent = ROOT / "data" / "processed" / "sentiment.parquet"
    if sent.exists():
        copy(sent, art, note)
        try:
            import pandas as pd
            df = pd.read_parquet(sent)
            (art / "sentiment_summary.json").write_text(json.dumps({
                "rows": int(len(df)), "tickers": int(df["ticker"].nunique()),
                "date_min": str(df["date"].min()), "date_max": str(df["date"].max()),
                "sentiment_mean": float(df["sentiment"].mean()),
                "sentiment_std": float(df["sentiment"].std()),
                "items_scored": int(df["n_items"].sum()), "items_failed": int(df["n_failed"].sum()),
            }, indent=2), encoding="utf-8")
            note.append("  + sentiment_summary.json")
        except Exception as e:
            note.append(f"  ! sentiment summary failed: {e}")
    h5 = ROOT / "data" / "processed" / "state.h5"
    if h5.exists():
        try:
            import h5py
            with h5py.File(h5, "r") as f:
                attrs = {k: (v.item() if hasattr(v, "item") else str(v)) for k, v in f.attrs.items()}
            (art / "state_h5_attrs.json").write_text(json.dumps(attrs, indent=2, default=str), encoding="utf-8")
            note.append("  + state_h5_attrs.json")
        except Exception as e:
            note.append(f"  ! state.h5 attrs failed: {e}")
    (art / "environment.json").write_text(json.dumps(environment_json(), indent=2), encoding="utf-8")
    note.append("  + environment.json")

    curves = export_tensorboard(art, note)
    plot_curves(curves, art, note)

    zip_path = None
    if not args.no_zip:
        zip_path = shutil.make_archive(str(REPORT_ROOT / run_dir.name), "zip", root_dir=run_dir)
        note.append(f"zipped -> {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")
    print("\n".join(note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
