"""Unattended runner for the training-PC runbook (no Claude Code needed).

Runs the CLAUDE.md steps in order, one subprocess per stage, and stores
EVERYTHING under reports/strongpc/run_<timestamp>/:
    <NN>_<stage>.log     full stdout/stderr of that stage (also shown live)
    status.json          per-stage status / timing / exit code, updated live
    gpu_usage.csv        nvidia-smi samples every 30 s during heavy stages
    preflight.txt        the pre-run environment check
    tests_before.xml     pytest junit before training, tests_after.xml after
    artifacts/           copies of REPORT.md, equity curves, PPO metrics,
                         TensorBoard scalars as CSV, training-curve plot ...
    SUMMARY.md           one-page human summary (written even on failure)

Stages (keys usable with --only / --from / --skip):
    preflight        strongpc/preflight.py (aborts if NOT READY unless --force)
    tests_before     pytest tests (junit xml)
    news_finnhub     scripts/fetch_finnhub_news.py  (skips itself without API key)
    news_google      scripts/fetch_google_news.py
    sentiment        scripts/run_sentiment.py       (needs Ollama)
    build_state      scripts/build_state.py         GATE: must print source: llm_cache
    baselines        python -m src.rl_agent.baselines
    ppo_full, ppo_no-gnn, ppo_no-sentiment, ppo_neither
                     scripts/train_ppo.py --ablation <arm> (skipped if model.zip exists)
    evaluate         scripts/evaluate.py
    report           scripts/report.py
    tests_after      pytest tests (expect 90 passed)
    gnn_report       scripts/gnn_test_report.py + gnn_test_pdf.py
    collect          strongpc/collect_reports.py (bundle + zip)

    python strongpc\\run_pipeline.py                 # everything
    python strongpc\\run_pipeline.py --from ppo_full # resume from a stage
    python strongpc\\run_pipeline.py --only tests_before,gnn_report
    python strongpc\\run_pipeline.py --retrain       # ignore existing model.zip
    python strongpc\\run_pipeline.py --dry-run
Stops at the first failing stage; SUMMARY.md says which log to read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIN = os.name == "nt"
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if WIN else "bin/python")
REPORT_ROOT = ROOT / "reports" / "strongpc"
ARMS = ["full", "no-gnn", "no-sentiment", "neither"]
PPO_LOGS = ROOT / "src" / "rl_agent" / "logs"


class Stage:
    def __init__(self, key: str, cmd: list[str], heavy: bool = False, gate=None, skip_if=None,
                 optional: bool = False):
        self.key, self.cmd, self.heavy, self.gate, self.skip_if, self.optional = (
            key, cmd, heavy, gate, skip_if, optional)


def build_stages(py: str, args) -> list[Stage]:
    pf = [py, "strongpc/preflight.py", "--skip-tests"]
    stages = [
        Stage("preflight", pf),
        Stage("tests_before", [py, "-m", "pytest", "tests", "-q", "-rA", "-W", "ignore",
                               "-p", "no:cacheprovider", "--junitxml=__RUN__/tests_before.xml"]),
        Stage("news_finnhub", [py, "scripts/fetch_finnhub_news.py"], optional=True),
        Stage("news_google", [py, "scripts/fetch_google_news.py"], optional=True),
        Stage("sentiment", [py, "scripts/run_sentiment.py"], heavy=True,
              gate=lambda out: sentiment_gate(py)),
        Stage("build_state", [py, "scripts/build_state.py"],
              gate=lambda out: "source: llm_cache" in out),
        Stage("baselines", [py, "-m", "src.rl_agent.baselines"]),
    ]
    for arm in ARMS:
        cmd = [py, "scripts/train_ppo.py", "--ablation", arm]
        if args.timesteps:
            cmd += ["--timesteps", str(args.timesteps)]
        model = PPO_LOGS / f"ppo_{arm}" / "model.zip"
        stages.append(Stage(f"ppo_{arm}", cmd, heavy=True,
                            skip_if=(lambda m=model: m.exists()) if not args.retrain else None))
    stages += [
        Stage("evaluate", [py, "scripts/evaluate.py"]),
        Stage("report", [py, "scripts/report.py"]),
        Stage("tests_after", [py, "-m", "pytest", "tests", "-q", "-rA", "-W", "ignore",
                              "-p", "no:cacheprovider", "--junitxml=__RUN__/tests_after.xml"]),
        Stage("gnn_report", [py, "scripts/gnn_test_report.py", "--seeds", "7", "2026"], optional=True),
        Stage("gnn_pdf", [py, "scripts/gnn_test_pdf.py"], optional=True),
        Stage("collect", [py, "strongpc/collect_reports.py", "--run-dir", "__RUN__"]),
    ]
    return stages


def sentiment_gate(py: str) -> bool:
    """A dead Ollama still yields a sentiment.parquet (all neutral, every item
    failed) which build_state would happily label llm_cache. Refuse to go on if
    more than half of the scored items failed."""
    code = ("import pandas as pd; d = pd.read_parquet('data/processed/sentiment.parquet'); "
            "n = int(d['n_items'].sum()); f = int(d['n_failed'].sum()); "
            "print(f'{f}/{n}'); raise SystemExit(0 if n and f / n <= 0.5 else 3)")
    p = subprocess.run([py, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    ratio = p.stdout.strip()
    if p.returncode != 0:
        print(f"\n!!! sentiment gate: {ratio or p.stderr.strip()[-300:]} items failed -> neutral. "
              f"Ollama was probably not serving. Delete data/processed/sentiment.parquet, fix "
              f"Ollama (strongpc/fix_04_ollama.bat), and resume from 'sentiment'.")
        return False
    print(f"\nsentiment gate ok: {ratio} items failed")
    return True


# ------------------------------------------------------------------ gpu monitor

class GpuMonitor:
    def __init__(self, csv_path: Path, interval: int = 30):
        self.path, self.interval = csv_path, interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.enabled = shutil.which("nvidia-smi") is not None

    def start(self, stage: str):
        if not self.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(stage,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self, stage: str):
        new = not self.path.exists()
        with self.path.open("a", encoding="utf-8") as f:
            if new:
                f.write("time,stage,gpu_util_pct,mem_used_mb,mem_total_mb,temp_c,power_w\n")
            while not self._stop.is_set():
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
                         "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=15).stdout.strip().splitlines()[0]
                    f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S},{stage},{out.replace(' ', '')}\n")
                    f.flush()
                except Exception:
                    pass
                self._stop.wait(self.interval)


# ------------------------------------------------------------------ runner

def run_stage(stage: Stage, run_dir: Path, idx: int, py: str) -> tuple[str, int, float, str]:
    log = run_dir / f"{idx:02d}_{stage.key}.log"
    cmd = [c.replace("__RUN__", str(run_dir)) for c in stage.cmd]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
           "PYTHONWARNINGS": "ignore", "STRONGPC_RUN_DIR": str(run_dir)}
    t0 = time.time()
    tail: list[str] = []
    with log.open("w", encoding="utf-8", errors="replace") as f:
        f.write(f"$ {' '.join(cmd)}\n# started {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                             bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            f.write(line)
            tail.append(line)
            if len(tail) > 400:
                tail.pop(0)
        rc = p.wait()
        f.write(f"\n# exit {rc} after {time.time() - t0:.0f}s\n")
    return str(log.relative_to(run_dir)), rc, time.time() - t0, "".join(tail)


def write_status(run_dir: Path, status: dict):
    (run_dir / "status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")


def write_summary(run_dir: Path, status: dict):
    st = status["stages"]
    lines = [f"# Strong-PC run {status['run_id']}", "",
             f"Host: {status['host']} · started {status['started']} · "
             f"finished {status.get('finished', '—')} · result **{status.get('result', 'RUNNING')}**", ""]
    lines += ["| # | stage | status | time | exit | log |", "|---|---|---|---|---|---|"]
    for i, (k, v) in enumerate(st.items()):
        secs = v.get("seconds")
        t = f"{secs / 60:.1f} min" if secs is not None else "—"
        lines.append(f"| {i:02d} | {k} | {v['status']} | {t} | {v.get('rc', '—')} | {v.get('log', '—')} |")
    if status.get("failure"):
        lines += ["", f"**Failed at `{status['failure']['stage']}`** — read "
                      f"`{status['failure']['log']}`. Last lines:", "```",
                  status["failure"]["tail"].strip()[-2000:], "```"]
    pf = run_dir / "preflight.txt"
    if pf.exists():
        m = re.search(r"SCORE:.*\nVERDICT:.*", pf.read_text(encoding="utf-8", errors="replace"))
        if m:
            lines += ["", "## Preflight", "```", m.group(0), "```"]
    for name in ("tests_before", "tests_after"):
        log = next(run_dir.glob(f"*_{name}.log"), None)
        if log:
            m = re.search(r"\d+ passed.*", log.read_text(encoding="utf-8", errors="replace"))
            if m:
                lines += [f"- {name}: `{m.group(0).strip()}`"]
    rep = ROOT / "REPORT.md"
    if rep.exists() and (run_dir / "artifacts").exists():
        body = rep.read_text(encoding="utf-8")
        table = "\n".join(l for l in body.splitlines() if l.startswith("|"))
        lines += ["", "## Out-of-sample results (REPORT.md)", "", table]
    gpu = run_dir / "gpu_usage.csv"
    if gpu.exists():
        rows = gpu.read_text(encoding="utf-8").splitlines()[1:]
        vals = [r.split(",") for r in rows if r.count(",") >= 6]
        if vals:
            try:
                util = [float(v[2]) for v in vals]; mem = [float(v[3]) for v in vals]
                temp = [float(v[5]) for v in vals]
                lines += ["", "## GPU during heavy stages",
                          f"- samples: {len(vals)} · util mean {sum(util) / len(util):.0f}% / max {max(util):.0f}%"
                          f" · VRAM max {max(mem) / 1024:.1f} GB · temp max {max(temp):.0f} °C"]
            except ValueError:
                pass
    lines += ["", f"_Everything for this run is under `{run_dir}`; artifacts/ holds the copies to take back._"]
    (run_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated stage keys to run")
    ap.add_argument("--from", dest="from_", help="start at this stage key")
    ap.add_argument("--skip", default="", help="comma-separated stage keys to skip")
    ap.add_argument("--retrain", action="store_true", help="retrain PPO arms even if model.zip exists")
    ap.add_argument("--timesteps", type=int, help="DEBUG ONLY: override rl.total_timesteps")
    ap.add_argument("--force", action="store_true", help="continue even if preflight says NOT READY")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default=os.environ.get("STRONGPC_PYTHON"))
    args = ap.parse_args()

    py = args.python or str(VENV_PY)
    if not Path(py).exists() and not shutil.which(py):
        print(f"python not found: {py}\nrun strongpc\\fix_02_venv.bat first")
        return 2
    if args.timesteps:
        print(f"!! --timesteps {args.timesteps}: DEBUG run, results are NOT valid for the report")

    stages = build_stages(py, args)
    keys = [s.key for s in stages]
    if args.only:
        wanted = [k.strip() for k in args.only.split(",")]
        bad = [k for k in wanted if k not in keys]
        if bad:
            print(f"unknown stage(s): {bad}\nknown: {keys}"); return 2
        stages = [s for s in stages if s.key in wanted]
    if args.from_:
        if args.from_ not in keys:
            print(f"unknown stage: {args.from_}\nknown: {keys}"); return 2
        stages = stages[keys.index(args.from_):]
    skip = {k.strip() for k in args.skip.split(",") if k.strip()}
    stages = [s for s in stages if s.key not in skip]

    if args.dry_run:
        for s in stages:
            print(f"{s.key:16} {' '.join(s.cmd)}")
        return 0

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = REPORT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    import platform
    status = {"run_id": run_id, "host": platform.node(), "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
              "python": py, "args": vars(args), "stages": {s.key: {"status": "pending"} for s in stages}}
    write_status(run_dir, status)
    print(f"=== {run_id} -> {run_dir}\n")
    monitor = GpuMonitor(run_dir / "gpu_usage.csv")

    result = "OK"
    for i, s in enumerate(stages):
        rec = status["stages"][s.key]
        if s.skip_if and s.skip_if():
            rec.update(status="skipped (output exists)")
            print(f"--- [{i:02d}] {s.key}: skipped, output already exists (use --retrain to redo)")
            write_status(run_dir, status); continue
        print(f"\n=== [{i:02d}] {s.key}  {datetime.now():%H:%M:%S}\n$ {' '.join(s.cmd)}\n")
        rec.update(status="running", started=f"{datetime.now():%Y-%m-%d %H:%M:%S}")
        write_status(run_dir, status)
        if s.heavy:
            monitor.start(s.key)
        log, rc, secs, tail = run_stage(s, run_dir, i, py)
        if s.heavy:
            monitor.stop()
        rec.update(log=log, rc=rc, seconds=round(secs), ended=f"{datetime.now():%Y-%m-%d %H:%M:%S}")

        if s.key == "preflight":
            latest = REPORT_ROOT / "preflight_latest.json"
            if latest.exists():
                shutil.copy(latest, run_dir / "preflight.json")
                txts = sorted(REPORT_ROOT.glob("preflight_*.txt"))
                if txts:
                    shutil.copy(txts[-1], run_dir / "preflight.txt")
            if rc != 0 and not args.force:
                rec.update(status="NOT READY")
                status["failure"] = {"stage": s.key, "log": log, "tail": tail}
                result = "ABORTED (preflight NOT READY — fix items or pass --force)"
                break
            rec.update(status="ok" if rc == 0 else "not ready (forced on)")
            write_status(run_dir, status); continue

        ok = rc == 0
        if ok and s.gate and not s.gate(tail):
            ok = False
            tail += ("\n\nGATE FAILED: build_state.py did not print 'source: llm_cache' — sentiment "
                     "was not produced. Do not train PPO on a neutral placeholder. Check the "
                     "sentiment stage log and Ollama." if s.key == "build_state" else
                     "\n\nGATE FAILED: see message above.")
        if not ok and s.optional:
            rec.update(status=f"failed (optional, rc={rc}) — continuing")
            print(f"\n!!! {s.key} failed (rc={rc}) but is optional; continuing")
            write_status(run_dir, status); continue
        if not ok:
            rec.update(status=f"FAILED (rc={rc})")
            status["failure"] = {"stage": s.key, "log": log, "tail": tail}
            result = f"FAILED at {s.key}"
            write_status(run_dir, status)
            break
        rec.update(status="ok")
        write_status(run_dir, status)

    status["finished"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    status["result"] = result
    write_status(run_dir, status)
    write_summary(run_dir, status)
    print(f"\n=== {result}\nsummary -> {run_dir / 'SUMMARY.md'}")
    return 0 if result == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
