"""Pre-flight checks for the training PC: is everything in place to run the
runbook (Ollama sentiment -> state gate -> PPO x4 on the GPU -> evaluate)?

Standard library only, on purpose: it must run even when the project venv is
missing or broken. Anything that needs project packages is probed by running
the venv's python in a subprocess.

    python strongpc\\preflight.py            # full check incl. the test suite
    python strongpc\\preflight.py --skip-tests
    python strongpc\\preflight.py --python C:\\path\\to\\python.exe   # override venv

Prints a scored table, names the fix script for every failed item, and writes
reports/strongpc/preflight_<timestamp>.txt + .json.
Exit code: 0 = READY (all required checks pass), 1 = NOT READY.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIN = os.name == "nt"
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if WIN else "bin/python")
REPORT_DIR = ROOT / "reports" / "strongpc"
OLLAMA_URL = "http://127.0.0.1:11434"

# import name -> pip distribution name (what requirements.txt lists)
PACKAGES = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"), ("yaml", "pyyaml"),
    ("pydantic", "pydantic"), ("h5py", "h5py"), ("yfinance", "yfinance"),
    ("networkx", "networkx"), ("torch", "torch"), ("torch_geometric", "torch-geometric"),
    ("langchain_core", "langchain-core"), ("langchain_ollama", "langchain-ollama"),
    ("ollama", "ollama"), ("requests", "requests"), ("gymnasium", "gymnasium"),
    ("stable_baselines3", "stable-baselines3"), ("tensorboard", "tensorboard"),
    ("tqdm", "tqdm"), ("rich", "rich"), ("pytest", "pytest"), ("matplotlib", "matplotlib"),
    ("reportlab", "reportlab"),
]
MIN_DRIVER_FOR_CU121 = 531   # NVIDIA driver needed by the cu121 torch wheels


@dataclass
class Check:
    key: str
    name: str
    ok: bool | None            # None = informational (never counts)
    detail: str
    weight: int = 1
    required: bool = False
    fix: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.ok is None:
            return "INFO"
        if self.ok:
            return "OK"
        return "FAIL" if self.required else "WARN"


# ------------------------------------------------------------------ helpers

def run(cmd: list[str], timeout: int = 60, cwd: Path | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=str(cwd or ROOT), encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"


def py_eval(python: Path | None, code: str, timeout: int = 120) -> tuple[int, str]:
    if python is None or not Path(python).exists():
        return 127, "no venv python"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
    try:
        p = subprocess.run([str(python), "-c", code], capture_output=True, text=True,
                           timeout=timeout, cwd=str(ROOT), env=env, encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout.strip() or p.stderr.strip())
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"


def http_get(url: str, timeout: int = 5) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, str(e)
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def http_post_json(url: str, payload: dict, timeout: int = 120) -> tuple[int, str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def find_ollama() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe
    if WIN:
        cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if cand.exists():
            return str(cand)
    return None


def config_value(section: str, key: str) -> str | None:
    """Tiny YAML-free reader: first `key:` under `section:` in config.yaml."""
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    m = re.search(rf"^{section}:\s*\n(.*?)(?=^\S|\Z)", text, re.S | re.M)
    if not m:
        return None
    k = re.search(rf"^\s+{key}:\s*([^\s#]+)", m.group(1), re.M)
    return k.group(1).strip('"\'') if k else None


def gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


# ------------------------------------------------------------------ checks

def check_host_python() -> Check:
    v = sys.version_info
    ok = (3, 9) <= (v.major, v.minor) <= (3, 12)
    return Check("host_python", "Python launcher on this PC", ok,
                 f"{platform.python_version()} at {sys.executable}"
                 + ("" if ok else " — need 3.9–3.12 (torch<2.6 has no wheels outside that; 3.11 recommended)"),
                 weight=3, required=True, fix="fix_01_python.bat")


def check_venv(python: Path) -> Check:
    if not python.exists():
        return Check("venv", "Project virtualenv (.venv)", False, f"missing: {python}",
                     weight=5, required=True, fix="fix_02_venv.bat")
    cfg = python.parents[1] / "pyvenv.cfg"
    home = ""
    if cfg.exists():
        m = re.search(r"^home\s*=\s*(.+)$", cfg.read_text(errors="replace"), re.M)
        home = m.group(1).strip() if m else ""
        if home and not Path(home).exists():
            return Check("venv", "Project virtualenv (.venv)", False,
                         f"pyvenv.cfg points at {home} which does not exist — this .venv was copied "
                         f"from another machine. Delete .venv and recreate.",
                         weight=5, required=True, fix="fix_02_venv.bat")
    rc, out = run([str(python), "--version"], timeout=30)
    if rc != 0:
        return Check("venv", "Project virtualenv (.venv)", False, f"python does not run: {out}",
                     weight=5, required=True, fix="fix_02_venv.bat")
    ver = out.replace("Python ", "")
    mm = tuple(int(x) for x in ver.split(".")[:2])
    ok = (3, 9) <= mm <= (3, 12)
    return Check("venv", "Project virtualenv (.venv)", ok,
                 f"Python {ver} (base: {home or 'n/a'})" + ("" if ok else " — need 3.9–3.12"),
                 weight=5, required=True, fix="fix_02_venv.bat")


def check_packages(python: Path) -> list[Check]:
    code = (
        "import importlib, json\n"
        "names = " + json.dumps([p[0] for p in PACKAGES]) + "\n"
        "out = {}\n"
        "for n in names:\n"
        "    try:\n"
        "        m = importlib.import_module(n); out[n] = getattr(m, '__version__', 'ok')\n"
        "    except Exception as e:\n"
        "        out[n] = None\n"
        "print(json.dumps(out))"
    )
    rc, out = py_eval(python, code, timeout=180)
    if rc != 0 or not out.startswith("{"):
        return [Check("packages", "Required packages importable", False,
                      f"could not probe venv: {out[:200]}", weight=8, required=True,
                      fix="fix_02_venv.bat")]
    versions = json.loads(out)
    missing = [dist for imp, dist in PACKAGES if versions.get(imp) is None]
    checks = [Check("packages", "Required packages importable", not missing,
                    (f"all {len(PACKAGES)} present" if not missing else
                     f"missing {len(missing)}: {', '.join(missing)}"),
                    weight=8, required=True, fix="fix_02_venv.bat", extra=versions)]
    tv = versions.get("torch")
    if tv:
        base = tv.split("+")[0]
        mm = tuple(int(x) for x in base.split(".")[:2])
        ok = (2, 3) <= mm < (2, 6)
        checks.append(Check("torch_version", "torch version in requirements range", ok,
                            f"torch {tv}" + ("" if ok else " — requirements pin >=2.3,<2.6"),
                            weight=2, required=True, fix="fix_03_torch_cuda.bat"))
    if versions.get("tqdm") is None or versions.get("rich") is None:
        checks.append(Check("sb3_progress", "tqdm + rich (SB3 progress bar)", False,
                            "train_ppo.py uses progress_bar=True and will crash without them",
                            weight=3, required=True, fix="fix_02_venv.bat"))
    return checks


def check_nvidia() -> tuple[Check, dict]:
    rc, out = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,"
                   "compute_cap", "--format=csv,noheader,nounits"], timeout=30)
    info: dict = {}
    if rc != 0:
        return Check("nvidia_driver", "NVIDIA driver / nvidia-smi", False,
                     "nvidia-smi not found or failed — no usable NVIDIA driver",
                     weight=6, required=False, fix="fix_08_nvidia_driver.bat"), info
    first = out.splitlines()[0].split(",")
    name, drv, total, used, cc = [x.strip() for x in first[:5]]
    info = {"name": name, "driver": drv, "vram_total_mb": float(total), "vram_used_mb": float(used),
            "compute_capability": cc}
    try:
        drv_major = int(drv.split(".")[0])
    except ValueError:
        drv_major = 0
    ok = drv_major >= MIN_DRIVER_FOR_CU121
    detail = f"{name}, driver {drv}, VRAM {float(total) / 1024:.1f} GB, compute {cc}"
    if not ok:
        detail += f" — driver < {MIN_DRIVER_FOR_CU121}, too old for cu121 torch wheels"
    return Check("nvidia_driver", "NVIDIA driver / nvidia-smi", ok, detail,
                 weight=6, required=False, fix="fix_08_nvidia_driver.bat", extra=info), info


def check_torch_cuda(python: Path) -> Check:
    code = (
        "import json, time, torch\n"
        "d = {'available': torch.cuda.is_available(), 'torch': torch.__version__,\n"
        "     'cuda': torch.version.cuda, 'cudnn': None, 'device': None, 'matmul_ms': None}\n"
        "if d['available']:\n"
        "    d['device'] = torch.cuda.get_device_name(0)\n"
        "    d['cudnn'] = torch.backends.cudnn.version()\n"
        "    a = torch.randn(4096, 4096, device='cuda'); torch.cuda.synchronize()\n"
        "    t = time.time(); b = a @ a; torch.cuda.synchronize(); d['matmul_ms'] = (time.time()-t)*1000\n"
        "    d['result_finite'] = bool(torch.isfinite(b).all())\n"
        "print(json.dumps(d))"
    )
    rc, out = py_eval(python, code, timeout=300)
    if rc != 0 or not out.startswith("{"):
        return Check("torch_cuda", "torch can use the GPU (CUDA)", False,
                     f"probe failed: {out[-200:]}", weight=10, required=False,
                     fix="fix_03_torch_cuda.bat")
    d = json.loads(out)
    if not d["available"]:
        cpu_build = "+cpu" in d["torch"] or d["cuda"] is None
        why = ("CPU-only torch build installed" if cpu_build
               else f"torch {d['torch']} (cuda {d['cuda']}) cannot see a GPU — driver problem?")
        return Check("torch_cuda", "torch can use the GPU (CUDA)", False, why,
                     weight=10, required=False,
                     fix="fix_03_torch_cuda.bat" if cpu_build else "fix_08_nvidia_driver.bat",
                     extra=d)
    ok = d.get("result_finite", False)
    return Check("torch_cuda", "torch can use the GPU (CUDA)", ok,
                 f"{d['device']}, torch {d['torch']}, CUDA {d['cuda']}, cuDNN {d['cudnn']}, "
                 f"4096² matmul {d['matmul_ms']:.0f} ms" + ("" if ok else " — GPU result not finite"),
                 weight=10, required=False, fix="fix_03_torch_cuda.bat", extra=d)


def check_vram(info: dict) -> Check:
    if not info:
        return Check("vram", "Free VRAM >= 8 GB", None, "no GPU info", weight=0)
    free = (info["vram_total_mb"] - info["vram_used_mb"]) / 1024
    ok = free >= 8
    return Check("vram", "Free VRAM >= 8 GB (8B LLM ~5 GB + PPO)", ok,
                 f"{free:.1f} GB free of {info['vram_total_mb'] / 1024:.1f} GB"
                 + ("" if ok else " — close other GPU apps"), weight=3, required=False, fix="")


def check_ollama_cli() -> Check:
    exe = find_ollama()
    if not exe:
        return Check("ollama_cli", "Ollama installed", False, "ollama.exe not found",
                     weight=5, required=True, fix="fix_04_ollama.bat")
    rc, out = run([exe, "--version"], timeout=30)
    return Check("ollama_cli", "Ollama installed", rc == 0, f"{out.splitlines()[0] if out else exe}",
                 weight=5, required=True, fix="fix_04_ollama.bat", extra={"exe": exe})


def check_ollama_server() -> tuple[Check, list[str]]:
    status, body = http_get(f"{OLLAMA_URL}/api/tags", timeout=5)
    if status != 200:
        return Check("ollama_server", "Ollama server reachable (port 11434)", False,
                     f"{body[:120]} — start it: run 'ollama serve' or launch the Ollama app",
                     weight=5, required=True, fix="fix_04_ollama.bat"), []
    try:
        models = [m["name"] for m in json.loads(body).get("models", [])]
    except Exception:
        models = []
    return Check("ollama_server", "Ollama server reachable (port 11434)", True,
                 f"{len(models)} model(s) available", weight=5, required=True,
                 fix="fix_04_ollama.bat", extra={"models": models}), models


def check_ollama_model(models: list[str], wanted: str) -> Check:
    ok = wanted in models
    return Check("ollama_model", f"LLM pulled ({wanted})", ok,
                 "present" if ok else f"not pulled (have: {', '.join(models) or 'none'})",
                 weight=5, required=True, fix="fix_05_llama_model.bat")


def check_ollama_smoke(wanted: str) -> Check:
    payload = {
        "model": wanted, "stream": False, "format": "json",
        "options": {"temperature": 0.0, "seed": 42, "num_predict": 64},
        "messages": [
            {"role": "system", "content": "Reply with JSON only: {\"sentiment\": float in [-1,1], "
                                          "\"confidence\": float in [0,1]}."},
            {"role": "user", "content": "Maruti Suzuki reports record quarterly profit, raises guidance."},
        ],
    }
    t0 = time.time()
    status, body = http_post_json(f"{OLLAMA_URL}/api/chat", payload, timeout=300)
    dt = time.time() - t0
    if status != 200:
        return Check("ollama_smoke", "LLM answers a sentiment prompt", False, body[:160],
                     weight=5, required=False, fix="fix_05_llama_model.bat")
    try:
        resp = json.loads(body)
        content = resp["message"]["content"]
        parsed = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
        s = float(parsed["sentiment"])
        ev = resp.get("eval_count", 0); ed = resp.get("eval_duration", 0)
        tps = ev / (ed / 1e9) if ed else 0
    except Exception as e:
        return Check("ollama_smoke", "LLM answers a sentiment prompt", False,
                     f"unparseable reply ({type(e).__name__}): {body[:120]}",
                     weight=5, required=False, fix="")
    # is it on the GPU?
    st, ps = http_get(f"{OLLAMA_URL}/api/ps", timeout=5)
    gpu_note = ""
    if st == 200:
        for m in json.loads(ps).get("models", []):
            if m.get("name") == wanted:
                sv, sz = m.get("size_vram", 0), m.get("size", 1)
                pct = 100 * sv / sz if sz else 0
                gpu_note = f", {pct:.0f}% in VRAM" + (" (GPU)" if pct > 90 else " — mostly CPU! slow")
    ok = -1 <= s <= 1 and dt < 60
    return Check("ollama_smoke", "LLM answers a sentiment prompt", ok,
                 f"sentiment={s:+.2f} in {dt:.1f}s, {tps:.0f} tok/s{gpu_note}",
                 weight=5, required=False, fix="", extra={"latency_s": dt, "tok_s": tps})


def check_files(key: str, name: str, files: list[Path], fix: str, weight: int = 5) -> Check:
    missing = [f for f in files if not f.exists()]
    if missing:
        return Check(key, name, False, "missing: " + ", ".join(str(m.relative_to(ROOT)) for m in missing),
                     weight=weight, required=True, fix=fix)
    sizes = ", ".join(f"{f.name} {f.stat().st_size / 1e6:.1f} MB" for f in files)
    return Check(key, name, True, sizes, weight=weight, required=True, fix=fix)


def check_news() -> list[Check]:
    g = ROOT / "data" / "raw" / "google_news"
    f = ROOT / "data" / "raw" / "finnhub"
    g_n = sum(1 for _ in g.glob("*/*.json")) if g.exists() else 0
    f_n = sum(1 for _ in f.glob("*/*.json")) if f.exists() else 0
    out = [Check("news_google", "Google News corpus cached", g_n > 0,
                 f"{g_n} month files" if g_n else "empty — run the backfill",
                 weight=4, required=True, fix="fix_07_news.bat")]
    out.append(Check("news_finnhub", "Finnhub corpus cached (optional)", f_n > 0 or None,
                     f"{f_n} month files" if f_n else "none (needs FINNHUB_API_KEY; optional)",
                     weight=0, fix="fix_07_news.bat"))
    key = os.environ.get("FINNHUB_API_KEY")
    out.append(Check("finnhub_key", "FINNHUB_API_KEY set (optional)", bool(key),
                     "set" if key else "not set — Finnhub backfill will be skipped (allowed)",
                     weight=1, required=False, fix="set FINNHUB_API_KEY=... then fix_07_news.bat"))
    return out


def check_internet() -> Check:
    status, _ = http_get("https://news.google.com/rss/search?q=test&hl=en-IN&gl=IN", timeout=8)
    return Check("internet", "Internet (Google News RSS) reachable", status == 200,
                 "ok" if status == 200 else "unreachable — only matters if news backfill is incomplete",
                 weight=2, required=False, fix="")


def check_disk() -> Check:
    u = shutil.disk_usage(ROOT)
    ok = u.free >= 10e9
    return Check("disk", "Free disk >= 10 GB", ok, f"{gb(u.free)} free of {gb(u.total)}",
                 weight=2, required=False, fix="free up space")


def check_ram() -> Check:
    total = None
    try:
        if WIN:
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS(); ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            total = ms.ullTotalPhys
        elif hasattr(os, "sysconf"):
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        total = None
    if total is None:
        return Check("ram", "RAM >= 16 GB", None, "could not determine", weight=0)
    ok = total >= 15e9
    return Check("ram", "RAM >= 16 GB", ok, gb(total), weight=2, required=False, fix="")


def check_cpu() -> Check:
    n = os.cpu_count() or 0
    n_envs = int(config_value("rl", "n_envs") or 8)
    return Check("cpu", f"CPU cores >= rl.n_envs ({n_envs})", n >= n_envs, f"{n} logical cores",
                 weight=1, required=False, fix="")


def check_sleep() -> Check:
    if not WIN:
        return Check("sleep", "PC will not sleep during training", None, "Windows-only check", weight=0)
    rc, out = run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"], timeout=30)
    m = re.search(r"Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)", out)
    if rc != 0 or not m:
        return Check("sleep", "PC will not sleep during training", None, "could not read powercfg", weight=0)
    secs = int(m.group(1), 16)
    ok = secs == 0
    return Check("sleep", "PC will not sleep during training", ok,
                 "sleep disabled on AC power" if ok else f"sleeps after {secs // 60} min on AC — "
                 f"the multi-hour sentiment run WILL be interrupted",
                 weight=3, required=False, fix="fix_09_no_sleep.bat")


def check_tests(python: Path) -> Check:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    try:
        p = subprocess.run([str(python), "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
                            "-W", "ignore"], capture_output=True, text=True, timeout=1200,
                           cwd=str(ROOT), env=env, encoding="utf-8", errors="replace")
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return Check("tests", "Test suite (pytest tests)", False, "timed out after 20 min",
                     weight=8, required=True, fix="read the pytest output")
    dt = time.time() - t0
    m = re.search(r"(\d+) passed", out); passed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) failed", out); failed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) skipped", out); skipped = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) error", out); errors = int(m.group(1)) if m else 0
    ok = failed == 0 and errors == 0 and passed >= 102
    detail = f"{passed} passed, {failed} failed, {skipped} skipped, {errors} errors in {dt:.0f}s"
    if ok and skipped == 8:
        detail += " — expected pre-sentiment state"
    elif ok and skipped <= 1:
        detail += " — sentiment already built"
    elif not ok:
        detail += " — expected 102 passed / 8 skipped (or 109 / 1 after sentiment)"
    return Check("tests", "Test suite (pytest tests)", ok, detail, weight=8, required=True,
                 fix="fix_06_regenerate_data.bat if >8 skipped; otherwise read the log",
                 extra={"passed": passed, "failed": failed, "skipped": skipped, "output_tail": out[-1500:]})


def check_state_and_ppo(python: Path) -> list[Check]:
    out = []
    h5 = ROOT / "data" / "processed" / "state.h5"
    if h5.exists():
        rc, res = py_eval(python, "import h5py; f=h5py.File('data/processed/state.h5','r'); "
                                  "print(f.attrs.get('sentiment_source','?'))", timeout=60)
        src = res if rc == 0 else "unreadable"
        out.append(Check("state_h5", "state.h5 sentiment source", None,
                         f"{src} — {'ready for PPO' if src == 'llm_cache' else 'STEP 3+4 still needed'}",
                         weight=0))
    else:
        out.append(Check("state_h5", "state.h5 sentiment source", None, "not built yet (STEP 4)", weight=0))
    sent = ROOT / "data" / "processed" / "sentiment.parquet"
    out.append(Check("sentiment_cache", "LLM sentiment cache", None,
                     f"present ({sent.stat().st_size / 1e6:.1f} MB)" if sent.exists() else "not computed yet (STEP 3)",
                     weight=0))
    arms = ["full", "no-gnn", "no-sentiment", "neither"]
    done = [a for a in arms if (ROOT / "src" / "rl_agent" / "logs" / f"ppo_{a}" / "model.zip").exists()]
    out.append(Check("ppo_arms", "PPO arms already trained", None,
                     f"{len(done)}/4: {', '.join(done) or 'none'}", weight=0))
    rc, res = run(["git", "diff", "--quiet", "HEAD", "--", "config.yaml"], timeout=30)
    if rc in (0, 1):
        out.append(Check("config_clean", "config.yaml unmodified vs git", rc == 0,
                         "unchanged" if rc == 0 else "MODIFIED — results will not be reproducible; "
                         "the runbook forbids editing it", weight=1, required=False, fix="git checkout config.yaml"))
    return out


def check_ppo_device(python: Path) -> Check:
    code = ("import torch\n"
            "from stable_baselines3.common.utils import get_device\n"
            "print(get_device('auto'))")
    rc, out = py_eval(python, code, timeout=120)
    if rc != 0:
        return Check("ppo_device", "PPO device (SB3 'auto')", None, f"probe failed: {out[-100:]}", weight=0)
    dev = out.strip()
    note = ("GPU — note SB3 prints a warning that MlpPolicy can be faster on CPU; that is expected"
            if "cuda" in dev else "CPU — install the CUDA torch wheel to use the GPU")
    return Check("ppo_device", "PPO device (SB3 'auto')", None, f"{dev}: {note}", weight=0)


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-tests", action="store_true", help="skip the pytest run (fast mode)")
    ap.add_argument("--skip-llm", action="store_true", help="skip the live Ollama prompt")
    ap.add_argument("--python", default=os.environ.get("STRONGPC_PYTHON"),
                    help="python to probe instead of .venv (testing only)")
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args()

    python = Path(args.python) if args.python else VENV_PY
    if args.python and not python.exists():
        w = shutil.which(args.python)
        python = Path(w) if w else python
    wanted_model = config_value("llm", "model") or "llama3:8b-instruct-q4_K_M"

    checks: list[Check] = []
    t0 = time.time()
    checks.append(check_host_python())
    checks.append(check_venv(python))
    venv_ok = checks[-1].ok
    if venv_ok:
        checks += check_packages(python)
    nv, gpu_info = check_nvidia()
    checks.append(nv)
    if venv_ok:
        checks.append(check_torch_cuda(python))
    checks.append(check_vram(gpu_info))
    checks.append(check_ollama_cli())
    srv, models = check_ollama_server()
    checks.append(srv)
    checks.append(check_ollama_model(models, wanted_model))
    if srv.ok and wanted_model in models and not args.skip_llm:
        checks.append(check_ollama_smoke(wanted_model))
    d = ROOT / "data"
    checks.append(check_files("data_artifacts", "Price data + features + GNN inputs + GNN scores", [
        d / "raw" / "relationships.csv", d / "raw" / "corporate_actions.csv",
        d / "processed" / "ohlcv_panel.parquet", d / "processed" / "features.parquet",
        d / "processed" / "gnn_node_features.parquet", d / "processed" / "market_context.parquet",
        d / "processed" / "gnn_scores.parquet",
    ], fix="fix_06_regenerate_data.bat"))
    w = ROOT / "src" / "models" / "gnn" / "weights"
    checks.append(check_files("gnn_weights", "Trained GNN weights", [
        w / "propagation_gnn.pt", w / "feature_norm.pt", w / "train_meta.json",
    ], fix="fix_06_regenerate_data.bat"))
    checks += check_news()
    checks.append(check_internet())
    checks.append(check_disk())
    checks.append(check_ram())
    checks.append(check_cpu())
    checks.append(check_sleep())
    if venv_ok:
        checks += check_state_and_ppo(python)
        checks.append(check_ppo_device(python))
        if not args.skip_tests:
            checks.append(check_tests(python))
        else:
            checks.append(Check("tests", "Test suite (pytest tests)", None, "skipped (--skip-tests)", weight=0))

    # ---- score
    scored = [c for c in checks if c.ok is not None and c.weight > 0]
    total_w = sum(c.weight for c in scored)
    got_w = sum(c.weight for c in scored if c.ok)
    pct = 100 * got_w / total_w if total_w else 0
    req_fail = [c for c in checks if c.required and c.ok is False]
    gpu_ready = any(c.key == "torch_cuda" and c.ok for c in checks)
    llm_ready = all((c.ok is not False) for c in checks if c.key in ("ollama_cli", "ollama_server", "ollama_model"))
    verdict = "READY" if not req_fail else "NOT READY"

    # ---- render
    lines = []
    lines.append(f"STRONG-PC PREFLIGHT  {datetime.now():%Y-%m-%d %H:%M}  {platform.node()}  "
                 f"{platform.system()} {platform.release()}")
    lines.append(f"project: {ROOT}")
    lines.append("")
    lines.append(f"{'STATUS':6} {'CHECK':46} DETAIL")
    lines.append("-" * 110)
    for c in checks:
        lines.append(f"[{c.status:4}] {c.name[:46]:46} {c.detail}")
        if c.ok is False and c.fix:
            lines.append(f"{'':6} {'':46} -> fix: strongpc\\{c.fix}" if c.fix.startswith("fix_")
                         else f"{'':6} {'':46} -> fix: {c.fix}")
    lines.append("-" * 110)
    n_ok = sum(1 for c in scored if c.ok); n_fail = sum(1 for c in scored if not c.ok)
    lines.append(f"SCORE: {got_w}/{total_w} weighted points = {pct:.0f}%   "
                 f"({n_ok} passed, {n_fail} failed, {len(checks) - len(scored)} informational)")
    lines.append(f"VERDICT: {verdict}   GPU training: {'yes' if gpu_ready else 'NO (CPU fallback)'}   "
                 f"LLM sentiment: {'yes' if llm_ready else 'NO'}")
    if req_fail:
        lines.append("")
        lines.append("Required checks failing — run these, in order, then re-run preflight:")
        seen = []
        for c in req_fail:
            if c.fix and c.fix not in seen:
                seen.append(c.fix)
                lines.append(f"  - {c.fix}   ({c.name})")
    else:
        warns = [c for c in checks if c.ok is False]
        if warns:
            lines.append("")
            lines.append("Optional items you may still want to fix:")
            for c in warns:
                lines.append(f"  - {c.fix or c.name}   ({c.detail[:80]})")
        lines.append("")
        lines.append("Next: strongpc\\10_run_pipeline.bat")
    lines.append(f"(preflight took {time.time() - t0:.0f}s)")
    text = "\n".join(lines)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (REPORT_DIR / f"preflight_{stamp}.txt").write_text(text, encoding="utf-8")
    (REPORT_DIR / f"preflight_{stamp}.json").write_text(json.dumps({
        "timestamp": stamp, "host": platform.node(), "verdict": verdict, "score_pct": pct,
        "points": [got_w, total_w], "gpu_ready": gpu_ready, "llm_ready": llm_ready,
        "checks": [asdict(c) | {"status": c.status} for c in checks],
    }, indent=2, default=str), encoding="utf-8")
    (REPORT_DIR / "preflight_latest.json").write_text(
        (REPORT_DIR / f"preflight_{stamp}.json").read_text(encoding="utf-8"), encoding="utf-8")

    if not args.json_only:
        print(text)
        print(f"\nsaved -> {REPORT_DIR / f'preflight_{stamp}.txt'}")
    return 0 if verdict == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
