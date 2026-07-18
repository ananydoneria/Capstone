"""Global determinism control (academic reproducibility guardrail).

Call ``set_global_seed(cfg.project.seed)`` as the FIRST action of every
entrypoint script. Covers: python ``random``, ``PYTHONHASHSEED``, NumPy,
PyTorch (CPU + all CUDA devices, deterministic kernels, cuDNN).

Gymnasium and Stable-Baselines3 do not have process-global seeds; they are
seeded at their own boundaries, always from the same config value:
  * ``env.reset(seed=cfg.project.seed)`` / ``env.action_space.seed(...)``
  * ``PPO(..., seed=cfg.project.seed)``
The LLM layer is made deterministic separately via temperature=0.0 and a
fixed sampler seed in config.yaml (enforced by config validation).
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return  # torch-free contexts (e.g. pure data-pipeline scripts) are fine

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # Required by CUDA >= 10.2 for deterministic cuBLAS matmuls
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
