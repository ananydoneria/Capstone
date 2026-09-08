================================================================================
 STRONG PC TOOLKIT  -  run the training without Claude Code
================================================================================
Double-click the .bat files in this folder in the order below. Every script
prints what it is doing, saves its output under  reports\strongpc\  and waits
for a key press at the end so the window stays open.

 STEP 1  00_preflight.bat
         Checks everything the training needs and prints a SCORE, e.g.
             SCORE: 61/74 weighted points = 82%
             VERDICT: NOT READY   GPU training: yes   LLM sentiment: NO
         Every failed line names the fix script to run. Required items
         (python, venv, packages, Ollama + model, data, weights, tests) decide
         READY / NOT READY; GPU items only decide fast vs slow.

 STEP 2  fix_XX_*.bat  -  ONLY the ones preflight told you to run, each once:
         fix_01_python.bat          install Python 3.11 (winget)
         fix_02_venv.bat            create .venv + install requirements
                                    (installs the CUDA torch wheel first when
                                    an NVIDIA GPU is present)
         fix_03_torch_cuda.bat      swap CPU torch for the CUDA 12.1 build
         fix_04_ollama.bat          install Ollama + start the server
         fix_05_llama_model.bat     pull llama3:8b-instruct-q4_K_M (4.7 GB)
         fix_06_regenerate_data.bat prices + GNN  (ONLY if they did not copy;
                                    it overwrites the shipped GNN weights)
         fix_07_news.bat            news corpus backfill (resume-safe)
         fix_08_nvidia_driver.bat   opens the driver download page (manual)
         fix_09_no_sleep.bat        stop Windows sleeping mid-training
         Re-run 00_preflight.bat until it says READY.

 STEP 3  10_run_pipeline.bat
         Runs the whole runbook unattended and logs every stage:
             preflight -> tests -> news backfill -> LLM sentiment (Ollama)
             -> build state (GATE: must say source: llm_cache) -> baselines
             -> PPO x4 ablation arms (2M steps each, HOURS) -> evaluate
             -> REPORT.md -> tests again -> GNN report -> bundle everything
         Output: reports\strongpc\run_<timestamp>\
             NN_<stage>.log   full log per stage
             status.json      live progress (open it any time)
             gpu_usage.csv    GPU utilisation / VRAM / temp every 30 s
             SUMMARY.md       one-page summary, written even on failure
             artifacts\       REPORT.md, equity_curves.png, per-arm metrics,
                              TensorBoard scalars as CSV, training-curve plot,
                              GNN report md/pdf, sentiment summary ...
         plus reports\strongpc\run_<timestamp>.zip  <- copy this back.

         If it stops: read SUMMARY.md (it names the failing stage and shows the
         last lines of its log), fix the cause, then run 11_resume_pipeline.bat
         and type the stage to resume from. PPO arms that already finished are
         skipped automatically (model.zip exists).

 ANY TIME
         20_run_tests.bat       tests + GNN report only (no training)
         30_collect_reports.bat re-bundle results into a zip
         40_tensorboard.bat     watch PPO training live at localhost:6006

--------------------------------------------------------------------------------
 RULES (from CLAUDE.md - the scripts enforce these, do not work around them)
--------------------------------------------------------------------------------
 - Do not edit config.yaml (seed, timesteps, model sizes). Preflight warns if
   it differs from git.
 - Do not lower total_timesteps for the real run. run_pipeline.py --timesteps N
   exists for a 5-minute smoke test only and prints a loud warning.
 - The state gate is real: if build_state prints neutral_placeholder the
   pipeline stops. Fix Ollama/sentiment, do not skip it.
 - Do not re-run fix_06 (regenerate) if the GNN weights are present.
 - No real broker orders, ever.

--------------------------------------------------------------------------------
 COMMAND-LINE USE (optional; same scripts, from a terminal in the project root)
--------------------------------------------------------------------------------
   .venv\Scripts\python strongpc\preflight.py [--skip-tests] [--skip-llm]
   .venv\Scripts\python strongpc\run_pipeline.py [--from STAGE] [--only a,b]
                                                 [--skip a,b] [--retrain]
                                                 [--force] [--dry-run]
   .venv\Scripts\python strongpc\collect_reports.py [--run-dir DIR]

 Smoke test of the full chain in a few minutes (results are NOT valid):
   .venv\Scripts\python strongpc\run_pipeline.py --timesteps 20000 --retrain
   (then delete src\rl_agent\logs\ppo_* before the real run)
================================================================================
