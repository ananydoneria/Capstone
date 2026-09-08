# GNN Test Report

Generated 2026-09-02 19:05 · python 3.9.6 · torch 2.8.0 · Darwin arm64

**Suite verdict: PASS** — 58 passed, 0 failed, 1 skipped (`pytest tests/gnn`: 58 passed, 1 skipped in 4.15s)

## Headline

| Metric | Value |
|---|---|
| Val AUC (trained checkpoint) | 0.6100 |
| Val AUC recorded at training time | 0.6100 |
| Train AUC | 0.6885 |
| Train/val gap | 0.0785 |
| Baseline: train-period return correlation | 0.5784 |
| Baseline: dst previous-day volatility | 0.3523 |
| Baseline: src previous-day volatility | 0.4231 |
| Val positive rate (chance predictor AUC = 0.5) | 0.2259 |
| Best epoch (early stopping) | 5 |

GNN beats the strongest naive baseline by **+0.0316** AUC on identical validation samples.

## Findings

- Signal is real but modest: val AUC 0.610 vs 23% base rate. The static return-correlation baseline gets 0.578, so the GNN's edge over 'pairs that co-move' is +0.032 AUC — thin; most of the ranking power is structure the correlation already captures.
- Volatility baselines are BELOW 0.5 (dst 0.352, src 0.423): in this val window, high-vol names cascaded LESS often. The vol-scaled shock threshold is doing its job (no 'volatile = shocks' shortcut).
- Model leans on `vol_21` (+0.138 AUC when shuffled) and `volz_21`; `logret_1`, `logret_5`, `logret_21`, `mom_63` contribute ≈ nothing. Return/momentum features are effectively unused — candidate for feature work later.
- Seed sensitivity: 0.610–0.629 across 3 seeds. Shipped seed 42 ranks 3/3 (seed 7 reached 0.629). Spread 0.019 is larger than the margin over the correlation baseline — the headline number is noisy at the ±0.01 level. Shipped epoch 5 is early; other seeds stopped later.
- Not stable month to month: 2026-04 0.32 (n=33) to 2025-08 0.77 (n=76); 3/11 months below 0.5 (2025-10, 2025-11, 2026-04). Monthly n is 6–90 samples so noise is large, but the RL agent should not treat GNN scores as uniformly reliable.
- Direction: downstream (supplier→OEM) AUC 0.622 vs upstream 0.630 — bidirectional supervision is justified; neither direction is dead weight.
- Weakest pairs cluster around `BOSCHLTD.NS` (6 of the below-0.5 pairs touch it). With ~9 val samples per pair this is a hint, not a verdict — worth a look at that node's curated edges in relationships.csv.
- Calibration bins are monotone: bottom quintile 13% actual cascades, top quintile 31%. Absolute p is inflated (mean 0.56) by pos_weight — fine for ranking, do not read as probability.
- Inference cache is bit-identical to a fresh forward pass and provably free of look-ahead (future-day perturbation test). Downstream RL state is safe to trust on that front.
- Train/val gap 0.079 with early stop at epoch 5 — no overfitting concern.

## Data under test

| Item | Value |
|---|---|
| Timeline | 1174 days, 2021-10-01 -> 2026-06-30 |
| Nodes / features | 15 / 6 |
| Curated edges / message-passing edges / scored pairs | 32 / 64 / 64 |
| Train samples | 2133 (17.8% positive), 2021-10-06 -> 2025-05-15 |
| Val samples | 571 (22.6% positive), 2025-07-03 -> 2026-06-17 |
| Overall shock-day rate | 3.61% |
| Config | arch=graphsage, hidden=64, layers=2, dropout=0.4, window=10, gru=32, k=5, shock_z=2.5, min_move=0.02, seed=42 |

Shock days per ticker:

| Ticker | Shock days |
|---|---|
| EXIDEIND.NS | 51 |
| TMPV.NS | 50 |
| EICHERMOT.NS | 45 |
| APOLLOTYRE.NS | 43 |
| ASHOKLEY.NS | 42 |
| BOSCHLTD.NS | 42 |
| MARUTI.NS | 42 |
| MOTHERSON.NS | 42 |
| BAJAJ-AUTO.NS | 41 |
| HEROMOTOCO.NS | 41 |
| M&M.NS | 41 |
| SONACOMS.NS | 41 |
| TVSMOTOR.NS | 41 |
| BHARATFORG.NS | 37 |
| UNOMINDA.NS | 37 |

## Direction split (val)

| Direction | n | pos rate | AUC |
|---|---|---|---|
| downstream | 272 | 19.1% | 0.6224 |
| upstream | 299 | 25.8% | 0.6303 |

`downstream` = curated supplier→buyer edge; `upstream` = reversed edge.

## Stability across the val period (monthly AUC)

| Month | n | pos | AUC |
|---|---|---|---|
| 2025-07 | 73 | 15 | 0.6391 |
| 2025-08 | 76 | 18 | 0.7711 |
| 2025-09 | 30 | 5 | 0.6080 |
| 2025-10 | 39 | 8 | 0.4718 |
| 2025-11 | 49 | 10 | 0.3769 |
| 2025-12 | 6 | 0 | n/a |
| 2026-01 | 86 | 34 | 0.5639 |
| 2026-02 | 39 | 8 | 0.5605 |
| 2026-03 | 67 | 20 | 0.7245 |
| 2026-04 | 33 | 4 | 0.3190 |
| 2026-05 | 47 | 5 | 0.7333 |
| 2026-06 | 26 | 2 | 0.6458 |

Months with a single class show `n/a`. Small n → noisy.

## Calibration (val, quintiles of predicted probability)

| Bin | n | mean predicted p | actual cascade rate |
|---|---|---|---|
| 0 | 115 | 0.2870 | 0.1304 |
| 1 | 114 | 0.4072 | 0.1316 |
| 2 | 114 | 0.5698 | 0.2632 |
| 3 | 114 | 0.7161 | 0.2982 |
| 4 | 114 | 0.8159 | 0.3070 |

Score distribution: mean 0.559, std 0.199, range [0.155, 0.864]. Training uses `pos_weight` (class re-balancing), so absolute p is shifted upward relative to the 22.6% base rate by design — ranking (AUC) is what the RL state consumes, not calibrated probability. Monotone bins = healthy.

## Per-pair val AUC (pairs with ≥8 samples and both classes)

Top 5:

| Pair | n | pos | AUC |
|---|---|---|---|
| BHARATFORG.NS->EICHERMOT.NS | 9 | 2 | 1.0000 |
| BHARATFORG.NS->M&M.NS | 9 | 2 | 1.0000 |
| EXIDEIND.NS->TMPV.NS | 9 | 1 | 1.0000 |
| EXIDEIND.NS->EICHERMOT.NS | 9 | 4 | 0.9500 |
| EXIDEIND.NS->MARUTI.NS | 9 | 1 | 0.8750 |

Bottom 5:

| Pair | n | pos | AUC |
|---|---|---|---|
| BOSCHLTD.NS->ASHOKLEY.NS | 9 | 2 | 0.3571 |
| BOSCHLTD.NS->TMPV.NS | 9 | 2 | 0.3571 |
| HEROMOTOCO.NS->BOSCHLTD.NS | 9 | 2 | 0.3571 |
| BOSCHLTD.NS->EICHERMOT.NS | 9 | 2 | 0.2857 |
| BOSCHLTD.NS->MARUTI.NS | 9 | 4 | 0.2000 |

36 pairs evaluable; 7 below 0.5 (per-pair n is small — treat as indicative, not conclusive).

## Feature permutation importance (val AUC drop when feature is shuffled)

| Feature | AUC drop |
|---|---|
| vol_21 | +0.1376 |
| volz_21 | +0.0446 |
| logret_1 | +0.0062 |
| logret_5 | -0.0015 |
| logret_21 | -0.0083 |
| mom_63 | -0.0083 |

Positive = model relies on it. Near-zero/negative = ignored or noise.

## Inference cache

`gnn_scores.parquet`: 1174 days × 64 pairs, mean 0.556, std 0.186. Max |cached − recomputed| = 0.00e+00 (bit-for-bit reproducible).

## Seed robustness

| Seed | Val AUC | Best epoch | Train time (s) |
|---|---|---|---|
| 42 | 0.6100 | 5 | shipped |
| 7 | 0.6291 | 15 | 32.6 |
| 2026 | 0.6189 | 20 | 38.4 |

Range across seeds: 0.6100 – 0.6291 (spread 0.0191). Retrains went to a temp dir; shipped weights untouched.

## Test inventory

| Status | Test |
|---|---|
| PASSED | tests/gnn/test_dataset.py::test_features_tensor_layout |
| PASSED | tests/gnn/test_dataset.py::test_features_tensor_respects_ticker_order |
| PASSED | tests/gnn/test_dataset.py::test_window_features_full_and_padded |
| PASSED | tests/gnn/test_dataset.py::test_window_width_one_is_the_day_itself |
| PASSED | tests/gnn/test_dataset.py::test_graphdata_window_agrees_with_standalone |
| PASSED | tests/gnn/test_dataset.py::test_real_shapes |
| PASSED | tests/gnn/test_dataset.py::test_real_edge_index_symmetric_and_in_range |
| PASSED | tests/gnn/test_dataset.py::test_real_supervision_pairs_bidirectional |
| PASSED | tests/gnn/test_dataset.py::test_real_train_normalization_uses_train_dates_only |
| PASSED | tests/gnn/test_dataset.py::test_real_split_is_chronological_with_gap |
| PASSED | tests/gnn/test_dataset.py::test_real_split_counts_match_train_meta |
| PASSED | tests/gnn/test_dataset.py::test_real_both_classes_present_in_val |
| PASSED | tests/gnn/test_dataset.py::test_real_samples_reference_valid_pairs |
| PASSED | tests/gnn/test_labels.py::test_shock_threshold_scales_with_previous_day_vol |
| PASSED | tests/gnn/test_labels.py::test_shock_day_zero_never_flags |
| PASSED | tests/gnn/test_labels.py::test_shock_min_move_floor |
| PASSED | tests/gnn/test_labels.py::test_shock_uses_previous_not_same_day_vol |
| PASSED | tests/gnn/test_labels.py::test_shock_matrix_shape_and_dtype |
| PASSED | tests/gnn/test_labels.py::test_label_window_is_exclusive_of_shock_day |
| PASSED | tests/gnn/test_labels.py::test_label_window_boundary_inclusive_at_k |
| PASSED | tests/gnn/test_labels.py::test_only_shocked_source_generates_samples |
| PASSED | tests/gnn/test_labels.py::test_sample_count_equals_shock_days_times_edges |
| PASSED | tests/gnn/test_labels.py::test_truncated_tail_excluded |
| PASSED | tests/gnn/test_labels.py::test_samples_are_hashable_and_frozen |
| PASSED | tests/gnn/test_labels.py::test_temporal_split_fraction_and_gap |
| PASSED | tests/gnn/test_labels.py::test_temporal_split_never_shuffles |
| PASSED | tests/gnn/test_model.py::test_output_shape_and_finite |
| PASSED | tests/gnn/test_model.py::test_seed_determinism |
| PASSED | tests/gnn/test_model.py::test_eval_mode_is_dropout_free |
| PASSED | tests/gnn/test_model.py::test_train_mode_dropout_is_stochastic |
| PASSED | tests/gnn/test_model.py::test_all_parameters_receive_gradients |
| PASSED | tests/gnn/test_model.py::test_pair_scores_independent_of_batching |
| PASSED | tests/gnn/test_model.py::test_direction_matters |
| PASSED | tests/gnn/test_model.py::test_node_permutation_equivariance |
| PASSED | tests/gnn/test_model.py::test_temporal_order_matters |
| PASSED | tests/gnn/test_model.py::test_graph_structure_matters |
| PASSED | tests/gnn/test_model.py::test_zero_padded_window_is_valid_input |
| PASSED | tests/gnn/test_model.py::test_gat_variant_builds_and_runs |
| PASSED | tests/gnn/test_model.py::test_unknown_arch_rejected |
| PASSED | tests/gnn/test_model.py::test_parameter_count_matches_config |
| PASSED | tests/gnn/test_train_utils.py::test_auc_perfect_inverted_and_ties |
| PASSED | tests/gnn/test_train_utils.py::test_auc_single_class_is_nan |
| PASSED | tests/gnn/test_train_utils.py::test_auc_is_rank_based |
| PASSED | tests/gnn/test_train_utils.py::test_batch_by_day_groups_and_orders |
| PASSED | tests/gnn/test_train_utils.py::test_pos_weight_balances_classes |
| PASSED | tests/gnn/test_train_utils.py::test_model_can_overfit_synthetic_cascades |
| PASSED | tests/gnn/test_trained_artifacts.py::test_meta_matches_config |
| PASSED | tests/gnn/test_trained_artifacts.py::test_state_dict_shapes_match_config |
| PASSED | tests/gnn/test_trained_artifacts.py::test_persisted_norm_stats_match_dataset |
| PASSED | tests/gnn/test_trained_artifacts.py::test_val_auc_reproduces_train_meta |
| PASSED | tests/gnn/test_trained_artifacts.py::test_val_auc_above_chance |
| PASSED | tests/gnn/test_trained_artifacts.py::test_gnn_beats_naive_baselines |
| PASSED | tests/gnn/test_trained_artifacts.py::test_train_auc_not_wildly_above_val |
| PASSED | tests/gnn/test_trained_artifacts.py::test_scores_are_probabilities_with_spread |
| PASSED | tests/gnn/test_trained_artifacts.py::test_cached_scores_integrity |
| PASSED | tests/gnn/test_trained_artifacts.py::test_cached_scores_reproducible_from_weights |
| PASSED | tests/gnn/test_trained_artifacts.py::test_inference_has_no_lookahead |
| PASSED | tests/gnn/test_trained_artifacts.py::test_inference_depends_on_recent_history |
| SKIPPED | tests/gnn/test_train_utils.py:36: could not import 'sklearn.metrics': No module named 'sklearn' |

## What the suite covers

- **Labels** (`test_labels.py`): volatility-scaled shock threshold, min-move floor,
  previous-day vol (no self-inflation), day-0 guard, cascade window `(t, t+k]`
  boundaries, truncated-tail exclusion, chronological split with a k-day gap.
- **Dataset bridge** (`test_dataset.py`): tensor layout `[day, node, feature]`,
  ticker-order enforcement, zero-padded windows that never look ahead; on real
  data: shapes, symmetric message passing, bidirectional supervision pairs,
  train-only normalization stats, split hygiene, counts matching `train_meta.json`.
- **Model** (`test_model.py`): determinism, dropout on/off behaviour, every
  parameter receives gradient, batching invariance, direction sensitivity,
  node-permutation equivariance, temporal-order sensitivity, graph-structure
  sensitivity, GAT variant, parameter count vs config.
- **Training utils** (`test_train_utils.py`): rank AUC correctness (incl. ties,
  degenerate classes, sklearn cross-check when installed), per-day batching,
  class-weight balance, and an overfit smoke test proving the full training
  loop can learn a planted rule.
- **Trained artifacts** (`test_trained_artifacts.py`): meta/config agreement,
  state-dict shapes, persisted norm stats, val AUC reproduces the recorded
  value, beats both naive baselines, train/val gap bounded, score spread, cache
  integrity + bitwise reproducibility, no-lookahead under future perturbation,
  and confirmation that the temporal window is actually used.
