# GNN Test Report

Generated 2026-09-15 23:52 · python 3.9.6 · torch 2.8.0 · Darwin arm64

**Suite verdict: PASS** — 76 passed, 0 failed, 1 skipped (`pytest tests/gnn`: 76 passed, 1 skipped in 7.53s)

## Headline

| Metric | Value |
|---|---|
| Val AUC (trained checkpoint) | 0.6002 |
| Val AUC recorded at training time | 0.6002 |
| Train AUC | 0.6721 |
| Train/val gap | 0.0719 |
| Baseline: train-period return correlation | 0.5269 |
| Baseline: dst previous-day volatility (raw AUC) | 0.4868 |
| Baseline: dst volatility rule, best direction | 0.5132 |
| Baseline: src previous-day volatility (raw AUC) | 0.4772 |
| Val positive rate (chance predictor AUC = 0.5) | 0.1683 |
| Best epoch (early stopping) | 3 |

GNN beats the strongest naive baseline by **+0.0733** AUC on identical validation samples.

## Out-of-sample: walk-forward comparison

Five half-year test windows (2023H2–2025H2), expanding-window retraining, 3 seeds, 1131 samples over 183 shock days. 2026 (PPO test) never used. Current config = **V4**.

| Variant | Description | AUC | 95% CI | P(better than V0) |
|---|---|---|---|---|
| V0 | previous model shape (2021+ data, SAGE, no pair features) | 0.5384 | 0.474–0.593 | — |
| V1 | long history 2011+ (adjusted, masked) | 0.5823 | 0.531–0.639 | 83% |
| V2 | V1 + edge weights (GraphConv) | 0.5691 | 0.521–0.623 | 77% |
| V3 | V1 + pair features | 0.5886 | 0.530–0.646 | 88% |
| **V4** | V1 + edge weights + pair features (idea 2) | 0.5842 | 0.535–0.636 | 86% |
| V5 | V1 + India context + breadth | 0.5636 | 0.482–0.633 | 76% |
| V6 | V1 + full context + breadth, PCA-8 | 0.5150 | 0.456–0.580 | 35% |
| V7 | V4 + India context + breadth | 0.5660 | 0.494–0.632 | 79% |
| V8 | V4 + full context + breadth, PCA-8 | 0.5646 | 0.510–0.625 | 74% |
| V9 | V4 + full context + breadth (raw 151) | 0.5000 | 0.430–0.566 | 11% |

Full tables: `reports/gnn_walkforward/WALKFORWARD.md`.

## Findings

- Signal is real but modest: val AUC 0.600 vs 17% base rate. The static return-correlation baseline gets 0.527, so the GNN's edge over 'pairs that co-move' is +0.073 AUC.
- Volatility shortcut: ranking pairs by low previous-day volatility of the receiving stock scores 0.513 on its own (raw dst-vol AUC 0.487, src 0.477); the GNN clears it by +0.087. Under the raw-return / 21d-sigma label this rule scored 0.64 out-of-sample, which is why the label moved to sector-excess returns with a 63d sigma (config.yaml gnn.shock_basis / shock_vol_window).
- Model leans on `volz_21` (+0.078 AUC when shuffled) and `logret_21`; `logret_5`, `present`, `pair:edge_weight`, `pair:direction`, `mom_63`, `pair:rolling_corr`, `logret_1` contribute ≈ nothing on their own (shuffling one input at a time understates correlated inputs).
- Seed sensitivity: 0.600–0.605 across 3 seeds (spread 0.005). Shipped seed 42 ranks 3/3 (seed 2026 reached 0.605); the spread is well inside the +0.073 margin over the correlation baseline — quote the AUC as a range, but the edge itself does not depend on the seed. Shipped seed stopped at epoch 3; other seeds at 35, 31.
- Not stable month to month: 2025-10 0.20 (n=21) to 2025-03 0.87 (n=47); 12/26 months below 0.5 (2023-05, 2023-07, 2023-10, 2023-11, 2023-12, 2024-04, 2024-06, 2024-08, 2024-09, 2025-08, 2025-10, 2025-11). Monthly n is 19–92 samples so noise is large, but the RL agent should not treat GNN scores as uniformly reliable.
- Direction: downstream (supplier→OEM) AUC 0.581 vs upstream 0.618 — bidirectional supervision is justified; neither direction is dead weight.
- Weakest pairs cluster around `BOSCHLTD.NS` (5 of the below-0.5 pairs touch it). With ~9 val samples per pair this is a hint, not a verdict — worth a look at that node's curated edges in relationships.csv.
- Calibration bins are monotone: bottom quintile 10% actual cascades, top quintile 25%. Absolute p is inflated (mean 0.40) by pos_weight — fine for ranking, do not read as probability.
- Inference cache is bit-identical to a fresh forward pass and provably free of look-ahead (future-day perturbation test). Downstream RL state is safe to trust on that front.
- Train/val gap 0.072 with early stop at epoch 3 — no overfitting concern.
- Out-of-sample (walk-forward, 183 shock days in 2023H2–2025H2): current config (V4) 0.584 vs previous model 0.538 (better in 86% of day-block bootstrap draws). See reports/gnn_walkforward/WALKFORWARD.md.

## Data under test

| Item | Value |
|---|---|
| Timeline | 3635 days, 2011-10-03 -> 2026-06-30 |
| Nodes / node features | 15 / 7 |
| Pair-head inputs | rolling_corr, edge_weight, direction |
| History start | 2011-10-01 |
| Curated edges / message-passing edges / scored pairs | 32 / 64 / 64 |
| Train samples | 4357 (16.9% positive), 2012-01-06 -> 2023-04-10 |
| Val samples | 1206 (16.8% positive), 2023-05-12 -> 2025-12-03 |
| Overall shock-day rate | 2.70% |
| Config | arch=graphconv, hidden=64, layers=2, dropout=0.4, window=10, gru=32, k=5, shock_z=2.5, min_move=0.02, shock_basis=excess, shock_sigma=63d, seed=42 |

Shock days per ticker:

| Ticker | Shock days |
|---|---|
| ASHOKLEY.NS | 110 |
| TMPV.NS | 105 |
| EXIDEIND.NS | 105 |
| MOTHERSON.NS | 101 |
| EICHERMOT.NS | 100 |
| M&M.NS | 99 |
| TVSMOTOR.NS | 99 |
| BAJAJ-AUTO.NS | 97 |
| HEROMOTOCO.NS | 97 |
| BHARATFORG.NS | 95 |
| BOSCHLTD.NS | 92 |
| UNOMINDA.NS | 92 |
| MARUTI.NS | 91 |
| APOLLOTYRE.NS | 91 |
| SONACOMS.NS | 34 |

## Direction split (val)

| Direction | n | pos rate | AUC |
|---|---|---|---|
| downstream | 619 | 14.9% | 0.5807 |
| upstream | 587 | 18.9% | 0.6179 |

`downstream` = curated supplier→buyer edge; `upstream` = reversed edge.

## Stability across the val period (monthly AUC)

| Month | n | pos | AUC |
|---|---|---|---|
| 2023-05 | 51 | 6 | 0.4630 |
| 2023-06 | 30 | 4 | 0.5481 |
| 2023-07 | 49 | 4 | 0.4222 |
| 2023-08 | 38 | 5 | 0.5333 |
| 2023-09 | 14 | 0 | n/a |
| 2023-10 | 26 | 3 | 0.4638 |
| 2023-11 | 59 | 19 | 0.3724 |
| 2023-12 | 42 | 2 | 0.4750 |
| 2024-01 | 59 | 12 | 0.5674 |
| 2024-02 | 79 | 16 | 0.5208 |
| 2024-03 | 27 | 4 | 0.7717 |
| 2024-04 | 24 | 4 | 0.4875 |
| 2024-05 | 38 | 6 | 0.6250 |
| 2024-06 | 37 | 4 | 0.4318 |
| 2024-07 | 17 | 0 | n/a |
| 2024-08 | 22 | 2 | 0.3750 |
| 2024-09 | 41 | 7 | 0.4496 |
| 2024-10 | 27 | 1 | 0.6538 |
| 2024-11 | 31 | 3 | 0.5595 |
| 2025-01 | 77 | 38 | 0.7045 |
| 2025-02 | 92 | 16 | 0.5493 |
| 2025-03 | 47 | 1 | 0.8696 |
| 2025-04 | 19 | 1 | 0.5556 |
| 2025-05 | 6 | 0 | n/a |
| 2025-07 | 45 | 3 | 0.6111 |
| 2025-08 | 76 | 19 | 0.4718 |
| 2025-09 | 40 | 6 | 0.7255 |
| 2025-10 | 21 | 1 | 0.2000 |
| 2025-11 | 66 | 16 | 0.4725 |
| 2025-12 | 6 | 0 | n/a |

Months with a single class show `n/a`. Small n → noisy.

## Calibration (val, quintiles of predicted probability)

| Bin | n | mean predicted p | actual cascade rate |
|---|---|---|---|
| 0 | 242 | 0.3387 | 0.0992 |
| 1 | 241 | 0.3690 | 0.1286 |
| 2 | 241 | 0.3955 | 0.1743 |
| 3 | 241 | 0.4283 | 0.1909 |
| 4 | 241 | 0.4858 | 0.2490 |

Score distribution: mean 0.403, std 0.054, range [0.297, 0.579]. Training uses `pos_weight` (class re-balancing), so absolute p is shifted upward relative to the 16.8% base rate by design — ranking (AUC) is what the RL state consumes, not calibrated probability. Monotone bins = healthy.

## Per-pair val AUC (pairs with ≥8 samples and both classes)

Top 5:

| Pair | n | pos | AUC |
|---|---|---|---|
| BAJAJ-AUTO.NS->EXIDEIND.NS | 13 | 1 | 1.0000 |
| SONACOMS.NS->TMPV.NS | 18 | 1 | 1.0000 |
| UNOMINDA.NS->MARUTI.NS | 12 | 1 | 1.0000 |
| EXIDEIND.NS->TMPV.NS | 21 | 1 | 0.9500 |
| EXIDEIND.NS->BAJAJ-AUTO.NS | 21 | 1 | 0.9000 |

Bottom 5:

| Pair | n | pos | AUC |
|---|---|---|---|
| BHARATFORG.NS->M&M.NS | 19 | 3 | 0.3958 |
| BHARATFORG.NS->ASHOKLEY.NS | 19 | 4 | 0.3833 |
| MARUTI.NS->UNOMINDA.NS | 16 | 1 | 0.3333 |
| TMPV.NS->BOSCHLTD.NS | 15 | 3 | 0.3333 |
| MOTHERSON.NS->TMPV.NS | 17 | 1 | 0.1875 |

61 pairs evaluable; 14 below 0.5 (per-pair n is small — treat as indicative, not conclusive).

## Feature permutation importance (val AUC drop when feature is shuffled)

| Feature | AUC drop |
|---|---|
| volz_21 | +0.0779 |
| logret_21 | +0.0350 |
| vol_21 | +0.0333 |
| logret_5 | +0.0006 |
| present | +0.0002 |
| pair:edge_weight | +0.0000 |
| pair:direction | +0.0000 |
| mom_63 | -0.0005 |
| pair:rolling_corr | -0.0021 |
| logret_1 | -0.0044 |

Positive = model relies on it. Near-zero/negative = ignored or noise.

## Inference cache

`gnn_scores.parquet`: 3635 days × 64 pairs, mean 0.405, std 0.055. Max |cached − recomputed| = 0.00e+00 (bit-for-bit reproducible).

## Seed robustness

| Seed | Val AUC | Best epoch | Train time (s) |
|---|---|---|---|
| 42 | 0.6002 | 3 | shipped |
| 7 | 0.6037 | 35 | 141.2 |
| 2026 | 0.6047 | 31 | 137.2 |

Range across seeds: 0.6002 – 0.6047 (spread 0.0045). Retrains went to a temp dir; shipped weights untouched.

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
| PASSED | tests/gnn/test_extended_inputs.py::test_demerger_back_adjustment_neutralises_the_ex_date |
| PASSED | tests/gnn/test_extended_inputs.py::test_asof_lag_zero_uses_same_day_lag_one_uses_previous_day |
| PASSED | tests/gnn/test_extended_inputs.py::test_asof_marks_stale_values_missing |
| PASSED | tests/gnn/test_extended_inputs.py::test_series_features_level_vs_price |
| PASSED | tests/gnn/test_extended_inputs.py::test_graphconv_uses_edge_weights_sage_does_not |
| PASSED | tests/gnn/test_extended_inputs.py::test_head_accepts_pair_features_and_context |
| PASSED | tests/gnn/test_extended_inputs.py::test_presence_masking |
| PASSED | tests/gnn/test_extended_inputs.py::test_extended_shapes_and_finiteness |
| PASSED | tests/gnn/test_extended_inputs.py::test_pca_basis_fit_on_train_days_only |
| PASSED | tests/gnn/test_extended_inputs.py::test_inference_inputs_match_training_inputs |
| PASSED | tests/gnn/test_extended_inputs.py::test_node_features_no_lookahead |
| PASSED | tests/gnn/test_extended_inputs.py::test_market_context_no_lookahead |
| PASSED | tests/gnn/test_extended_inputs.py::test_rolling_corr_pair_feature_no_lookahead |
| PASSED | tests/gnn/test_extended_inputs.py::test_labels_never_read_past_label_end_date |
| PASSED | tests/gnn/test_labels.py::test_shock_threshold_scales_with_previous_day_vol |
| PASSED | tests/gnn/test_labels.py::test_shock_day_zero_never_flags |
| PASSED | tests/gnn/test_labels.py::test_shock_min_move_floor |
| PASSED | tests/gnn/test_labels.py::test_shock_uses_previous_not_same_day_vol |
| PASSED | tests/gnn/test_labels.py::test_shock_matrix_shape_and_dtype |
| PASSED | tests/gnn/test_labels.py::test_shock_vol_window_overrides_feature_window |
| PASSED | tests/gnn/test_labels.py::test_excess_basis_uses_sector_excess_return_and_its_own_sigma |
| PASSED | tests/gnn/test_labels.py::test_excess_basis_requires_excess_column |
| PASSED | tests/gnn/test_labels.py::test_unknown_shock_basis_rejected |
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
