# THASCAB — Final Report

**Three Heterogeneous Architectures, Shared Cognitive workspAce Bridge**
Run on a MacBook Air M4 (16 GB), July 15–19 2026. All numbers are mean ± std over 5 seeds unless noted. All raw results in `runs/<name>/result.json`; figures in `figures/`.

---

## Verdict up front

**The pre-registered kill criterion triggered.** At ~15 M total parameters on MiniCLEVR-R, the shared-workspace union of three heterogeneous specialists lost to every null hypothesis: the best solo, the ensemble, and the parameter/compute-matched monolith — on every split. Per the plan (§8.2): *"the finding is: at ~15 M params on this task, monoliths win. That is itself a result."*

The mechanistic analysis below is the actual contribution: the union failed for a **specific, diagnosable reason** — representational illegibility between frozen modules ("lazy fusion") — and explicitly **not** for lack of workspace bandwidth, rounds, or channel format. The plan's core premise (bandwidth-limited negotiation) was never stressed, because the bottleneck turned out to live elsewhere.

## 1. Main results (accuracy %, chance ≈ 20–30% mixed; yes/no families 50%)

| System | iid | comp_attr | comp_tmpl | extrap |
|---|---|---|---|---|
| B1 Perceiver solo (frozen JEPA + head) | 49.9 ± 0.4 | 47.7 ± 1.0 | 44.2 ± 0.3 | 43.7 ± 1.0 |
| B1 Perceiver solo (finetuned) | 53.6 ± 0.8 | 50.2 ± 0.3 | 44.2 ± 0.5 | 44.5 ± 0.5 |
| B2 Reasoner solo (GNN) | 62.2 ± 2.2 | 61.3 ± 2.2 | 50.9 ± 4.7 | 51.6 ± 1.4 |
| B3 Interpreter solo (AR transformer) | **68.3 ± 0.7** | **67.2 ± 1.1** | 44.6 ± 1.6 | 52.5 ± 1.4 |
| B4 union, frozen cores | 50.4 ± 0.3 | 47.8 ± 0.2 | 43.6 ± 0.5 | 43.4 ± 0.9 |
| B4.5 union, pretrain→finetune | 50.6 ± 1.2 | 47.9 ± 1.2 | 43.4 ± 0.8 | 44.2 ± 1.1 |
| B5 union, end-to-end scratch | 41.7 ± 0.5 | 41.2 ± 0.7 | 33.8 ± 7.5 | 38.7 ± 0.6 |
| B6 text bridge (Gumbel, bandwidth-matched) | 48.3 ± 0.6 | 46.3 ± 0.7 | 45.3 ± 0.8 | 42.7 ± 0.9 |
| B7 ensemble of solos (learned gate) | **71.8 ± 1.5** | **71.4 ± 2.0** | **52.1 ± 3.7** | 54.8 ± 1.2 |
| B8 monolith (6L×256d, matched) | 68.7 ± 1.2 | 67.0 ± 1.2 | 46.0 ± 0.7 | **57.8 ± 2.4** |

(`figures/main_results.png`)

Success-criteria ladder (§8.2): **all four rungs failed.** B4.5 < max(solo) by ~18 pts; < B7 by ~21 pts; < B8 by ~18 pts; B6 ≈ B4 (the latent-vs-symbolic comparison is moot when neither works).

Notable side-results:
- **The ensemble, not the monolith, is the iid/comp_attr champion** — cheap specialist averaging beat everything, including the co-trained monolith.
- **The monolith wins extrapolation** (57.8% on 6-object scenes) — capacity helps most where the distribution shifts.
- **comp_tmpl (held-out templates) broke everyone** (≤52%): compositional generalization over query structure is the genuinely hard axis of this task, unsolved by any architecture here.
- B5 (scratch union) was worst overall with huge comp_tmpl variance (±7.5) — "modules melt together," as the plan's risk table predicted.

## 2. Why the union failed: lazy fusion, not bandwidth

**Causal write-masking** (`figures/damage_matrix.png`; zero one module's workspace writes at eval):

| iid drop when masked | B4 frozen | B4.5 finetune | B5 scratch | B6 text |
|---|---|---|---|---|
| A Perceiver | 0.1 ± 0.1 | 0.4 ± 1.1 | 0.5 ± 0.8 | 1.8 ± 0.8 |
| B Reasoner | 8.8 ± 0.6 | 8.1 ± 1.0 | 0.1 ± 0.3 | 9.2 ± 2.2 |
| C Interpreter | 2.9 ± 2.0 | 2.9 ± 2.2 | 7.2 ± 8.2 | 1.0 ± 0.9 |

The bridge learned to rely on the Reasoner's geometry tokens and the Interpreter's query — and **never learned to read the Perceiver**, in any regime, despite the JEPA latents *provably containing* the needed information: linear probes decode object color at ~100% and shape at 73–81% from patch neighborhoods. The information was there; the bridge couldn't afford the translation. Gradient descent found freeloading cheaper than language acquisition. Under the input partition (P2), losing A's slice caps attainable accuracy near exactly where all unions plateaued (~50%).

Observer probes corroborate: by end of training, slot write-attention mass is A: 0.7%, B: 24%, C: 75%; writer identity is decodable from slots at 88–94% (module-ID embeddings work); answer decodability from slots *declines* over training (44% → 39%) while object-count stays ~97% — the workspace settles into carrying B's cheap geometry summary rather than fused task evidence (`figures/observer_curve.png`).

**Bandwidth and iteration were never the constraint** (`figures/kr_sweep.png`): K ∈ {1,2,4,8,16} → 50.1–50.8% flat; R ∈ {1,2,3,5} → 50.1–50.6% flat. A *single slot* and a *single round* do as well as sixteen and five. The Goyal-style bottleneck hypothesis got no traction here because the system never had more to say than the channel could carry.

**Seed-0 exploratory ladder** (1 seed each, diag_*): module-dropout 0.3 (50.8%), R=5 (50.3%), K=16 (50.4%) — nothing moved the plateau. The single deep-dive B4.5 (10 ep, bs64) reached 52.5% with a genuine A-lesion effect (−2.5), showing co-adaptation *can* wake the Perceiver — but the effect washed out across the 5-seed fleet (−0.4 ± 1.1).

**Compensation test** (module-dropout training, per plan §8.4.2): dropout-trained bridges lose less when C is masked (−0.5 vs −2.9) — responsibilities *are* shareable among modules the bridge already reads. It does not create readings that never existed: A stays dead (−0.0). Redundancy probes agree: post-communication, color information in Reasoner tokens stays at chance (~36% vs 33% chance) — appearance never diffused into the graph side.

## 3. Solo bias signatures (per plan §9.2 — these worked as designed)

Per-family iid accuracy confirmed each architecture's predicted blind spot: the Interpreter aces attribute counting (97%) but is at chance on relational counting (33%) — parsing beats grounding; the Reasoner is balanced but mediocre on query structure; the Perceiver solo can't do multi-hop logic. The task instrument (difficulty band 60–85% for best solo: B3 at 68%) calibrated correctly on the first try, and split hygiene held (scene-hash-disjoint splits verified; comp_attr scenes 100% held-out-combo).

## 4. Interpretation

1. **The hard problem of modular AI is shared semantics, not channels.** A 1 M-param bridge + 40k examples could not learn to translate between three mature, mutually foreign representation systems. Every working analog (CLIP alignment, LLM vision-adapters, biological cortex) buys its translator either with massive paired-data alignment or lifelong co-development. Post-hoc integration of frozen strangers — the transplant scenario — got the transplant-rejection outcome.
2. **Lazy fusion is the default failure mode of learned communication.** Given any sufficient subset of inputs, optimization freeloads on the cheapest-to-read module and abandons the rest. Emergent division of labor requires *pressure that makes freeloading impossible* — our module-dropout was too weak, too late.
3. **Modularity's advantages are regime-dependent, and 15 M params on one task is the wrong regime.** No capacity scarcity (B8 fit everything), no multi-task pressure, no lifelong constraint. The concept was tested below its operating range; that bounds where workspace architectures can pay off, and the flat K/R sweeps show the premise (bandwidth-limited negotiation) never even engaged.
4. **Ensembles are an underrated null.** B7's win over B8 says: on single-task iid performance, *independent* specialists + trivial output fusion beat both joint training schemes. Communication has to buy something ensembling can't — which at this scale it didn't.

## 5. What would give the concept its fair fight (future work)

- **Legibility pressure from birth**: joint pretraining with the workspace present from init *plus* aggressive scheduled module-dropout (so no module's slice is reliably present), making freeloading structurally impossible.
- **Explicit alignment losses**: contrastive slot-to-module objectives so writes must be decodable by other readers — pay the translator bill directly.
- **A regime with real capacity pressure**: multi-task suites at a scale where the matched monolith *can't* absorb everything.
- Multi-token answers/justifications, where iterative workspace negotiation has room to matter (R-sweep may not stay flat).

## 6. Reproduction

```
python3 -m thascab.data --out data/full --seed 0          # dataset (+ curriculum variant)
python3 scripts/stage1_solos.py                            # B1/B2/B3 + JEPA, 5 seeds
python3 scripts/overnight_driver.py                        # everything else, resumable
python3 scripts/aggregate.py                               # tables + figures
```
Wall clock on the M4 Air: ~2.5 days total, of which the honest budget matched the plan's prediction — code was ~1 day; calibration, diagnosis, and baseline tuning were the majority. Peak memory after fixes ~8 GB (fp16 CPU-side activation cache, per-process run isolation, MPS watermark caps).

Operational lessons for this hardware logged in `runs/`: MPS memory needs explicit caps + per-run process isolation (Metal pool leaks accumulate); unfrozen-core regimes need micro-batches (bs 64); and the two big slowdowns during the project were a lid-close sleep and an unrelated 79 GB zombie-process swap storm — not thermal throttling.

---

*Design credit where due: the input-partition table (P1/P2), the pre-registered nulls (B7/B8), and the kill criterion are what made this negative result diagnostic instead of ambiguous. The instrument worked; the hypothesis, at this scale, did not.*
