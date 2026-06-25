# CD-002 Completion Report: Scalable ST Direction Audit

Branch status: DONE_WITH_CONCERNS

Recommendation: prioritize a graph-aware Robust-LargeST mini-benchmark for W2; keep the virtual-node/bipartite prototype as a conditional method option; do not start a new patch/expert method yet.

Date: 2026-06-24

Scope followed: research/report only. No experiments, model-code edits, dataset creation, or remote jobs were run.

## Result Summary

The broad scalable-ST method space is crowded. Generic virtual nodes, generic auxiliary-node bipartite inference, and generic patch/expert scaling are not safe novelty claims. The strongest surviving W2 options are:

1. **Graph-aware Robust-LargeST benchmark: go with constraints.** This is the clearest gap if the benchmark is traffic-graph-aware: corridor, region, freeway-class, upstream/downstream, sensor-type, and time-correlated faults. It must not be framed as "Gaussian noise on LargeST" or generic sensor-fault robustness.
2. **LargeST-scale sparse geography-aware virtual bottleneck: go with constraints, secondary.** It can be a W2 prototype only if framed as a narrow engineering/scaling contribution: full-node forecasting with sparse `N -> M -> N` geography-aware assignments, compared against PatchSTG and MAGE. Do not call generic virtual nodes novel.
3. **New patch/expert/anchor model: hold.** Use PatchSTG, MAGE, BiST, and dynamic patching as baselines/comparators. A new method here needs a sharper mechanism than "patch sensors" or "route nodes to experts."

The missing soft dependency is the final `CD-001` baseline matrix. This report did not read sibling raw context; the master should review `CD-001` before binding a W2 implementation choice.

## Evidence Sources

Allowed branch files read:

- `branch-brief.md`
- `../branch-map.md`
- `../snapshots/snap-2026-06-24-001.md`
- `../../../largest2019_virtual_node/conductor/CD-002_virtual_node_bipartite/completion-report.md`
- `../../../largest2019_virtual_node/conductor/CD-E01_bist_paper_explainer/reading-notes.md`

Zotero used:

- Semantic search for LargeST-scale virtual-node, PatchSTG, MAGE, SensorFault, and scalability directions.
- Item metadata for PatchSTG / "Efficient Large-Scale Traffic Forecasting with Transformers" (`2JJBXCS6`), BiST (`DBBINN8I`), and dynamic patching / latent graph structure learning (`9I527PL5`).

Web primary sources used:

- [LargeST, arXiv:2306.08259](https://arxiv.org/abs/2306.08259)
- [Virtual Nodes Improve Long-term Traffic Prediction, arXiv:2501.10048](https://arxiv.org/abs/2501.10048)
- [Multivariate Time Series Forecasting with Latent Graph Inference, arXiv:2203.03423](https://arxiv.org/abs/2203.03423)
- [Set Transformer, arXiv:1810.00825](https://arxiv.org/abs/1810.00825)
- [Perceiver, arXiv:2103.03206](https://arxiv.org/abs/2103.03206)
- [Perceiver IO, arXiv:2107.14795](https://arxiv.org/abs/2107.14795)
- [PatchSTG / Efficient Large-Scale Traffic Forecasting with Transformers, arXiv:2412.09972](https://arxiv.org/abs/2412.09972)
- [PatchSTG: Scalable Spatiotemporal Graph Transformers, arXiv:2606.09872](https://arxiv.org/abs/2606.09872)
- [MAGE / Less but More, OpenReview](https://openreview.net/forum?id=jCGwSLwOt9)
- [SensorFault-Bench, arXiv:2605.10822](https://arxiv.org/abs/2605.10822)
- [TS-Fault, arXiv:2606.18539](https://arxiv.org/abs/2606.18539)
- [CA-VMGCN / robust LargeST, arXiv:2504.06660](https://arxiv.org/abs/2504.06660)
- [Metropolis-scale road network benchmark, arXiv:2510.02278](https://arxiv.org/abs/2510.02278)

## Literature and Collision Table

| Work | Status | Mechanism | Complexity / scaling claim | Relationship to W2 target | Collision level |
|---|---|---|---|---|---|
| LargeST, Liu et al. 2023 | arXiv benchmark | 8,600 California sensors, 5-year coverage, sensor metadata | Establishes scale and metadata-rich setting | Target dataset family | Foundation |
| Virtual Nodes Improve Long-term Traffic Prediction, Cao et al. 2025 | arXiv preprint | Adds virtual nodes to traffic graphs via semi-adaptive distance/adaptive adjacency | Improves global relay capacity and long-range traffic prediction | Direct virtual-node traffic prior | Direct collision |
| LSTGI, Satorras et al. 2022 | arXiv preprint | Latent graph inference with fully connected or bipartite auxiliary-node graph | Reduces graph inference from `O(N^2)` to `O(NK)` in bipartite mode | Direct `N -> K -> N` auxiliary-node precedent | Direct collision |
| Set Transformer, Lee et al. 2019 | ICML / arXiv | Inducing-point attention for set interactions | Reduces set self-attention from quadratic to linear for fixed inducing count | Foundational inducing-point analogy | Inspiration, not traffic-specific |
| Perceiver / Perceiver IO, Jaegle et al. 2021/2022 | ICML/ICLR / arXiv | Cross-attends large inputs into a latent bottleneck; IO queries structured outputs | Scales through latent array and output queries | Strong architectural analogy for `N -> M -> N` | Reviewer analogy risk |
| PatchSTG, Fang et al. 2024/2025 | KDD 2025 / arXiv; Zotero item `2JJBXCS6` | KDTree irregular spatial patching plus depth/breadth attention | Reports up to `10x` training speed and `4x` memory improvement | Closest LargeST-style patch scalability comparator | Strong adjacent competitor |
| PatchSTG, Li and Shi 2026 | arXiv preprint | Hierarchical balanced, locality-preserving geographic patches with dual attention | Claims quadratic to near-linear spatial scaling | Shows patch naming and mechanism are even more crowded | Strong adjacent competitor |
| MAGE / Less but More, Ma et al. 2025 | NeurIPS 2025 poster / OpenReview | Sparse balanced mixture-of-experts for adaptive graph learning | Each expert operates with linear node complexity | Closest sparse expert/adaptive-graph comparator | Strong adjacent competitor |
| Latent Graph Structure Learning for Large-Scale Traffic Forecasting, Wang et al. 2025 | ACM DOI 10.1145/3746252.3760937; Zotero item `9I527PL5` | Data-driven dynamic patch assignment and forecasting in one framework | Learns patch assignments for large-scale traffic | Direct pressure on learned patch/anchor variants | Strong adjacent competitor |
| BiST, Ma et al. 2025 | PVLDB; Zotero item `DBBINN8I` | MLP forward base predictor plus backward correction | Claims competitive accuracy with much lower training time and memory | Lightweight efficiency baseline, not a graph bottleneck | Baseline pressure |
| SensorFault-Bench, Windmann et al. 2026 | arXiv preprint | CPS sensor-fault stress-test protocol with scored scenarios and clean/corrupt comparison | Reports clean MSE, worst-scenario degradation, fault-time MSE | Direct generic sensor-fault robustness benchmark prior | Direct benchmark collision |
| TS-Fault, Zhao et al. 2026 | arXiv preprint | Structural TSF fault benchmark: temporal shape, broken dependencies, regime/missingness, causal propagation | 21 models, 6 datasets, paired clean/corrupt protocol | Broad structured-fault benchmark prior | Strong adjacent benchmark |
| CA-VMGCN, Ahmad and Khalid 2025 | IJCNN / arXiv | Variational-mode GCN plus 3D attention under i.i.d. Gaussian corruption on LargeST | Claims robust/noise-resilient LargeST long-term prediction | Direct LargeST Gaussian-noise robustness prior | Direct collision for Gaussian-noise framing |
| Metropolis-scale road network benchmark, Velikonivtsev et al. 2025/2026 | arXiv preprint | Up to 100,000 road segments with real connectivity and road attributes | Exposes scalability issues in current ST models | Shows "LargeST-scale" is no longer the outer frontier | Adjacent scale pressure |

## Direction Audit

### 1. Virtual-Node / Bipartite Bottleneck

Decision: **go with constraints**, but only as a secondary W2 method option.

What survives:

- A fixed or lightly learned small virtual set can be useful if it preserves full-node forecasts and makes geography-aware sparse assignments explicit.
- The defensible mechanism is not "virtual nodes improve traffic prediction." It is "LargeST-compatible sparse geography-aware `N -> M -> N` communication with full-node output and measurable scaling behavior."
- The implementation should be additive and rollback-friendly: a local branch plus virtual residual, not a replacement for all spatial modeling.

Main objections:

- Cao et al. 2025 already uses virtual nodes in traffic forecasting and explicitly motivates global dependency / over-squashing.
- LSTGI 2022 already gives the bipartite auxiliary-node graph-inference complexity story.
- Set Transformer and Perceiver make the latent bottleneck analogy obvious to reviewers.
- PatchSTG and MAGE are stronger LargeST-scale comparators than old full `N x N` adaptive graph baselines.

Minimum W2 credibility test:

- Compare against `STID`, `PatchSTG`, `BiST`, and `MAGE` on `SD / GLA / GBA`.
- Report accuracy, training time, peak memory, and parameter/runtime deltas.
- Ablate `M in {8,16,32,64}`, top-r assignment, geography initialization, learned assignment, load balance, virtual update, and local residual.
- Show full-node output shape remains `[B,H,N,C]`.
- Provide geography/load visualizations; otherwise interpretability claims are weak.

Unsafe claims:

- "Virtual nodes for traffic forecasting are novel."
- "Bipartite auxiliary nodes for forecasting are novel."
- "The method is Perceiver-like but reviewers will not notice."
- "LargeST-scale alone is enough without PatchSTG/MAGE comparisons."

### 2. Patch / Expert / Anchor Scalable Spatial Modeling

Decision: **hold** for new method development.

What survives:

- Patch/expert/anchor methods are essential comparators and design references.
- A W2 virtual-bottleneck prototype should borrow the discipline of bounded local groups, balanced routing, and explicit scaling metrics.

Main objections:

- PatchSTG KDD 2025 already targets large-scale traffic with irregular KDTree spatial patching and depth/breadth attention.
- The June 2026 PatchSTG preprint independently reinforces balanced geographic patches and dual attention.
- MAGE already frames sparse balanced experts as linear adaptive graph learning.
- Zotero surfaced another ACM large-scale traffic paper focused on data-driven patch assignment.

Minimum W2 credibility test if reopened:

- The mechanism must not be another KDTree/hierarchical patching variant or generic sparse MoE.
- It would need a new unit of structure, such as traffic-corridor constrained routing, flow-direction-aware anchors, or benchmark-driven robustness routing.
- It must beat or clearly complement PatchSTG/MAGE on efficiency and accuracy.

Unsafe claims:

- "Patch-based scalable traffic forecasting is open."
- "Sparse experts are new for ST graph learning."
- "A learned patch assignment alone is enough novelty."

### 3. Graph-Aware Robust-LargeST / SensorFault Benchmark

Decision: **go with constraints** and recommended as the primary W2 branch if the master wants the most defensible contribution.

What survives:

- Generic sensor-fault robustness is not novel, but a LargeST traffic-graph-aware protocol can still be defensible.
- CA-VMGCN already covers i.i.d. Gaussian corruption on LargeST, so W2 must use structured traffic faults, not generic random noise.
- SensorFault-Bench and TS-Fault make the right protocol shape clear: paired clean/corrupt evaluation, fixed fault banks, scenario severity, degradation metrics, and separation of clean accuracy from robustness.

Recommended graph-aware fault families:

- Corridor faults: contiguous sensors along a freeway/corridor biased, missing, delayed, or drifting together.
- Region faults: geographic polygons or districts with shared outage/noise.
- Sensor-type / road-class faults: stations grouped by metadata such as freeway class or measurement type if available.
- Directional propagation faults: upstream/downstream disturbance patterns rather than independent sensor corruption.
- Temporal burst faults: rush-hour-only, incident-window-only, or maintenance-window corruption.
- Hybrid availability/value faults: dropout plus bias, drift plus delay, or stuck-at-value plus missingness.

Minimum W2 credibility test:

- Keep the clean test set as the reference; inject faults into historical inputs only; keep labels unchanged.
- Use fixed seeds, documented severity levels, paired clean/corrupt batches, and shared fault banks across models.
- Report clean MAE/RMSE/MAPE, degradation ratio, worst-scenario error, fault-time error, and per-region/corridor breakdowns.
- Include `STID`, `PatchSTG`, `BiST`, and `MAGE` once `CD-001` confirms available baselines.
- Include at least one simple robustness intervention, such as fault augmentation or randomized training, to show the benchmark can distinguish mitigation strategies.

Unsafe claims:

- "Sensor-fault robustness benchmark is new."
- "LargeST robustness under noise is new."
- "Gaussian noise is realistic traffic graph failure."
- "Clean leaderboard rank implies deployment reliability."

## W2 Recommendation

Recommended W2 path:

1. **Primary:** build a graph-aware Robust-LargeST mini-benchmark after master approval. It has the clearest defensible gap because it changes the fault structure, not just the model architecture.
2. **Secondary:** if the master wants a method prototype instead, implement the sparse geography-aware virtual bottleneck with strict constraints and make PatchSTG/MAGE/BiST mandatory comparators.
3. **Do not start:** a fresh patch/expert/anchor method unless a sharper, traffic-specific mechanism is proposed after reviewing the final `CD-001` matrix.

This is a branch recommendation, not a binding master decision.

## Local Decisions

- Treated broad virtual-node and broad auxiliary-node novelty as already collided.
- Treated patch/expert/anchor modeling as a comparator family rather than a W2 implementation target.
- Treated generic Gaussian or i.i.d. fault robustness as too crowded for a W2 benchmark claim.
- Did not read `CD-001` raw files because the branch brief forbids sibling raw context; noted the baseline-matrix dependency for the master.

## Proposed Global Decisions

- Choose graph-aware Robust-LargeST as the default W2 implementation direction if the master wants the safest novelty/positioning trade-off.
- Keep any W2 virtual-node work under a narrow name such as "sparse geography-aware latent bottleneck" rather than generic "virtual nodes."
- Make `PatchSTG`, `MAGE`, `BiST`, and `STID` mandatory W2 comparators once `CD-001` is complete.
- Do not open a new patch/expert method branch unless master approves a mechanism that is clearly not KDTree patching, dynamic patch assignment, or sparse MoE adaptive graphs.

## Validation or Checks

- Checked the required allowed branch files.
- Checked Zotero for local/library context and metadata.
- Checked primary arXiv/OpenReview pages for the closest anchors and recent adjacent papers.
- Used an arXiv helper for a structured PatchSTG/MAGE search where permitted; one broader network helper query was blocked by the sandbox and was replaced with primary web-source verification.
- No code, datasets, experiment configs, remote jobs, or model files were touched.

## Risks

- `CD-001` may change the priority if PatchSTG/MAGE/BiST results are missing, weak, or unexpectedly strong.
- Robust-LargeST can still be rejected as "benchmark-only" unless the fault bank is graph-aware, reproducible, and hard enough to reveal model differences.
- A virtual bottleneck can easily become a thin LSTGI/Perceiver adaptation if the geography/sparsity/interpretability parts are not central.
- LargeST `SD / GLA / GBA` may not expose the full scaling benefit; master may later need CA-scale evidence for strong claims.

## Suggested Merge Note

- CD-002 status: DONE_WITH_CONCERNS.
- Primary W2 recommendation: graph-aware Robust-LargeST mini-benchmark, go with constraints.
- Virtual-node/bipartite method: secondary go with constraints; only as sparse geography-aware `N -> M -> N` full-node bottleneck.
- New patch/expert/anchor method: hold; use PatchSTG, MAGE, BiST, and dynamic patching as comparators.
- Unsafe claims: generic virtual nodes, auxiliary-node bipartite inference, Gaussian LargeST robustness, and generic sensor-fault benchmarking are not novel.
- W2 robustness must use traffic graph-aware faults: corridor, region, road-class/sensor-type, directional propagation, and temporal burst faults.
- W2 method prototype must report accuracy, train time, memory, scaling, assignment ablations, and geography/load interpretability.
- Master should review `CD-001` baseline matrix before making the binding W2 decision.
