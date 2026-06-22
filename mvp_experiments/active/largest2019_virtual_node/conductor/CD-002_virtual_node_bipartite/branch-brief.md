# Branch Brief: CD-002 - Virtual-Node Bipartite Graph Research

- Branch ID: `CD-002`
- Stable session title: `[CD-002][W1][research] Virtual-node bipartite graph`
- Type: `branch`
- Role: research
- Based on snapshot: `snap-2026-06-22-001`
- Brief version: 1
- Created: 2026-06-22
- Parent: `[CD-MAIN][master] LargeST2019 virtual-node benchmark`

## Purpose Card

You are branch CD-002: Virtual-node bipartite graph.

Purpose: research and evaluate the idea of using a fixed number of latent/virtual/auxiliary nodes connected to real sensors through a bipartite graph for scalable spatiotemporal forecasting.

Not for: running experiments, implementing the prototype, or reviving dynamic-threshold modules.

Input: master snapshot `snap-2026-06-22-001`; public literature and local notes only.

Output: `completion-report.md`.

Return to master when: there is a clear literature map, method taxonomy, risk assessment, and minimal prototype recommendation.

## Why This Branch Exists

The project needs to know whether fixed-size virtual nodes are a novel and practical modeling path for LargeST-scale forecasting, or whether the idea is already covered by existing latent graph / inducing point / spatial patch methods.

## Initial Anchors

- Virtual Nodes Improve Long-term Traffic Prediction: `https://arxiv.org/abs/2501.10048`
- Multivariate Time Series Forecasting with Latent Graph Inference: `https://arxiv.org/abs/2203.03423`
- Set Transformer: `https://arxiv.org/abs/1810.00825`
- PatchSTG: `https://arxiv.org/abs/2412.09972`
- MAGE: `https://openreview.net/forum?id=jCGwSLwOt9`

## Tasks

1. Search recent and foundational work around:
   - virtual nodes in traffic/STGNNs;
   - latent/auxiliary/inducing nodes for MTS forecasting;
   - bipartite graph inference for scalable forecasting;
   - Perceiver/Set Transformer/Nystrom-style bottleneck attention;
   - spatial patches, experts, anchors, landmarks, and pooling as adjacent ideas.
2. Produce a taxonomy with at least three buckets:
   - virtual nodes / global relay nodes;
   - latent inducing points / auxiliary nodes / bipartite inference;
   - spatial patches / experts / coarsening.
3. Compare each bucket against the target idea:
   - complexity in terms of `N` sensors and `M` virtual nodes;
   - whether it preserves node-level forecasts;
   - whether assignments are static, dynamic, learned, or geography-aware;
   - interpretability and scalability risks.
4. Draft a minimal LargeST-compatible prototype:
   - real-to-virtual aggregation;
   - virtual temporal/spatial update;
   - virtual-to-real broadcast;
   - expected complexity and possible ablations.
5. State whether the direction is worth entering W2 implementation.

## Completion Criteria

- Literature table includes title, year/venue/status, mechanism, complexity claim, relevance, and risk.
- Method spectrum makes clear how this differs from PatchSTG and MAGE.
- Final recommendation is one of: `go`, `go with constraints`, `hold`, or `drop`.
- No repo code is modified beyond this branch's completion report if needed.

## Expected Completion Report

Create `completion-report.md` with:

- Branch status: `DONE`, `DONE_WITH_CONCERNS`, or `BLOCKED`.
- Literature table.
- Taxonomy and relationship map.
- Minimal prototype sketch with formulas.
- Reviewer-risk assessment.
- Suggested Merge Note, no more than 8 bullets.

