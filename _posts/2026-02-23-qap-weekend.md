---
layout: distill
title: Can a Weekend + LLMs Produce a New Neural Solver for the Quadratic Assignment Problem?
description: Weekend project with LLMs.
giscus_comments: true
date: 2026-02-23
tags:
  - nco
authors:
  - name: Andoni Irazusta Garmendia
    url: "https://theleprechaun25.github.io/"
    affiliations:
      name: University of the Basque Country (UPV/EHU)

bibliography: 2026-02-23-qap-weekend.bib

toc:
  - name: Motivation
  - name: Preliminaries
  - name: "Stage 1: Literature Review"
  - name: "Stage 2: Gaps within Neural QAP"
  - name: "Stage 3: The Proposal"
  - name: "Stage 4: Implementation"
  - name: "Stage 5: Experiments"
  - name: "Stage 6: Results"
  - name: "Stage 7: Write-up"
  - name: Wrapping up
  - name: References

_styles: >
  .note {
    padding: 12px 14px;
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,.12);
    background: rgba(0,0,0,.03);
    margin: 14px 0;
  }
  html[data-theme='dark'] .note{
    border: 1px solid rgba(255,255,255,.12);
    background: rgba(255,255,255,.05);
  }
  .anim-wrap{
    border-radius: 18px;
    border: 1px solid rgba(0,0,0,.12);
    overflow: hidden;
    background: white;
  }
  html[data-theme='dark'] .anim-wrap{
    border: 1px solid rgba(255,255,255,.12);
    background: rgba(20,20,20,.55);
  }
  .anim-toolbar{
    display:flex;
    gap:10px;
    flex-wrap:wrap;
    align-items:center;
    padding: 10px 12px;
    border-bottom: 1px solid rgba(0,0,0,.08);
    background: rgba(0,0,0,.02);
  }
  html[data-theme='dark'] .anim-toolbar{
    border-bottom: 1px solid rgba(255,255,255,.10);
    background: rgba(255,255,255,.04);
  }
  .anim-toolbar button{
    padding: 7px 10px;
    border-radius: 12px;
    border: 1px solid rgba(0,0,0,.18);
    background: rgba(255,255,255,.7);
    cursor:pointer;
    font-weight: 600;
  }
  html[data-theme='dark'] .anim-toolbar button{
    border: 1px solid rgba(255,255,255,.16);
    background: rgba(0,0,0,.25);
    color: rgba(255,255,255,.92);
  }
  .anim-toolbar input[type="range"]{
    width: 150px;
  }
  .anim-toolbar .pill{
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid rgba(0,0,0,.12);
    background: rgba(0,0,0,.03);
    font-size: 13px;
  }
  html[data-theme='dark'] .anim-toolbar .pill{
    border: 1px solid rgba(255,255,255,.12);
    background: rgba(255,255,255,.05);
  }
  .anim-canvas{
    width:100%;
    height: 520px;
    display:block;
  }
  @media (max-width: 680px){
    .anim-canvas{ height: 420px; }
  }
  
---

## Motivation

LLMs have gotten surprisingly competent at the core task of research projects: reading papers, writing code, and turning messy thoughts into structured text. 
That naturally raises a question I couldn’t stop thinking about:

If I compress the entire research loop into a single weekend—survey → idea → implementation → experiments → write-up—how far can I get with LLMs as a co-pilot?

This post is that experiment.

I’m going to try it on the Quadratic Assignment Problem (QAP): an NP-hard optimization problem that’s notoriously hard. 
QAP is a “permutation” problem (like TSP), but the objective is quadratic, choices interact with each other in a way that quickly becomes globally entangled. 
Many well-known combinatorial problems can be expressed as special cases or close relatives of QAP, including variants related to routing, matching, layout, and assignment.

Despite that importance, QAP has received much less attention in neural combinatorial optimization than problems like TSP. 
One reason is practical: QAP is harder to “package” into the neural architectures that work nicely for other permutation problems. 
The signal you need depends on a lot of pairwise interactions, so learning good local decisions from partial context is simply harder. 
I’ll unpack this properly later, because it’s exactly where a new model design might help.

So the goal is simple and slightly reckless:

Can I, with the help of LLMs, produce a working neural method for QAP in a single weekend? And if I can, how competitive will it be?
We’ll find out.

---

## Preliminaries

Before diving into the sprint, a quick note on how I’ll run it.

This is a time-boxed project: the point is not to claim a polished, state-of-the-art solver, but to see what kind of research artifact can be produced under extreme constraints with modern LLM support. The output I’m aiming for is:
- a working implementation,
- a reproducible experimental setup,
- and a coherent write-up (call it a short paper or technical report).

I’ll use the following tools:
- ChatGPT (GPT-5.2 Thinking) for quick literature scanning, problem framing, and architecture brainstorming.
- Codex (gpt-5.3-codex) for coding iterations and refinements.
- Prism (LLM-assisted LaTeX projects) for the final report formatting.

I’ll split the project into explicit stages:

1) Literature scan and baseline selection

2) Where current neural approaches struggle on QAP (and why)

3) Method proposal (model + training/inference recipe)

4) Implementation plan and coding

5) Experiments + evaluation protocol

6) Results, ablations, and failure modes

7) Write-up

---

## Stage 1: Literature Review

Before reading papers, I wanted to make sure I had the QAP formulation fresh in my mind.
It’s been a while, so I simply asked the LLM for a quick refresh.

**Q.** *Remind me of the Quadratic Assignment Problem (QAP) formulation.*

**A.** The **Quadratic Assignment Problem (QAP)** assigns $n$ facilities to $n$ locations. Let:

- $F \in \mathbb{R}^{n\times n}$ be the **flow** matrix, where $F_{ij}$ quantifies the interaction between facilities $i$ and $j$.
- $D \in \mathbb{R}^{n\times n}$ be the **distance** matrix, where $D_{ab}$ is the distance between locations $a$ and $b$.
- $\pi \in S_n$ be a **permutation**, where $\pi(i)$ is the location assigned to facility $i$.

The QAP objective is:

$$
\min_{\pi \in S_n} \ \sum_{i=1}^{n}\sum_{j=1}^{n} F_{ij}\,D_{\pi(i)\,\pi(j)}.
$$

Equivalently, using a permutation matrix $P \in \{0,1\}^{n\times n}$ with $P_{i,\pi(i)}=1$ and the feasible set $\Pi$ of all permutation matrices:

$$
\min_{P \in \Pi} \ \langle F,\ P D P^\top \rangle,
$$

where the Frobenius inner product is:

$$
\langle A,B \rangle = \sum_{i=1}^{n}\sum_{j=1}^{n} A_{ij}B_{ij}.
$$

---

Next, I asked the LLM to do what I’d normally do first: a fast literature scan, focused specifically on **neural / deep learning methods for QAP**.

**Q.** *Give me a concise literature review of deep learning approaches for the Quadratic Assignment Problem (QAP): main method families, key papers, benchmarks/baselines, and the most promising gaps to explore.*

**A.** *(Literature review.)*

After a fairly broad scan (24 minutes, 277 sources), it returned a structured summary and a handful of recent papers that were new to me.

<div class="table-wrapper" markdown="1">

| Paper / year | Method family | Core modeling choice | Training paradigm and loss | Data / benchmark | Sizes reported | Performance vs baselines (as reported) | Key limitations (as reported/visible) |
|---|---|---|---|---|---|---|---|
| Revised note on learning QAP with GNNs (2018) [@nowak2018revised_qap_gnn] | Supervised learning + GNN | Siamese GNN embeddings; outer product to score matches; row/col softmax to approximate permutation | Supervised planted-solution training; cross-entropy on recovered permutation | Synthetic Erdős–Rényi / random regular graph alignment | Example: $n=50$ | High recovery rates across noise in tested regimes; outperforms some relaxation baselines on random regular graphs in their experiments | Focused on planted/random-graph regimes; not a direct QAPLIB facility-layout evaluation; requires supervised targets from the generator |
| Neural graph matching network for Lawler QAP (2019; TPAMI 2022) [@wang2019ngm_lawler_qap; @wang2022ngm_tpami] | GNN solver + differentiable projection | Association graph embedding (can be $O(n^2)$ nodes); projection to a doubly-stochastic solution; sampling to explore feasible space | Supervised (cross-entropy) for labeled matching; unsupervised objective-driven training for QAPLIB setting | QAPLIB + vision graph matching benchmarks (in the same pipeline) | QAPLIB sizes up to 256 mentioned; excludes the hardest tai256c due to memory burden | Reports best-performing counts on QAPLIB (e.g., best on 72/133 instances with heavy sampling in their table) | Association-graph scaling and memory bottlenecks; evaluation uses normalized scoring rather than uniform optimality-gap reporting |
| Deep RL for QAP with double pointer network (2023) [@bagga2023drl_qap_double_pointer] | RL (constructive) | Alternating pointer decoding over locations/facilities with GRUs + attention; location encoding + facility GCN embeds | Actor–critic training; evaluated as % gap vs a swap heuristic baseline | Synthetic symmetric Koopmans–Beckmann instances with Euclidean locations | $n=10,20$ | Average gap ~7–9% vs swap heuristic; faster than a time-limited MILP solver in their setup | Small sizes only; comparison target is a heuristic baseline; unclear scaling to $n\ge 50$ |
| Solution-aware transformer learn-to-improve (ICML 2024) [@tan2024sawt_qap] | RL (learn-to-improve) + attention | Separate facility/location encoding; “solution-aware” attention uses incumbent-dependent matrix; proposes swap actions | Policy-gradient with value baseline + entropy; reward tied to improvement vs best-so-far | Synthetic QAP10/20/50/100 + QAPLIB generalization | Up to $n=100$ synthetic; QAPLIB includes up to $n=256$ | Synthetic: near-zero gaps at higher search steps; QAPLIB: mean gap reported as 26.8% with per-instance time ~12.11s | Large distribution shift across QAPLIB categories; needs better cross-category generalization |
| Unsupervised learning for QAP as tabu-search initialization (2025) [@min2025ul_qap_plume] | Unsupervised + hybrid heuristic | Permutation-equivariant network outputs logits; entropic relaxation gives soft permutation; discretize then tabu search | Unsupervised objective; no solved-instance labels; evaluated by tabu-search improvement | Synthetic instances across sizes/densities | $n=100,200$ | Improves over random initialization and improves downstream tabu-search final cost under multiple budgets | Gains after strong tabu can be modest; trained per regime; not benchmarked on QAPLIB |
| Bi-level differentiable framework (2025) [@shi2025biqap] | Differentiable relaxation + bi-level | Network modifies an instance; inner solver uses entropic relaxation + iterative normalization; discretize | Bi-level training: outer loss is original QAP objective; inner solved via differentiable layer; multi-sampling | Case studies incl. QAPLIB + large synthetic | Reports up to sizes like 500–1000 in a “large random dataset” | QAPLIB: best on 92/134 instances; strong on many instances vs a lifted-relaxation baseline | Normalized scoring + per-category training; cross-category transfer and consistent gap reporting unclear |
| Warm-started MCMC finetuning + cross-graph attention (OpenReview 2026) [@pan2026plma_qap_mcmc] | Learning-to-optimize + hybrid search | Cross-graph attention heatmap; additive energy model enables fast 2-swap proposal scoring; warm-start refinement | Pretrain then inference-time adaptation; evaluates runtime–gap tradeoffs | Synthetic + QAPLIB + Taixxeyy | Synthetic up to at least $n=500$ | QAPLIB: avg gap ~0.10% with faster time than a strong tabu baseline (as reported) | Sensitive to inference-time budget + time accounting; depends on reference-solution definitions |

</div>

---

## Stage 2: Gaps within Neural QAP

The literature report came back with the usual “checkbox gaps”: scalability beyond small $n$, generalization to QAPLIB, hybridization with exact solvers, lack of standardized benchmarks, and a *“theory and guarantees”* bucket (help!). All true, but also a bit too generic.

What I *did* take seriously is the evaluation angle. 
Seems like QAP papers often look hard to compare because each one uses different evaluation protocols: different instance generators, different time budgets, different baselines. 
So rather than jumping straight into model design, I decided to first produce a **clean benchmark + reproducible baselines**. 

That said, I didn’t want this weekend to become an exercise of reviewing theoretical proofs, or re-implementing tabu search for the tenth time. 
My point is to explore something **algorithmic**: a representation choice, an inference procedure, or a learning pipeline that is genuinely better aligned with QAP’s structure.

So I asked the LLM again:


<b>Q.</b> <i>Ignore generic “future work” (scale, more benchmarks, theory). Based on existing neural QAP methods, list 5–8 specific, mechanism-level gaps and a concrete improvement for each.

You can propose ideas like:
- representation of (F, D, permutation/incumbent),
- inference/search procedure,
- learning pipeline/objective aligned with QAP structure.

For each item give:
- Failure mode → Fix → Minimal experiment (dataset/baseline/metric/budget).
</i>

**A.** It came with a bunch of interesting gaps, here the most interesting ones in my opinion:

- **Association-graph materialization blows up memory.** Methods that build the full association graph quickly hit GPU-memory limits as $n$ grows, making large QAPLIB instances impractical.

- **2-swap-only neighborhoods dominate inference.** Many pipelines rely almost exclusively on facility–facility swaps, which mixes slowly when good improvements require coordinated multi-facility reassignment.

- **Additive scoring/energy misses quadratic coupling.** Surrogates that score assignments as $\sum_i \phi_{i,\pi(i)}$ cannot represent the true quadratic interactions, biasing move ranking and exploration, especially on dense instances.

- **Sparse/ambiguous reward in learn-to-improve RL.** Treating non-improving steps as “zero signal” (and not explicitly penalizing worsening moves) makes plateaus look identical and wastes steps under tight budgets.

- **Initializer training is decoupled from downstream search.** Models trained to output a “good” permutation often optimize a surrogate objective, while actual performance comes from a later tabu/local-search stage—so learning is misaligned with what’s deployed.

- **Entropic/Sinkhorn inner solvers are local-optimum prone.** Differentiable relaxation methods can get stuck; quality becomes heavily dependent on stochastic multi-start, increasing variance and compute.

- **Soft training vs hard rounding mismatch.** Training optimizes a continuous doubly-stochastic solution, but evaluation uses a discrete permutation after Hungarian rounding; small changes in the soft matrix can cause large discrete jumps, misaligning the optimized loss with the measured objective.

---


## Stage 3: The Proposal: Architecture and Training/Inference Recipe

Talk about how we will first code the benchmark: in distribution performance (syhthetic instances), generalization to larger sizes and to QAPLIB, computational cost of training and inference.

The second step is how to represent the problem with a neural network

Then we need to think about decoding constructive vs improvement?

Finally the learning: supervised? RL? unsupervised?

---


## Stage 4: From Idea to Code: Implementation Plan


---


## Stage 5: Experiments and Evaluation Protocol


---


## Stage 6: What Worked, What Didn’t: Results, Ablations, and Failure Modes


---


## Stage 7: Packaging It: Final Write-up

---

## Wrapping up


<d-bibliography></d-bibliography>
