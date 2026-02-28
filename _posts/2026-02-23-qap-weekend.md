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

Let's start.

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

| Paper / year (real title) | Cite | Short description |
|---|---|---|
| **Revised Note on Learning Algorithms for Quadratic Assignment with Graph Neural Networks** (2018) | <d-cite key="nowak2018revised_qap_gnn"></d-cite> | Supervised GNN method trained on synthetic/planted regimes to learn QAP-like assignment behavior (primarily studied outside “full” QAPLIB facility-layout evaluation). |
| **Neural Graph Matching Network: Learning Lawler’s Quadratic Assignment Problem With Extension to Hypergraph and Multiple-Graph Matching** (2019; TPAMI 2022) | <d-cite key="wang2019ngm_lawler_qap,wang2022ngm_tpami"></d-cite> | Association-graph neural matching + Sinkhorn projection to predict permutations for Lawler-QAP / graph matching; extends to hypergraph and multi-graph matching settings. |
| **Solving the Quadratic Assignment Problem using Deep Reinforcement Learning** (2023) | <d-cite key="bagga2023drl_qap_double_pointer"></d-cite> | Constructive RL for Koopmans–Beckmann QAP using an alternating “double pointer” style policy over facilities/locations, trained with actor–critic. |
| **Learning Solution-Aware Transformers for Efficiently Solving Quadratic Assignment Problem** (ICML 2024) | <d-cite key="tan2024sawt_qap"></d-cite> | Learn-to-improve transformer that conditions on the incumbent assignment (“solution-aware”) and outputs swap-style improvement actions; evaluated on synthetic KB-QAP and QAPLIB generalization. |
| **Unsupervised Learning for Quadratic Assignment** (2025) | <d-cite key="min2025ul_qap_plume"></d-cite> | Unsupervised permutation learning (no solved targets) to generate strong initial solutions that are refined by tabu search; studies instance-regime and size variation. |
| **BiQAP: Neural Bi-level Optimization-based Framework for Solving Quadratic Assignment Problems** (KDD 2025) | <d-cite key="shi2025biqap"></d-cite> | Bi-level framework where a network guides a differentiable entropic relaxation (inner solver) while optimizing the original QAP objective at the outer level. |
| **Learning the Quadratic Assignment Problem with Warm-Started MCMC Finetuning and Cross-Graph Attention** (ICLR 2026 submission) | <d-cite key="pan2026plma_qap_mcmc"></d-cite> | Cross-graph attention + warm-started MCMC finetuning/adaptation at inference time; energy-based formulation for efficient swap proposals and runtime–quality tradeoffs. |
| **Revocable Deep Reinforcement Learning with Affinity Regularization for Outlier-Robust Graph Matching** (ICLR 2023) | <d-cite key="liu2023rgm_affinity"></d-cite> | RL-based solver for Lawler-QAP/graph matching with outliers; uses revocable decisions and affinity regularization to improve robustness to unmatched nodes. |

</div>


I need to focus on a specific family so I decided to focus on improvement based methods, since I remember Tabu Search works pretty well in the QAP. To do so, the ICML 2024 work seems a good starting point <d-cite key="tan2024sawt_qap"></d-cite>.

---

## Stage 2: Gaps within Neural QAP

The literature report came back with the usual “checkbox gaps”: scalability beyond small $n$, generalization to QAPLIB, hybridization with exact solvers, lack of standardized benchmarks, and a *“theory and guarantees”* bucket (help!). All true, but also a bit too generic.

What I *did* take seriously is the evaluation angle. 
Seems like QAP papers often look hard to compare because each one uses different evaluation protocols: different instance generators, different time budgets, different baselines. 
So rather than jumping straight into model design, I decided to first produce a **clean benchmark + reproducible baselines**. 

That said, I didn’t want this weekend to become an exercise of reviewing theoretical proofs, or re-implementing plain tabu search for the tenth time. 
My point is to explore something **algorithmic**: a representation choice, an inference procedure, or a learning pipeline that is genuinely better aligned with QAP’s structure.

So I asked the LLM again:


<b>Q.</b> <i>Ignore generic “future work” (scale, more benchmarks, theory). Based on existing neural QAP methods, list 5–8 specific, mechanism-level gaps and a concrete improvement for each.

You can propose ideas like:
- representation of (F, D, permutation/incumbent),
- inference/search procedure,
- learning pipeline/objective aligned with QAP structure.
- Focus on these two papers: <d-cite key="tan2024sawt_qap"></d-cite>

</i>

**A.** It came with a bunch of interesting gaps, here the most interesting ones in my opinion:

- **Association-graph materialization blows up memory.** Methods that build the full association graph quickly hit GPU-memory limits as $n$ grows, making large QAPLIB instances impractical.

- **Additive scoring/energy misses quadratic coupling.** Surrogates that score assignments as $\sum_i \phi_{i,\pi(i)}$ cannot represent the true quadratic interactions, biasing move ranking and exploration, especially on dense instances.

- **Sparse/ambiguous reward in learn-to-improve RL.** Treating non-improving steps as “zero signal” (and not explicitly penalizing worsening moves) makes plateaus look identical and wastes steps under tight budgets.

I had some additional gaps in my mind, which the LLM did not mention:
- **Current neural approaches where implemented for the euclidean QAP**. Methods that take coordinates (as done in euclidean TSP implementations), to measure distances. However, they are limited to instances in which distances are euclidean (computed from euclidean coordinates).

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

Train with instances of size 5, generalize to sizes, 5, 10, 15, 20

Compare to two baselines (trained with instances of size 20), greedy construction, and local search.

<p style="text-align:center;">
  <img src="/assets/img/anytime_euclid_panel.png" alt="Alt text" style="width:80%; max-width:900px;">
</p>


---


## Stage 6: What Worked, What Didn’t: Results, Ablations, and Failure Modes


---


## Stage 7: Packaging It: Final Write-up

---

## Wrapping up


<d-bibliography></d-bibliography>
