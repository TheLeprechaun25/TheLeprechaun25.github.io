---
layout: distill
title: Scale Generalization Problem in NCO
description: An introduction to the cross-scale generalization via the Traveling Salesperson Problem.
giscus_comments: true
date: 2026-02-17
tags:
  - nco
authors:
  - name: Andoni Irazusta Garmendia
    url: "https://theleprechaun25.github.io/"
    affiliations:
      - name: University of the Basque Country (UPV/EHU)
bibliography: 2026-02-17-scale-generalization-problem.bib
toc:
  - name: Neural Improvement for the TSP
  - name: Why scale generalization is hard
  - name: 2-step imitation learning across sizes
  - name: Starting positions, symmetries, and label noise
  - name: What to measure and how to stress-test
  - name: References
---

## Neural Improvement for the TSP

Neural Combinatorial Optimization (NCO) often gets introduced through *constructive* models: a policy that builds a solution from scratch, one decision at a time. Neural Improvement (NI) flips the script: start from an existing solution and repeatedly apply small edits that make it better.

The Traveling Salesperson Problem (TSP) will be our playground to explain NI.

A TSP instance with $N$ cities is a set of coordinates

$$
X^{(N)} = (x_1,\ldots,x_N)\in \mathbb{R}^{N\times 2}.
$$

A tour is a Hamiltonian cycle, commonly represented as a permutation $\pi\in S_N$, with cyclic indexing $\pi_{N+1}=\pi_1$. The tour length is

$$
L(\pi; X^{(N)}) = \sum_{t=1}^{N} \|x_{\pi_t}-x_{\pi_{t+1}}\|_2.
$$

In NI, we also define a *move operator* $\Phi$ (e.g., 2-opt), and an action $a_t$ that selects a particular move. Starting from an initial tour $\pi^{(0)}$, NI produces a sequence

$$
\pi^{(t+1)} = \Phi(\pi^{(t)}, a_t),\qquad t=0,1,\ldots,T-1.
$$

Given a step budget $T$, the goal is to quickly drive the tour cost down.

> **One sentence mental model:** NI is a learned local-search heuristic whose policy is applied repeatedly; scale generalization asks whether that heuristic remains valid when $N$ grows.

---

## Why scale generalization is hard

If you train on TSP instances with $N\in[20,100]$, you’d love the same learned improver to work on $N=500$ or $N=1000$. That is the *cross-scale generalization* problem.

### 1) The action set grows with $N$

For 2-opt, an action is typically “pick two breakpoints” $(i,j)$, leading to $|\mathcal{A}_N|=\Theta(N^2)$. Even if your model outputs a score for every candidate move, the decision surface changes with $N$.

In plain terms: at $N=50$ you might have a handful of obviously good 2-opt moves; at $N=500$, you have *many* plausible moves, and the best ones can be annoyingly close.

### 2) The state distribution shifts with $N$

A random tour at $N=50$ looks different than at $N=500$. So does a “partially improved” tour after $t$ edits. The density of crossings, typical edge lengths, and the distribution of “available easy wins” all change.

So even if your architecture can technically process any $N$, the model still faces a distribution shift:

$$
s \sim \mathcal{D}_N \quad \text{changes with } N.
$$

### 3) The horizon grows

If your inference budget scales like $T(N)=4N$ for example, then the policy is used for much longer rollouts on large instances. Small systematic biases that are harmless at $N=50$ can compound at $N=500$.

---

## 2-step imitation learning across sizes

Let's think how could we train a NI policy using imitation learning (IL).

We could build a “teacher” that chooses the best move, then train a network to imitate it. But in NI, a *myopic* teacher can be misleading because some moves are “setup moves” that enable better improvements later.

That motivates a **k-step optimal teacher**. Let's use **k=2** to start.

Let $s=(X^{(N)},\pi)$ be the state. For a first action $a_1$, define the 2-step lookahead value

$$
Q^{(2)}(s,a_1) = \min_{a_2\in \mathcal{A}_N(\Phi(\pi,a_1))}
L\big(\Phi(\Phi(\pi,a_1),a_2); X^{(N)}\big).
$$

Then the teacher action is

$$
a^\star(s) = \arg\min_{a_1\in\mathcal{A}_N(\pi)} Q^{(2)}(s,a_1).
$$

Your IL objective becomes

$$
\min_\theta\; \mathbb{E}_{s\sim\mathcal{D}}
\big[ -\log \pi_\theta(a^\star(s)\mid s) \big].
$$

### A simple “setup move” example

Suppose three candidate first moves have immediate improvements:

- $a$: improves by $+1.0$, but then nothing good remains $\Rightarrow$ total $+1.0$.
- $b$: improves by $+0.2$, but unlocks a second move of $+3.0$ $\Rightarrow$ total $+3.2$.
- $c$: improves by $+0.8$, then $+0.1$ $\Rightarrow$ total $+0.9$.

A greedy teacher picks $a$. A 2-step teacher picks $b$.  
So 2-step IL teaches “planning” even in a local-search setting.

A useful tweak is **soft imitation**:

$$
p_T(a\mid s)\propto \exp\!\big(-Q^{(2)}(s,a)/\tau\big),
\qquad
\min_\theta\; \mathbb{E}\big[ \mathrm{KL}(p_T(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s))\big].
$$

> **Why this matters for scale:** the number of near-tied 2-step choices tends to increase with $N$, so one-hot labels become brittle. A soft teacher distribution can stabilize training.

---

## What to measure and how to stress-test

If your goal is *cross-scale generalization*, you want evaluations that separate “works on the training distribution” from “learned the right invariances and planning heuristics”.

### 1) Cross-scale rollout curves (anytime performance)

For each $N$, plot best-so-far tour length versus steps $t$, normalized by a strong reference:

$$
\text{gap}(t) = \frac{L(\pi^{(t)}) - L_{\text{ref}}}{L_{\text{ref}}}\times 100\%.
$$

The reference can be a classical solver/heuristic (or best-known on synthetic benchmarks).

The key is not only final gap, but how the curve degrades as $N$ increases.

### 2) Teacher consistency across sizes

If the teacher changes implementation with $N$ (e.g., exact depth-2 at small sizes, approximate at large sizes), quantify the discrepancy on overlapping sizes. This helps you separate “student failed to generalize” from “teacher changed”.

### 3) Start from different initializations

NI performance depends heavily on the initial tour distribution: random, greedy, constructive neural, POMO-style sampling, etc. Cross-scale generalization can look strong from one initializer and collapse from another.

A simple takeaway: **changing the initializer changes the state distribution**, and the learned improver is only as robust as the diversity of states it saw during training.

---

## References
