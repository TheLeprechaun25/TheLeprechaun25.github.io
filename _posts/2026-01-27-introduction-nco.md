---
layout: distill
title: Introduction to Neural Combinatorial Optimization
description: an example with the Travelling Salesperson Problem
giscus_comments: true
date: 2026-01-27

bibliography: 2026-01-27-introduction-nco.bib

toc:
  - name: Why Neural Combinatorial Optimization
  - name: The Traveling Salesperson Problem
  - name: Two families of neural solvers
    subsections:
      - name: Neural constructive models
      - name: Neural improvement models
  - name: Training signals
  - name: Inference under a compute budget
  - name: When to use which approach
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

## Why Neural Combinatorial Optimization

Combinatorial optimization sits underneath a huge range of real systems: routing and logistics, scheduling, allocation, graph partitioning, and many more. Classical solvers rely on decades of algorithmic craftsmanship—greedy heuristics, local search, branch-and-bound, cutting planes—often tuned to specific instance structures.

**Neural Combinatorial Optimization (NCO)** asks a different question: can we learn a solver from data and interaction, so that the *algorithm itself* adapts to a problem family? In the best case, a learned solver does more than memorize training instances: it learns **search priors**, **representations**, and **decision rules** that generalize across instances and scale with compute at inference time.

The field took off once sequence models and attention made it practical to map *sets/graphs → permutations/structures*, starting with pointer-network style decoders for routing problems <d-cite key="vinyals2015pointer"></d-cite> and then attention-based constructive solvers for TSP <d-cite key="bello2016neural,kool2019attention"></d-cite>. Since then, NCO has broadened to include non-autoregressive models (heatmaps, diffusion) <d-cite key="joshi2019efficient,sun2023difusco"></d-cite>, multi-start decoding like POMO <d-cite key="kwon2020pomo"></d-cite>, and—crucially for real deployment—**neural improvement** methods that explicitly learn local moves under a budget <d-cite key="chen2021learning"></d-cite>.

<div class="note">
<strong>Two practical regimes.</strong><br/>
(1) You want a strong <em>first</em> solution quickly (constructive).<br/>
(2) You want the best solution you can reach within a fixed compute/time budget (improvement / search).
</div>

We will use the Traveling Salesperson Problem (TSP) as a running example because it is simple to state, hard to solve exactly at scale, and rich enough to expose most of the key design choices.

---

## The Traveling Salesperson Problem

Given cities with coordinates \(x_1,\dots,x_N \in \mathbb{R}^2\), the symmetric Euclidean TSP asks for a permutation \(\pi\) that minimizes the tour length:

$$
\min_{\pi \in S_N}\; C(\pi)
\;=\;
\sum_{t=1}^{N} \left\|x_{\pi_t} - x_{\pi_{t+1}}\right\|_2,
\qquad \pi_{N+1} := \pi_1.
$$

Even though the objective is “just” a sum of distances, the search space is enormous: \((N-1)!\) tours (fixing a start city). This is why most scalable solvers are **heuristics** that trade optimality for speed, and why the *budgeted* setting (what can you do in 50ms, 1s, 10s?) matters so much.

---

## Two families of neural solvers

At a high level, NCO methods for routing often fall into two families:

1. **Neural constructive models**: output a full tour in one decoding run.
2. **Neural improvement models**: start from a tour and iteratively apply local modifications.

They can be combined, but it helps to understand their inductive biases separately.

### Neural constructive models

A constructive policy builds a solution one decision at a time. The canonical example is an attention-based encoder–decoder:

- **Encoder**: maps the set of cities to embeddings (often with self-attention).
- **Decoder**: autoregressively selects the next city conditioned on the partial tour and visited mask.

In the attention model for TSP <d-cite key="kool2019attention"></d-cite>, the decoder defines a distribution:
$$
p_\theta(\pi) = \prod_{t=1}^{N} p_\theta(\pi_t \mid \pi_{<t}, x),
$$
and you decode either greedily, by sampling, or with beam search / multi-start.

Training is typically reinforcement learning (REINFORCE) with a baseline:
$$
\nabla_\theta \mathbb{E}_{\pi \sim p_\theta(\cdot|x)}[C(\pi)]
=
\mathbb{E}\left[(C(\pi)-b(x)) \nabla_\theta \log p_\theta(\pi|x)\right].
$$
Practical upgrades include multi-start decoding and instance-dependent baselines (e.g., POMO) <d-cite key="kwon2020pomo"></d-cite>.

**Intuition.** Constructive models learn a *prior over good tours* and can be extremely fast at producing decent solutions. But they are often optimized for the “first-shot” output; getting from “good” to “excellent” under a budget may require explicit search.

---

### Neural improvement models

Neural improvement (sometimes framed as learned local search) treats optimization as a sequential decision process:

- **State**: current solution (tour) + instance (cities) + optional history.
- **Action**: a local move (e.g., 2-opt swap).
- **Transition**: apply the move to get a new tour.
- **Reward**: improvement in tour cost (dense) or best-so-far improvement (sparse).

A classic local move for TSP is **2-opt**: pick two edges \((i,i+1)\) and \((j,j+1)\) and reverse the segment between them, removing crossings and often reducing cost. The neighborhood size is \(O(N^2)\), so selecting moves intelligently matters.

Neural improvement policies can be trained by:
- **Imitation**: predict the best move given a teacher (exact or heuristic).
- **RL**: optimize expected improvement over a rollout budget.
- **Hybrid**: imitation warm-start + RL fine-tuning.

This family connects to “learning to search” and learned heuristics more broadly <d-cite key="chen2021learning,mazyavkina2021reinforcement,bengio2021machine"></d-cite>.

**Intuition.** Improvement models are naturally **anytime**: with more steps, they (usually) keep getting better. This aligns well with real constraints where you can spend a fixed inference budget.

---

## Two animations: construct vs improve

The demos below illustrate the *algorithmic shape* of the two families:

- **Constructive**: pick next node sequentially (a stand-in for an autoregressive policy).
- **Improvement**: start from a tour and apply 2-opt improvements.

These are illustrative animations (not running your trained network), but the compute/budget behavior mirrors what you typically see in practice.

<div class="l-page">
  <div class="anim-wrap" id="ncoAnimWrap">
    <div class="anim-toolbar">
      <button id="modeConstruct">Constructive</button>
      <button id="modeImprove">Improvement (2-opt)</button>
      <button id="btnReset">Reset</button>
      <button id="btnStep">Step</button>
      <button id="btnPlay">Play</button>
      <span class="pill" id="pillInfo">Ready.</span>
      <span class="pill" id="pillCost">Cost: –</span>
      <span class="pill" id="pillStep">Step: –</span>
    </div>
    <canvas class="anim-canvas" id="ncoCanvas"></canvas>
  </div>
</div>

<script>
(function(){
  // ---- helpers ----
  function mulberry32(a){
    return function(){
      var t = a += 0x6D2B79F5;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
  }
  function dist(a,b){
    const dx=a[0]-b[0], dy=a[1]-b[1];
    return Math.hypot(dx,dy);
  }
  function tourCost(pts, tour){
    let c=0;
    for(let k=0;k<tour.length;k++){
      const i=tour[k], j=tour[(k+1)%tour.length];
      c += dist(pts[i], pts[j]);
    }
    return c;
  }
  function nearestNeighborTour(pts, start=0){
    const n=pts.length;
    const used=new Array(n).fill(false);
    const tour=[start];
    used[start]=true;
    for(let t=1;t<n;t++){
      const last=tour[tour.length-1];
      let best=-1, bestd=1e18;
      for(let j=0;j<n;j++){
        if(used[j]) continue;
        const d=dist(pts[last], pts[j]);
        if(d<bestd){ bestd=d; best=j; }
      }
      tour.push(best);
      used[best]=true;
    }
    return tour;
  }
  function twoOptBestMove(pts, tour){
    // returns best (i,j,delta) for 2-opt (reverse segment between i+1..j)
    const n=tour.length;
    let bestDelta=0, bestI=-1, bestJ=-1;
    for(let i=0;i<n;i++){
      const a=tour[i], b=tour[(i+1)%n];
      for(let j=i+2;j<n;j++){
        // avoid adjacent + full wrap
        if(i===0 && j===n-1) continue;
        const c=tour[j], d=tour[(j+1)%n];
        const before = dist(pts[a], pts[b]) + dist(pts[c], pts[d]);
        const after  = dist(pts[a], pts[c]) + dist(pts[b], pts[d]);
        const delta = after - before; // negative is improvement
        if(delta < bestDelta){
          bestDelta=delta; bestI=i; bestJ=j;
        }
      }
    }
    return {i:bestI, j:bestJ, delta:bestDelta};
  }
  function applyTwoOpt(tour, i, j){
    // reverse segment (i+1..j)
    const n=tour.length;
    const out=tour.slice();
    let l=i+1, r=j;
    while(l<r){
      const tmp=out[l]; out[l]=out[r]; out[r]=tmp;
      l++; r--;
    }
    return out;
  }

  // ---- DOM ----
  const canvas = document.getElementById("ncoCanvas");
  const ctx = canvas.getContext("2d");
  const pillInfo = document.getElementById("pillInfo");
  const pillCost = document.getElementById("pillCost");
  const pillStep = document.getElementById("pillStep");

  const btnReset = document.getElementById("btnReset");
  const btnStep = document.getElementById("btnStep");
  const btnPlay = document.getElementById("btnPlay");
  const modeConstruct = document.getElementById("modeConstruct");
  const modeImprove = document.getElementById("modeImprove");

  // ---- state ----
  let mode = "construct";
  let rng = mulberry32(1337);
  let pts = [];
  let tour = [];
  let partial = [];
  let visited = [];
  let step = 0;
  let playing = false;
  let timer = null;
  let lastMove = null;

  function resize(){
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);
    draw();
  }
  window.addEventListener("resize", resize);

  function makeInstance(n=32){
    pts=[];
    for(let i=0;i<n;i++){
      // nice margin, deterministic seed
      const x = 0.08 + 0.84 * rng();
      const y = 0.08 + 0.84 * rng();
      pts.push([x,y]);
    }
  }

  function initConstruct(){
    tour = [];
    partial = [0];
    visited = new Array(pts.length).fill(false);
    visited[0]=true;
    step = 0;
    lastMove = null;
    pillInfo.textContent = "Constructive: step-by-step selection.";
    updatePills();
  }

  function initImprove(){
    tour = nearestNeighborTour(pts, 0); // start from a quick heuristic
    partial = [];
    visited = [];
    step = 0;
    lastMove = null;
    pillInfo.textContent = "Improvement: greedy 2-opt steps.";
    updatePills();
  }

  function resetAll(){
    stop();
    rng = mulberry32(1337);
    makeInstance(34);
    if(mode==="construct") initConstruct(); else initImprove();
    draw();
  }

  function stop(){
    playing=false;
    if(timer){ clearInterval(timer); timer=null; }
    btnPlay.textContent = "Play";
  }

  function play(){
    if(playing){ stop(); return; }
    playing=true;
    btnPlay.textContent = "Pause";
    timer = setInterval(()=>{
      const progressed = stepOnce();
      if(!progressed) stop();
    }, 220);
  }

  function updatePills(){
    if(mode==="construct"){
      const cur = partial.length >= 2 ? tourCost(pts, partial) : 0;
      pillCost.textContent = partial.length>=2 ? ("Partial cost: " + cur.toFixed(2)) : "Partial cost: –";
      pillStep.textContent = "Step: " + step + "/" + (pts.length-1);
    }else{
      const c = tourCost(pts, tour);
      pillCost.textContent = "Cost: " + c.toFixed(2);
      pillStep.textContent = "Step: " + step;
    }
  }

  function stepOnce(){
    if(mode==="construct"){
      if(partial.length === pts.length){
        // close tour
        pillInfo.textContent = "Constructive: tour completed.";
        updatePills();
        draw();
        return false;
      }
      const last = partial[partial.length-1];
      let best=-1, bestd=1e18;
      for(let j=0;j<pts.length;j++){
        if(visited[j]) continue;
        const d=dist(pts[last], pts[j]);
        if(d<bestd){ bestd=d; best=j; }
      }
      partial.push(best);
      visited[best]=true;
      step++;
      lastMove = {type:"pick", a:last, b:best};
      pillInfo.textContent = "Constructive: chose next city (illustrative policy).";
      updatePills();
      draw();
      return true;
    }else{
      const mv = twoOptBestMove(pts, tour);
      if(mv.i<0){
        pillInfo.textContent = "Improvement: local optimum under 2-opt.";
        updatePills();
        draw();
        return false;
      }
      const oldCost = tourCost(pts, tour);
      const newTour = applyTwoOpt(tour, mv.i, mv.j);
      const newCost = tourCost(pts, newTour);
      lastMove = {type:"2opt", i:mv.i, j:mv.j, oldCost, newCost};
      tour = newTour;
      step++;
      pillInfo.textContent = "Improvement: applied a 2-opt swap (greedy).";
      updatePills();
      draw();
      return true;
    }
  }

  function draw(){
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;

    // background
    ctx.clearRect(0,0,W,H);

    // read theme
    const dark = document.documentElement.getAttribute("data-theme")==="dark";
    const bg = dark ? "rgba(0,0,0,0)" : "rgba(255,255,255,1)";
    const edge = dark ? "rgba(240,240,240,.35)" : "rgba(0,0,0,.18)";
    const edgeStrong = dark ? "rgba(240,240,240,.70)" : "rgba(0,0,0,.40)";
    const nodeFill = dark ? "rgba(230,230,230,.90)" : "rgba(30,30,30,.85)";
    const nodeStroke = dark ? "rgba(255,255,255,.25)" : "rgba(0,0,0,.15)";
    const accent = getComputedStyle(document.documentElement).getPropertyValue("--global-theme-color").trim() || (dark ? "#00C060" : "#00A550");

    // margins
    const pad = 26;
    function X(x){ return pad + x*(W-2*pad); }
    function Y(y){ return pad + y*(H-2*pad); }

    // draw tour / partial
    ctx.lineWidth = 2;
    ctx.strokeStyle = edge;

    function drawPath(path, close=false, strong=false){
      if(path.length<2) return;
      ctx.beginPath();
      ctx.moveTo(X(pts[path[0]][0]), Y(pts[path[0]][1]));
      for(let k=1;k<path.length;k++){
        ctx.lineTo(X(pts[path[k]][0]), Y(pts[path[k]][1]));
      }
      if(close) ctx.lineTo(X(pts[path[0]][0]), Y(pts[path[0]][1]));
      ctx.strokeStyle = strong ? edgeStrong : edge;
      ctx.stroke();
    }

    if(mode==="construct"){
      drawPath(partial, false, true);
      if(partial.length===pts.length){
        drawPath(partial, true, true);
      }
    }else{
      drawPath(tour, true, true);
    }

    // highlight last move
    if(lastMove && lastMove.type==="pick"){
      ctx.strokeStyle = accent;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(X(pts[lastMove.a][0]), Y(pts[lastMove.a][1]));
      ctx.lineTo(X(pts[lastMove.b][0]), Y(pts[lastMove.b][1]));
      ctx.stroke();
      ctx.lineWidth = 2;
    }
    if(lastMove && lastMove.type==="2opt"){
      // highlight the cut edges around i and j after swap (approx)
      const n=tour.length;
      const i=lastMove.i, j=lastMove.j;
      // reconstruct endpoints for visualization (post-swap)
      const a = tour[i], b = tour[(i+1)%n];
      const c = tour[j], d = tour[(j+1)%n];
      ctx.strokeStyle = accent;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(X(pts[a][0]), Y(pts[a][1])); ctx.lineTo(X(pts[b][0]), Y(pts[b][1]));
      ctx.moveTo(X(pts[c][0]), Y(pts[c][1])); ctx.lineTo(X(pts[d][0]), Y(pts[d][1]));
      ctx.stroke();
      ctx.lineWidth = 2;
    }

    // nodes
    for(let i=0;i<pts.length;i++){
      const px = X(pts[i][0]), py = Y(pts[i][1]);
      const r = (mode==="construct" && visited[i]) ? 5.5 : 5.0;

      ctx.beginPath();
      ctx.arc(px,py,r,0,Math.PI*2);
      ctx.fillStyle = nodeFill;
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = nodeStroke;
      ctx.stroke();
    }

    // start node marker
    ctx.beginPath();
    ctx.arc(X(pts[0][0]), Y(pts[0][1]), 8, 0, Math.PI*2);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.lineWidth = 2;

    // footer text
    ctx.fillStyle = dark ? "rgba(255,255,255,.60)" : "rgba(0,0,0,.55)";
    ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
    const tag = (mode==="construct")
      ? "Constructive demo: nearest-neighbor-like selection (illustrative)"
      : "Improvement demo: greedy 2-opt until local optimum";
    ctx.fillText(tag, 18, H-14);
  }

  // ---- wire up ----
  modeConstruct.addEventListener("click", ()=>{
    stop();
    mode="construct";
    initConstruct();
    draw();
  });
  modeImprove.addEventListener("click", ()=>{
    stop();
    mode="improve";
    initImprove();
    draw();
  });
  btnReset.addEventListener("click", resetAll);
  btnStep.addEventListener("click", ()=>{ stop(); stepOnce(); });
  btnPlay.addEventListener("click", play);

  // ---- init ----
  resetAll();
  resize();
})();
</script>

<div class="note">
<strong>How to read the demos.</strong><br/>
Constructive policies are optimized to output a strong tour quickly (and can be boosted with sampling/beam/multi-start). Improvement policies are optimized to <em>spend</em> steps: each step is a learned or heuristic move that hopefully decreases cost. Under tight budgets the first can win; under larger budgets the second often dominates.
</div>

---

## Training signals

NCO sits at the intersection of optimization and learning, so “training signal” is the main design lever.

### Reinforcement learning (policy gradients)

The simplest approach is to treat the solver as a stochastic policy and optimize expected cost. For constructive decoding, REINFORCE is the standard baseline <d-cite key="bello2016neural,kool2019attention"></d-cite>. For improvement, you can define dense rewards like
$$
r_t = C(\pi_t) - C(\pi_{t+1}),
$$
or best-so-far rewards that focus learning on the steps that first achieve a new best value.

Practical patterns:
- normalize advantages, use baselines, clip gradients;
- train on a distribution of sizes for robustness;
- separate “behavior policy” and “update policy” (PPO-style) if doing multi-step rollouts.

### Supervised learning (imitation)

When you have a strong teacher (exact for small \(N\), heuristic for large \(N\)), imitation can make training stable and fast. For constructive models, the teacher might be an optimal tour (but there are many optimal permutations); for improvement models, the teacher might be “best move in a neighborhood”.

The tricky bit is that in many combinatorial settings there are **many equivalent targets**. A good training recipe handles ties (e.g., set-valued targets) rather than forcing a single arbitrary label.

### Hybrid: imitation → RL fine-tune

A common, effective recipe is:
1) imitate for stability and a strong starting policy, then  
2) fine-tune with RL to match the evaluation metric and budget.

---

## Inference under a compute budget

The most deployment-relevant view of NCO is: **what do you get per unit of compute?** A few practical observations:

- **Constructive**: one-shot decoding is fast; more compute typically means more *restarts* (sampling, beam, POMO-style multi-start) <d-cite key="kwon2020pomo"></d-cite>.
- **Improvement**: more compute means more steps; you want policies that keep producing improvements rather than stagnating early.
- **Hybrids**: generate a good initial tour (constructive) and then refine with learned improvement—often the best of both worlds.

A helpful mental model is an anytime curve: solution quality versus time/steps. Constructive methods often start strong but flatten; improvement methods may start weaker but keep improving.

---

## When to use which approach

A practical rule of thumb:

- If you need a strong answer **immediately** (tight latency), start with a **constructive** solver (possibly with a small number of restarts).
- If you have a moderate/large budget and care about **best attainable quality**, use **improvement** (or constructive → improvement).
- If your instance distribution shifts (new geometry, new constraints), improvement methods can be easier to adapt because they learn **local decision rules** that transfer across sizes and structures.

---

## References

<d-bibliography></d-bibliography>
