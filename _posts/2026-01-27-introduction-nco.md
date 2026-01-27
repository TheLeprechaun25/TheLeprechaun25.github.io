---
layout: distill
title: Introduction to Neural Combinatorial Optimization (NCO)
description: A visual introduction to NCO via the Traveling Salesperson Problem.
giscus_comments: true
date: 2026-01-27
tags:
  - nco
authors:
  - name: Andoni Irazusta Garmendia
    url: "https://theleprechaun25.github.io/"
    affiliations:
      name: University of the Basque Country (UPV/EHU)

bibliography: 2026-01-27-introduction-nco.bib

toc:
  - name: Combinatorial Optimization
  - name: The Traveling Salesperson Problem
  - name: Why Neural Combinatorial Optimization
  - name: Two families of neural solvers
    subsections:
      - name: Neural constructive models
      - name: Neural improvement models
  - name: Training signals
  - name: Inference under a compute budget
  - name: Pitfalls and evaluation
  - name: Where the field is going
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

## Combinatorial Optimization

Combinatorial optimization (CO) sits underneath a huge range of real systems: routing and logistics, scheduling, allocation, packing, and graph problems such as partitioning or cuts. What makes these problems challenging is not that the objective is mysterious, but that the number of feasible solutions typically grows **combinatorially** with instance size.

Classical solvers and heuristics are the result of decades of human algorithmic effort. They are not “generic black boxes”: they encode substantial **problem structure** and **domain knowledge**—which neighborhoods to search, which relaxations to solve, which cuts to add, which branching rules work, which invariances matter, which parameters to tune. This accumulated craft is a major reason why mature optimization toolchains remain extremely strong in practice.

Surveys provide a broad picture of how machine learning has entered this landscape, and how CO problems are used as testbeds for learning-based decision-making <d-cite key="bengio2021machine,mazyavkina2021reinforcement"></d-cite>.

---

## The Traveling Salesperson Problem

The Traveling Salesperson Problem (TSP) is a classical benchmark in combinatorial optimization. The name comes from its canonical story: a salesperson wants to plan a trip that visits a set of cities exactly once and returns home, while minimizing total travel distance.

Formally, given $N$ cities represented by coordinates $x_1,\dots,x_N \in \mathbb{R}^2$, a tour is an ordering in which each city is visited exactly once and the route returns to the start. The symmetric Euclidean TSP asks for the tour of minimum total length, where the cost between two cities is their Euclidean distance (and is symmetric). Writing a tour as a permutation of cities $\pi$, its length is the sum of distances between consecutive cities in that order, including the closing edge back to the first city:

$$
\min_{\pi \in S_N}\; C(\pi)
\;=\;
\sum_{t=1}^{N} \left\|x_{\pi_t} - x_{\pi_{t+1}}\right\|_2,
\qquad \pi_{N+1} := \pi_1.
$$

It is often useful to view a TSP instance as a complete weighted graph: each city is a node, and the cost of traveling between cities $i$ and $j$ is an edge weight $d_{ij}$. The animation below shows a random TSP instance. Hover a city to reveal its distances to others; use the slider to control how many edges are drawn.

<div class="l-page">
  <div class="anim-wrap" id="tspAnimWrap">
    <div class="anim-toolbar">
      <button id="tspNew">New instance</button>
      <span class="pill">Hover cities to see distances</span>
      <span class="pill">Edge threshold</span>
      <input id="tspThresh" type="range" min="0" max="100" value="38" />
      <span class="pill" id="tspInfo">Ready.</span>
    </div>
    <canvas class="anim-canvas" id="tspCanvas"></canvas>
  </div>
</div>

<script>
(function(){
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

  const canvas = document.getElementById("tspCanvas");
  const ctx = canvas.getContext("2d");
  const btnNew = document.getElementById("tspNew");
  const slider = document.getElementById("tspThresh");
  const info = document.getElementById("tspInfo");

  let rng = mulberry32(20260127);
  let pts = [];
  let hoverIdx = -1;

  function resize(){
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);
    draw();
  }
  window.addEventListener("resize", resize);

  function newInstance(n=36){
    pts=[];
    for(let i=0;i<n;i++){
      const x = 0.08 + 0.84 * rng();
      const y = 0.08 + 0.84 * rng();
      pts.push([x,y]);
    }
    hoverIdx = -1;
    info.textContent = "New instance created.";
    draw();
  }

  function pickHover(mx, my){
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    const pad = 26;
    function X(x){ return pad + x*(W-2*pad); }
    function Y(y){ return pad + y*(H-2*pad); }
    let best=-1, bestd=1e18;
    for(let i=0;i<pts.length;i++){
      const dx = X(pts[i][0]) - mx;
      const dy = Y(pts[i][1]) - my;
      const d = Math.hypot(dx,dy);
      if(d < bestd){ bestd = d; best = i; }
    }
    return (bestd <= 14) ? best : -1;
  }

  canvas.addEventListener("mousemove", (e)=>{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const idx = pickHover(mx, my);
    if(idx !== hoverIdx){
      hoverIdx = idx;
      if(hoverIdx>=0) info.textContent = `Selected city ${hoverIdx+1}/${pts.length}.`;
      else info.textContent = "Hover cities to see distances.";
      draw();
    }
  });
  canvas.addEventListener("mouseleave", ()=>{
    hoverIdx = -1;
    info.textContent = "Hover cities to see distances.";
    draw();
  });

  function draw(){
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    ctx.clearRect(0,0,W,H);

    const dark = document.documentElement.getAttribute("data-theme")==="dark";
    const edge = dark ? "rgba(240,240,240,.20)" : "rgba(0,0,0,.10)";
    const edgeStrong = dark ? "rgba(240,240,240,.55)" : "rgba(0,0,0,.28)";
    const nodeFill = dark ? "rgba(230,230,230,.92)" : "rgba(30,30,30,.88)";
    const nodeStroke = dark ? "rgba(255,255,255,.25)" : "rgba(0,0,0,.12)";
    const label = dark ? "rgba(255,255,255,.70)" : "rgba(0,0,0,.60)";
    const accent = getComputedStyle(document.documentElement).getPropertyValue("--global-theme-color").trim() || (dark ? "#00C060" : "#00A550");

    const pad = 26;
    function X(x){ return pad + x*(W-2*pad); }
    function Y(y){ return pad + y*(H-2*pad); }

    // compute threshold based on slider percentile of pairwise distances
    const ds = [];
    for(let i=0;i<pts.length;i++){
      for(let j=i+1;j<pts.length;j++){
        ds.push(dist(pts[i], pts[j]));
      }
    }
    ds.sort((a,b)=>a-b);
    const q = Math.max(0, Math.min(100, +slider.value)) / 100.0;
    const thr = ds[Math.floor(q * (ds.length-1))];

    // draw "local" edges under threshold (gives intuition about local geometry)
    ctx.lineWidth = 1.2;
    for(let i=0;i<pts.length;i++){
      for(let j=i+1;j<pts.length;j++){
        const d = dist(pts[i], pts[j]);
        if(d > thr) continue;
        ctx.strokeStyle = edge;
        ctx.beginPath();
        ctx.moveTo(X(pts[i][0]), Y(pts[i][1]));
        ctx.lineTo(X(pts[j][0]), Y(pts[j][1]));
        ctx.stroke();
      }
    }

    // if hovering, draw distances from the selected city with stronger opacity
    if(hoverIdx >= 0){
      ctx.lineWidth = 2.6;
      for(let j=0;j<pts.length;j++){
        if(j===hoverIdx) continue;
        ctx.strokeStyle = edgeStrong;
        ctx.beginPath();
        ctx.moveTo(X(pts[hoverIdx][0]), Y(pts[hoverIdx][1]));
        ctx.lineTo(X(pts[j][0]), Y(pts[j][1]));
        ctx.stroke();
      }
    }

    // nodes
    for(let i=0;i<pts.length;i++){
      const px = X(pts[i][0]), py = Y(pts[i][1]);
      const r = (i===hoverIdx) ? 7.5 : 5.2;

      ctx.beginPath();
      ctx.arc(px,py,r,0,Math.PI*2);
      ctx.fillStyle = nodeFill;
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = nodeStroke;
      ctx.stroke();

      if(i===hoverIdx){
        ctx.beginPath();
        ctx.arc(px,py,r+4.5,0,Math.PI*2);
        ctx.strokeStyle = accent;
        ctx.lineWidth = 3;
        ctx.stroke();
      }
    }

    // small legend text
    ctx.fillStyle = label;
    ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
    ctx.fillText(`Edge threshold ≈ ${(thr).toFixed(3)} (quantile ${Math.round(q*100)}%)`, 18, H-14);
  }

  slider.addEventListener("input", draw);
  btnNew.addEventListener("click", ()=>{
    rng = mulberry32((Math.random()*1e9)>>>0);
    newInstance(36);
  });

  newInstance(36);
  resize();
})();
</script>

---

## Classical Algorithms for the TSP

Even for moderate $N$, the search space is enormous: fixing a start city still leaves $(N-1)!$ possible tours ($(N-1)!/2$ in the symmetric TSP, since reversing a tour gives the same length). For example, with $N=20$ this is $19! \approx 1.2\times 10^{17}$ candidate tours. Exhaustively evaluating every tour and selecting the best is therefore computationally infeasible.

There are **exact** solvers that can find optimal tours by combining relaxations, bounds, and systematic search (e.g., branch-and-cut). But that is not the focus of this blog. Instead, we will look at two classical strategies used to obtain (not optimal but high-quality) tours efficiently: **constructive heuristics** and **local search**.

### Constructive heuristics

A heuristic is a "strategy or rule" used when finding the optimal solution is computationally impossible. Constructive heuristics build a solution step-by-step by making choices following specific rules.

There are many constructive heuristics for the TSP, here two of the most well known:

- **Nearest Neighbor**: Starting from a city, repeatedly go to the nearest unvisited city, until all cities are visited, then go back to the initial city.  
- **Nearest Insertion**: Maintain a partial tour and insert the next city where it increases the tour length the least.

The primary drawback of these methods is that they are myopic. Because these algorithms never "look ahead," they often result in a logarithmic approximation factor. In plain English: as the map gets bigger, the gap between the heuristic's guess and the actual shortest path grows significantly. They are prone to "the lighthouse effect," where they travel efficiently for 90% of the trip but are forced to take a massive, inefficient leap at the end to close the loop.

### Local search

Once a constructive heuristic provides an initial feasible solution, **Local Search (LS)** takes over to optimize it. It operates on the principle of neighborhoods: it takes the current route and looks at "neighboring" routes that are only slightly different.

To move in the neighborhood of solutions, we define operators (or actions later). One of the most used operator in the TSP is the **2-opt**. The 2-opt selects two non-adjacent edges, (A,B) and (C,D), deletes them, and replaces them with (A,C) and (B,D).

Therefore, the LS takes a solution and test several 2-opt moves, if any of those moves improves the quality of the tour it moves there, and continues repeatedly improving the solution until there is no possible improving 2-opt move in the neighborhood (we reached a local optima).

### Demo
In the demo below we show how these two families work in TSP:

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
        if(i===0 && j===n-1) continue; // avoid full reversal
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

  function makeInstance(n=34){
    pts=[];
    for(let i=0;i<n;i++){
      const x = 0.08 + 0.84 * rng();
      const y = 0.08 + 0.84 * rng();
      pts.push([x,y]);
    }
  }

  function initConstruct(){
    partial = [0];
    visited = new Array(pts.length).fill(false);
    visited[0]=true;
    step = 0;
    lastMove = null;
    pillInfo.textContent = "Constructive: sequential selection (illustrative).";
    updatePills();
  }

  function initImprove(){
    tour = nearestNeighborTour(pts, 0);
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
      pillCost.textContent = partial.length>=2 ? ("Partial cost: " + tourCost(pts, partial).toFixed(2)) : "Partial cost: –";
      pillStep.textContent = "Step: " + step + "/" + (pts.length-1);
    }else{
      pillCost.textContent = "Cost: " + tourCost(pts, tour).toFixed(2);
      pillStep.textContent = "Step: " + step;
    }
  }

  function stepOnce(){
    if(mode==="construct"){
      if(partial.length === pts.length){
        pillInfo.textContent = "Constructive: tour completed.";
        updatePills(); draw();
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
      pillInfo.textContent = "Constructive: picked next city.";
      updatePills(); draw();
      return true;
    }else{
      const mv = twoOptBestMove(pts, tour);
      if(mv.i<0){
        pillInfo.textContent = "Improvement: reached a 2-opt local optimum.";
        updatePills(); draw();
        return false;
      }
      lastMove = {type:"2opt", i:mv.i, j:mv.j};
      tour = applyTwoOpt(tour, mv.i, mv.j);
      step++;
      pillInfo.textContent = "Improvement: applied a 2-opt swap.";
      updatePills(); draw();
      return true;
    }
  }

  function draw(){
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    ctx.clearRect(0,0,W,H);

    const dark = document.documentElement.getAttribute("data-theme")==="dark";
    const edge = dark ? "rgba(240,240,240,.35)" : "rgba(0,0,0,.18)";
    const edgeStrong = dark ? "rgba(240,240,240,.70)" : "rgba(0,0,0,.40)";
    const nodeFill = dark ? "rgba(230,230,230,.90)" : "rgba(30,30,30,.85)";
    const nodeStroke = dark ? "rgba(255,255,255,.25)" : "rgba(0,0,0,.15)";
    const accent = getComputedStyle(document.documentElement).getPropertyValue("--global-theme-color").trim() || (dark ? "#00C060" : "#00A550");

    const pad = 26;
    function X(x){ return pad + x*(W-2*pad); }
    function Y(y){ return pad + y*(H-2*pad); }

    function drawPath(path, close=false){
      if(path.length<2) return;
      ctx.beginPath();
      ctx.moveTo(X(pts[path[0]][0]), Y(pts[path[0]][1]));
      for(let k=1;k<path.length;k++){
        ctx.lineTo(X(pts[path[k]][0]), Y(pts[path[k]][1]));
      }
      if(close) ctx.lineTo(X(pts[path[0]][0]), Y(pts[path[0]][1]));
      ctx.strokeStyle = edgeStrong;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    if(mode==="construct"){
      drawPath(partial, false);
      if(partial.length===pts.length) drawPath(partial, true);
    }else{
      drawPath(tour, true);
    }

    // highlight last move
    if(lastMove && lastMove.type==="pick"){
      ctx.strokeStyle = accent;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(X(pts[lastMove.a][0]), Y(pts[lastMove.a][1]));
      ctx.lineTo(X(pts[lastMove.b][0]), Y(pts[lastMove.b][1]));
      ctx.stroke();
    }
    if(lastMove && lastMove.type==="2opt"){
      const n=tour.length;
      const i=lastMove.i, j=lastMove.j;
      const a = tour[i], b = tour[(i+1)%n];
      const c = tour[j], d = tour[(j+1)%n];
      ctx.strokeStyle = accent;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(X(pts[a][0]), Y(pts[a][1])); ctx.lineTo(X(pts[b][0]), Y(pts[b][1]));
      ctx.moveTo(X(pts[c][0]), Y(pts[c][1])); ctx.lineTo(X(pts[d][0]), Y(pts[d][1]));
      ctx.stroke();
    }

    // nodes
    for(let i=0;i<pts.length;i++){
      const px = X(pts[i][0]), py = Y(pts[i][1]);
      ctx.beginPath();
      ctx.arc(px,py,5.2,0,Math.PI*2);
      ctx.fillStyle = nodeFill;
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = nodeStroke;
      ctx.stroke();
    }

    // start marker
    ctx.beginPath();
    ctx.arc(X(pts[0][0]), Y(pts[0][1]), 8, 0, Math.PI*2);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 3;
    ctx.stroke();

    // footer
    ctx.fillStyle = dark ? "rgba(255,255,255,.60)" : "rgba(0,0,0,.55)";
    ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
    const tag = (mode==="construct")
      ? "Constructive: one-pass tour construction (illustrative policy)"
      : "Improvement: iterative local search (greedy 2-opt)";
    ctx.fillText(tag, 18, H-14);
  }

  modeConstruct.addEventListener("click", ()=>{
    stop(); mode="construct"; initConstruct(); draw();
  });
  modeImprove.addEventListener("click", ()=>{
    stop(); mode="improve"; initImprove(); draw();
  });
  btnReset.addEventListener("click", resetAll);
  btnStep.addEventListener("click", ()=>{ stop(); stepOnce(); });
  btnPlay.addEventListener("click", play);

  resetAll();
  resize();
})();
</script>

<div class="note">
<strong>How to interpret this contrast.</strong><br/>
Constructive solvers invest modeling capacity into producing a good solution in one decoding pass (optionally with restarts). Improvement solvers invest modeling capacity into choosing <em>moves</em> that refine an candidate solution. Both can be made budget-aware; they just spend compute differently.
</div>

---

## Why Neural Combinatorial Optimization

**Neural Combinatorial Optimization (NCO)** adopts a learning perspective on solver design: instead of hand-coding every heuristic rule, we ask whether a model can **learn search strategies** from data and interaction.

Concretely, an NCO method typically learns one or more of the following:

- **Representations** that make the relevant structure easy to reason about (sets, graphs, constraints).
- **Decision rules** that choose actions during construction or improvement (which city to visit next, which local move, which edge to cut).
- **Search priors** that bias exploration toward promising regions of the solution space.

This is not meant to “replace” classical optimization. In practice, many successful NCO systems reuse classical ingredients (feasibility checks, neighborhood operators, decoding schemes) and learn the parts where *good choices are hard to specify* but easy to evaluate with data.

Historically, NCO accelerated once sequence models and attention made it practical to map **sets/graphs → permutations/structures**, starting with pointer-network style decoders for routing problems <d-cite key="vinyals2015pointer"></d-cite> and then attention-based constructive solvers for TSP <d-cite key="bello2016neural,kool2019attention"></d-cite>. Since then, NCO has expanded to non-autoregressive approaches (heatmaps, diffusion) <d-cite key="joshi2019efficient,sun2023difusco"></d-cite> and to methods that explicitly learn *iterative refinement* (learned local search / neural improvement) <d-cite key="chen2021learning"></d-cite>.

---

## Two families of neural solvers

At a high level, and imitating the classical optimization algorithms, many NCO solvers for routing can be grouped into:

1. **Neural constructive models**: produce a tour in a single decoding run (sometimes with multiple starts).
2. **Neural improvement models**: iteratively refine a tour using local moves.

<div class="note">
<strong>Construct vs improve is a design choice, not a verdict.</strong><br/>
Constructive models output a full solution in one decoding run (often with optional restarts). Improvement models start from a solution and apply local modifications over time. Which is preferable depends on the instance family, the compute budget, and what “good” looks like in deployment. In fact, they can be combined (construct → improve).
</div>

### Neural constructive models

A constructive policy builds a solution one decision at a time. The canonical example is an attention-based encoder–decoder:

- **Encoder**: maps the set of cities to embeddings (often using self-attention).
- **Decoder**: autoregressively selects the next city conditioned on the partial tour and a visited mask.

In the attention model for TSP <d-cite key="kool2019attention"></d-cite>, the decoder defines a distribution:
$$
p_\theta(\pi) = \prod_{t=1}^{N} p_\theta(\pi_t \mid \pi_{<t}, x),
$$
and at inference you decode greedily, by sampling, or via structured search (beam, multi-start).

**Typical strength.** Constructive models are excellent at amortizing: after training, a single forward pass can output a good tour quickly. Additional compute is usually spent on *restarts* (sample more tours, keep the best) rather than deeper reasoning within one run.

---

### Neural improvement models

Neural improvement (learned local search) treats optimization as a sequential decision process:

- **State**: current solution (tour) + instance (cities) + optional history.
- **Action**: a local move (e.g., a 2-opt swap).
- **Transition**: apply the move to get a new tour.
- **Reward**: improvement in tour cost (dense) or best-so-far improvement (sparse).

A classic local operator is **2-opt**: pick two tour edges \((i,i+1)\) and \((j,j+1)\), remove them, and reconnect in the other way (equivalently reverse the segment between them). This removes crossings and frequently reduces cost. The neighborhood size is \(O(N^2)\), so a learned policy can matter simply by prioritizing which moves to examine.

This connects to broader “learning to search” approaches in CO <d-cite key="chen2021learning,mazyavkina2021reinforcement,bengio2021machine"></d-cite>.

**Typical strength.** Improvement models naturally define an anytime procedure: you can stop at any step, keep the best-so-far tour, and trade compute for quality.


---

## Training signals

NCO sits at the intersection of optimization and learning, so the training signal is a central design lever.

### Reinforcement learning (policy gradients)

A common path is to treat the solver as a stochastic policy and optimize expected cost. For constructive decoding, REINFORCE-style training is a standard baseline <d-cite key="bello2016neural,kool2019attention"></d-cite>. For improvement, you can define dense rewards such as
$$
r_t = C(\pi_t) - C(\pi_{t+1}),
$$
or best-so-far rewards that focus learning on steps that first achieve a new candidate.

Practical patterns include advantage normalization, strong baselines, entropy regularization, and (for multi-step rollouts) PPO-style updates.

### Supervised learning (imitation)

When you have a strong teacher (exact solvers for small $N$, heuristics for large $N$), imitation can be stable and efficient. For constructive models the teacher might be an optimal tour (but there are many equivalent permutations); for improvement models the teacher is often “best move in a neighborhood”.

A recurring issue is **ambiguity**: many actions can be equally good. Good objectives handle ties explicitly (set-valued targets, soft aggregation) rather than forcing an arbitrary single label.

### Hybrid: imitation → RL fine-tuning

A practical recipe is to imitate for fast convergence and then fine-tune with RL to align optimization with the evaluation budget and metric.

---

## Inference under a compute budget

A deployment-relevant view of NCO is: *what do you get per unit of compute?* Typical knobs differ by family:

- **Constructive**: spend extra compute on *restarts* (sampling more tours, multi-start, beam search) <d-cite key="kwon2020pomo"></d-cite>.
- **Improvement**: spend extra compute on *more steps* (more local moves, larger neighborhoods, deeper rollouts).
- **Hybrid**: construct a good initial tour and then refine it—often a strong baseline when latency is moderate.

A helpful diagnostic is an **anytime curve**: best-so-far cost vs time/steps. Different methods can cross depending on (i) budget, (ii) instance distribution, and (iii) implementation details.

---

## Pitfalls and evaluation

A few common traps in NCO experiments:

- **Budget mismatch.** Training/evaluating “one-shot” while deploying with heavy search (or vice versa) can hide the real tradeoffs.
- **Unfair comparisons.** Wall-clock time, batch sizes, and hardware matter. Report compute and decoding settings, not just final cost.
- **Distribution shift.** Learned policies can be sensitive to geometry, constraint changes, or size scaling; include out-of-distribution tests.
- **Leakage through labels.** If you train on optimal tours for small \(N\), be explicit about what supervision is used and where it comes from.
- **Ablation blindness.** Representation choices (positional encodings, invariances, neighborhood definitions) often matter as much as the backbone network.

---

## Where the field is going

Several directions are especially active:

- **Better inductive biases** for graphs and constraints (invariances, symmetries, structured decoding).
- **Test-time adaptation** and memory (learned heuristics that improve as they search).
- **Non-autoregressive generation** (heatmaps, diffusion) as proposal mechanisms <d-cite key="joshi2019efficient,sun2023difusco"></d-cite>.
- **Hybridization** with classical optimization (learn the hard decisions, keep the hard constraints exact).
- **Budget-aware training** so that the policy allocates computation effectively rather than only optimizing end-of-rollout quality.

<d-bibliography></d-bibliography>
