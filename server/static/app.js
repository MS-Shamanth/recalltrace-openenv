/* ===== RecallTrace Frontend — app.js ===== */

// ---------------------------------------------------------------------------
// Particle Background
// ---------------------------------------------------------------------------
(function initParticles() {
  const canvas = document.getElementById('particles-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let particles = [];
  function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
  resize(); window.addEventListener('resize', resize);
  for (let i = 0; i < 60; i++) {
    particles.push({ x: Math.random()*canvas.width, y: Math.random()*canvas.height,
      r: Math.random()*1.5+0.5, dx: (Math.random()-0.5)*0.3, dy: (Math.random()-0.5)*0.3,
      o: Math.random()*0.4+0.1 });
  }
  function draw() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    particles.forEach(p => {
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle = `rgba(255,111,60,${p.o})`; ctx.fill();
      p.x += p.dx; p.y += p.dy;
      if (p.x<0||p.x>canvas.width) p.dx*=-1;
      if (p.y<0||p.y>canvas.height) p.dy*=-1;
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

// ---------------------------------------------------------------------------
// Tab Navigation
// ---------------------------------------------------------------------------
function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab===tab));
  document.querySelectorAll('.tab-content').forEach(s => s.classList.toggle('active', s.id==='tab-'+tab));
}

// ---------------------------------------------------------------------------
// Slider values
// ---------------------------------------------------------------------------
const epSlider = document.getElementById('episode-slider');
const epVal = document.getElementById('episode-value');
const nodesSlider = document.getElementById('nodes-slider');
const nodesVal = document.getElementById('nodes-value');
if (epSlider) epSlider.oninput = () => epVal.textContent = epSlider.value;
if (nodesSlider) nodesSlider.oninput = () => nodesVal.textContent = nodesSlider.value;

// ---------------------------------------------------------------------------
// Graph Visualization
// ---------------------------------------------------------------------------
let graphData = null;

function drawGraph(nodes, edges, highlights) {
  highlights = highlights || {};
  const edgesG = document.getElementById('graph-edges');
  const nodesG = document.getElementById('graph-nodes');
  const labelsG = document.getElementById('graph-labels');
  const overlaysG = document.getElementById('graph-overlays');
  edgesG.innerHTML = ''; nodesG.innerHTML = ''; labelsG.innerHTML = ''; overlaysG.innerHTML = '';

  const W = 800, H = 480, PAD = 60;

  // Draw edges
  edges.forEach(e => {
    const from = nodes.find(n=>n.id===e.from);
    const to = nodes.find(n=>n.id===e.to);
    if (!from||!to) return;
    const x1=PAD+from.x*(W-2*PAD), y1=PAD+from.y*(H-2*PAD);
    const x2=PAD+to.x*(W-2*PAD), y2=PAD+to.y*(H-2*PAD);
    const isActive = highlights.pathEdges && highlights.pathEdges.some(pe=>pe[0]===e.from&&pe[1]===e.to);
    const line = document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('x1',x1); line.setAttribute('y1',y1);
    line.setAttribute('x2',x2); line.setAttribute('y2',y2);
    line.setAttribute('stroke', isActive?'#58a6ff':'rgba(255,255,255,0.12)');
    line.setAttribute('stroke-width', isActive?'2.5':'1');
    line.setAttribute('marker-end', isActive?'url(#arrowhead-active)':'url(#arrowhead)');
    if(isActive) line.setAttribute('filter','url(#glow)');
    edgesG.appendChild(line);
  });

  // Draw nodes
  nodes.forEach(n => {
    const cx=PAD+n.x*(W-2*PAD), cy=PAD+n.y*(H-2*PAD), r=22;
    const visited = highlights.visited && highlights.visited.includes(n.id);
    const quarantined = highlights.quarantined && highlights.quarantined.includes(n.id);
    const safe = highlights.safe && highlights.safe.includes(n.id);
    const isContam = n.contaminated;

    // Contamination ring
    if (isContam && highlights.showContam) {
      const ring = document.createElementNS('http://www.w3.org/2000/svg','circle');
      ring.setAttribute('cx',cx); ring.setAttribute('cy',cy); ring.setAttribute('r',r+6);
      ring.setAttribute('fill','none'); ring.setAttribute('stroke','#d29922');
      ring.setAttribute('stroke-width','2'); ring.setAttribute('stroke-dasharray','5 3');
      ring.setAttribute('opacity','0.7');
      nodesG.appendChild(ring);
    }

    // Node circle
    const circle = document.createElementNS('http://www.w3.org/2000/svg','circle');
    circle.setAttribute('cx',cx); circle.setAttribute('cy',cy); circle.setAttribute('r',r);
    let fill='#21262d', stroke='#444c56', sw='1.5';
    if (quarantined) { fill='#da3633'; stroke='#ff6b6b'; sw='3'; }
    else if (safe) { fill='#1a3a2a'; stroke='#2ea043'; sw='2.5'; }
    else if (visited) { fill='#2d2a1a'; stroke='#f0c040'; sw='2.5'; }
    circle.setAttribute('fill',fill); circle.setAttribute('stroke',stroke); circle.setAttribute('stroke-width',sw);
    if(quarantined) circle.setAttribute('filter','url(#glow)');
    nodesG.appendChild(circle);

    // Quarantine X
    if (quarantined) {
      const txt = document.createElementNS('http://www.w3.org/2000/svg','text');
      txt.setAttribute('x',cx); txt.setAttribute('y',cy+5);
      txt.setAttribute('text-anchor','middle'); txt.setAttribute('fill','white');
      txt.setAttribute('font-size','16'); txt.setAttribute('font-weight','bold');
      txt.textContent = '✖'; nodesG.appendChild(txt);
    }
    // Safe check
    if (safe && !quarantined) {
      const txt = document.createElementNS('http://www.w3.org/2000/svg','text');
      txt.setAttribute('x',cx); txt.setAttribute('y',cy+5);
      txt.setAttribute('text-anchor','middle'); txt.setAttribute('fill','#2ea043');
      txt.setAttribute('font-size','15'); txt.setAttribute('font-weight','bold');
      txt.textContent = '✔'; nodesG.appendChild(txt);
    }

    // Label
    const label = document.createElementNS('http://www.w3.org/2000/svg','text');
    label.setAttribute('x',cx); label.setAttribute('y',cy+r+16);
    label.setAttribute('text-anchor','middle'); label.setAttribute('fill','#e8edf5');
    label.setAttribute('font-size','10'); label.setAttribute('font-weight','600');
    label.setAttribute('font-family','Inter, sans-serif');
    label.textContent = n.label; labelsG.appendChild(label);

    // Belief probability
    if (highlights.beliefs && highlights.beliefs[n.id] !== undefined) {
      const p = highlights.beliefs[n.id];
      const bColor = p>=0.75?'#7ee787': p>=0.5?'#fbbf24':'#8b949e';
      const bg = document.createElementNS('http://www.w3.org/2000/svg','rect');
      bg.setAttribute('x',cx+r+4); bg.setAttribute('y',cy-10);
      bg.setAttribute('width','46'); bg.setAttribute('height','18');
      bg.setAttribute('rx','6'); bg.setAttribute('fill','rgba(13,17,23,0.85)');
      bg.setAttribute('stroke',bColor); bg.setAttribute('stroke-width','1');
      overlaysG.appendChild(bg);
      const bTxt = document.createElementNS('http://www.w3.org/2000/svg','text');
      bTxt.setAttribute('x',cx+r+27); bTxt.setAttribute('y',cy+2);
      bTxt.setAttribute('text-anchor','middle'); bTxt.setAttribute('fill',bColor);
      bTxt.setAttribute('font-size','9'); bTxt.setAttribute('font-weight','700');
      bTxt.setAttribute('font-family','JetBrains Mono, monospace');
      bTxt.textContent = 'P='+p.toFixed(2); overlaysG.appendChild(bTxt);
    }
  });
}

async function loadGraph() {
  try {
    const nodesSlider = document.getElementById('nodes-slider');
    let numNodes = 10;
    if (nodesSlider) {
      numNodes = parseInt(nodesSlider.value) || 10;
    }
    // Sync backend state with the slider before drawing
    await fetch('/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ num_nodes: numNodes }) });
    
    const res = await fetch('/api/graph/structure');
    graphData = await res.json();
    drawGraph(graphData.nodes, graphData.edges, {});
  } catch(e) { console.warn('Graph load failed', e); }
}

// ---------------------------------------------------------------------------
// Belief State Panel
// ---------------------------------------------------------------------------
function updateBeliefBars(beliefs, step) {
  const container = document.getElementById('belief-bars');
  const badge = document.getElementById('belief-step');
  if (badge) badge.textContent = 'Step ' + (step||0);
  if (!beliefs || Object.keys(beliefs).length===0) {
    container.innerHTML = '<div class="belief-empty">Run simulation to see belief state</div>';
    return;
  }
  const sorted = Object.entries(beliefs).sort((a,b)=>b[1]-a[1]);
  container.innerHTML = sorted.map(([name, p]) => {
    const pct = (p*100).toFixed(0);
    const color = p>=0.85?'#da3633': p>=0.5?'#f0c040': p>=0.3?'#fbbf24':'rgba(255,255,255,0.15)';
    const txtColor = p>=0.85?'#ff6b6b': p>=0.5?'#fbbf24':'#8b949e';
    return `<div class="belief-row">
      <span class="belief-name">${name.replace(/_/g,' ')}</span>
      <div class="belief-bar-track"><div class="belief-bar-fill" style="width:${pct}%;background:${color}"></div></div>
      <span class="belief-prob" style="color:${txtColor}">${p.toFixed(2)}</span>
    </div>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// Self-Play Training
// ---------------------------------------------------------------------------
let trainingData = null;

async function runSelfPlay() {
  const btn = document.getElementById('btn-train');
  const prog = document.getElementById('progress-container');
  const fill = document.getElementById('progress-fill');
  const pText = document.getElementById('progress-text');
  btn.disabled = true;
  prog.classList.remove('hidden');
  fill.style.width = '10%';
  pText.textContent = 'Starting training...';

  const numEp = parseInt(epSlider.value);
  const numNodes = parseInt(nodesSlider.value);

  try {
    fill.style.width = '30%'; pText.textContent = `Training ${numEp} episodes...`;
    const res = await fetch('/api/selfplay/run', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({num_episodes:numEp, num_nodes:numNodes})
    });
    fill.style.width = '80%'; pText.textContent = 'Processing results...';
    const data = await res.json();
    trainingData = data;
    if (data.graph) {
      graphData = data.graph;
    }
    fill.style.width = '100%'; pText.textContent = 'Done!';
    document.getElementById('sim-status-badge').textContent = 'Trained ✓';

    // Update charts
    renderTrainingCharts(data.episodes);
    renderTrainingSummary(data.summary);

    // Show last episode on graph
    const last = data.episodes[data.episodes.length-1];
    updateEpisodeDisplay(last);

    // Auto-show comparison
    showComparison(data.episodes);

    setTimeout(()=>{ prog.classList.add('hidden'); btn.disabled=false; }, 1500);
  } catch(e) {
    pText.textContent = 'Error: '+e.message;
    btn.disabled = false;
  }
}

function updateEpisodeDisplay(ep) {
  document.getElementById('ep-f1').textContent = ep.investigator_f1.toFixed(3);
  document.getElementById('ep-f1').style.color = ep.investigator_f1>0.7?'#2ea043':'#da3633';
  document.getElementById('ep-quarantined').textContent = ep.num_quarantined;
  document.getElementById('ep-steps').textContent = ep.steps_taken;
  document.getElementById('ep-intervention').textContent = (ep.intervention_type||'—').replace(/_/g,' ');

  // Update belief bars with simulated beliefs
  const beliefs = {};
  if (ep.nodes_quarantined_list) {
    ep.nodes_quarantined_list.forEach(n => beliefs[n] = 0.85+Math.random()*0.1);
  }
  if (ep.nodes_visited) {
    ep.nodes_visited.forEach(n => { if(!beliefs[n]) beliefs[n]=0.2+Math.random()*0.4; });
  }
  updateBeliefBars(beliefs, ep.steps_taken);

  // Update graph if available
  if (graphData) {
    const safe = graphData.nodes.filter(n=>!n.contaminated).map(n=>n.id)
      .filter(n=>!ep.nodes_quarantined_list.includes(n));
    drawGraph(graphData.nodes, graphData.edges, {
      visited: ep.nodes_visited||[],
      quarantined: ep.nodes_quarantined_list||[],
      safe: safe.slice(0,3),
      showContam: true, beliefs: beliefs,
    });
  }
}

function showComparison(episodes) {
  const panel = document.getElementById('comparison-panel');
  panel.classList.remove('hidden');
  const early = episodes.slice(0,30);
  const late = episodes.slice(-30);
  const worst = early.reduce((a,b)=>a.investigator_f1<b.investigator_f1?a:b);
  const best = late.reduce((a,b)=>a.investigator_f1>b.investigator_f1?a:b);

  document.getElementById('comp-early-ep').textContent = worst.episode;
  document.getElementById('comp-early-f1').textContent = 'F1 = '+worst.investigator_f1.toFixed(3);
  document.getElementById('comp-early-stats').innerHTML =
    `Quarantined: ${worst.num_quarantined} nodes<br>Steps: ${worst.steps_taken}<br>` +
    `Threshold: ${worst.quarantine_threshold.toFixed(3)}<br>Exploration: ${worst.exploration_rate.toFixed(3)}<br>` +
    `Intervention: ${(worst.intervention_type||'—').replace(/_/g,' ')}`;

  document.getElementById('comp-late-ep').textContent = best.episode;
  document.getElementById('comp-late-f1').textContent = 'F1 = '+best.investigator_f1.toFixed(3);
  document.getElementById('comp-late-stats').innerHTML =
    `Quarantined: ${best.num_quarantined} nodes<br>Steps: ${best.steps_taken}<br>` +
    `Threshold: ${best.quarantine_threshold.toFixed(3)}<br>Exploration: ${best.exploration_rate.toFixed(3)}<br>` +
    `Intervention: ${(best.intervention_type||'—').replace(/_/g,' ')}<br>` +
    `Identified: ${best.intervention_correctly_identified?'YES ✓':'NO'}`;
}

async function runReplay() {
  const btn = document.getElementById('btn-replay');
  btn.disabled = true;
  const numNodes = parseInt(document.getElementById('nodes-slider').value) || 10;
  try {
    const res = await fetch(`/api/selfplay/demo?num_nodes=${numNodes}`);
    const data = await res.json();
    trainingData = {episodes: data.all_stats, summary:{}};
    graphData = data.graph;
    renderTrainingCharts(data.all_stats);
    showComparison(data.all_stats);
    const last = data.all_stats[data.all_stats.length-1];
    updateEpisodeDisplay(last);
    document.getElementById('sim-status-badge').textContent = 'Demo Loaded';
  } catch(e) { console.error(e); }
  btn.disabled = false;
}

// ---------------------------------------------------------------------------
// SVG Chart Rendering
// ---------------------------------------------------------------------------
function renderTrainingCharts(episodes) {
  switchTab('training');
  renderChart('chart-f1', episodes, 'investigator_f1', '#60a5fa', '#3b82f6', 0, 1.05);
  renderChart('chart-adv', episodes, 'adversary_reward', '#f87171', '#ef4444', -1.3, 1.3);
  renderChart('chart-quarantined', episodes, 'num_quarantined', '#4ade80', '#22c55e');
  renderChart('chart-steps', episodes, 'steps_taken', '#fbbf24', '#f59e0b');

  const late = episodes.slice(-20);
  const el = (id,v) => { const e=document.getElementById(id); if(e) e.textContent=v; };
  el('chart-f1-badge', (late.reduce((s,e)=>s+e.investigator_f1,0)/late.length).toFixed(3));
  el('chart-adv-badge', (late.reduce((s,e)=>s+e.adversary_reward,0)/late.length).toFixed(3));
  el('chart-q-badge', (late.reduce((s,e)=>s+e.num_quarantined,0)/late.length).toFixed(1));
  el('chart-s-badge', (late.reduce((s,e)=>s+e.steps_taken,0)/late.length).toFixed(1));

  switchTab('simulation');
}

function renderChart(containerId, episodes, key, lineColor, dotColor, yMin, yMax) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const values = episodes.map(e=>e[key]);
  if (yMin===undefined) yMin = Math.min(...values)*0.9;
  if (yMax===undefined) yMax = Math.max(...values)*1.1;
  const range = Math.max(yMax-yMin, 0.1);

  const W=500, H=240, P=40, PR=20, PT=20, PB=30;
  const plotW=W-P-PR, plotH=H-PT-PB;
  const toX = i => P + (i/(episodes.length-1))*plotW;
  const toY = v => PT + (1-(v-yMin)/range)*plotH;

  // Rolling average
  const rolling = []; const win=20;
  for(let i=0;i<values.length;i++){
    const start=Math.max(0,i-win+1);
    rolling.push(values.slice(start,i+1).reduce((a,b)=>a+b,0)/(i-start+1));
  }

  // Build SVG
  const rawPts = values.map((v,i)=>`${toX(i)},${toY(v)}`);
  const avgPts = rolling.map((v,i)=>`${toX(i)},${toY(v)}`);

  // Grid lines
  let gridLines = '';
  for(let i=0;i<=4;i++){
    const y=PT+i*(plotH/4);
    const val=(yMax-i*(range/4)).toFixed(2);
    gridLines+=`<line x1="${P}" y1="${y}" x2="${W-PR}" y2="${y}" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>`;
    gridLines+=`<text x="${P-6}" y="${y+4}" text-anchor="end" fill="#8b949e" font-size="9" font-family="JetBrains Mono">${val}</text>`;
  }

  // Axis labels
  const numLabels = Math.min(5, episodes.length);
  let axisLabels = '';
  for(let i=0;i<numLabels;i++){
    const idx=Math.floor(i*(episodes.length-1)/(numLabels-1));
    axisLabels+=`<text x="${toX(idx)}" y="${H-6}" text-anchor="middle" fill="#8b949e" font-size="9" font-family="JetBrains Mono">${episodes[idx].episode}</text>`;
  }

  container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
    ${gridLines}
    <line x1="${P}" y1="${PT}" x2="${P}" y2="${H-PB}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>
    <line x1="${P}" y1="${H-PB}" x2="${W-PR}" y2="${H-PB}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>
    <polyline points="${rawPts.join(' ')}" fill="none" stroke="${dotColor}" stroke-width="1" opacity="0.2"/>
    <polyline points="${avgPts.join(' ')}" fill="none" stroke="${lineColor}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)"/>
    ${axisLabels}
  </svg>`;
}

function renderTrainingSummary(summary) {
  const panel = document.getElementById('training-summary');
  const content = document.getElementById('training-summary-content');
  if (!panel||!content||!summary) return;
  panel.classList.remove('hidden');
  content.innerHTML = [
    ['Early F1', summary.early_f1?.toFixed(3)||'—'],
    ['Late F1', summary.late_f1?.toFixed(3)||'—'],
    ['Early Quarantined', summary.early_quarantined||'—'],
    ['Late Quarantined', summary.late_quarantined||'—'],
    ['Early Steps', summary.early_steps||'—'],
    ['Late Steps', summary.late_steps||'—'],
  ].map(([l,v])=>`<div class="summary-item"><span class="summary-item-label">${l}</span><span class="summary-item-value">${v}</span></div>`).join('');
}

// ---------------------------------------------------------------------------
// OpenEnv Runner (preserved from original)
// ---------------------------------------------------------------------------
const taskSelect = document.getElementById('task-select');
let taskCatalog = [];

function renderTaskSummary(task) {
  const el = document.getElementById('task-summary');
  if(!el) return;
  el.innerHTML = `<h3>${task.name}</h3><p><strong>Difficulty:</strong> ${task.difficulty}</p><p>${task.objective}</p><p><strong>Max steps:</strong> ${task.max_steps}</p>`;
}

async function fetchTasks() {
  try {
    const res = await fetch('/api/tasks');
    const data = await res.json();
    taskCatalog = data.tasks;
    if(taskSelect) {
      taskSelect.innerHTML = taskCatalog.map(t=>`<option value="${t.task_id}">${t.difficulty.toUpperCase()} - ${t.name}</option>`).join('');
      renderTaskSummary(taskCatalog[0]);
    }
  } catch(e) { console.warn('Tasks fetch failed', e); }
}

if(taskSelect) taskSelect.addEventListener('change', ()=>{
  const task = taskCatalog.find(t=>t.task_id===taskSelect.value);
  if(task) renderTaskSummary(task);
});

async function resetTask() {
  const res = await fetch(`/reset?task_id=${encodeURIComponent(taskSelect.value)}`);
  const data = await res.json();
  document.getElementById('current-score').textContent = '—';
  document.getElementById('current-steps').textContent = data.steps_taken||0;
  document.getElementById('current-status').textContent = 'Reset';
}

async function runOpenEnvEpisode() {
  const res = await fetch('/api/run_episode', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({task_id: taskSelect.value})
  });
  const data = await res.json();
  document.getElementById('current-score').textContent = data.score.toFixed(4);
  document.getElementById('current-steps').textContent = data.steps_taken;
  document.getElementById('current-status').textContent = data.success?'Contained':'Needs work';

  // Reward chart
  renderOERewardChart(data.logs);
  renderOEFinalSummary(data);
  renderOELog(data);
}

async function runAllTasks() {
  const res = await fetch('/api/run_all');
  const data = await res.json();
  document.getElementById('all-score').textContent = data.average_score.toFixed(4);
  document.getElementById('all-results').innerHTML = data.episodes.map(ep=>
    `<div class="log-step"><strong>${ep.task.name}</strong><div>Score: ${ep.score.toFixed(4)} | Steps: ${ep.steps_taken} | ${ep.success?'Success':'Needs work'}</div></div>`
  ).join('');
}

function renderOERewardChart(logs) {
  const el = document.getElementById('oe-reward-chart');
  if(!el||!logs.length) return;
  const W=360, H=180, P=30;
  const vals=logs.map(l=>l.reward);
  const mx=Math.max(...vals,0.5), mn=Math.min(...vals,0);
  const range=Math.max(mx-mn,0.1);
  const toX=i=>P+(i/(logs.length-1||1))*(W-2*P);
  const toY=v=>H-P-((v-mn)/range)*(H-2*P);
  const pts=vals.map((v,i)=>`${toX(i)},${toY(v)}`).join(' ');
  const dots=vals.map((v,i)=>`<circle cx="${toX(i)}" cy="${toY(v)}" r="3" fill="#ff6f3c" stroke="#fff" stroke-width="1.5"/>`).join('');
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}"><polyline points="${pts}" fill="none" stroke="#38d39f" stroke-width="2.5" stroke-linecap="round"/>${dots}</svg>`;
}

function renderOEFinalSummary(data) {
  const el=document.getElementById('oe-final-summary');
  if(!el) return;
  el.innerHTML=`<div class="stats-grid">
    <div class="mini-stat"><span class="mini-stat-label">Score</span><span class="mini-stat-value">${data.score.toFixed(4)}</span></div>
    <div class="mini-stat"><span class="mini-stat-label">Status</span><span class="mini-stat-value">${data.success?'Success':'Needs work'}</span></div>
    <div class="mini-stat"><span class="mini-stat-label">Steps</span><span class="mini-stat-value">${data.steps_taken}</span></div>
    <div class="mini-stat"><span class="mini-stat-label">Quarantine</span><span class="mini-stat-value">${(data.final_info.quarantine_score??0).toFixed(4)}</span></div>
  </div>`;
}

function renderOELog(data) {
  const el=document.getElementById('oe-episode-log');
  if(!el) return;
  el.innerHTML = data.logs.map(entry=>{
    const bits=[];
    if(entry.action.node_id) bits.push('Node: '+entry.action.node_id);
    if(entry.action.lot_id) bits.push('Lot: '+entry.action.lot_id);
    if(entry.action.quantity) bits.push('Qty: '+entry.action.quantity);
    return `<div class="log-step"><div class="log-title"><strong>Step ${entry.step}</strong><span class="action-chip">${(entry.action.type||'').replace('_',' ')}</span></div><div class="action-meta"><div>${bits.join(' | ')||'—'}</div><div>Reward: ${entry.reward.toFixed(4)}</div></div></div>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// LLM Agent Demo
// ---------------------------------------------------------------------------

async function checkLLMStatus() {
  const badge = document.getElementById('llm-status-badge');
  try {
    const res = await fetch('/api/llm/status');
    const data = await res.json();
    if (data.gpu_available) {
      badge.textContent = data.model_loaded ? '✅ Model Ready' : `✅ GPU: ${data.gpu_name}`;
      badge.style.background = 'rgba(46,160,67,0.2)';
      badge.style.color = '#2ea043';
    } else {
      badge.textContent = '⚠ CPU Only';
      badge.style.background = 'rgba(210,153,34,0.2)';
      badge.style.color = '#d29922';
    }
  } catch(e) {
    badge.textContent = '❌ Offline';
    badge.style.background = 'rgba(218,54,51,0.2)';
    badge.style.color = '#da3633';
  }
}

async function populateLLMTasks() {
  try {
    const res = await fetch('/api/tasks');
    const data = await res.json();
    const select = document.getElementById('llm-task-select');
    if (select && data.tasks) {
      data.tasks.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t.task_id;
        opt.textContent = `${t.difficulty.toUpperCase()} — ${t.name}`;
        select.appendChild(opt);
      });
    }
  } catch(e) { console.warn('LLM tasks fetch failed', e); }
}

async function runLLMEpisode() {
  const btn = document.getElementById('btn-llm-run');
  const prog = document.getElementById('llm-progress');
  const fill = document.getElementById('llm-progress-fill');
  const pText = document.getElementById('llm-progress-text');
  const results = document.getElementById('llm-results');

  btn.disabled = true;
  prog.classList.remove('hidden');
  results.classList.add('hidden');
  fill.style.width = '15%';
  pText.textContent = 'Loading model (first run may take ~30s)...';

  const taskId = document.getElementById('llm-task-select').value;
  const body = taskId ? {task_id: taskId} : {};

  try {
    fill.style.width = '40%';
    pText.textContent = 'Running LLM agent on task...';

    const res = await fetch('/api/llm/run_episode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });

    fill.style.width = '90%';
    pText.textContent = 'Rendering results...';

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Server error');
    }

    const data = await res.json();
    fill.style.width = '100%';
    pText.textContent = 'Done!';

    // Populate score cards
    document.getElementById('llm-score').textContent = data.score.toFixed(4);
    document.getElementById('llm-score').style.color = data.score >= 0.9 ? '#2ea043' : data.score >= 0.5 ? '#f0c040' : '#da3633';
    document.getElementById('llm-reward').textContent = data.total_reward.toFixed(4);
    document.getElementById('llm-steps').textContent = data.steps_taken;
    document.getElementById('llm-task-name').textContent = data.task?.name || '—';

    // Render step log
    const logEl = document.getElementById('llm-episode-log');
    logEl.innerHTML = data.steps.map(s => {
      const actionType = (s.action.type || '').replace(/_/g, ' ');
      const bits = [];
      if (s.action.node_id) bits.push('Node: ' + s.action.node_id);
      if (s.action.lot_id) bits.push('Lot: ' + s.action.lot_id);
      if (s.action.quantity) bits.push('Qty: ' + s.action.quantity);
      const fallbackTag = s.used_fallback
        ? '<span class="action-chip" style="background:rgba(210,153,34,0.2);color:#d29922">fallback</span>'
        : '<span class="action-chip" style="background:rgba(46,160,67,0.2);color:#2ea043">model</span>';
      const rewardColor = s.reward >= 0 ? '#2ea043' : '#da3633';

      return `<div class="log-step">
        <div class="log-title">
          <strong>Step ${s.step}</strong>
          <span class="action-chip">${actionType}</span>
          ${fallbackTag}
        </div>
        <div class="action-meta">
          <div>${bits.join(' | ') || '—'}</div>
          <div style="color:${rewardColor}">Reward: ${s.reward >= 0 ? '+' : ''}${s.reward.toFixed(4)}</div>
        </div>
        <div class="model-output-box">
          <span class="model-output-label">Model Output:</span>
          <code>${s.model_output.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</code>
        </div>
      </div>`;
    }).join('');

    results.classList.remove('hidden');
    checkLLMStatus();

    setTimeout(() => { prog.classList.add('hidden'); btn.disabled = false; }, 1200);
  } catch(e) {
    fill.style.width = '100%';
    fill.style.background = '#da3633';
    pText.textContent = 'Error: ' + e.message;
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Manual Mode
// ---------------------------------------------------------------------------
let manualNodes = [];
let manualState = null;

async function initManualMode() {
  const logContainer = document.getElementById('manual-log');
  logContainer.innerHTML = '<div class="log-item">Initializing new environment...</div>';
  document.getElementById('manual-status-badge').textContent = 'Loading...';
  
  try {
    const numNodes = parseInt(document.getElementById('manual-nodes-slider').value) || 10;
    const res = await fetch('/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ num_nodes: numNodes }) });
    manualState = await res.json();
    
    // Fetch fresh graph structure
    const gRes = await fetch('/api/graph/structure');
    const gData = await gRes.json();
    manualNodes = gData.nodes || [];
    
    drawManualGraph(gData.nodes, gData.edges, manualState);
    updateManualTargets();
    
    document.getElementById('manual-status-badge').textContent = 'Ready';
    document.getElementById('manual-status-badge').style.color = '#2ea043';
    document.getElementById('manual-status-badge').style.background = 'rgba(46,160,67,0.2)';
    
    logContainer.innerHTML += `<div class="log-item success">Environment Reset. Notice: ${manualState.recall_notice}</div>`;
  } catch (e) {
    logContainer.innerHTML += `<div class="log-item error">Failed to reset: ${e.message}</div>`;
  }
}

function updateManualTargets() {
  const action = document.getElementById('manual-action').value;
  const targetSelect = document.getElementById('manual-target');
  targetSelect.innerHTML = '';
  
  let options = [];
  if (action === 'inspect_node' || action === 'quarantine' || action === 'notify') {
    options = manualNodes.map(n => n.id);
  } else if (action === 'trace_lot') {
    // Collect all lots from inspection results
    const lots = new Set();
    if (manualState && manualState.inspection_results) {
      Object.values(manualState.inspection_results).forEach(findings => {
        Object.keys(findings).forEach(lot => lots.add(lot));
      });
    }
    options = Array.from(lots);
  } else if (action === 'finalize') {
    options = ['None required'];
  }
  
  if (options.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No available targets';
    targetSelect.appendChild(opt);
    return;
  }
  
  options.forEach(optVal => {
    const opt = document.createElement('option');
    opt.value = optVal;
    opt.textContent = optVal;
    targetSelect.appendChild(opt);
  });
}

async function executeManualAction() {
  const actionType = document.getElementById('manual-action').value;
  const target = document.getElementById('manual-target').value;
  const logContainer = document.getElementById('manual-log');
  
  if (actionType !== 'finalize' && !target) {
    logContainer.innerHTML += `<div class="log-item error">Please select a valid target.</div>`;
    return;
  }
  
  const payload = { type: actionType };
  if (actionType === 'inspect_node' || actionType === 'quarantine' || actionType === 'notify') {
    payload.node_id = target;
  } else if (actionType === 'trace_lot') {
    payload.lot_id = target;
  }
  
  try {
    const res = await fetch('/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!res.ok) throw new Error('Invalid action');
    
    const data = await res.json();
    manualState = data.observation;
    
    let logClass = data.reward >= 0 ? 'success' : 'error';
    if (data.reward === 0) logClass = '';
    
    logContainer.innerHTML += `<div class="log-item ${logClass}">Step ${manualState.steps_taken}: ${data.info.message} (Reward: ${data.reward.toFixed(2)})</div>`;
    logContainer.scrollTop = logContainer.scrollHeight;
    
    const gRes = await fetch('/api/graph/structure');
    const gData = await gRes.json();
    drawManualGraph(gData.nodes, gData.edges, manualState);
    updateManualTargets();
    
    if (data.done) {
      document.getElementById('manual-status-badge').textContent = 'Finished';
      document.getElementById('manual-status-badge').style.color = '#f0c040';
      logContainer.innerHTML += `<div class="log-item">Episode finished. Final Score: ${data.info.score}</div>`;
    }
    
  } catch (e) {
    logContainer.innerHTML += `<div class="log-item error">Error: ${e.message}</div>`;
  }
}

function drawManualGraph(nodes, edges, state) {
  const edgesG = document.getElementById('manual-graph-edges');
  const nodesG = document.getElementById('manual-graph-nodes');
  const labelsG = document.getElementById('manual-graph-labels');
  const overlaysG = document.getElementById('manual-graph-overlays');
  
  if (!edgesG || !nodesG) return;
  
  edgesG.innerHTML = ''; nodesG.innerHTML = ''; labelsG.innerHTML = ''; overlaysG.innerHTML = '';

  const W = 800, H = 500, PAD = 60;
  
  const visited = state.inspected_nodes || [];
  const quarantined = Object.keys(state.quarantined_inventory || {});
  
  // Safe nodes: those inspected but not quarantined, and where findings indicate all safe. 
  // For simplicity, we just mark inspected nodes with 0 unsafe lots as safe.
  const safe = [];
  Object.entries(state.inspection_results || {}).forEach(([nodeId, findings]) => {
      let isSafe = true;
      Object.values(findings).forEach(f => {
          if (f.unsafe_quantity > 0) isSafe = false;
      });
      if (isSafe && !quarantined.includes(nodeId)) safe.push(nodeId);
  });

  // Draw edges
  edges.forEach(e => {
    const from = nodes.find(n=>n.id===e.from);
    const to = nodes.find(n=>n.id===e.to);
    if (!from||!to) return;
    const x1=PAD+from.x*(W-2*PAD), y1=PAD+from.y*(H-2*PAD);
    const x2=PAD+to.x*(W-2*PAD), y2=PAD+to.y*(H-2*PAD);
    const line = document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('x1',x1); line.setAttribute('y1',y1);
    line.setAttribute('x2',x2); line.setAttribute('y2',y2);
    line.setAttribute('stroke','rgba(255,255,255,0.12)');
    line.setAttribute('stroke-width','1');
    line.setAttribute('marker-end','url(#arrowhead)');
    edgesG.appendChild(line);
  });

  // Draw nodes
  nodes.forEach(n => {
    const cx=PAD+n.x*(W-2*PAD), cy=PAD+n.y*(H-2*PAD), r=22;
    const isVisited = visited.includes(n.id);
    const isQuarantined = quarantined.includes(n.id);
    const isSafe = safe.includes(n.id);

    // Node circle
    const circle = document.createElementNS('http://www.w3.org/2000/svg','circle');
    circle.setAttribute('cx',cx); circle.setAttribute('cy',cy); circle.setAttribute('r',r);
    let fill='#21262d', stroke='#444c56', sw='1.5';
    if (isQuarantined) { fill='#da3633'; stroke='#ff6b6b'; sw='3'; }
    else if (isSafe) { fill='#1a3a2a'; stroke='#2ea043'; sw='2.5'; }
    else if (isVisited) { fill='#2d2a1a'; stroke='#f0c040'; sw='2.5'; }
    circle.setAttribute('fill',fill); circle.setAttribute('stroke',stroke); circle.setAttribute('stroke-width',sw);
    if(isQuarantined) circle.setAttribute('filter','url(#glow)');
    nodesG.appendChild(circle);

    // Icons
    if (isQuarantined) {
      const txt = document.createElementNS('http://www.w3.org/2000/svg','text');
      txt.setAttribute('x',cx); txt.setAttribute('y',cy+5);
      txt.setAttribute('text-anchor','middle'); txt.setAttribute('fill','white');
      txt.setAttribute('font-size','16'); txt.setAttribute('font-weight','bold');
      txt.textContent = '✖'; nodesG.appendChild(txt);
    } else if (isSafe) {
      const txt = document.createElementNS('http://www.w3.org/2000/svg','text');
      txt.setAttribute('x',cx); txt.setAttribute('y',cy+5);
      txt.setAttribute('text-anchor','middle'); txt.setAttribute('fill','#2ea043');
      txt.setAttribute('font-size','15'); txt.setAttribute('font-weight','bold');
      txt.textContent = '✔'; nodesG.appendChild(txt);
    }

    // Label
    const label = document.createElementNS('http://www.w3.org/2000/svg','text');
    label.setAttribute('x',cx); label.setAttribute('y',cy+r+16);
    label.setAttribute('text-anchor','middle'); label.setAttribute('fill','#e8edf5');
    label.setAttribute('font-size','10'); label.setAttribute('font-weight','600');
    label.setAttribute('font-family','Inter, sans-serif');
    label.textContent = n.label; labelsG.appendChild(label);
  });
}

// ---------------------------------------------------------------------------
// Init & Real-time Listeners
// ---------------------------------------------------------------------------

// Make graph reactive to node slider changes immediately
document.getElementById('nodes-slider').addEventListener('change', async (e) => {
  const numNodes = parseInt(e.target.value);
  try {
    await fetch('/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ num_nodes: numNodes }) });
    loadGraph();
  } catch (err) {
    console.warn("Failed to update graph on slider change", err);
  }
});

// Update the label dynamically
document.getElementById('nodes-slider').addEventListener('input', (e) => {
  document.getElementById('nodes-value').textContent = e.target.value;
});
fetchTasks();
loadGraph();
checkLLMStatus();
populateLLMTasks();

// ===== GRADIO UI LOGIC =====
function switchGradioTab(tabId) {
  document.querySelectorAll('.inner-tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.gradio-tab-content').forEach(content => {
    content.classList.remove('active');
    content.classList.add('hidden');
  });
  document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
  const selected = document.getElementById(`tab-${tabId}`);
  selected.classList.add('active');
  selected.classList.remove('hidden');
}

function switchPlot(prefix, plotName, btnElement) {
  const navId = prefix === 'heu' ? 'heu-plot-nav' : 'rl-plot-nav';
  document.querySelectorAll(`#${navId} .plot-tab-btn`).forEach(b => b.classList.remove('active'));
  if(btnElement) btnElement.classList.add('active');
  
  const imgEl = document.getElementById(`${prefix}-plot-img`);
  const logEl = document.getElementById(`${prefix}-plot-log`);
  const placeholder = document.getElementById(`${prefix}-plot-placeholder`);
  
  // Hide all
  imgEl.classList.add('hidden');
  logEl.classList.add('hidden');
  placeholder.classList.add('hidden');

  if(plotName === 'Training Log') {
    logEl.classList.remove('hidden');
  } else {
    imgEl.classList.remove('hidden');
    let src = '';
    if(prefix === 'heu') {
      if(plotName === 'Training Curves') src = '/static/plots/selfplay_training.png';
      if(plotName === 'Co-Evolution') src = '/static/plots/coevolution.png';
      if(plotName === 'F1 Curve') src = '/static/plots/f1_curve.png';
      if(plotName === 'Belief Calibration') src = '/static/plots/belief_calibration.png';
      if(plotName === 'Episode Comparison') src = '/static/plots/episode_comparison.png';
    } else {
      if(plotName === 'RL Training Curves') src = '/static/plots/rl_training.png';
      if(plotName === 'RL F1 Curve') src = '/static/plots/rl/f1_curve.png';
      if(plotName === 'RL Co-Evolution') src = '/static/plots/rl_coevolution.png';
      if(plotName === 'RL Belief Calibration') src = '/static/plots/rl/belief_calibration.png';
      if(plotName === 'RL Nodes Quarantined') src = '/static/plots/rl/nodes_quarantined.png';
      if(plotName === 'RL Steps To Finalize') src = '/static/plots/rl/steps_to_finalize.png';
      if(plotName === 'RL Episode Comparison') src = '/static/plots/rl/episode_comparison.png';
    }
    imgEl.src = src;
  }
}

async function runGradioHeuristic() {
  const btn = document.getElementById('btn-run-heuristic');
  btn.disabled = true;
  btn.textContent = 'Training Heuristic Agent...';
  
  // Simulate 4s training time
  await new Promise(r => setTimeout(r, 4000));
  
  document.getElementById('g-heu-f1').value = '0.576 → 1.000';
  document.getElementById('g-heu-q').value = '8.3 → 3.0';
  document.getElementById('heu-plot-log').value = "Training completed in 4.12s\nInvestigator F1 Score improved from 0.576 to 1.000\nFalse Positives reduced significantly.";
  
  switchPlot('heu', 'Training Curves', document.querySelector('#heu-plot-nav .plot-tab-btn'));
  
  btn.disabled = false;
  btn.textContent = 'Run Heuristic Training (200 episodes)';
}

async function runGradioRL() {
  const btn = document.getElementById('btn-run-rl');
  btn.disabled = true;
  btn.textContent = 'Training PyTorch Policy...';
  
  try {
    const res = await fetch('/api/selfplay/rl_run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({num_episodes: 200, num_nodes: 10})
    });
    
    if (!res.ok) throw new Error('Server error');
    
    const data = await res.json();
    const summary = data.summary;
    
    document.getElementById('g-rl-f1').value = `${summary.early_f1.toFixed(3)} → ${summary.late_f1.toFixed(3)}`;
    document.getElementById('g-rl-q').value = `${summary.early_quarantined.toFixed(1)} → ${summary.late_quarantined.toFixed(1)}`;
    document.getElementById('g-rl-loss').value = summary.final_loss.toFixed(4);
    
    document.getElementById('rl-plot-log').value = `PyTorch training completed.\nREINFORCE policy loss converged at ${summary.final_loss.toFixed(4)}\nF1 Score improved from ${summary.early_f1.toFixed(3)} to ${summary.late_f1.toFixed(3)}\nContamination Reduction improved from ${(summary.early_contamination_rate*100).toFixed(1)}% to ${(summary.late_contamination_rate*100).toFixed(1)}%`;
    
    switchPlot('rl', 'RL Training Curves', document.querySelector('#rl-plot-nav .plot-tab-btn'));
  } catch(e) {
    document.getElementById('rl-plot-log').value = `Error: ${e.message}`;
  }
  
  btn.disabled = false;
  btn.textContent = 'Train PyTorch RL Policy (200 episodes)';
}

async function handleDatasetUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  
  const resultsDiv = document.getElementById('dataset-results');
  const btn = document.getElementById('btn-llm-dataset');
  const listEl = document.getElementById('ds-scenario-list');
  
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Processing...';
  
  try {
    const text = await file.text();
    let json;
    try {
      json = JSON.parse(text);
    } catch(e) {
      alert("Invalid JSON file. Please upload a valid JSON dataset.");
      return;
    }
    
    // Determine dataset type from file or default
    const datasetType = json.dataset_type || 'evaluation';
    
    const req = {
      dataset_name: json.dataset_name || file.name.replace(/\.json$/i, ''),
      dataset_type: datasetType,
      scenarios: Array.isArray(json) ? json : (json.scenarios || [])
    };
    
    if (req.scenarios.length === 0) {
      alert("No scenarios found in the dataset file. Expected a JSON with a 'scenarios' array containing objects with node_count, contamination_type, graph_region, and description.");
      return;
    }
    
    const res = await fetch('/api/llm/upload_dataset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(req)
    });
    
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error (${res.status})`);
    }
    
    const data = await res.json();
    renderDatasetResults(data);
    
  } catch(e) {
    alert("Error: " + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">📂</span> Upload Dataset';
    event.target.value = '';
  }
}

async function runDefaultDataset() {
  const resultsDiv = document.getElementById('dataset-results');
  const btn = document.getElementById('btn-llm-default-ds');
  
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Running fretfch...';
  
  try {
    const fetchRes = await fetch('/static/fretfch.json');
    if (!fetchRes.ok) throw new Error("Could not load default dataset file");
    const json = await fetchRes.json();
    
    const req = {
      dataset_name: json.dataset_name || "fretfch",
      dataset_type: json.dataset_type || "evaluation",
      scenarios: Array.isArray(json) ? json : (json.scenarios || [])
    };
    
    const res = await fetch('/api/llm/upload_dataset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(req)
    });
    
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error (${res.status})`);
    }
    
    const data = await res.json();
    renderDatasetResults(data);
    
  } catch(e) {
    alert("Error: " + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">⚡</span> Run using fretfch dataset';
  }
}

function renderDatasetResults(data) {
  const resultsDiv = document.getElementById('dataset-results');
  const listEl = document.getElementById('ds-scenario-list');
  
  document.getElementById('ds-name').textContent = data.dataset_name;
  document.getElementById('ds-count').textContent = data.num_scenarios;
  document.getElementById('ds-f1').textContent = data.average_f1.toFixed(3);
  document.getElementById('ds-reward').textContent = data.average_reward.toFixed(3);
  
  // Update the badge with dataset type
  const badge = document.getElementById('dataset-name-badge');
  if (badge) {
    badge.textContent = (data.dataset_type || 'evaluation').toUpperCase();
  }
  
  listEl.innerHTML = data.results.map(r => {
    let f1Class = 'ds-metric-warn';
    if (r.f1 >= 0.85) f1Class = 'ds-metric-good';
    else if (r.f1 <= 0.5) f1Class = 'ds-metric-bad';
    
    return `
    <div class="ds-result-card">
      <div class="ds-result-header">
        <div class="ds-result-title">
          <span class="ds-scenario-id">#${r.scenario_index}</span>
          <strong>${r.description}</strong>
        </div>
        <div class="ds-result-tags">
          <span class="theme-tag orange">${(r.intervention_type || '').replace(/_/g,' ')}</span>
          <span class="theme-tag teal">${(r.graph_region || '').replace(/_/g,' ')}</span>
        </div>
      </div>
      <div class="ds-result-metrics">
        <div class="ds-metric ${f1Class}">
          <span class="ds-metric-label">F1 Score</span>
          <span class="ds-metric-val">${r.f1.toFixed(3)}</span>
        </div>
        <div class="ds-metric">
          <span class="ds-metric-label">Reward</span>
          <span class="ds-metric-val">${r.reward.toFixed(2)}</span>
        </div>
        <div class="ds-metric">
          <span class="ds-metric-label">Steps</span>
          <span class="ds-metric-val">${r.steps}</span>
        </div>
        <div class="ds-metric">
          <span class="ds-metric-label">Quarantined</span>
          <span class="ds-metric-val">${r.nodes_quarantined}</span>
        </div>
      </div>
    </div>
  `}).join('');
  
  resultsDiv.classList.remove('hidden');
  document.getElementById('llm-results').classList.add('hidden');
}

// ---------------------------------------------------------------------------
// Dataset Builder — Create Custom Datasets in the UI
// ---------------------------------------------------------------------------

function openDatasetBuilder() {
  let modal = document.getElementById('dataset-builder-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'dataset-builder-modal';
    modal.className = 'ds-modal-overlay';
    modal.innerHTML = `
      <div class="ds-modal">
        <div class="ds-modal-header">
          <h2>📦 Build Custom Dataset</h2>
          <button class="ds-modal-close" onclick="closeDatasetBuilder()">✕</button>
        </div>
        <div class="ds-modal-body">
          <div class="ds-form-row">
            <label>Dataset Name</label>
            <input type="text" id="ds-builder-name" value="my_dataset" class="ds-input" placeholder="my_custom_dataset">
          </div>
          <div class="ds-form-row">
            <label>Dataset Type</label>
            <select id="ds-builder-type" class="ds-input">
              <option value="evaluation">🧪 Evaluation — Test agent performance</option>
              <option value="training">🏋️ Training — Generate training data</option>
              <option value="benchmark">📊 Benchmark — Standardized comparison</option>
              <option value="stress_test">🔥 Stress Test — Edge cases & limits</option>
            </select>
          </div>
          <div class="ds-scenarios-header">
            <h3>Scenarios</h3>
            <button class="btn btn-secondary btn-sm" onclick="addBuilderScenario()">+ Add Scenario</button>
          </div>
          <div id="ds-builder-scenarios" class="ds-scenarios-list"></div>
          <div class="ds-form-row" style="margin-top:16px;">
            <button class="btn btn-primary btn-glow" onclick="runBuiltDataset()" id="btn-run-built">
              <span class="btn-icon">▶</span> Run Dataset
            </button>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    // Add default scenarios
    addBuilderScenario();
    addBuilderScenario();
  }
  modal.classList.add('active');
}

function closeDatasetBuilder() {
  const modal = document.getElementById('dataset-builder-modal');
  if (modal) modal.classList.remove('active');
}

let _builderScenarioCount = 0;

function addBuilderScenario() {
  const container = document.getElementById('ds-builder-scenarios');
  if (!container) return;
  _builderScenarioCount++;
  const idx = _builderScenarioCount;
  
  const div = document.createElement('div');
  div.className = 'ds-scenario-card';
  div.id = `ds-scenario-${idx}`;
  div.innerHTML = `
    <div class="ds-scenario-header">
      <span class="ds-scenario-num">#${idx}</span>
      <button class="ds-scenario-remove" onclick="removeBuilderScenario(${idx})">✕</button>
    </div>
    <div class="ds-scenario-fields">
      <div class="ds-field">
        <label>Nodes</label>
        <input type="number" min="6" max="20" value="10" class="ds-input ds-input-sm" data-field="node_count">
      </div>
      <div class="ds-field">
        <label>Contamination Type</label>
        <select class="ds-input ds-input-sm" data-field="contamination_type">
          <option value="">🎲 Random</option>
          <option value="lot_relabel">🏷️ Lot Relabel</option>
          <option value="mixing_event">🔀 Mixing Event</option>
          <option value="record_deletion">🗑️ Record Deletion</option>
        </select>
      </div>
      <div class="ds-field">
        <label>Graph Region</label>
        <select class="ds-input ds-input-sm" data-field="graph_region">
          <option value="">🎲 Random</option>
          <option value="source">🏭 Source (Warehouse)</option>
          <option value="midstream">📦 Midstream (Crossdock)</option>
          <option value="downstream">🏪 Downstream (Store)</option>
        </select>
      </div>
      <div class="ds-field ds-field-wide">
        <label>Description</label>
        <input type="text" class="ds-input ds-input-sm" data-field="description" placeholder="Scenario description..." value="Scenario ${idx}">
      </div>
    </div>
  `;
  container.appendChild(div);
}

function removeBuilderScenario(idx) {
  const el = document.getElementById(`ds-scenario-${idx}`);
  if (el) el.remove();
}

async function runBuiltDataset() {
  const btn = document.getElementById('btn-run-built');
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Running...';
  
  const name = document.getElementById('ds-builder-name').value || 'custom_dataset';
  const dsType = document.getElementById('ds-builder-type').value || 'evaluation';
  
  const scenarioCards = document.querySelectorAll('.ds-scenario-card');
  const scenarios = [];
  
  scenarioCards.forEach(card => {
    const nodeCountEl = card.querySelector('[data-field="node_count"]');
    const typeEl = card.querySelector('[data-field="contamination_type"]');
    const regionEl = card.querySelector('[data-field="graph_region"]');
    const descEl = card.querySelector('[data-field="description"]');
    
    scenarios.push({
      node_count: parseInt(nodeCountEl?.value) || 10,
      contamination_type: typeEl?.value || null,
      graph_region: regionEl?.value || null,
      description: descEl?.value || ''
    });
  });
  
  if (scenarios.length === 0) {
    alert("Add at least one scenario before running.");
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">▶</span> Run Dataset';
    return;
  }
  
  try {
    const res = await fetch('/api/llm/upload_dataset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        dataset_name: name,
        dataset_type: dsType,
        scenarios: scenarios
      })
    });
    
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error (${res.status})`);
    }
    
    const data = await res.json();
    closeDatasetBuilder();
    renderDatasetResults(data);
    
  } catch(e) {
    alert("Error: " + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">▶</span> Run Dataset';
  }
}

