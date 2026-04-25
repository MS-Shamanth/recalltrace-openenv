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
  try {
    const res = await fetch('/api/selfplay/demo');
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
// Init
// ---------------------------------------------------------------------------
fetchTasks();
loadGraph();
