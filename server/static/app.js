const taskSelect = document.getElementById("task-select");
const taskSummary = document.getElementById("task-summary");
const currentScore = document.getElementById("current-score");
const currentSteps = document.getElementById("current-steps");
const currentStatus = document.getElementById("current-status");
const allScore = document.getElementById("all-score");
const allResults = document.getElementById("all-results");
const episodeLog = document.getElementById("episode-log");

let taskCatalog = [];

function renderTaskSummary(task) {
  taskSummary.innerHTML = `
    <h3>${task.name}</h3>
    <p><strong>Difficulty:</strong> ${task.difficulty}</p>
    <p>${task.objective}</p>
    <p><strong>Max steps:</strong> ${task.max_steps}</p>
  `;
}

function renderEpisode(data) {
  currentScore.textContent = data.score.toFixed(4);
  currentSteps.textContent = String(data.steps_taken);
  currentStatus.textContent = data.success ? "Contained" : "Needs work";

  const logMarkup = data.logs.map((entry) => {
    const actionType = entry.action.type || "action";
    const detailBits = [];
    if (entry.action.node_id) detailBits.push(`Node: ${entry.action.node_id}`);
    if (entry.action.lot_id) detailBits.push(`Lot: ${entry.action.lot_id}`);
    if (entry.action.quantity) detailBits.push(`Qty: ${entry.action.quantity}`);

    return `
      <div class="log-step">
        <div class="log-title">
          <strong>Step ${entry.step}</strong>
          <span class="action-chip">${actionType.replace("_", " ")}</span>
        </div>
        <div class="action-meta">
          <div>${detailBits.length ? detailBits.join(" • ") : "No extra parameters"}</div>
          <div>Reward: ${entry.reward.toFixed(4)}</div>
        </div>
        <div>Message: ${entry.message || "-"}</div>
      </div>
    `;
  }).join("");

  const rewardChartMarkup = data.logs.map((entry) => {
    const normalized = Math.min(Math.abs(entry.reward), 1);
    const width = Math.max(4, normalized * 50);
    const polarity = entry.reward >= 0 ? "positive" : "negative";
    return `
      <div class="reward-row">
        <span>S${entry.step}</span>
        <div class="reward-track">
          <div class="reward-bar ${polarity}" style="width:${width}%"></div>
        </div>
        <strong>${entry.reward.toFixed(4)}</strong>
      </div>
    `;
  }).join("");

  document.getElementById("reward-chart").innerHTML = rewardChartMarkup || "No step rewards yet.";

  const finalSummary = `
    <div class="summary-grid">
      <div class="summary-pill">
        <span>Final score</span>
        <strong>${data.score.toFixed(4)}</strong>
      </div>
      <div class="summary-pill">
        <span>Status</span>
        <strong>${data.success ? "Success" : "Needs improvement"}</strong>
      </div>
      <div class="summary-pill">
        <span>Steps used</span>
        <strong>${data.steps_taken}</strong>
      </div>
      <div class="summary-pill">
        <span>Quarantine quality</span>
        <strong>${(data.final_info.quarantine_score ?? 0).toFixed(4)}</strong>
      </div>
    </div>
    <div class="summary-card">
      <strong>Containment outcome</strong>
      <div>All affected nodes notified: ${data.final_info.all_affected_nodes_notified ? "Yes" : "No"}</div>
      <div>All affected stock quarantined: ${data.final_info.all_affected_stock_quarantined ? "Yes" : "No"}</div>
    </div>
    <div class="summary-card">
      <strong>Grader focus</strong>
      <div>Notification score: ${(data.final_info.notification_score ?? 0).toFixed(4)}</div>
      <div>Investigation score: ${(data.final_info.investigation_score ?? 0).toFixed(4)}</div>
      <div>Efficiency score: ${(data.final_info.efficiency_score ?? 0).toFixed(4)}</div>
    </div>
  `;
  document.getElementById("final-summary").innerHTML = finalSummary;

  episodeLog.innerHTML = `
    <div class="log-step">
      <strong>Task:</strong> ${data.task.name}
    </div>
    ${logMarkup}
  `;
}

function renderRunAll(data) {
  allScore.textContent = data.average_score.toFixed(4);
  allResults.innerHTML = data.episodes.map((episode) => `
    <div class="log-step">
      <strong>${episode.task.name}</strong>
      <div>Difficulty: ${episode.task.difficulty}</div>
      <div>Score: ${episode.score.toFixed(4)}</div>
      <div>Steps: ${episode.steps_taken}</div>
      <div>Status: ${episode.success ? "success" : "needs work"}</div>
    </div>
  `).join("");
}

async function fetchTasks() {
  const response = await fetch("/api/tasks");
  const data = await response.json();
  taskCatalog = data.tasks;

  taskSelect.innerHTML = taskCatalog.map((task) => `
    <option value="${task.task_id}">${task.difficulty.toUpperCase()} - ${task.name}</option>
  `).join("");

  renderTaskSummary(taskCatalog[0]);
}

async function resetTask() {
  const taskId = taskSelect.value;
  const response = await fetch(`/reset?task_id=${encodeURIComponent(taskId)}`);
  const data = await response.json();
  currentScore.textContent = "-";
  currentSteps.textContent = String(data.steps_taken || 0);
  currentStatus.textContent = "Reset";
  episodeLog.textContent = JSON.stringify(data, null, 2);
}

async function runEpisode() {
  const response = await fetch("/api/run_episode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task_id: taskSelect.value }),
  });
  const data = await response.json();
  renderEpisode(data);
}

async function runAllTasks() {
  const response = await fetch("/api/run_all");
  const data = await response.json();
  renderRunAll(data);
}

taskSelect.addEventListener("change", () => {
  const task = taskCatalog.find((item) => item.task_id === taskSelect.value);
  if (task) {
    renderTaskSummary(task);
  }
});

document.getElementById("reset-button").addEventListener("click", resetTask);
document.getElementById("run-button").addEventListener("click", runEpisode);
document.getElementById("run-all-button").addEventListener("click", runAllTasks);

fetchTasks();
