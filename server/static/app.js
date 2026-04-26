const taskSelect = document.getElementById("task-select");
const taskSummary = document.getElementById("task-summary");
const currentScore = document.getElementById("current-score");
const currentSteps = document.getElementById("current-steps");
const currentStatus = document.getElementById("current-status");
const allScore = document.getElementById("all-score");
const allResults = document.getElementById("all-results");
const episodeLog = document.getElementById("episode-log");
const rewardChart = document.getElementById("reward-chart");
const finalSummary = document.getElementById("final-summary");

let taskCatalog = [];

function renderTaskSummary(task) {
  taskSummary.innerHTML = `
    <h3>${task.name}</h3>
    <p><strong>Difficulty:</strong> ${task.difficulty}</p>
    <p>${task.objective}</p>
    <p><strong>Max steps:</strong> ${task.max_steps}</p>
  `;
}

function buildLineChart(logs) {
  if (!logs.length) {
    rewardChart.innerHTML = "No rewards available.";
    return;
  }

  const width = 380;
  const height = 220;
  const padding = 28;
  const values = logs.map((entry) => entry.reward);
  const maxReward = Math.max(...values, 1);
  const minReward = Math.min(...values, 0);
  const range = Math.max(maxReward - minReward, 0.25);

  const toX = (index) => {
    if (logs.length === 1) {
      return width / 2;
    }
    return padding + (index * (width - padding * 2)) / (logs.length - 1);
  };

  const toY = (value) => {
    return height - padding - ((value - minReward) / range) * (height - padding * 2);
  };

  const linePoints = logs
    .map((entry, index) => `${toX(index)},${toY(entry.reward)}`)
    .join(" ");

  const horizontalGuides = [0, 0.25, 0.5, 0.75, 1]
    .map((ratio) => {
      const y = padding + ratio * (height - padding * 2);
      return `<line class="chart-grid" x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}"></line>`;
    })
    .join("");

  const labels = logs
    .map((entry, index) => {
      const x = toX(index);
      return `<text class="chart-label" x="${x}" y="${height - 8}" text-anchor="middle">S${entry.step}</text>`;
    })
    .join("");

  const points = logs
    .map((entry, index) => {
      const x = toX(index);
      const y = toY(entry.reward);
      return `
        <circle class="chart-point" cx="${x}" cy="${y}" r="5"></circle>
        <text class="chart-label" x="${x}" y="${y - 10}" text-anchor="middle">${entry.reward.toFixed(2)}</text>
      `;
    })
    .join("");

  rewardChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" aria-label="Reward line chart">
      ${horizontalGuides}
      <line class="chart-axis" x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}"></line>
      <line class="chart-axis" x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}"></line>
      <polyline class="chart-line" points="${linePoints}"></polyline>
      ${points}
      ${labels}
    </svg>
  `;
}

function renderEpisode(data) {
  currentScore.textContent = data.score.toFixed(4);
  currentSteps.textContent = String(data.steps_taken);
  currentStatus.textContent = data.success ? "Contained" : "Needs work";

  buildLineChart(data.logs);

  finalSummary.innerHTML = `
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
          <div>${detailBits.length ? detailBits.join(" | ") : "No extra parameters"}</div>
          <div>Reward: ${entry.reward.toFixed(4)}</div>
          <div>Message: ${entry.message || "-"}</div>
        </div>
      </div>
    `;
  }).join("");

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
      <div>Status: ${episode.success ? "Success" : "Needs work"}</div>
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
  rewardChart.innerHTML = "Task reset. Run a task to render the reward trajectory.";
  finalSummary.innerHTML = "Readable scoring highlights will appear here.";
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
