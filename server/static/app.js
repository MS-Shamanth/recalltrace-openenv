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
    const action = JSON.stringify(entry.action);
    return `
      <div class="log-step">
        <strong>Step ${entry.step}</strong>
        <div>Action: ${action}</div>
        <div>Reward: ${entry.reward.toFixed(4)}</div>
        <div>Message: ${entry.message || "-"}</div>
      </div>
    `;
  }).join("");

  const finalInfo = JSON.stringify(data.final_info, null, 2);
  episodeLog.innerHTML = `
    <div class="log-step">
      <strong>Task:</strong> ${data.task.name}
    </div>
    ${logMarkup}
    <div class="log-step">
      <strong>Final score:</strong> ${data.score.toFixed(4)}
      <div><strong>Final grading payload:</strong></div>
      <div>${finalInfo}</div>
    </div>
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
