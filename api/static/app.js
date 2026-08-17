"use strict";

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const SUPPORTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const DIAGNOSTIC_KEYS = [
  "blur",
  "glare",
  "darkness",
  "overexposure",
  "motion_artifacts",
  "occlusion",
  "poor_framing",
  "low_resolution",
];

const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const selectedFile = document.querySelector("#selectedFile");
const imagePreview = document.querySelector("#imagePreview");
const fileName = document.querySelector("#fileName");
const fileMeta = document.querySelector("#fileMeta");
const resetButton = document.querySelector("#resetButton");
const analyzeButton = document.querySelector("#analyzeButton");
const buttonLabel = analyzeButton.querySelector(".button-label");
const errorMessage = document.querySelector("#errorMessage");
const modelStatus = document.querySelector("#modelStatus");
const modelStatusText = modelStatus.querySelector(".status-text");
const resultsEmpty = document.querySelector("#resultsEmpty");
const resultsContent = document.querySelector("#resultsContent");
const resultState = document.querySelector("#resultState");
const diagnosticsGrid = document.querySelector("#diagnosticsGrid");
const diagnosticCards = new Map(
  [...diagnosticsGrid.querySelectorAll("[data-diagnostic-key]")].map((card) => [
    card.dataset.diagnosticKey,
    card,
  ]),
);

let currentFile = null;
let previewUrl = null;
let modelReady = false;

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}

function updateAnalyzeAvailability() {
  analyzeButton.disabled = !currentFile || !modelReady;
}

function setSelectedFile(file) {
  clearError();

  if (!SUPPORTED_TYPES.has(file.type)) {
    showError("Please choose a JPEG, PNG, or WebP image.");
    return;
  }
  if (file.size === 0) {
    showError("The selected file is empty.");
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    showError("The selected image is larger than the 10 MB upload limit.");
    return;
  }

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  currentFile = file;
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  imagePreview.alt = `Preview of ${file.name}`;
  fileName.textContent = file.name;
  fileMeta.textContent = `${formatBytes(file.size)} - ${file.type.replace("image/", "").toUpperCase()}`;

  dropZone.hidden = true;
  selectedFile.hidden = false;
  resetButton.hidden = false;
  resultsContent.hidden = true;
  resultsEmpty.hidden = false;
  resultState.textContent = "Waiting";
  resultState.classList.remove("complete");
  updateAnalyzeAvailability();
}

function resetSelection() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = null;
  currentFile = null;
  fileInput.value = "";
  imagePreview.removeAttribute("src");
  dropZone.hidden = false;
  selectedFile.hidden = true;
  resetButton.hidden = true;
  resultsContent.hidden = true;
  resultsEmpty.hidden = false;
  resultState.textContent = "Waiting";
  resultState.classList.remove("complete");
  clearError();
  updateAnalyzeAvailability();
}

function setLoading(isLoading) {
  analyzeButton.classList.toggle("loading", isLoading);
  analyzeButton.disabled = isLoading || !currentFile || !modelReady;
  buttonLabel.textContent = isLoading ? "Analyzing image" : "Analyze image quality";
}

function qualityBand(score) {
  if (score >= 0.8) return ["Excellent quality", "The model places this image in its strongest quality band."];
  if (score >= 0.7) return ["Good quality", "The model predicts a clear, generally usable image."];
  if (score >= 0.6) return ["Moderate quality", "The image passes the current model threshold, but may still need review."];
  if (score >= 0.4) return ["Low quality", "The model score is below the current suitability threshold."];
  return ["Very low quality", "The model detects substantial overall quality degradation."];
}

function setCheck(iconElement, passed) {
  iconElement.classList.toggle("fail", !passed);
}

function formatMeasured(value) {
  if (value === null || value === undefined || value === "") {
    return "Not available";
  }
  if (typeof value === "number") {
    return Number.isFinite(value)
      ? value.toLocaleString(undefined, { maximumSignificantDigits: 6, useGrouping: false })
      : "Not available";
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  if (Array.isArray(value)) {
    return value.map(formatMeasured).join(", ");
  }
  if (typeof value === "object") {
    return Object.entries(value)
      .map(([key, item]) => {
        const label = key.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
        return `${label}: ${formatMeasured(item)}`;
      })
      .join(" · ");
  }
  return String(value);
}

function parseDiagnostics(diagnostics) {
  if (!diagnostics || typeof diagnostics !== "object") {
    throw new Error("The API response did not include image diagnostics.");
  }

  return DIAGNOSTIC_KEYS.map((key) => {
    const diagnostic = diagnostics[key];
    if (!diagnostic || typeof diagnostic !== "object") {
      throw new Error(`The API response is missing the ${key.replaceAll("_", " ")} diagnostic.`);
    }

    const score = Number(diagnostic.score);
    if (!Number.isFinite(score)) {
      throw new Error(`The ${key.replaceAll("_", " ")} diagnostic has an invalid score.`);
    }

    const thresholds = diagnostic.thresholds;
    if (
      !thresholds
      || typeof thresholds !== "object"
      || Array.isArray(thresholds)
      || Object.keys(thresholds).length === 0
    ) {
      throw new Error(`The ${key.replaceAll("_", " ")} diagnostic has no decision limits.`);
    }

    return {
      key,
      score: Math.max(0, Math.min(1, score)),
      passes: diagnostic.passes === true,
      status: String(diagnostic.status || (diagnostic.passes ? "Within range" : "Needs review")),
      explanation: String(diagnostic.explanation || "No explanation was provided."),
      measured: formatMeasured(diagnostic.measured),
      thresholds: formatMeasured(thresholds),
    };
  });
}

function renderDiagnostics(diagnostics) {
  for (const diagnostic of diagnostics) {
    const card = diagnosticCards.get(diagnostic.key);
    if (!card) continue;

    const severityIndex = diagnostic.score * 100;
    const displayedIndex = severityIndex.toFixed(1);
    const withinBaseline = diagnostic.score === 0;
    const verdict = withinBaseline ? "Within baseline" : diagnostic.passes ? "Pass" : "Review";

    card.classList.toggle("attention", !diagnostic.passes);
    card.querySelector('[data-field="verdict"]').textContent = verdict;
    card.querySelector('[data-field="score"]').textContent = `${displayedIndex} / 100`;
    card.querySelector('[data-field="status"]').textContent = diagnostic.status;
    card.querySelector('[data-field="explanation"]').textContent = diagnostic.explanation;
    card.querySelector('[data-field="measured"]').textContent = diagnostic.measured;
    card.querySelector('[data-field="thresholds"]').textContent = diagnostic.thresholds;

    const track = card.querySelector('[data-field="track"]');
    track.style.setProperty("--severity-size", `${severityIndex}%`);
    track.setAttribute("aria-valuenow", displayedIndex);
    track.setAttribute(
      "aria-valuetext",
      `Severity index ${displayedIndex} out of 100, ${withinBaseline ? "within baseline" : diagnostic.passes ? "check passed" : "review recommended"}`,
    );
  }
}

function renderResult(data) {
  const diagnostics = parseDiagnostics(data.diagnostics);
  const score = Math.max(0, Math.min(1, Number(data.quality_score)));
  const scorePercent = Math.round(score * 100);
  const [band, description] = qualityBand(score);

  const scoreRing = document.querySelector("#scoreRing");
  scoreRing.style.setProperty("--score-angle", `${score * 360}deg`);
  scoreRing.setAttribute("aria-label", `Overall quality score ${scorePercent} out of 100`);
  document.querySelector("#scoreValue").textContent = String(scorePercent);
  document.querySelector("#qualityBand").textContent = band;
  document.querySelector("#scoreDescription").textContent = description;

  const decisionChip = document.querySelector("#decisionChip");
  decisionChip.textContent = data.suitable
    ? "Suitable for computer vision"
    : "Not suitable at current thresholds";
  decisionChip.className = `decision-chip ${data.suitable ? "pass" : "fail"}`;

  setCheck(document.querySelector("#modelCheckIcon"), data.model_check.passes);
  document.querySelector("#modelCheckValue").textContent = data.model_check.passes ? "Passed" : "Below threshold";
  document.querySelector("#modelCheckDetail").textContent =
    `Score ${score.toFixed(3)} / threshold ${Number(data.model_check.threshold).toFixed(2)}`;

  setCheck(document.querySelector("#resolutionCheckIcon"), data.resolution_check.passes);
  document.querySelector("#resolutionCheckValue").textContent = data.resolution_check.passes ? "Passed" : "Too small";
  document.querySelector("#resolutionCheckDetail").textContent =
    `Minimum ${data.resolution_check.minimum_width} x ${data.resolution_check.minimum_height} px`;

  document.querySelector("#imageDimensions").textContent = `${data.image.width} x ${data.image.height} px`;
  document.querySelector("#imageFormat").textContent = data.image.format || "Unknown";
  document.querySelector("#mosValue").textContent = Number(data.mos_equivalent).toFixed(2);
  renderDiagnostics(diagnostics);

  resultsEmpty.hidden = true;
  resultsContent.hidden = false;
  resultState.textContent = "Complete";
  resultState.classList.add("complete");
}

async function analyzeImage() {
  if (!currentFile || !modelReady) return;

  clearError();
  setLoading(true);
  const formData = new FormData();
  formData.append("file", currentFile, currentFile.name);

  try {
    const response = await fetch("/analyze-quality", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `Analysis failed with status ${response.status}.`);
    }
    renderResult(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "The image could not be analyzed.";
    showError(message);
  } finally {
    setLoading(false);
  }
}

async function checkModel() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    if (!response.ok) throw new Error("Health check failed");
    const health = await response.json();
    modelReady = Boolean(health.model_available);
    modelStatus.classList.toggle("ready", modelReady);
    modelStatus.classList.toggle("unavailable", !modelReady);
    modelStatusText.textContent = modelReady ? "Model ready" : "Model unavailable";
    if (!modelReady) {
      showError("The model checkpoint is missing. Train the model before analyzing images.");
    }
  } catch {
    modelReady = false;
    modelStatus.classList.add("unavailable");
    modelStatusText.textContent = "Server unavailable";
    showError("The API health check failed. Make sure the FastAPI server is running.");
  } finally {
    updateAnalyzeAvailability();
  }
}

fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  if (file) setSelectedFile(file);
});

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
}

dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) setSelectedFile(file);
});

resetButton.addEventListener("click", resetSelection);
analyzeButton.addEventListener("click", analyzeImage);
window.addEventListener("beforeunload", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});

checkModel();
