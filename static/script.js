// ---------------------------------------------------------------------
// CONFIG
// If your FastAPI server runs somewhere other than the same origin as
// this page (e.g. you open index.html directly as a file, or host the
// UI separately from the API), set the full API URL here.
// Example: "http://127.0.0.1:8000"
// Leave it as an empty string if this page is served BY the FastAPI
// app itself (e.g. via StaticFiles) — then "/predict" already works.
// ---------------------------------------------------------------------
const API_BASE_URL = "https://demo-dmz5.onrender.com";

// Emoji + color per emotion, purely cosmetic.
const EMOTION_META = {
  sadness: { emoji: "😢", color: "#5b8def" },
  joy: { emoji: "😄", color: "#f4c542" },
  love: { emoji: "❤️", color: "#ef5c8e" },
  anger: { emoji: "😠", color: "#e0553d" },
  fear: { emoji: "😨", color: "#9a5bd6" },
  surprise: { emoji: "😲", color: "#3ad0a3" },
};

// Grab all the elements we'll need to update.
const textInput = document.getElementById("textInput");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");
const loadingEl = document.getElementById("loading");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const barsEl = document.getElementById("bars");
const topEmoji = document.getElementById("topEmoji");
const topLabel = document.getElementById("topLabel");
const topConfidence = document.getElementById("topConfidence");

// Update the character counter as the user types.
textInput.addEventListener("input", () => {
  charCount.textContent = textInput.value.length;
});

// Clicking an example chip fills the textarea and analyzes it right away.
document.querySelectorAll(".example-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    textInput.value = chip.textContent;
    charCount.textContent = textInput.value.length;
    analyze();
  });
});

// Clicking the button, or pressing Ctrl/Cmd+Enter, triggers analysis.
analyzeBtn.addEventListener("click", analyze);
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyze();
});

async function analyze() {
  const text = textInput.value.trim();

  errorEl.style.display = "none";
  if (!text) {
    errorEl.textContent = "Please type something first.";
    errorEl.style.display = "block";
    return;
  }

  resultEl.style.display = "none";
  loadingEl.style.display = "block";
  analyzeBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed (status ${response.status})`);
    }

    const data = await response.json();
    renderResult(data);
  } catch (err) {
    errorEl.textContent = "Something went wrong: " + err.message;
    errorEl.style.display = "block";
  } finally {
    loadingEl.style.display = "none";
    analyzeBtn.disabled = false;
  }
}

function renderResult(data) {
  const meta = EMOTION_META[data.predicted_emotion] || { emoji: "🎭", color: "#7c8cff" };

  topEmoji.textContent = meta.emoji;
  topLabel.textContent = data.predicted_emotion;
  topConfidence.textContent = `${(data.confidence * 100).toFixed(1)}% confident`;

  // Sort emotions from highest to lowest probability.
  const sorted = Object.entries(data.all_probabilities).sort((a, b) => b[1] - a[1]);

  barsEl.innerHTML = "";
  for (const [label, prob] of sorted) {
    const m = EMOTION_META[label] || { color: "#7c8cff" };
    const pct = (prob * 100).toFixed(1);

    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-label">${label}</div>
      <div class="bar-track"><div class="bar-fill" style="background:${m.color}"></div></div>
      <div class="bar-value">${pct}%</div>
    `;
    barsEl.appendChild(row);

    // Animate the bar width in on the next frame.
    requestAnimationFrame(() => {
      row.querySelector(".bar-fill").style.width = pct + "%";
    });
  }

  resultEl.style.display = "block";
}
