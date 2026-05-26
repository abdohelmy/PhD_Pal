const form = document.querySelector("#profile-form");
const results = document.querySelector("#results");
const message = document.querySelector("#message");
const refreshButton = document.querySelector("#refresh");
const courseCount = document.querySelector("#course-count");
const matchCount = document.querySelector("#match-count");
const lastScraped = document.querySelector("#last-scraped");
const recommenderMode = document.querySelector("#recommender-mode");
const reasoningMode = document.querySelector("#reasoning-mode");
const universityList = document.querySelector("#university-list");
const areaSelect = document.querySelector("#area");
const loadClustersButton = document.querySelector("#load-clusters");
const clusters = document.querySelector("#clusters");
const insights = document.querySelector("#insights");
const includeOtherSchools = document.querySelector("#include-other-schools");
const otherSchoolsField = document.querySelector("#other-schools-field");
const phdSchoolList = document.querySelector("#phd-school-list");

function setMessage(text, tone = "neutral") {
  message.textContent = text;
  message.dataset.tone = tone;
}

function formatDate(value) {
  if (!value) return "Not scraped yet";
  return new Intl.DateTimeFormat("en-DK", {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(value));
}

function profileFromForm() {
  const data = new FormData(form);
  return {
    email: data.get("email").trim(),
    profile: {
      preferredUniversities: data.getAll("preferredUniversities"),
      school: data.get("school").trim(),
      includeOtherSchools: data.get("includeOtherSchools") === "on",
      allowedOtherSchools: data.getAll("allowedOtherSchools"),
      studyProgram: data.get("studyProgram").trim(),
      area: data.get("area"),
      researchDirection: data.get("researchDirection").trim(),
      interests: data.get("interests").trim(),
      topic: data.get("topic").trim(),
      keywords: data.get("keywords").trim(),
      location: data.get("location").trim(),
      minEcts: data.get("minEcts"),
      includeGeneric: data.get("includeGeneric") === "on"
    }
  };
}

function escapeHtml(value = "") {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderChecklist(container, name, values) {
  container.innerHTML = values
    .map(
      (value) => `
        <label class="check-option">
          <input type="checkbox" name="${name}" value="${escapeHtml(value)}" />
          <span>${escapeHtml(value)}</span>
        </label>
      `
    )
    .join("");
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

function updateCacheStats(cache) {
  courseCount.textContent = cache?.count || 0;
  lastScraped.textContent = formatDate(cache?.scrapedAt);
}

function updateRecommender(recommender) {
  recommenderMode.textContent =
    recommender?.mode === "semantic-llm-rerank"
      ? "LLM rerank"
      : recommender?.mode === "semantic"
      ? "Semantic"
      : recommender?.mode === "lexical-fallback"
        ? "Fallback"
        : "Lexical";
  recommenderMode.title =
    [recommender?.model, recommender?.rerankerModel && `reranker: ${recommender.rerankerModel}`]
      .filter(Boolean)
      .join(" · ") || "Set HF_TOKEN or EMBEDDING_ENDPOINT to enable semantic embeddings";
}

function updateReasoning(reasoning) {
  reasoningMode.textContent = reasoning?.enabled || reasoning?.mode ? "On" : "Off";
  reasoningMode.title = reasoning?.model || "Set HF_TOKEN or LLM_CHAT_ENDPOINT to enable reasoning";
}

function renderInsights(reasoning) {
  const data = reasoning?.insights;
  if (!data) {
    insights.innerHTML = "";
    return;
  }
  const groups = [
    ["Inferred subjects", data.inferredSubjects],
    ["Adjacent subjects", data.adjacentSubjects],
    ["Methods", data.methods],
    ["Search phrases", data.searchKeywords]
  ].filter(([, values]) => values?.length);

  insights.innerHTML = `
    <article>
      <div>
        <p class="eyebrow">Reasoning expansion</p>
        <strong>${reasoning.mode === "cached" ? "Cached LLM analysis" : "Fresh LLM analysis"}</strong>
      </div>
      ${data.rationale ? `<p>${data.rationale}</p>` : ""}
      ${groups
        .map(
          ([label, values]) => `
            <section>
              <span>${label}</span>
              <div>${values.map((value) => `<em>${value}</em>`).join("")}</div>
            </section>
          `
        )
        .join("")}
    </article>
  `;
}

function renderRecommendations(recommendations) {
  matchCount.textContent = recommendations.length;
  results.innerHTML = recommendations
    .map(
      (course, index) => `
        <article class="course-card">
          <div class="rank">${index + 1}</div>
          <div class="course-main">
            <div class="course-head">
              <div>
                <a href="${course.sourceUrl}" target="_blank" rel="noreferrer">${course.title}</a>
                <p>${course.phdSchool}</p>
              </div>
              <strong>${course.score}</strong>
            </div>
            <p class="description">${course.description}</p>
            <div class="meta">
              <span>${course.startDate || "Date TBA"}</span>
              <span>${course.ects || "ECTS TBA"}</span>
              <span>${course.city || "Location TBA"}</span>
            </div>
            <div class="reasons">
              ${course.reasons.map((reason) => `<span>${reason}</span>`).join("")}
            </div>
          </div>
        </article>
      `
    )
    .join("");
}

async function loadMetadata() {
  const metadata = await requestJson("/api/metadata");
  renderChecklist(universityList, "preferredUniversities", metadata.universities);
  renderChecklist(phdSchoolList, "allowedOtherSchools", metadata.phdSchools || []);
  areaSelect.innerHTML = metadata.fields.map((name) => `<option value="${name}">${name}</option>`).join("");
  updateCacheStats(metadata.cache);
  updateRecommender(metadata.recommender);
  updateReasoning(metadata.reasoning);
}

includeOtherSchools.addEventListener("change", () => {
  otherSchoolsField.hidden = !includeOtherSchools.checked;
});

function renderClusters(payload) {
  if (!payload.clusters.length) {
    clusters.innerHTML = '<p class="empty">No subscribers saved yet.</p>';
    return;
  }

  clusters.innerHTML = payload.clusters
    .map(
      (cluster) => `
        <article class="cluster-card">
          <div>
            <strong>${cluster.id}</strong>
            <span>${cluster.size} user${cluster.size === 1 ? "" : "s"} · ${payload.mode}</span>
          </div>
          <p>${cluster.insight?.label || cluster.themes.slice(0, 6).join(", ") || "No dominant themes yet"}</p>
          ${
            cluster.insight?.searchKeywords?.length
              ? `<p>Search: ${cluster.insight.searchKeywords.slice(0, 8).join(", ")}</p>`
              : ""
          }
          ${
            cluster.insight?.digestStrategy
              ? `<p>Digest: ${cluster.insight.digestStrategy}</p>`
              : ""
          }
        </article>
      `
    )
    .join("");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const { email, profile } = profileFromForm();
  setMessage("Matching your profile against the course cache...");
  results.innerHTML = "";

  try {
    const payload = await requestJson("/api/recommend", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(profile)
    });
    updateCacheStats(payload.cache);
    updateRecommender(payload.recommender);
    updateReasoning(payload.reasoning);
    renderInsights(payload.reasoning);
    renderRecommendations(payload.recommendations);

    if (email) {
      await requestJson("/api/subscribe", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ email, profile })
      });
      setMessage("Recommendations ready. Your weekly digest profile has been saved.", "success");
    } else {
      setMessage("Recommendations ready. Add an email to save this as a weekly digest.", "success");
    }
  } catch (error) {
    setMessage(error.message, "error");
  }
});

refreshButton.addEventListener("click", async () => {
  refreshButton.disabled = true;
  setMessage("Refreshing course data from phdcourses.dk. This may take a little while...");
  try {
    const cache = await requestJson("/api/courses?refresh=true");
    updateCacheStats(cache);
    setMessage(`Refreshed ${cache.count} courses from ${cache.totalPages} result pages.`, "success");
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    refreshButton.disabled = false;
  }
});

loadClustersButton.addEventListener("click", async () => {
  loadClustersButton.disabled = true;
  clusters.innerHTML = '<p class="empty">Clustering saved subscriber profiles...</p>';
  try {
    renderClusters(await requestJson("/api/clusters"));
  } catch (error) {
    clusters.innerHTML = `<p class="empty">${error.message}</p>`;
  } finally {
    loadClustersButton.disabled = false;
  }
});

loadMetadata().catch((error) => setMessage(error.message, "error"));
