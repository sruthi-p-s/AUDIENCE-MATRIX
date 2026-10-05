/**
 * AUDIENCE MATRIX - Dashboard Controller
 * Dynamic Telemetry Visualization & Personalization Client
 */

document.addEventListener("DOMContentLoaded", () => {
  fetchOverviewData();
});

let systemData = null;

async function fetchOverviewData() {
  try {
    const res = await fetch("/api/overview");
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: Failed to load overview`);
    }
    const data = await res.json();
    systemData = data;
    renderOverview(data);
    renderSegments(data.segments);
    renderDnaMatrix(data.segments);
    renderModelQuality(data.k_sweep, data.selected_k);
    renderRobustness(data.evaluator_robustness);
  } catch (err) {
    console.error("[AUDIENCE MATRIX] Error loading system overview:", err);
    document.getElementById("system-status-text").innerText = "Degraded / Offline";
    document.getElementById("system-status-text").parentElement.style.borderColor = "rgba(239, 68, 68, 0.4)";
    document.getElementById("system-status-text").parentElement.style.color = "#F87171";
  }
}

function renderOverview(data) {
  document.getElementById("kpi-total-users").innerText = (data.total_users || 0).toLocaleString();
  document.getElementById("kpi-segments-count").innerText = data.selected_k || "--";
  document.getElementById("kpi-silhouette").innerText = (data.best_silhouette_score !== undefined) ? data.best_silhouette_score.toFixed(4) : "--";
  document.getElementById("kpi-selected-k").innerText = `K = ${data.selected_k}`;
}

function renderSegments(segments) {
  const container = document.getElementById("segments-container");
  if (!segments || segments.length === 0) {
    container.innerHTML = `<div style="color: var(--text-muted);">No segments discovered.</div>`;
    return;
  }

  container.innerHTML = segments.map(seg => {
    const traitsHtml = (seg.traits || []).map(t => `<li>${t}</li>`).join("");
    return `
      <div class="segment-card">
        <div class="segment-header-row">
          <div class="segment-name">${seg.name}</div>
          <span class="segment-id-tag">ID: ${seg.segment_id}</span>
        </div>
        <div class="segment-stat-row">
          <div class="segment-stat">Audience: <span>${(seg.user_count || 0).toLocaleString()}</span></div>
          <div class="segment-stat">Share: <span>${seg.percentage}%</span></div>
        </div>
        <div class="segment-bar-container">
          <div class="segment-bar-fill" style="width: ${seg.percentage}%;"></div>
        </div>
        <p class="segment-desc">${seg.description}</p>
        <div>
          <div class="traits-title">Distinguishing Behavioral Traits</div>
          <ul class="traits-list">${traitsHtml}</ul>
        </div>
      </div>
    `;
  }).join("");
}

function renderDnaMatrix(segments) {
  const tbody = document.getElementById("dna-table-body");
  if (!segments || segments.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="color: var(--text-muted);">Awaiting telemetry clustering...</td></tr>`;
    return;
  }

  tbody.innerHTML = segments.map(seg => {
    const c = seg.centroid || {};
    const wt = c.watch_time_hours ? c.watch_time_hours.toFixed(1) : "--";
    const sd = c.avg_session_minutes ? c.avg_session_minutes.toFixed(0) : "--";
    const spw = c.sessions_per_week ? c.sessions_per_week.toFixed(1) : "--";
    const cr = c.completion_rate ? `${(c.completion_rate * 100).toFixed(0)}%` : "--";
    const wr = c.weekend_ratio ? `${(c.weekend_ratio * 100).toFixed(0)}%` : "--";
    const div = c.genre_diversity ? c.genre_diversity.toFixed(1) : "--";

    // Percentages for mini visual bars
    const wtPct = Math.min(100, ((c.watch_time_hours || 0) / 45) * 100);
    const sdPct = Math.min(100, ((c.avg_session_minutes || 0) / 120) * 100);
    const crPct = Math.min(100, (c.completion_rate || 0) * 100);
    const wrPct = Math.min(100, (c.weekend_ratio || 0) * 100);

    return `
      <tr>
        <td style="font-weight: 700; color: #FFFFFF;">
          ${seg.name}
          <div style="font-size: 11px; color: var(--text-muted); font-weight: normal;">Cluster ${seg.segment_id}</div>
        </td>
        <td class="dna-bar-cell">
          ${wt} hrs/wk
          <div class="dna-mini-bar-bg"><div class="dna-mini-bar-fill" style="width: ${wtPct}%;"></div></div>
        </td>
        <td class="dna-bar-cell">
          ${sd} mins
          <div class="dna-mini-bar-bg"><div class="dna-mini-bar-fill" style="width: ${sdPct}%;"></div></div>
        </td>
        <td>${spw} /wk</td>
        <td class="dna-bar-cell">
          ${cr}
          <div class="dna-mini-bar-bg"><div class="dna-mini-bar-fill" style="width: ${crPct}%;"></div></div>
        </td>
        <td class="dna-bar-cell">
          ${wr}
          <div class="dna-mini-bar-bg"><div class="dna-mini-bar-fill" style="width: ${wrPct}%;"></div></div>
        </td>
        <td>${div} / 5</td>
      </tr>
    `;
  }).join("");
}

function renderModelQuality(kSweep, selectedK) {
  const tbody = document.getElementById("k-sweep-body");
  if (!kSweep || kSweep.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" style="color: var(--text-muted);">No hyperparameter evaluation records found.</td></tr>`;
    return;
  }

  tbody.innerHTML = kSweep.map(row => {
    const isSelected = row.k === selectedK;
    const selectedBadge = isSelected ? ` <span style="color: #A78BFA; font-size: 11px;">(Selected)</span>` : "";
    return `
      <tr class="${isSelected ? 'selected-k-row' : ''}">
        <td><strong>K = ${row.k}</strong>${selectedBadge}</td>
        <td style="font-family: var(--font-mono);">${row.inertia.toLocaleString()}</td>
        <td style="font-family: var(--font-mono); font-weight: 600;">${row.silhouette_score.toFixed(4)}</td>
        <td>${(row.balance_ratio * 100).toFixed(0)}% Min/Max</td>
      </tr>
    `;
  }).join("");
}

function renderRobustness(metrics) {
  if (!metrics) {
    document.getElementById("rob-api-tests").innerText = "Pending";
    document.getElementById("rob-edge-tests").innerText = "Pending";
    document.getElementById("rob-api-rate").innerText = "Awaiting Evaluator";
    document.getElementById("rob-edge-rate").innerText = "Awaiting Evaluator";
    document.getElementById("rob-test-list").innerHTML = `
      <div style="color: var(--text-muted); font-size: 12px;">
        Evaluator results not yet registered in metrics.json. Run evaluator service to populate.
      </div>
    `;
    return;
  }

  const apiTests = metrics.api_tests || { passed: 0, total: 0 };
  const edgeTests = metrics.edge_case_tests || { passed: 0, total: 0 };

  document.getElementById("rob-api-tests").innerText = `${apiTests.passed}/${apiTests.total}`;
  document.getElementById("rob-api-rate").innerText = `${apiTests.pass_rate_percent}% Passed`;
  document.getElementById("rob-edge-tests").innerText = `${edgeTests.passed}/${edgeTests.total}`;
  document.getElementById("rob-edge-rate").innerText = `${edgeTests.pass_rate_percent}% Passed`;

  const allDetails = [
    ...(apiTests.details || []).map(t => ({ name: t.name, passed: t.passed, latency: `${t.latency_ms}ms`, type: "Persona" })),
    ...(edgeTests.details || []).map(t => ({ name: t.description, passed: t.passed, latency: `${t.latency_ms}ms`, type: "Edge Case" }))
  ];

  document.getElementById("rob-test-list").innerHTML = allDetails.map(t => `
    <div class="test-row">
      <div>
        <span style="color: var(--text-primary); font-weight: 500;">${t.name}</span>
        <span style="color: var(--text-muted); font-size: 11px; margin-left: 6px;">[${t.type}]</span>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);">${t.latency}</span>
        <span class="test-status-pill ${t.passed ? 'pass' : 'fail'}">${t.passed ? 'PASS' : 'FAIL'}</span>
      </div>
    </div>
  `).join("");
}

function loadPreset(presetKey) {
  const presets = {
    marathon: {
      userId: "viewer_weekend_marathon",
      watchTime: 34.0,
      avgSession: 95,
      sessions: 5,
      completion: 0.90,
      weekend: 0.82,
      diversity: 3,
      genres: ["Action", "Sci-Fi"]
    },
    casual: {
      userId: "viewer_casual_mobile",
      watchTime: 6.5,
      avgSession: 24,
      sessions: 4,
      completion: 0.54,
      weekend: 0.28,
      diversity: 2,
      genres: ["Comedy", "Drama"]
    },
    drama: {
      userId: "viewer_drama_devotee",
      watchTime: 18.5,
      avgSession: 56,
      sessions: 6,
      completion: 0.91,
      weekend: 0.35,
      diversity: 2,
      genres: ["Drama", "Thriller"]
    },
    omnivore: {
      userId: "viewer_eclectic_omnivore",
      watchTime: 22.0,
      avgSession: 48,
      sessions: 9,
      completion: 0.76,
      weekend: 0.44,
      diversity: 5,
      genres: ["Action", "Comedy", "Thriller", "Drama", "Sci-Fi"]
    }
  };

  const p = presets[presetKey];
  if (!p) return;

  document.getElementById("input-user-id").value = p.userId;
  document.getElementById("input-watch-time").value = p.watchTime;
  document.getElementById("input-avg-session").value = p.avgSession;
  document.getElementById("input-sessions-week").value = p.sessions;
  document.getElementById("input-completion-rate").value = p.completion;
  document.getElementById("input-weekend-ratio").value = p.weekend;
  document.getElementById("input-genre-diversity").value = p.diversity;

  document.querySelectorAll("input[name='genres']").forEach(cb => {
    cb.checked = p.genres.includes(cb.value);
  });
}

async function handleRecommendSubmit(event) {
  event.preventDefault();

  const selectedGenres = Array.from(document.querySelectorAll("input[name='genres']:checked")).map(cb => cb.value);
  if (selectedGenres.length === 0) {
    alert("Please select at least 1 preferred genre.");
    return;
  }

  const payload = {
    user_id: document.getElementById("input-user-id").value.trim(),
    watch_time_hours: parseFloat(document.getElementById("input-watch-time").value),
    avg_session_minutes: parseFloat(document.getElementById("input-avg-session").value),
    sessions_per_week: parseFloat(document.getElementById("input-sessions-week").value),
    completion_rate: parseFloat(document.getElementById("input-completion-rate").value),
    weekend_ratio: parseFloat(document.getElementById("input-weekend-ratio").value),
    genre_diversity: parseInt(document.getElementById("input-genre-diversity").value, 10),
    top_genres: selectedGenres
  };

  const submitBtn = document.getElementById("btn-submit");
  submitBtn.disabled = true;
  submitBtn.innerText = "Analyzing Telemetry...";

  try {
    const res = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (!res.ok) {
      alert(`Error (${res.status}): ${data.message || data.error}`);
      return;
    }

    renderRecommendResult(data);
  } catch (err) {
    console.error("Inference request failed:", err);
    alert("Network or inference error occurred.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerText = "Execute Segmentation & Personalization";
  }
}

function renderRecommendResult(data) {
  document.getElementById("result-placeholder").style.display = "none";
  const resultContent = document.getElementById("result-content");
  resultContent.style.display = "flex";

  document.getElementById("res-segment-name").innerText = data.segment_name;
  document.getElementById("res-segment-id").innerText = `Cluster ID: ${data.segment_id}`;
  document.getElementById("res-segment-desc").innerText = data.segment_description;
  document.getElementById("res-distance").innerText = (data.distance_to_centroid !== undefined) ? data.distance_to_centroid.toFixed(4) : "--";
  document.getElementById("res-user-id").innerText = data.user_id;
  document.getElementById("res-explanation").innerText = data.explanation;

  const recsContainer = document.getElementById("res-recommendations-list");
  if (!data.recommendations || data.recommendations.length === 0) {
    recsContainer.innerHTML = `<div style="color: var(--text-muted); font-size: 13px;">No catalog matches found.</div>`;
    return;
  }

  recsContainer.innerHTML = data.recommendations.map(r => `
    <div class="rec-item">
      <div class="rec-item-main">
        <div class="rec-title-row">
          <span class="rec-title">${r.title}</span>
          <span class="rec-badge">${r.type}</span>
          <span class="rec-badge">${r.duration}</span>
        </div>
        <div style="font-size: 11px; color: var(--accent-light); margin-top: 2px;">
          Genres: ${(r.genres || []).join(", ")}
        </div>
        <div class="rec-reason">${r.match_reason}</div>
      </div>
      <div class="rec-score-pill">
        ${Math.round(r.match_score * 100)}% Match
      </div>
    </div>
  `).join("");
}
