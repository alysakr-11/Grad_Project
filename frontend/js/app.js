const API = "/api";
let sessionId = null;
let currentTab = "kpi";
let viewKeys = ["unified"];
let viewTitles = {};
let tripleMode = false;
let sharedCols = [];

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

function showLoading(msg = "Processing...") {
  const el = document.getElementById("loadingOverlay");
  el.querySelector("p").textContent = msg;
  el.classList.remove("hidden");
}
function hideLoading() { document.getElementById("loadingOverlay").classList.add("hidden"); }

function showStatus(msg, type = "info") {
  const el = document.getElementById("statusMessage");
  el.className = `status-msg ${type}`;
  el.innerHTML = msg;
  el.classList.remove("hidden");
}
function clearStatus() { document.getElementById("statusMessage").classList.add("hidden"); }

// ─── Tabs ───
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    $$(".tab-panel").forEach((p) => p.classList.add("hidden"));
    const panel = document.getElementById(`panel-${tab.dataset.tab}`);
    if (panel) panel.classList.remove("hidden");
    currentTab = tab.dataset.tab;
  });
});

// ─── File Upload ───
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault(); uploadZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFiles(fileInput.files);
});

async function handleFiles(files) {
  showLoading("Uploading files...");
  const form = new FormData();
  for (const f of files) form.append("files", f);
  try {
    const res = await fetch(`${API}/upload`, { method: "POST", body: form });
    const data = await res.json();
    sessionId = data.session_id;
    renderFileBadges(data.files);
    await preprocess();
    await loadDataPreview();
    hideLoading();
    showStatus("Files uploaded and preprocessed!", "success");
  } catch (err) {
    hideLoading();
    showStatus(`Upload failed: ${err.message}`, "error");
  }
}

function renderFileBadges(files) {
  document.getElementById("fileBadges").innerHTML = files.map((f) => `<span class="file-badge">📊 ${f}</span>`).join("");
}

async function preprocess() {
  const form = new FormData();
  form.append("session_id", sessionId);
  const res = await fetch(`${API}/preprocess`, { method: "POST", body: form });
  const data = await res.json();
  renderSidebarStats(data.summary);
  renderPrepReport(data.reports);

  tripleMode = data.triple_mode;
  sharedCols = data.shared_cols || [];
  viewKeys = Object.keys(data.views || { "unified": true });
  viewTitles = data.view_titles || {};

  // Show intersection / no-intersection message
  const msgContainer = document.getElementById("viewMessage");
  msgContainer.innerHTML = "";
  if (tripleMode) {
    if (sharedCols.length === 0) {
      msgContainer.innerHTML = `<div class="status-msg warning" style="margin-bottom:16px;">⚠️ No Common Columns Detected — The two datasets don't share column names. They will be analyzed independently as Dataset A and Dataset B.</div>`;
    } else {
      msgContainer.innerHTML = `<div class="status-msg info" style="margin-bottom:16px;">🔀 Triple-View Active — All tabs contain sub-tabs: Dataset A, Dataset B, and their Intersection on <strong>${sharedCols.length} shared column${sharedCols.length > 1 ? 's' : ''}</strong>: ${sharedCols.map(c => `<code style="color:var(--accent-cyan)">${c}</code>`).join(', ')}</div>`;
    }
  }

  // Render view sub-tabs
  renderViewSubtabs();
  // Initialize anomaly panel
  renderAnomalyPanel();
}

function renderSidebarStats(summary) {
  let totalRows = 0, totalCols = 0, dsCount = 0;
  for (const info of Object.values(summary)) {
    totalRows += info.rows; totalCols += info.cols; dsCount++;
  }
  document.getElementById("statDatasets").textContent = dsCount;
  document.getElementById("statColumns").textContent = totalCols;
  document.getElementById("statRecords").textContent = totalRows.toLocaleString();
}

function renderPrepReport(reports) {
  const container = document.getElementById("prepReport");
  container.innerHTML = "";
  for (const [name, lines] of Object.entries(reports)) {
    const h = document.createElement("h4");
    h.style.cssText = "font-family:'Space Grotesk',sans-serif;margin:12px 0 6px;font-size:0.9rem;color:var(--accent-cyan);";
    h.textContent = `📂 ${name}`;
    container.appendChild(h);
    for (const line of lines) {
      const p = document.createElement("p");
      p.style.cssText = "font-size:0.82rem;color:var(--text-secondary);padding:2px 0;font-family:'JetBrains Mono',monospace;";
      p.textContent = line;
      container.appendChild(p);
    }
  }
}

// ─── View Sub-Tabs ───
function renderViewSubtabs() {
  const containers = $$(".view-subtabs");
  containers.forEach((container) => {
    container.innerHTML = "";
    if (viewKeys.length <= 1) return;

    const tabBar = document.createElement("div");
    tabBar.className = "tabs view-tabs";
    tabBar.style.marginBottom = "12px";

    viewKeys.forEach((vk, i) => {
      const label = viewTitles[vk] || vk;
      const short = label.length > 30 ? label.substring(0, 28) + "…" : label;
      const btn = document.createElement("button");
      btn.className = `tab view-tab ${i === 0 ? "active" : ""}`;
      btn.dataset.viewKey = vk;
      btn.textContent = short;
      btn.title = label;
      btn.addEventListener("click", () => {
        container.querySelectorAll(".view-tab").forEach(t => t.classList.remove("active"));
        btn.classList.add("active");
        // Hide all view panels in this group
        const parent = container.closest(".view-group") || container.parentElement;
        parent.querySelectorAll(".view-panel").forEach(p => p.classList.add("hidden"));
        const panel = parent.querySelector(`.view-panel[data-view="${vk}"]`);
        if (panel) panel.classList.remove("hidden");
      });
      tabBar.appendChild(btn);
    });
    container.appendChild(tabBar);
  });
}

function makeViewPanels(containerId, htmlFn) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  if (viewKeys.length <= 1) {
    // Single view — no sub-tabs
    container.innerHTML = `<div class="view-group">${htmlFn("unified")}</div>`;
    return;
  }
  // Triple view — add sub-tab bar and panels
  const subtabDiv = document.createElement("div");
  subtabDiv.className = "view-subtabs";
  container.appendChild(subtabDiv);

  const group = document.createElement("div");
  group.className = "view-group";
  viewKeys.forEach((vk, i) => {
    const panel = document.createElement("div");
    panel.className = `view-panel ${i === 0 ? "" : "hidden"}`;
    panel.dataset.view = vk;
    panel.innerHTML = htmlFn(vk);
    group.appendChild(panel);
  });
  container.appendChild(group);
  renderViewSubtabs();
}

// ─── Data Preview ───
async function loadDataPreview() {
  try {
    const res = await fetch(`${API}/data/${sessionId}?rows=5`);
    const data = await res.json();
    const container = document.getElementById("dataPreview");
    container.innerHTML = "";
    for (const [name, info] of Object.entries(data)) {
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML = `<div class="card-title">📂 ${name} <span style="margin-left:auto;font-size:0.75rem;color:var(--text-muted);font-family:'JetBrains Mono',monospace;">${info.shape[0]} rows × ${info.shape[1]} cols</span></div>`;
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      const table = document.createElement("table");
      table.className = "data-table";
      const headers = Object.keys(info.data[0] || {});
      table.innerHTML = `<thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead>`;
      let tbody = "<tbody>";
      for (const row of info.data) {
        tbody += `<tr>${headers.map(h => `<td>${row[h] ?? ""}</td>`).join("")}</tr>`;
      }
      tbody += "</tbody>";
      table.innerHTML += tbody;
      wrap.appendChild(table);
      card.appendChild(wrap);
      container.appendChild(card);
    }
    renderDatasetSelector(data);
  } catch (err) { console.error("Preview error:", err); }
}

function renderDatasetSelector(datasets) {
  const sel = document.getElementById("dsSelect");
  sel.innerHTML = Object.keys(datasets).map(n => `<option value="${n}">${n}</option>`).join("");
}

// ─── Pipeline ───
async function runPipeline() {
  if (!sessionId) { showStatus("Upload files first!", "warning"); return; }
  showLoading("Running Multi-Agent Pipeline across all views...");
  clearStatus();

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("run_insight", "true");

  try {
    const res = await fetch(`${API}/pipeline/run`, { method: "POST", body: form });
    const data = await res.json();
    if (data.status === "complete") {
      await loadAllResults();
      hideLoading();
      showStatus("✅ Pipeline complete! Results loaded for all views.", "success");
    }
  } catch (err) {
    hideLoading();
    showStatus(`Pipeline failed: ${err.message}`, "error");
  }
}

async function loadAllResults() {
  try {
    const [resData, resCharts, resInsight] = await Promise.all([
      fetch(`${API}/results/${sessionId}`).then(r => r.json()),
      fetch(`${API}/charts/${sessionId}`).then(r => r.json()),
      fetch(`${API}/insight/${sessionId}`).then(r => r.json()),
    ]);

    if (resData.status !== "complete") return;
    viewKeys = resData.view_keys || viewKeys;
    viewTitles = resData.view_titles || viewTitles;
    tripleMode = resData.triple_mode || false;

    const views = resData.views || {};

    // KPIs
    makeViewPanels("kpiContent", (vk) => {
      const v = views[vk];
      if (!v) return "<p style='color:var(--text-muted)'>No results for this view.</p>";
      let html = '<div class="kpi-grid">';
      for (const [name, info] of Object.entries(v.kpi || {})) {
        html += `<div class="kpi-card">
          <div class="kpi-label">${name}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;">
            <div><div style="font-size:0.6rem;color:var(--text-muted);">ROWS</div><div style="font-size:1.1rem;font-weight:700;color:var(--accent-cyan);">${info.rows.toLocaleString()}</div></div>
            <div><div style="font-size:0.6rem;color:var(--text-muted);">COLS</div><div style="font-size:1.1rem;font-weight:700;color:var(--accent-violet);">${info.cols}</div></div>
            <div><div style="font-size:0.6rem;color:var(--text-muted);">NUMERIC</div><div style="font-size:1.1rem;font-weight:700;color:var(--accent-pink);">${info.numeric_cols}</div></div>
            <div><div style="font-size:0.6rem;color:var(--text-muted);">NULLS</div><div style="font-size:1.1rem;font-weight:700;color:var(--accent-yellow);">${info.nulls}</div></div>
          </div>
        </div>`;
      }
      html += '</div>';
      return html;
    });

    // Logs
    makeViewPanels("logsContent", (vk) => {
      const v = views[vk];
      if (!v) return "";
      const logs = v.logs || {};
      const agents = [
        { key: "retriever", label: "🔍 Retriever Agent", cls: "" },
        { key: "planner", label: "🎯 Planner Agent", cls: "planner" },
        { key: "stylist", label: "🎨 Stylist Agent", cls: "stylist" },
        { key: "visualizer", label: "📈 Visualizer Agent", cls: "visualizer" },
        { key: "critic", label: "🧪 Critic Agent", cls: "critic" },
      ];
      let html = "";
      for (const a of agents) {
        if (logs[a.key]) html += `<div class="log-entry ${a.cls}"><strong>${a.label}</strong><br>${logs[a.key].replace(/\n/g, "<br>")}</div>`;
      }
      return html;
    });

    // Outliers
    makeViewPanels("outlierContent", (vk) => {
      const v = views[vk];
      if (!v) return "";
      const outliers = v.outliers || {};
      let html = "", hasAny = false;
      for (const [ds, cols] of Object.entries(outliers)) {
        for (const [col, count] of Object.entries(cols)) {
          if (count > 0) hasAny = true;
          html += `<p style="font-size:0.82rem;color:var(--text-secondary);padding:4px 0;font-family:'JetBrains Mono',monospace;">📂 ${ds} → ${col}: ${count} outlier(s)</p>`;
        }
      }
      return hasAny ? html : '<p style="color:var(--accent-green);">✅ No outliers detected.</p>';
    });

    // Charts
    const chartsByView = resCharts.views || {};
    makeViewPanels("chartContent", (vk) => {
      const cv = chartsByView[vk];
      if (!cv || !cv.charts) return "";
      let html = "";
      for (const [dsName, cols] of Object.entries(cv.charts)) {
        html += `<h3 style="font-family:'Space Grotesk',sans-serif;font-size:1rem;margin:16px 0 12px;color:var(--accent-cyan);">📂 ${dsName}</h3><div class="chart-grid">`;
        let count = 0;
        for (const [col, chart] of Object.entries(cols)) {
          if (count >= 6) break;
          const overridden = chart.original_type !== chart.type;
          const badgeCls = overridden ? "corrected" : chart.guard_warning ? "warning" : "approved";
          const badgeText = overridden ? `🔄 ${chart.original_type} → ${chart.type}` : chart.guard_warning ? "🛡️ Guard" : `✅ ${chart.type}`;
          const chartId = `chart-${vk}-${dsName.replace(/\s/g, "")}-${col.replace(/\s/g, "")}`;
          html += `<div class="chart-container">
            <div class="chart-header"><span class="chart-title">${col}</span><span class="chart-badge ${badgeCls}">${badgeText}</span></div>
            <div class="plotly-container" id="${chartId}" style="width:100%;height:380px;"></div>
          </div>`;
          count++;
        }
        html += '</div>';
      }
      return html;
    });

    // Insights
    const insightByView = resInsight.views || {};
    makeViewPanels("insightContent", (vk) => {
      const iv = insightByView[vk] || {};
      const insightText = iv.insight || "";
      const recText = iv.recommender || "";

      // Clean up insight text - remove error prefixes
      const cleanInsight = insightText.replace(/\[.*?\]\s*/g, "").trim();
      const cleanRec = recText.replace(/\[.*?\]\s*/g, "").trim();

      const hasInsight = cleanInsight && !cleanInsight.startsWith("[") && cleanInsight.length > 20;
      const hasRec = cleanRec && !cleanRec.startsWith("[") && cleanRec.length > 20;

      let html = '<div class="card"><div class="card-title">📋 Strategic Report</div>';
      if (hasInsight) {
        html += `<div class="insight-box">${cleanInsight.replace(/\n/g, "<br>")}</div>`;
      } else if (insightText.includes("Offline") || insightText.includes("offline")) {
        html += `<div class="status-msg warning">🤖 AI Engine Offline — Start Ollama with <code>llama3</code> to generate AI insights. Without it, the Insight Agent uses rule-based fallbacks.</div>`;
      } else {
        html += "<p style='color:var(--text-muted);'>No insight available. Run the pipeline with AI enabled.</p>";
      }
      html += '</div><div class="card"><div class="card-title" style="color:var(--accent-yellow);">💡 Recommendations</div>';
      if (hasRec) {
        html += `<div class="insight-box" style="border-left-color:var(--accent-yellow);">${cleanRec.replace(/\n/g, "<br>")}</div>`;
      } else if (recText.includes("Offline") || recText.includes("offline")) {
        html += `<div class="status-msg warning">🤖 AI Engine Offline — Start Ollama to generate strategic recommendations.</div>`;
      } else {
        html += "<p style='color:var(--text-muted);'>No recommendations available.</p>";
      }
      html += '</div>';
      return html;
    });

    // Anomaly panel
    renderAnomalyPanel();

    // Render Plotly charts after DOM update
    setTimeout(() => {
      for (const [vk, cv] of Object.entries(chartsByView)) {
        if (!cv || !cv.charts) continue;
        for (const [dsName, cols] of Object.entries(cv.charts)) {
          for (const [col, chart] of Object.entries(cols)) {
            const chartId = `chart-${vk}-${dsName.replace(/\s/g, "")}-${col.replace(/\s/g, "")}`;
            const el = document.getElementById(chartId);
            if (el && chart.json) {
              try {
                const layout = chart.json.layout || {};
                layout.autosize = true;
                layout.width = undefined;
                layout.margin = layout.margin || {};
                layout.margin.l = Math.min(layout.margin.l || 50, 60);
                layout.margin.r = Math.min(layout.margin.r || 20, 30);
                Plotly.react(el, chart.json.data, layout, { displayModeBar: false, responsive: true });
              }
              catch (e) { console.error("Plotly error:", e); }
            }
          }
        }
      }
    }, 100);

  } catch (err) { console.error("Results error:", err); }
}

// ─── Anomaly Detection (per-view) ───
function renderAnomalyPanel() {
  makeViewPanels("anomalyContent", (vk) => {
    if (!vk) vk = viewKeys[0] || "unified";
    const resultsId = `anomaly-results-${vk}`;
    const explainId = `anomaly-explain-${vk}`;
    return `
      <div>
        <div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;">
          <button class="btn btn-primary" onclick="runAnomalyView('${vk}')">🚀 Run Anomaly Scanner</button>
          <button class="btn btn-secondary" onclick="explainAnomalyView('${vk}')">🧠 Explain with AI</button>
        </div>
        <div id="${resultsId}"><p style="color:var(--text-muted);">Click "Run Anomaly Scanner" to detect anomalies in this view.</p></div>
        <div id="${explainId}"></div>
      </div>`;
  });
}

async function runAnomalyView(vk) {
  if (!sessionId) return;
  showLoading("Running Isolation Forest...");
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("view_key", vk);
  try {
    const res = await fetch(`${API}/anomaly/run`, { method: "POST", body: form });
    const data = await res.json();
    hideLoading();
    const container = document.getElementById(`anomaly-results-${vk}`);
    if (!container) return;
    if (data.error) { container.innerHTML = `<div class="status-msg warning">${data.error}</div>`; return; }
    container.innerHTML = `
      <div class="kpi-grid" style="grid-template-columns:repeat(4,1fr);">
        <div class="kpi-card"><div class="kpi-label">SCANNED</div><div class="kpi-value" style="font-size:1.3rem;">${data.total.toLocaleString()}</div></div>
        <div class="kpi-card"><div class="kpi-label">ANOMALIES</div><div class="kpi-value" style="font-size:1.3rem;color:var(--accent-red);-webkit-text-fill-color:var(--accent-red);">${data.anomalies}</div></div>
        <div class="kpi-card"><div class="kpi-label">NORMAL</div><div class="kpi-value" style="font-size:1.3rem;">${(data.total - data.anomalies).toLocaleString()}</div></div>
        <div class="kpi-card"><div class="kpi-label">RATE</div><div class="kpi-value" style="font-size:1.3rem;">${data.total ? ((data.anomalies / data.total) * 100).toFixed(1) : 0}%</div></div>
      </div>`;
    if (data.breakdown) {
      container.innerHTML += `<div class="card" style="margin-top:12px;"><div class="card-title">📊 Breakdown</div>
        <div class="table-wrap"><table class="data-table"><thead><tr><th>Dataset</th><th>Status</th><th>Count</th></tr></thead>
        <tbody>${data.breakdown.map(r => `<tr><td>${r._source}</td><td>${r.Status}</td><td>${r.count}</td></tr>`).join("")}</tbody></table></div></div>`;
    }
  } catch (err) { hideLoading(); showStatus(`Anomaly failed: ${err.message}`, "error"); }
}

async function explainAnomalyView(vk) {
  if (!sessionId) return;
  showLoading("AI analyzing anomalies...");
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("view_key", vk);
  try {
    const res = await fetch(`${API}/anomaly/explain`, { method: "POST", body: form });
    const data = await res.json();
    hideLoading();
    const container = document.getElementById(`anomaly-explain-${vk}`);
    if (!container) return;
    if (data.explanation) container.innerHTML =
      `<div class="insight-box" style="border-left-color:var(--accent-pink);margin-top:12px;">${data.explanation.replace(/\n/g, "<br>")}</div>`;
  } catch (err) { hideLoading(); showStatus(`Explain failed: ${err.message}`, "error"); }
}

// ─── Recommender ───
async function findSimilar() {
  if (!sessionId) { showStatus("Upload files first!", "warning"); return; }
  const ds = document.getElementById("dsSelect").value;
  const rowIdx = parseInt(document.getElementById("rowIndex").value) || 0;
  showLoading("Finding similar records...");
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("dataset", ds);
  form.append("row_index", rowIdx);
  form.append("top_n", 5);
  try {
    const res = await fetch(`${API}/recommender/similar`, { method: "POST", body: form });
    const data = await res.json();
    hideLoading();
    const container = document.getElementById("similarResults");
    if (data.error) { container.innerHTML = `<div class="status-msg warning">${data.error}</div>`; return; }
    if (!data.matches || !data.matches.length) { container.innerHTML = '<div class="status-msg info">No similar records found.</div>'; return; }
    const headers = Object.keys(data.matches[0]);
    container.innerHTML = `
      <div class="status-msg success" style="margin-top:12px;">✅ Top ${data.matches.length} matches for Row ${rowIdx}</div>
      <div class="table-wrap"><table class="data-table"><thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead>
      <tbody>${data.matches.map(r => `<tr>${headers.map(h => `<td>${r[h] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  } catch (err) { hideLoading(); showStatus(`Similarity search failed: ${err.message}`, "error"); }
}

// ─── PDF Export ───
async function downloadPDF() {
  if (!sessionId) { showStatus("Upload and run pipeline first!", "warning"); return; }
  const activeView = getActiveView();
  showLoading("Generating PDF...");
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("view_key", activeView);
  try {
    const res = await fetch(`${API}/pdf/generate`, { method: "POST", body: form });
    if (!res.ok) throw new Error("PDF generation failed");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "Executive_Report.pdf"; a.click();
    URL.revokeObjectURL(url);
    hideLoading();
    showStatus("✅ PDF downloaded!", "success");
  } catch (err) { hideLoading(); showStatus(`PDF failed: ${err.message}`, "error"); }
}

// ─── LLM Status ───
async function checkLLM() {
  try {
    const res = await fetch(`${API}/llm/status`);
    const data = await res.json();
    const el = document.getElementById("llmStatus");
    el.className = `llm-status ${data.online ? "online" : "offline"}`;
    el.innerHTML = `<span class="dot"></span> ${data.online ? "ONLINE" : "OFFLINE"} <span style="margin-left:auto;font-size:0.65rem;color:var(--text-muted);">${data.online ? "Ollama Ready" : "Start Ollama"}</span>`;
  } catch {}
}

// ─── Init ───
checkLLM();
setInterval(checkLLM, 15000);
