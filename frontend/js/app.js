const API = "http://localhost:8000/api";
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

// ─── Minimal, safe Markdown → HTML renderer (no external dependencies) ───
// The AI Insight / Recommender reports are Markdown. Previously they were dumped
// with a plain "\n -> <br>" replace, so headings (##), bold (**) and bullet lists
// showed as literal characters. This renders them properly and escapes HTML first.
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function renderInline(s) {
  // input is already HTML-escaped
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");          // `code`
  s = s.replace(/\*\*([^\n]+?)\*\*/g, "<strong>$1</strong>"); // **bold** (may wrap *italics*)
  s = s.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>"); // *italic*
  s = s.replace(/(^|[^_])_([^_\n]+)_(?!_)/g, "$1<em>$2</em>");    // _italic_
  return s;
}
function renderMarkdown(md) {
  if (!md) return "";
  const lines = String(md).replace(/\r\n/g, "\n").split("\n");
  let html = "", listOpen = false;
  const closeList = () => { if (listOpen) { html += "</ul>"; listOpen = false; } };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    if (!line.trim()) { closeList(); continue; }
    if (/^\s*---+\s*$/.test(line)) { closeList(); html += "<hr>"; continue; }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { closeList(); const lv = Math.min(h[1].length, 6);
      html += `<h${lv}>${renderInline(escapeHtml(h[2]))}</h${lv}>`; continue; }
    const bq = line.match(/^\s*>\s?(.*)$/);
    if (bq) { closeList(); html += `<blockquote>${renderInline(escapeHtml(bq[1]))}</blockquote>`; continue; }
    const li = line.match(/^\s*[-*]\s+(.*)$/);
    if (li) { if (!listOpen) { html += "<ul>"; listOpen = true; }
      html += `<li>${renderInline(escapeHtml(li[1]))}</li>`; continue; }
    closeList();
    html += `<p>${renderInline(escapeHtml(line))}</p>`;
  }
  closeList();
  return html;
}
(function injectMarkdownStyles() {
  if (document.getElementById("md-style")) return;
  const el = document.createElement("style");
  el.id = "md-style";
  el.textContent = `
  .markdown-body h1,.markdown-body h2,.markdown-body h3,.markdown-body h4{
    font-family:'Space Grotesk',sans-serif;color:var(--accent-cyan,#00e5ff);margin:14px 0 8px;line-height:1.25;}
  .markdown-body h1{font-size:1.25rem;}
  .markdown-body h2{font-size:1.1rem;border-bottom:1px solid rgba(148,163,184,.18);padding-bottom:4px;}
  .markdown-body h3{font-size:1rem;}
  .markdown-body h4{font-size:.92rem;color:var(--text,#f8fafc);}
  .markdown-body p{margin:6px 0;line-height:1.6;}
  .markdown-body ul{margin:6px 0 10px;padding-left:20px;}
  .markdown-body li{margin:3px 0;line-height:1.55;}
  .markdown-body strong{color:var(--text,#f8fafc);font-weight:600;}
  .markdown-body em{color:var(--text-muted,#cbd5e1);font-style:italic;}
  .markdown-body code{font-family:'JetBrains Mono',monospace;font-size:.85em;background:rgba(148,163,184,.14);
    padding:1px 5px;border-radius:4px;color:var(--accent-cyan,#00e5ff);}
  .markdown-body blockquote{margin:8px 0;padding:8px 12px;border-left:3px solid var(--accent-cyan,#00e5ff);
    background:rgba(0,229,255,.07);border-radius:4px;}
  .markdown-body hr{border:none;border-top:1px solid rgba(148,163,184,.18);margin:14px 0;}`;
  document.head.appendChild(el);
})();

// ─── Tabs ───
function resizeCharts(container) {
  (container || document).querySelectorAll(".plotly-container .js-plotly-plot").forEach((el) => {
    Plotly.Plots.resize(el);
  });
}
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    $$(".tab-panel").forEach((p) => p.classList.add("hidden"));
    const panel = document.getElementById(`panel-${tab.dataset.tab}`);
    if (panel) panel.classList.remove("hidden");
    currentTab = tab.dataset.tab;
    setTimeout(() => resizeCharts(panel), 50);
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
    await loadChartColumns();
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
    const res = await fetch(`${API}/data/${sessionId}/preview?rows=5`);
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
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch (parseErr) {
      hideLoading();
      showStatus(`Pipeline failed! Server returned ${res.status}: "${text.substring(0, 200)}"`, "error");
      console.error("Raw response text:", text);
      return;
    }
    if (data.status === "complete") {
      await loadAllResults();
      hideLoading();
      showStatus("✅ Pipeline complete! Results loaded for all views.", "success");
    } else {
      hideLoading();
      showStatus(`Pipeline error: ${JSON.stringify(data)}`, "error");
    }
  } catch (err) {
    hideLoading();
    showStatus(`Pipeline network error: ${err.message}`, "error");
  }
}

async function loadAllResults() {
  try {
    const [resData, resCharts, resInsight, resRec] = await Promise.all([
      fetch(`${API}/results/${sessionId}`).then(r => r.json()),
      fetch(`${API}/charts/${sessionId}`).then(r => r.json()),
      fetch(`${API}/insight/${sessionId}`).then(r => r.json()),
      fetch(`${API}/recommender/${sessionId}`).then(r => r.json()),
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
    const recByView = resRec.views || {};
    makeViewPanels("insightContent", (vk) => {
      const iv = insightByView[vk] || {};
      const insightText = iv.insight || "";

      // strip only a LEADING "[Agent] " marker — keep numeric ranges like [10 – 90] in the body
      const cleanInsight = insightText.replace(/^\s*\[[^\]]*\]\s*/, "").trim();
      const hasInsight = cleanInsight && !cleanInsight.startsWith("[") && cleanInsight.length > 20;

      let html = '<div class="card"><div class="card-title">📋 Strategic Report</div>';
      if (hasInsight) {
        if (cleanInsight.toLowerCase().includes("smaller model") || cleanInsight.toLowerCase().includes("ollama pull")) {
          html += `<div class="status-msg warning">⚠️ ${renderMarkdown(cleanInsight)}</div>`;
        } else {
          html += `<div class="insight-box markdown-body">${renderMarkdown(cleanInsight)}</div>`;
        }
      } else if (insightText.includes("Offline") || insightText.includes("offline")) {
        html += `<div class="status-msg warning">🤖 AI Engine Offline — Install and start <a href="https://ollama.ai" style="color:var(--accent-cyan);" target="_blank">Ollama</a>, then pull a model: <code>ollama pull llama3.2:1b</code></div>`;
      } else {
        html += "<p style='color:var(--text-muted);'>No insight available. Run the pipeline with AI enabled.</p>";
      }
      html += '</div>';
      return html;
    });

    // Anomaly panel
    renderAnomalyPanel();

    // Recommender tab - show recommendations
    {
      let recHtml = '';
      for (const vk of Object.keys(recByView)) {
        const rv = recByView[vk] || {};
        const recText = rv.recommender || "";
        const cleanRec = recText.replace(/^\s*\[[^\]]*\]\s*/, "").trim();
        const hasRec = cleanRec && !cleanRec.startsWith("[") && cleanRec.length > 20;
        if (hasRec) {
          if (cleanRec.toLowerCase().includes("smaller model") || cleanRec.toLowerCase().includes("ollama pull")) {
            recHtml += `<div class="status-msg warning">⚠️ ${renderMarkdown(cleanRec)}</div>`;
          } else {
            recHtml += `<div class="insight-box markdown-body" style="border-left-color:var(--accent-yellow);margin-bottom:16px;">${renderMarkdown(cleanRec)}</div>`;
          }
        } else if (recText.includes("Offline") || recText.includes("offline")) {
          recHtml += `<div class="status-msg warning">🤖 AI Engine Offline — Install and start <a href="https://ollama.ai" style="color:var(--accent-cyan);" target="_blank">Ollama</a>, then pull a model: <code>ollama pull llama3.2:1b</code></div>`;
        } else {
          recHtml += "<p style='color:var(--text-muted);'>No recommendations available. Run the pipeline with AI enabled.</p>";
        }
      }
      document.getElementById("recContent").innerHTML = recHtml;
    }

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
                const layout = Object.assign({}, chart.json.layout || {});
                layout.autosize = true;
                layout.width = el.clientWidth || undefined;
                layout.margin = layout.margin || {};
                layout.margin.l = Math.min(layout.margin.l || 50, 55);
                layout.margin.r = Math.min(layout.margin.r || 20, 25);
                Plotly.react(el, chart.json.data, layout, { displayModeBar: false, responsive: true });
              }
              catch (e) { console.error("Plotly error:", e); }
            }
          }
        }
      }
    }, 200);

    // Refresh chart builder columns
    await loadChartColumns();
  } catch (err) { console.error("Results error:", err); }
}

// ─── Interactive Chart Builder ───
let chartColumnCache = {};  // {dsName: [{name, type}, ...]}

function updateYVisibility(chartType) {
  const yEl    = document.getElementById("chartY");
  const yLabel = document.getElementById("yOptional");
  const aggRow = document.getElementById("chartAggFuncRow");
  const ctEl   = document.getElementById("chartType");

  if (!yEl || !ctEl) return; // DOM not ready yet

  if (!chartType) chartType = ctEl.value;
  const yVal    = yEl.value;
  const yRow    = yEl.closest("div");
  const needsY  = ["scatter", "line", "bar"].includes(chartType);

  if (yRow) {
    if (chartType === "box") {
      yRow.style.opacity = "1";
      if (yLabel) yLabel.style.display = "inline";
    } else if (needsY) {
      yRow.style.opacity = "1";
      if (yLabel) yLabel.style.display = "none";
    } else {
      yRow.style.opacity = "0.4";
      if (yLabel) yLabel.style.display = "inline";
    }
  }

  // Show agg selector only when chart type supports grouping AND a Y column is selected
  if (aggRow) {
    const aggSupported = ["bar", "line", "pie", "histogram"].includes(chartType);
    aggRow.style.display = (aggSupported && yVal) ? "" : "none";
  }
}

function onChartDatasetChange() {
  const dsSel = document.getElementById("chartDataset");
  const xSel  = document.getElementById("chartX");
  const ySel  = document.getElementById("chartY");
  if (!dsSel || !xSel || !ySel) return;

  const ds       = dsSel.value;
  const cols     = chartColumnCache[ds] || [];
  const currentX = xSel.value;
  const currentY = ySel.value;

  // Populate X — all columns available
  if (cols.length === 0) {
    xSel.innerHTML = `<option value="">— No columns loaded —</option>`;
  } else {
    xSel.innerHTML = cols.map(c =>
      `<option value="${c.name}" ${c.name === currentX ? "selected" : ""}>${c.name} (${c.type})</option>`
    ).join("");
  }

  // Populate Y — numeric first, then categorical, with "None" default
  const numCols = cols.filter(c => c.type === "numeric");
  const catCols = cols.filter(c => c.type !== "numeric");
  let yOpts = `<option value="">— None (count only) —</option>`;
  if (numCols.length) {
    yOpts += `<optgroup label="📊 Numeric">` +
      numCols.map(c => `<option value="${c.name}" ${c.name === currentY ? "selected" : ""}>${c.name}</option>`).join("") +
      `</optgroup>`;
  }
  if (catCols.length) {
    yOpts += `<optgroup label="🔤 Categorical">` +
      catCols.map(c => `<option value="${c.name}" ${c.name === currentY ? "selected" : ""}>${c.name}</option>`).join("") +
      `</optgroup>`;
  }
  ySel.innerHTML = yOpts;
  updateYVisibility();
}

function onChartTypeChange() {
  updateYVisibility(document.getElementById("chartType").value);
}

async function loadChartColumns() {
  if (!sessionId) return;
  const dsSel = document.getElementById("chartDataset");
  const ySel  = document.getElementById("chartY");
  if (!dsSel) return;

  try {
    const res = await fetch(`${API}/data/${sessionId}/columns`);
    if (!res.ok) { console.warn("loadChartColumns: server returned", res.status); return; }
    const data = await res.json();
    chartColumnCache = {};

    // Flatten all view datasets into a single lookup
    for (const [vk, info] of Object.entries(data.views || {})) {
      for (const [dsName, dsInfo] of Object.entries(info.datasets || {})) {
        if (!chartColumnCache[dsName]) {
          chartColumnCache[dsName] = dsInfo.columns || [];
        }
      }
    }

    const dsNames = Object.keys(chartColumnCache);
    const currentDs = dsSel.value;

    if (dsNames.length === 0) {
      dsSel.innerHTML = `<option value="">— No datasets loaded —</option>`;
      if (ySel) ySel.innerHTML = `<option value="">— None —</option>`;
      return;
    }

    dsSel.innerHTML = dsNames.map(n =>
      `<option value="${n}" ${n === currentDs ? "selected" : ""}>${n}</option>`
    ).join("");

    onChartDatasetChange(); // populates X and Y dropdowns
  } catch (err) {
    console.error("Chart columns error:", err);
  }
}

// Color palette per chart type for history badges
const CHART_TYPE_COLORS = {
  bar: "var(--accent-cyan)", line: "var(--accent-violet)", scatter: "var(--accent-pink)",
  pie: "var(--accent-yellow)", histogram: "var(--accent-green)", box: "#f97316"
};

// Swap X ↔ Y columns
function swapXY() {
  const xSel = document.getElementById("chartX");
  const ySel = document.getElementById("chartY");
  if (!xSel || !ySel) return;
  const xVal = xSel.value;
  const yVal = ySel.value;
  if (!yVal) return;
  // Try to set X to old Y value
  const xOpt = Array.from(xSel.options).find(o => o.value === yVal);
  if (xOpt) xSel.value = yVal;
  // Try to set Y to old X value
  const yOpt = Array.from(ySel.options).find(o => o.value === xVal);
  if (yOpt) ySel.value = xVal;
  updateYVisibility();
}

// Clear all saved charts from history
function clearChartHistory() {
  const h = document.getElementById("chartHistory");
  if (h) h.innerHTML = "";
  const d = document.getElementById("chartBuilderDisplay");
  if (d) d.innerHTML = `<p style="color:var(--text-muted);">Select your options above and click "Build Chart".</p>`;
}

async function buildInteractiveChart() {
  // ── Safely read all inputs ──────────────────────────────────────────
  const dsEl    = document.getElementById("chartDataset");
  const xEl     = document.getElementById("chartX");
  const yEl     = document.getElementById("chartY");
  const ctypeEl = document.getElementById("chartType");
  const titleEl = document.getElementById("chartTitle");
  const aggEl   = document.getElementById("chartAggFunc");

  if (!dsEl || !xEl || !ctypeEl) {
    showStatus("Chart builder controls not found — please reload the page.", "error");
    return;
  }

  const ds      = dsEl.value;
  const xCol    = xEl.value;
  const yCol    = yEl ? yEl.value : "";
  const ctype   = ctypeEl.value;
  const title   = titleEl ? titleEl.value : "";
  const aggFunc = aggEl ? aggEl.value : "mean";

  // ── Validation ──────────────────────────────────────────────────────
  if (!sessionId) {
    showStatus("Upload a file first!", "warning");
    return;
  }
  if (!ds) {
    showStatus("Please select a dataset.", "warning");
    return;
  }
  if (!xCol) {
    showStatus("Please select an X axis column.", "warning");
    return;
  }

  // ── Build request ───────────────────────────────────────────────────
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("dataset_name", ds);
  form.append("x_column", xCol);
  form.append("y_column", yCol || "");
  form.append("chart_type", ctype);
  form.append("agg_func", aggFunc);
  form.append("title", title);

  showLoading("Building chart...");
  try {
    const res = await fetch(`${API}/chart/build`, { method: "POST", body: form });

    // Handle non-OK responses gracefully
    if (!res.ok) {
      let errMsg = `Server error ${res.status}`;
      try {
        const errData = await res.json();
        errMsg = errData.detail || errMsg;
      } catch (_) {}
      throw new Error(errMsg);
    }

    const data = await res.json();
    hideLoading();

    // ── Validate response structure ────────────────────────────────────
    if (!data || !data.fig) {
      throw new Error("Server returned an empty chart response.");
    }
    const figData   = Array.isArray(data.fig.data)   ? data.fig.data   : [];
    const figLayout = (data.fig.layout && typeof data.fig.layout === "object")
                        ? data.fig.layout : {};

    // ── Build display labels ───────────────────────────────────────────
    const ctypeLabel = ctypeEl.options[ctypeEl.selectedIndex]
                         ? ctypeEl.options[ctypeEl.selectedIndex].text
                         : ctype;
    const yLabel   = yCol ? ` vs ${yCol}` : "";
    const aggLabel = (yCol && aggFunc && aggFunc !== "count") ? ` [${aggFunc}]` : "";
    const autoTitle = title || `${ctypeLabel}: ${xCol}${yLabel}${aggLabel}`;
    const badgeColor = CHART_TYPE_COLORS[ctype] || "var(--accent-cyan)";

    // Unique ID for this chart in history
    const chartId = `custom-chart-${Date.now()}`;

    // ── Shared Plotly layout overrides ─────────────────────────────────
    const layout = Object.assign({}, figLayout);
    layout.autosize = true;
    layout.margin   = { l: 55, r: 20, t: 50, b: 60 };

    // ── Update the "latest chart" preview ─────────────────────────────
    const displayEl = document.getElementById("chartBuilderDisplay");
    if (displayEl) {
      displayEl.innerHTML = `
        <div class="chart-container">
          <div class="chart-header">
            <span class="chart-title">${autoTitle}</span>
            <span class="chart-badge approved" style="background:${badgeColor};color:#070912;">${ctype}</span>
          </div>
          <div class="plotly-container" id="latest-chart" style="width:100%;height:420px;"></div>
        </div>`;
      const latestEl = document.getElementById("latest-chart");
      if (latestEl && typeof Plotly !== "undefined") {
        Plotly.react(latestEl, figData, layout, { displayModeBar: true, responsive: true });
      }
    }

    // ── Prepend a copy to history ──────────────────────────────────────
    const historyContainer = document.getElementById("chartHistory");
    if (historyContainer) {
      const historyCard = document.createElement("div");
      historyCard.className = "chart-container";
      historyCard.style.marginBottom = "12px";
      historyCard.innerHTML = `
        <div class="chart-header">
          <span class="chart-title">${autoTitle}</span>
          <span style="display:flex;align-items:center;gap:6px;">
            <span class="chart-badge approved" style="background:${badgeColor};color:#070912;">${ctype}</span>
            <button onclick="this.closest('.chart-container').remove()"
              style="background:none;border:none;color:var(--accent-red);cursor:pointer;font-size:1.1rem;line-height:1;"
              title="Remove chart">✕</button>
          </span>
        </div>
        <div class="plotly-container" id="${chartId}" style="width:100%;height:360px;"></div>`;
      historyContainer.prepend(historyCard);

      // Render history card after DOM update
      setTimeout(() => {
        const histEl = document.getElementById(chartId);
        if (histEl && typeof Plotly !== "undefined") {
          Plotly.react(histEl, figData, layout, { displayModeBar: false, responsive: true });
        }
      }, 100);
    }

    showStatus(`✅ Chart built — ${autoTitle}`, "success");

  } catch (err) {
    hideLoading();
    showStatus(`Chart build failed: ${err.message}`, "error");
    console.error("buildInteractiveChart error:", err);
  }
}

// Set up chart builder event listeners (script runs at end of body, DOM is ready)
(function initChartBuilder() {
  const chartTypeSel = document.getElementById("chartType");
  const chartDsSel   = document.getElementById("chartDataset");
  const chartYSel    = document.getElementById("chartY");
  const chartXSel    = document.getElementById("chartX");

  if (chartTypeSel) {
    chartTypeSel.addEventListener("change", () => updateYVisibility(chartTypeSel.value));
    // Run once on load to set initial agg row visibility
    updateYVisibility(chartTypeSel.value);
  }
  if (chartDsSel) {
    chartDsSel.addEventListener("change", onChartDatasetChange);
  }
  if (chartYSel) {
    chartYSel.addEventListener("change", () => updateYVisibility());
  }
  // Allow pressing Enter in the title field to trigger build
  const titleEl = document.getElementById("chartTitle");
  if (titleEl) {
    titleEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") buildInteractiveChart();
    });
  }
})();

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
      `<div class="insight-box markdown-body" style="border-left-color:var(--accent-pink);margin-top:12px;">${renderMarkdown(data.explanation)}</div>`;
  } catch (err) { hideLoading(); showStatus(`Explain failed: ${err.message}`, "error"); }
}

// ─── Active View Helper ───
function getActiveView() {
  if (viewKeys.length <= 1) return viewKeys[0] || "unified";
  const activeTab = document.querySelector(".view-tab.active");
  return activeTab ? activeTab.dataset.viewKey : viewKeys[0];
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
  const el = document.getElementById("llmStatus");
  if (!el) return;
  try {
    const res = await fetch(`${API}/llm/status`);
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    const isOnline = data.online === true;
    const statusDetail = data.status || (isOnline ? "online" : "offline");
    el.className = `llm-status ${isOnline ? "online" : "offline"}`;
    let label = isOnline ? "ONLINE" : "OFFLINE";
    let detail = isOnline ? "Ollama Ready" : "Start Ollama";
    if (statusDetail.startsWith("error:")) {
      el.className = "llm-status offline";
      label = "ERROR";
      detail = statusDetail.replace("error:", "").trim();
    }
    el.innerHTML = `<span class="dot"></span> ${label} <span style="margin-left:auto;font-size:0.65rem;color:var(--text-muted);">${detail}</span>`;
  } catch {
    el.className = "llm-status offline";
    el.innerHTML = `<span class="dot"></span> OFFLINE <span style="margin-left:auto;font-size:0.65rem;color:var(--text-muted);">Server unreachable</span>`;
  }
}

// ─── Init ───
checkLLM();
setInterval(checkLLM, 15000);