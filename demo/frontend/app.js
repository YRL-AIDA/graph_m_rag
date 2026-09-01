/* Graph-M-RAG demo frontend */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  files: [],
  strategies: [],
  docBboxes: null,   // {file_hash, pages, byPage, all}
  docPage: 0,
  lastAnswer: null,
  lastFileHash: null, // file of the last answered question (for Graphs tab)
  graphPage: 0,
  showRegions: true,
  showEvidence: false,
  evidenceSet: null, // {pages:Set, indices:Set, byPage:Map}
};

/* ---------------- helpers ---------------- */
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"]/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;",
  }[c]));
}

function fmtBytes(n) {
  if (n == null) return "—";
  if (n > 1e6) return (n / 1e6).toFixed(1) + " MB";
  return (n / 1e3).toFixed(0) + " KB";
}

function timingHtml(md) {
  if (!md || !md.total_ms) return "";
  return `поиск ${md.search_ms ?? 0}мс · графы/контекст ${md.enrichment_ms ?? 0}мс · LLM ${md.llm_generation_ms ?? 0}мс · всего ${md.total_ms}мс`;
}

function descOf(name) {
  const s = (state.strategies || []).find((x) => x.name === name);
  return s ? s.description : "";
}

/* ---------------- tabs ---------------- */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tabview").forEach((v) => v.classList.toggle("active", v.id === "tab-" + name));
  if (name === "doc") renderDoc();
  if (name === "graphs") renderGraphs();
  if (name === "demo") renderDemoQuestions();
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});
document.querySelectorAll(".pipe-link").forEach((p) => {
  p.addEventListener("click", () => { if (p.dataset.target) switchTab(p.dataset.target); });
});

/* ---------------- pipeline animation ---------------- */
function animatePipeline() {
  const steps = document.querySelectorAll(".pstep");
  steps.forEach((s) => s.classList.remove("active"));
  for (let i = 0; i < steps.length; i++) {
    setTimeout(() => steps[i].classList.add("active"), 350 * (i + 1));
  }
}

/* ---------------- bootstrap ---------------- */
async function bootstrap() {
  try {
    const h = await api("/api/health");
    const pill = $("status-pill");
    if (h.mock) {
      pill.textContent = "демо-режим (mock)";
      pill.className = "pill pill-warn";
    } else if (h.backend_online) {
      pill.textContent = "бэкенд подключён";
      pill.className = "pill pill-on";
    } else {
      pill.textContent = "бэкенд недоступен";
      pill.className = "pill pill-off";
    }
  } catch (e) {
    $("status-pill").textContent = "демо-сервер недоступен";
  }
  try {
    state.strategies = (await api("/api/strategies")).strategies || [];
    $("ask-strategy").innerHTML = state.strategies
      .map((s) => `<option value="${s.name}">${s.label}</option>`)
      .join("");
  } catch (e) {}
  await Promise.all([loadFiles(), loadSampleQuestions()]);
}

async function loadFiles() {
  try {
    const data = await api("/api/files");
    state.files = data.files || [];
  } catch (e) {
    state.files = [];
  }
  const opts = state.files
    .map((f) => `<option value="${f.file_hash}">${f.filename}</option>`)
    .join("");
  const noDocs = '<option value="">нет документов</option>';
  $("ask-doc").innerHTML = opts || noDocs;
  $("cmp-doc").innerHTML = opts || noDocs;
  $("doc-select").innerHTML = opts || noDocs;
  $("demo-doc").innerHTML = opts || noDocs;
  renderFiles();
  renderDemoQuestions();
  if (state.files.length) renderDoc();
}

async function loadSampleQuestions() {
  try {
    const d = await api("/api/sample-questions");
    state.sampleQuestions = d.questions || [];
  } catch (e) {
    state.sampleQuestions = [];
  }
  renderDemoQuestions();
}

function renderDemoQuestions() {
  const catsEl = $("demo-categories");
  const qSel = $("demo-question");
  if (!catsEl || !qSel) return;
  const cats = state.sampleQuestions || [];
  if (!cats.length) {
    catsEl.innerHTML = "";
    qSel.innerHTML = '<option value="">выберите документ и введите свой вопрос</option>';
    return;
  }
  catsEl.innerHTML = cats
    .map((c, i) => `<button class="chip-chip" data-cat="${i}">${escapeHtml(c.category || ("Пример " + (i + 1)))}</button>`)
    .join("");
  const fill = (i) => {
    const cat = cats[i] || {};
    qSel.innerHTML = (cat.questions || [])
      .map((q) => `<option value="${escapeHtml(q)}">${escapeHtml(q)}</option>`)
      .join("");
  };
  catsEl.querySelectorAll("button").forEach((b) => {
    b.addEventListener("click", () => {
      catsEl.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      fill(Number(b.dataset.cat));
    });
  });
  const first = catsEl.querySelector("button");
  if (first) { first.classList.add("active"); fill(0); }
}

function runDemo() {
  const fh = $("demo-doc").value;
  const q = $("demo-question").value;
  if (!fh || !q) {
    $("ask-status").textContent = "Выберите документ и пример вопроса.";
    switchTab("ask");
    return;
  }
  $("ask-doc").value = fh;
  $("ask-question").value = q;
  switchTab("ask");
  ask();
}

function renderFiles() {
  const body = $("files-body");
  if (!state.files.length) {
    body.innerHTML = '<tr><td colspan="4" class="hint">Нет загруженных документов</td></tr>';
    return;
  }
  body.innerHTML = state.files
    .map((f) => `<tr>
        <td>${escapeHtml(f.filename)}</td>
        <td class="${f.status === "error" ? "status-err" : "status-ok"}">${f.status || "—"}</td>
        <td>${fmtBytes(f.file_size)}</td>
        <td>${(f.upload_date || "—").slice(0, 16).replace("T", " ")}</td>
      </tr>`)
    .join("");
}

/* ================= DOC / REGIONS ================= */

async function loadBboxes(fh) {
  const d = await api(`/api/demo/pdf/${fh}/mineru-bboxes`);
  const byPage = {};
  let pages = (d.pages || 0);
  (d.bboxes || []).forEach((b) => {
    const p = b.page_idx || 0;
    (byPage[p] = byPage[p] || []).push(b);
    pages = Math.max(pages, p + 1);
  });
  state.docBboxes = { file_hash: fh, pages: Math.max(pages, 1), byPage, all: d.bboxes || [] };
}

async function renderDoc() {
  const fh = $("doc-select").value;
  if (!fh) {
    $("doc-empty").style.display = "block";
    $("doc-empty").textContent = "Выберите документ.";
    return;
  }
  if (!state.docBboxes || state.docBboxes.file_hash !== fh) {
    try {
      await loadBboxes(fh);
    } catch (e) {
      $("doc-empty").style.display = "block";
      $("doc-empty").textContent = "Не удалось загрузить регионы: " + e.message;
      return;
    }
  }
  state.docPage = Math.min(state.docPage, state.docBboxes.pages - 1);
  drawPage();
}

function drawPage() {
  const { pages } = state.docBboxes;
  state.docPage = Math.max(0, Math.min(state.docPage, pages - 1));
  $("doc-page-ind").textContent = `стр. ${state.docPage + 1} / ${pages}`;
  const img = $("doc-img");
  const ov = $("doc-overlay");
  // Cache-bust so a fresh `load` event always fires.  Without this, re-rendering
  // the same page (e.g. when switching to this tab after bootstrap) reuses the
  // browser cache and onload never runs again, leaving the overlay computed for
  // the wrong (natural) image size.
  img.onload = () => positionOverlay(state.docPage);
  img.onerror = () => { ov.innerHTML = ""; };
  img.src = `/api/demo/pdf/${state.docBboxes.file_hash}/page/${state.docPage}?t=${Date.now()}`;
  $("doc-empty").style.display = "none";
  renderLegend();
}

function positionOverlay(page) {
  const img = $("doc-img");
  const ov = $("doc-overlay");
  const regions = (state.docBboxes.byPage[page] || []);
  // Defer to the next frame so layout has settled and offsetWidth/Height are
  // valid even when this fires straight from the image `load` event.
  requestAnimationFrame(() => {
    const W = img.offsetWidth;
    const H = img.offsetHeight;
    // Tab hidden (no layout yet) → nothing to draw; drawPage() will re-run
    // positionOverlay once the tab is visible and the image is laid out.
    if (!W || !H) { ov.innerHTML = ""; return; }
    // MinerU bboxes are normalized to 0-1000 independently on EACH axis, while
    // PDF pages are not square.  Scale x and y with the actual displayed image
    // width/height separately, otherwise every region is stretched/squished
    // vertically by the page aspect ratio.
    const scaleX = W / 1000;
    const scaleY = H / 1000;
    ov.style.left = (img.offsetLeft || 0) + "px";
    ov.style.top = (img.offsetTop || 0) + "px";
    ov.style.width = W + "px";
    ov.style.height = H + "px";
    ov.innerHTML = "";
    if (!regions.length) return;
    const contextOnly = $("doc-context-only") && $("doc-context-only").checked;
    regions.forEach((r) => {
      if (contextOnly && !isEvidenceRegion(r)) return;
      const [x1, y1, x2, y2] = r.bbox || [0, 0, 0, 0];
      const el = document.createElement("div");
      el.className = "reg";
      const color = r.color || "#ffffff";
      el.style.setProperty("--color", color);
      el.style.left = x1 * scaleX + "px";
      el.style.top = y1 * scaleY + "px";
      el.style.width = Math.max(4, (x2 - x1) * scaleX) + "px";
      el.style.height = Math.max(4, (y2 - y1) * scaleY) + "px";
      if (isEvidenceRegion(r)) el.classList.add("evidence");
      el.style.display = state.showRegions ? "block" : "none";
      el.title = `${r.element_type}${r.page_idx != null ? " · стр. " + (r.page_idx + 1) : ""}\n${(r.text || "").slice(0, 120)}`;
      el.addEventListener("click", (ev) => {
        ev.stopPropagation();
        showRegionInfo(r);
      });
      ov.appendChild(el);
    });
  });
}

function isEvidenceRegion(r) {
  if (!state.showEvidence || !state.evidenceSet) return false;
  return state.evidenceSet.indices.has(r.element_index) ||
    (r.page_idx != null && state.evidenceSet.pages.has(r.page_idx));
}

function showRegionInfo(r) {
  $("doc-region-info").innerHTML = `
    <div style="margin-bottom:6px"><span class="ptag" style="--c:${r.color || "#fff"}">${escapeHtml(r.element_type)}</span>
    <span class="hint">стр. ${(r.page_idx ?? 0) + 1}</span></div>
    <pre style="white-space:pre-wrap;font-size:13px;color:#c9d6e8;max-height:220px;overflow:auto">${escapeHtml(r.text || "")}</pre>`;
}

function renderLegend() {
  const types = {};
  (state.docBboxes.all || []).forEach((r) => {
    if (!types[r.element_type]) types[r.element_type] = r.color || "#fff";
  });
  const order = ["title", "text", "table", "table_caption", "table_footnote", "image", "image_caption", "image_footnote", "equation", "discarded"];
  const el = $("doc-legend");
  const keys = Object.keys(types).sort((a, b) => (order.indexOf(a) < 0 ? 99 : order.indexOf(a)) - (order.indexOf(b) < 0 ? 99 : order.indexOf(b)));
  el.innerHTML = keys.length
    ? keys.map((t) => `<div class="lg-row"><span class="lg-swatch" style="--c:${types[t]}"></span> ${escapeHtml(t)}</div>`).join("")
    : '<p class="hint">Нет данных о регионах.</p>';
}

$("doc-select").addEventListener("change", renderDoc);
$("doc-prev").addEventListener("click", () => { state.docPage--; drawPage(); });
$("doc-next").addEventListener("click", () => { state.docPage++; drawPage(); });
$("doc-show-regions").addEventListener("change", (e) => {
  state.showRegions = e.target.checked;
  positionOverlay(state.docPage);
});
$("doc-show-evidence").addEventListener("change", (e) => {
  state.showEvidence = e.target.checked;
  positionOverlay(state.docPage);
});
// Reposition the region overlay whenever the page image is resized by the
// browser (window resize / container reflow).
window.addEventListener("resize", () => {
  if (state.docBboxes) positionOverlay(state.docPage);
  if (state.graphData) graphPositionOverlay(state.graphPage);
});

/* ================= GRAPHS controls ================= */
$("graph-prev").addEventListener("click", () => { state.graphPage--; graphDrawPage(); });
$("graph-next").addEventListener("click", () => { state.graphPage++; graphDrawPage(); });
["gl-blocks", "gl-order", "gl-entities", "gl-communities"].forEach((id) => {
  $(id).addEventListener("change", () => {
    if (state.graphData) graphPositionOverlay(state.graphPage);
  });
});
document.querySelectorAll(".gtab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".gtab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".graph-tab-panel").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    const g = t.dataset.gtab;
    const panel = document.querySelector(".gpanel-" + g);
    if (panel) panel.classList.add("active");
    if (g === "struct" && state.lastAnswer && state.lastAnswer.context) {
      renderStructuralGraph(state.lastAnswer.context);
    }
  });
});

/* ================= ASK ================= */

async function ask() {
  const file_hash = $("ask-doc").value;
  const question = $("ask-question").value.trim();
  if (!file_hash || !question) {
    $("ask-status").textContent = "Выберите документ и введите вопрос.";
    return;
  }
  const strategy = $("ask-strategy").value;
  const useReranker = $("ask-reranker").checked;
  const useMmr = $("ask-mmr").checked;
  $("ask-btn").disabled = true;
  $("ask-status").textContent = "обработка…";
  startProgress();
  animatePipeline();
  try {
    const r = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_hash, question, strategy,
        use_reranker: useReranker,
        use_mmr_reranker: useMmr,
      }),
    });
    renderAnswer(r);
    state.lastAnswer = r;
    state.lastFileHash = file_hash;
    state.lastQuestion = question;
    buildEvidenceSet(r);
    $("ask-status").textContent = r.status === "completed"
      ? "готово"
      : friendlyError(r.message || r.status);
  } catch (e) {
    $("ask-status").textContent = "ошибка: " + friendlyError(e.message);
  } finally {
    stopProgress();
    $("ask-btn").disabled = false;
  }
}

function friendlyError(msg) {
  const m = String(msg || "").toLowerCase();
  if (m.includes("backend timeout") || m.includes("timed out")) {
    return "Сервис ответа перегружен — подождите пару секунд и попробуйте ещё раз.";
  }
  if (m.includes("connection refused") || m.includes("unreachable") || m.includes("connection aborted")) {
    return "Сервис временно недоступен. Попробуйте позже.";
  }
  if (m.includes("not answerable")) {
    return "В документе не найден ответ на этот вопрос.";
  }
  return msg || "Неизвестная ошибка.";
}

function startProgress() {
  const el = $("ask-progress");
  const bar = $("ask-progress-bar");
  const label = $("ask-progress-label");
  if (!el) return;
  el.classList.remove("hidden");
  bar.style.width = "3%";
  const stages = [
    "поиск фрагментов в документе…",
    "построение графов и связей…",
    "анализ таблиц и изображений…",
    "формирование ответа…",
  ];
  let w = 3;
  clearInterval(state.progressTimer);
  state.progressTimer = setInterval(() => {
    w = Math.min(w + 1 + Math.random() * 2.2, 97);
    bar.style.width = w + "%";
    label.textContent = stages[Math.min(Math.floor(w / 25), stages.length - 1)];
  }, 400);
}

function stopProgress() {
  clearInterval(state.progressTimer);
  state.progressTimer = null;
  const el = $("ask-progress");
  if (el) {
    el.classList.add("hidden");
    $("ask-progress-bar").style.width = "0%";
  }
}

async function compareWithBaseline() {
  const btn = $("ask-compare");
  const block = $("ask-compare-block");
  if (btn.disabled) return;
  btn.disabled = true;
  btn.textContent = "сравнение…";
  block.classList.remove("hidden");
  block.innerHTML = '<p class="hint">Сравниваем с версией без графов…</p>';
  try {
    const fh = state.lastFileHash;
    const q = state.lastQuestion || $("ask-question").value;
    const r = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_hash: fh, question: q, strategy: "baseline" }),
    });
    const withGraphs = (state.lastAnswer && state.lastAnswer.answer) || "—";
    const without = (r && r.answer) || (r && r.message) || "—";
    const same = String(withGraphs).trim().toLowerCase() === String(without).trim().toLowerCase();
    block.innerHTML = `
      <div class="cmp-pair">
        <div class="cmp-col">
          <div class="cmp-title">С графами <span class="hint">полная версия</span></div>
          <div class="cmp-answer">${escapeHtml(withGraphs)}</div>
        </div>
        <div class="cmp-col">
          <div class="cmp-title">Без графов <span class="hint">только поиск</span></div>
          <div class="cmp-answer">${escapeHtml(without)}</div>
        </div>
      </div>
      <p class="hint">${same
        ? "Ответы совпали — этот вопрос можно решить и простым поиском."
        : "Ответы различаются: графы добавили информацию, недоступную обычному поиску."}</p>`;
  } catch (e) {
    block.innerHTML = '<p class="hint">Не удалось выполнить сравнение: ' +
      escapeHtml(friendlyError(e.message)) + "</p>";
  } finally {
    btn.disabled = false;
    btn.textContent = "Что дают графы?";
  }
}

function renderAnswer(r) {
  $("ask-result").classList.remove("hidden");
  $("answer-strategy").textContent = `${r.strategy_label} · ${r.status === "completed" ? "ответ" : r.status}`;
  $("answer-timing").textContent = timingHtml(r.metadata);
  $("answer-text").textContent = r.answer || "—";

  const blocks = (r.context.blocks || []).slice(0, 15);
  const badge = $("answer-evidence-badge");
  if (blocks.length) {
    badge.textContent = `найдено ${blocks.length} доказательств`;
    badge.classList.remove("hidden");
  } else {
    badge.classList.add("hidden");
  }
  $("ask-compare").classList.toggle("hidden", !state.lastFileHash);

  renderComposition(r.context);
  renderEntities(r.context);

  const evEl = $("answer-evidence");
  evEl.innerHTML = "";
  if (!blocks.length) {
    evEl.innerHTML = '<p class="hint">Доказательства не найдены (возможно, mock-режим).</p>';
    return;
  }
  blocks.forEach((b) => {
    const item = document.createElement("div");
    item.className = "evidence-item";
    item.innerHTML = `
      <div class="evidence-head">
        <span>${escapeHtml(b.type || "block")} #${escapeHtml(b.id)}</span>
        <span class="meta">стр. ${escapeHtml(b.page)} · rel ${escapeHtml(b.relevance || "?")}</span>
      </div>
      <div class="evidence-body">${escapeHtml(b.text)}</div>`;
    item.querySelector(".evidence-head").addEventListener("click", () => item.classList.toggle("open"));
    evEl.appendChild(item);
  });
}

function renderComposition(ctx) {
  const s = ctx.sources || { qdrant: 0, structural: 0, semantic: 0 };
  const max = Math.max(1, s.qdrant, s.structural, s.semantic);
  const rows = [
    ["qdrant", "Фрагменты документа", s.qdrant, "векторный поиск", "ask"],
    ["structural", "Структурный граф", s.structural, "соседи по чтению, родители", "graphs"],
    ["semantic", "Семантический граф", s.semantic, "сущности и сообщества", "graphs"],
  ];
  $("ctx-composition").innerHTML = rows.map(([cls, label, count, sub, target]) => `
    <div class="ctx-row${count ? " clickable" : ""}" data-target="${target}">
      <div class="ctx-label">${label}<br><small class="hint">${sub}</small></div>
      <div class="ctx-bar"><div class="ctx-fill ${cls}" style="width:${(count / max) * 100}%"></div></div>
      <div class="ctx-count">${count}</div>
    </div>`).join("");
  $("ctx-composition").querySelectorAll(".ctx-row.clickable").forEach((row) => {
    row.addEventListener("click", () => {
      const t = row.dataset.target;
      if (t === "graphs") switchTab("graphs");
      else {
        const ev = $("answer-evidence");
        if (ev) ev.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });
}

function renderEntities(ctx) {
  const el = $("answer-entities");
  el.innerHTML = "";
  (ctx.entities || []).slice(0, 14).forEach((e) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.title = e.description || "";
    c.textContent = `${e.type}: ${e.name}`;
    el.appendChild(c);
  });
  (ctx.communities || []).slice(0, 5).forEach((c) => {
    const s = document.createElement("span");
    s.className = "chip comm";
    s.title = c.summary || "";
    s.textContent = `⚡ ${c.title}`;
    el.appendChild(s);
  });
  if (!el.children.length) el.innerHTML = '<span class="hint">Сущности не найдены.</span>';
}

function buildEvidenceSet(r) {
  const indices = new Set();
  const pages = new Set();
  const byPage = new Map();
  (r.context.blocks || []).forEach((b) => {
    let p = parseInt(b.page, 10);
    if (isNaN(p)) p = -1;
    pages.add(p);
    const txt = (b.text || "").slice(0, 40).toLowerCase();
    if (state.docBboxes && txt.length > 8) {
      (state.docBboxes.byPage[p] || []).forEach((rg) => {
        const rt = (rg.text || "").toLowerCase();
        if (rt.startsWith(txt) || txt.startsWith(rt.slice(0, 40))) indices.add(rg.element_index);
      });
    }
    if (!byPage.has(p)) byPage.set(p, []);
    byPage.get(p).push(b);
  });
  state.evidenceSet = { indices, pages, byPage };
}

$("ask-btn").addEventListener("click", ask);
$("ask-compare").addEventListener("click", compareWithBaseline);
$("demo-btn").addEventListener("click", runDemo);
$("demo-doc").addEventListener("change", () => {
  if (state.files.length) {
    $("ask-doc").value = $("demo-doc").value;
  }
});
$("doc-context-only").addEventListener("change", () => {
  positionOverlay(state.docPage);
});
$("ask-question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
});
$("ask-show-evidence").addEventListener("click", () => {
  if (!state.lastAnswer || !state.docBboxes) return;
  document.querySelector('.tab[data-tab="doc"]').click();
  const page = state.evidenceSet && state.evidenceSet.pages.size
    ? Math.max(0, [...state.evidenceSet.pages].find((p) => p >= 0) || 0)
    : state.docPage;
  state.docPage = page;
  state.showEvidence = true;
  $("doc-show-evidence").checked = true;
  drawPage();
});

/* ================= GRAPHS ================= */

function _gtitle(text) {
  return text ? `<title>${escapeHtml(text)}</title>` : "";
}

// Resolve a context element to a page bbox.
//   * Blocks carry an exact Qdrant region_id -> use it first.
//   * Graph elements (ORDER/PARENT/cross-graph/BFS) use a Neo4j region_id that
//     may be offset from the bbox counter on legacy documents, so the reliable
//     signal is page + text overlap.
function graphResolveBBox(el, bboxByRegion, byPage, allBboxes) {
  const rid = el.region_id || el.source_region;
  const num = rid ? String(rid).split("|").pop() : null;
  const page = el.page != null ? Number(el.page) : null;
  const text = (el.text || "").toLowerCase();
  const words = text.replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter((w) => w.length > 3);
  const isBlock = el.kind === "block";

  const matchOnPage = (list) => {
    if (!words.length) return null;
    let best = null, bestScore = 0;
    (list || []).forEach((b) => {
      const bt = (b.text_preview || b.text || "").toLowerCase();
      let score = 0;
      for (let i = 0; i < words.length && i < 12; i++) if (bt.includes(words[i])) score++;
      if (score > bestScore) { bestScore = score; best = b; }
    });
    return bestScore >= 2 ? best : null;
  };

  if (isBlock) {
    const hit = num ? (bboxByRegion.get(num) || bboxByRegion.get(rid)) : null;
    if (hit) return hit;
    return page != null ? matchOnPage(byPage[page]) : null;
  }

  if (page != null) {
    const t = matchOnPage(byPage[page]);
    if (t) return t;
  }
  const hit = num ? (bboxByRegion.get(num) || bboxByRegion.get(rid)) : null;
  if (hit) return hit;
  return matchOnPage(allBboxes);
}

function graphBuildContext() {
  const a = state.lastAnswer;
  const fh = state.lastFileHash;
  if (!a || !fh || !state.docBboxes || state.docBboxes.file_hash !== fh) return null;

  const bboxByRegion = new Map();
  (state.docBboxes.all || []).forEach((b) => {
    (b.region_ids || []).forEach((rid) => {
      const num = String(rid).split("|").pop();
      if (!bboxByRegion.has(num)) bboxByRegion.set(num, b);
      if (!bboxByRegion.has(rid)) bboxByRegion.set(rid, b);
    });
  });
  const byPage = state.docBboxes.byPage;
  const ctx = a.context;

  const items = [];
  (ctx.blocks || []).forEach((b) => items.push({
    kind: "block", id: b.id, source: "qdrant", color: "#3498db",
    page: b.page, text: b.text, region_id: b.region_id,
    relevance: b.relevance, type: b.type,
  }));
  (ctx.regions || []).forEach((r) => items.push({
    kind: "region", source: r.source === "order" ? "order" : (r.source || "link"),
    color: r.source === "order" ? "#3fb950" : "#8b6bff",
    label: r.label, page: r.page, text: r.text,
    region_id: r.region_id || r.source_region, source_region: r.source_region,
  }));
  (ctx.entities || []).forEach((e) => items.push({
    kind: "entity", name: e.name, type: e.type, color: "#7c5cff",
    description: e.description, region_id: e.region_id, score: e.score,
    related: e.related || "",
  }));
  (ctx.communities || []).forEach((c) => items.push({
    kind: "community", title: c.title, color: "#d29922",
    rating: c.rating, summary: c.summary, region_id: c.region_id,
  }));

  items.forEach((it) => {
    const b = graphResolveBBox(it, bboxByRegion, byPage, state.docBboxes.all);
    if (b) {
      it.bbox = b.bbox;
      it.page = b.page_idx;
    }
  });

  // ORDER chain edges (same-page, in reading order)
  const order = items.filter((i) => i.source === "order" && i.bbox != null);
  order.sort((a, b) => (a.bbox[1] - b.bbox[1]) || (a.bbox[0] - b.bbox[0]));
  state.graphEdges = order;
  return { items, byPage, pages: state.docBboxes.pages };
}

function renderGraphs() {
  // The document must always be visible here: pick a file from the last
  // answered question, else the question tab's selection, else the first file.
  const a = state.lastAnswer;
  let fh = state.lastFileHash;
  if (!fh && $("ask-doc") && $("ask-doc").value) fh = $("ask-doc").value;
  if (!fh && state.files.length) fh = state.files[0].file_hash;
  if (!fh) {
    $("graph-empty").style.display = "block";
    $("graph-empty").textContent = "Нет доступных документов.";
    return;
  }
  const load = state.docBboxes && state.docBboxes.file_hash === fh
    ? Promise.resolve()
    : loadBboxes(fh);
  load.then(() => {
    $("graph-empty").style.display = "none";
    if (a && a.context) {
      state.graphData = graphBuildContext();
      state.graphRegionMap = null;
      state.pendingGraphHighlight = null;
      renderStructuralGraph(a.context);
      renderSemanticGraph(a.context);
    }
    state.graphPage = Math.min(state.graphPage, state.docBboxes.pages - 1);
    graphDrawPage();
  }).catch((e) => {
    $("graph-empty").style.display = "block";
    $("graph-empty").textContent = "Не удалось загрузить регионы: " + e.message;
  });
}

function graphDrawPage() {
  const img = $("graph-img");
  const ov = $("graph-overlay");
  const pages = state.docBboxes ? state.docBboxes.pages : 1;
  state.graphPage = Math.max(0, Math.min(state.graphPage, pages - 1));
  $("graph-page-ind").textContent = `стр. ${state.graphPage + 1} / ${pages}`;
  img.onload = () => graphPositionOverlay(state.graphPage);
  img.onerror = () => { ov.innerHTML = ""; $("graph-edges").innerHTML = ""; };
  img.src = `/api/demo/pdf/${state.docBboxes.file_hash}/page/${state.graphPage}?t=${Date.now()}`;
}

function graphPositionOverlay(page) {
  const img = $("graph-img");
  const ov = $("graph-overlay");
  const edgesEl = $("graph-edges");
  requestAnimationFrame(() => {
    const W = img.offsetWidth;
    const H = img.offsetHeight;
    if (!W || !H) { ov.innerHTML = ""; edgesEl.innerHTML = ""; return; }
    const scaleX = W / 1000;
    const scaleY = H / 1000;
    ov.style.left = (img.offsetLeft || 0) + "px";
    ov.style.top = (img.offsetTop || 0) + "px";
    ov.style.width = W + "px";
    ov.style.height = H + "px";
    edgesEl.style.left = (img.offsetLeft || 0) + "px";
    edgesEl.style.top = (img.offsetTop || 0) + "px";
    edgesEl.style.width = W + "px";
    edgesEl.style.height = H + "px";
    ov.innerHTML = "";
    edgesEl.innerHTML = "";

    const data = state.graphData;
    if (!data) return;
    const onPage = data.items.filter((i) => i.bbox != null && i.page === page);

    const layers = {
      blocks: $("gl-blocks").checked,
      order: $("gl-order").checked,
      entities: $("gl-entities").checked,
      communities: $("gl-communities").checked,
    };

    // highlight rectangles for blocks / structural regions
    onPage.forEach((it) => {
      const show =
        (it.kind === "block" && layers.blocks) ||
        (it.source === "order" && layers.order) ||
        (["parent", "cross_graph", "bfs_crawler", "link"].includes(it.source) && layers.order);
      if (!show) return;
      const [x1, y1, x2, y2] = it.bbox;
      const el = document.createElement("div");
      el.className = "greg";
      el.style.setProperty("--c", it.color);
      el.dataset.region = itemRegionNum(it) || "";
      el.style.left = x1 * scaleX + "px";
      el.style.top = y1 * scaleY + "px";
      el.style.width = Math.max(4, (x2 - x1) * scaleX) + "px";
      el.style.height = Math.max(4, (y2 - y1) * scaleY) + "px";
      el.addEventListener("click", () => {
        graphDetail(it);
        const n = itemRegionNum(it);
        if (n) graphHighlightRegion(n);
      });
      ov.appendChild(el);
    });

    // entity / community badges
    onPage.forEach((it) => {
      if (it.kind === "entity" && !layers.entities) return;
      if (it.kind === "community" && !layers.communities) return;
      if (it.kind !== "entity" && it.kind !== "community") return;
      const [x1, y1, , ] = it.bbox;
      const el = document.createElement("div");
      el.className = "gbadge";
      el.style.setProperty("--c", it.color);
      const cx = x1 * scaleX;
      const cy = y1 * scaleY;
      el.style.left = cx + "px";
      el.style.top = cy + "px";
      el.dataset.region = itemRegionNum(it) || "";
      el.textContent = it.kind === "entity" ? `◇ ${it.name}` : `⚡ ${it.title}`;
      el.addEventListener("click", () => {
        graphDetail(it);
        const n = itemRegionNum(it);
        if (n) graphHighlightRegion(n);
      });
      ov.appendChild(el);
    });

    // ORDER chain edges (SVG lines between consecutive order regions on this page)
    const orderOnPage = state.graphEdges.filter((i) => i.bbox != null && i.page === page);
    if (orderOnPage.length > 1 && layers.order) {
      let svg = `<svg viewBox="0 0 1000 ${H / scaleY}" width="${W}" height="${H}" style="position:absolute;inset:0;pointer-events:none">`;
      for (let i = 1; i < orderOnPage.length; i++) {
        const a = orderOnPage[i - 1].bbox, b = orderOnPage[i].bbox;
        const x1 = (a[0] + a[2]) / 2 * scaleX, y1 = (a[1] + a[3]) / 2 * scaleY;
        const x2 = (b[0] + b[2]) / 2 * scaleX, y2 = (b[1] + b[3]) / 2 * scaleY;
        svg += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#3fb950" stroke-width="2" stroke-dasharray="6 4" opacity="0.85" marker-end="url(#gEdgeArrow)"/>`;
      }
      svg += `<defs><marker id="gEdgeArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#3fb950"/></marker></defs></svg>`;
      edgesEl.innerHTML = svg;
    }

    // apply a pending graph-node highlight (e.g. after navigating to a page)
    if (state.pendingGraphHighlight) {
      const n = state.pendingGraphHighlight;
      state.pendingGraphHighlight = null;
      graphHighlightRegion(n);
    }
  });
}

function itemRegionNum(it) {
  const r = it.region_id || it.source_region;
  return r ? String(r).split("|").pop() : null;
}

function graphHighlightRegion(num) {
  if (!num) return;
  const ov = $("graph-overlay");
  ov.querySelectorAll(".greg").forEach((el) => {
    el.classList.toggle("g-active", el.dataset.region === num);
  });
  if (state.semNodes) {
    state.semNodes.classed("g-node-active", (d) => d.region_id && String(d.region_id).split("|").pop() === num);
  }
}

function graphClearHighlight() {
  const ov = $("graph-overlay");
  ov.querySelectorAll(".greg.g-active").forEach((el) => el.classList.remove("g-active"));
  if (state.semNodes) state.semNodes.classed("g-node-active", false);
}

function graphNavigateRegion(rid) {
  const num = String(rid).split("|").pop();
  if (!state.graphRegionMap) {
    state.graphRegionMap = new Map();
    (state.graphData ? state.graphData.items : []).forEach((it) => {
      const n = itemRegionNum(it);
      if (n && it.page != null && !state.graphRegionMap.has(n)) state.graphRegionMap.set(n, it.page);
    });
  }
  const page = state.graphRegionMap.get(num);
  if (page == null) return;
  state.pendingGraphHighlight = num;
  if (state.graphPage !== page) {
    state.graphPage = page;
    graphDrawPage();
  } else {
    graphPositionOverlay(state.graphPage);
  }
}

function graphDetail(it) {
  const el = $("graph-detail");
  const lines = [];
  if (it.kind === "block") {
    lines.push(`<span class="ptag" style="--c:#3498db">Qdrant-блок #${escapeHtml(it.id)}</span> <span class="hint">стр. ${(Number(it.page) || 0) + 1} · rel ${escapeHtml(it.relevance || "?")}</span>`);
    lines.push(`<pre>${escapeHtml(it.text || "")}</pre>`);
  } else if (it.kind === "entity") {
    lines.push(`<span class="ptag" style="--c:#7c5cff">Сущность: ${escapeHtml(it.type)}</span> <span class="hint">${escapeHtml(it.name || "")}</span>`);
    lines.push(`<pre>${escapeHtml(it.description || "")}</pre>`);
  } else if (it.kind === "community") {
    lines.push(`<span class="ptag" style="--c:#d29922">Сообщество</span> <span class="hint">${escapeHtml(it.title || "")} · rating ${escapeHtml(it.rating || "?")}</span>`);
    lines.push(`<pre>${escapeHtml(it.summary || "")}</pre>`);
  } else {
    lines.push(`<span class="ptag" style="--c:${it.color}">${escapeHtml(it.source || it.label || "Регион")}</span> <span class="hint">стр. ${(Number(it.page) || 0) + 1}</span>`);
    lines.push(`<pre>${escapeHtml(it.text || "")}</pre>`);
  }
  el.innerHTML = lines.join("");
}

function renderStructuralGraph(ctx) {
  const box = $("struct-graph");
  const items = (state.graphData ? state.graphData.items : [])
    .filter((it) => it.kind === "block" || (it.kind === "region" && it.source));
  const seeds = items.filter((it) => it.kind === "block");
  const order = items.filter((it) => it.kind === "region" && it.source === "order");
  const others = items.filter((it) => it.kind === "region" && it.source !== "order");
  if (!seeds.length && !order.length && !others.length) {
    box.innerHTML = '<p class="hint">Структурные связи не найдены в контексте.</p>';
    return;
  }
  if (!window.d3) {
    box.innerHTML = '<p class="hint">Библиотека D3 не загрузилась.</p>';
    return;
  }
  const d3 = window.d3;
  const W = Math.max(box.clientWidth || 500, 380);
  const H = Math.max(box.clientHeight || 480, 340);

  const regNum = (r) => (r ? String(r).split("|").pop() : null);
  const bboxY = (it) => (it.item && it.item.bbox ? it.item.bbox[1] : 0);

  const nodes = [];
  const idSet = new Set();
  const addNode = (n) => { if (!idSet.has(n.id)) { idSet.add(n.id); nodes.push(n); } };

  addNode({ id: "hub", kind: "hub", label: "контекст", r: 24, color: "#8b96a5" });

  const seedNodes = [];
  seeds.forEach((s, i) => {
    const n = {
      id: "b:" + (s.id || i), kind: "block",
      label: (s.type || "блок") + " #" + (s.id || i + 1),
      r: 19, color: "#3498db", region_id: s.region_id, page: s.page, text: s.text, item: s,
    };
    addNode(n);
    seedNodes.push(n);
  });

  const orderNodes = order.map((o, i) => ({
    id: "o:" + i, kind: "order",
    label: (o.label || "ORDER").slice(0, 14),
    r: 15, color: "#3fb950", region_id: o.region_id || o.source_region,
    source_region: o.source_region, page: o.page, text: o.text, item: o,
  }));
  orderNodes.forEach((n) => addNode(n));

  others.forEach((x, i) => {
    const k = x.source;
    // Normalize BFS crawler sources (order_neighbor / direct_entity / ...) and
    // any unknown source (legacy cross-graph without a source attr) into a
    // connectable kind.
    const norm = k === "parent" ? "parent"
      : (k === "cross_graph" ? "cross_graph" : "bfs_crawler");
    addNode({
      id: "x:" + i, kind: norm, r: 13,
      label: (k || "link").slice(0, 12),
      color: norm === "parent" ? "#8b6bff"
        : (norm === "cross_graph" ? "#9b59b6" : "#1abc9c"),
      region_id: x.region_id || x.source_region, source_region: x.source_region,
      page: x.page, text: x.text, item: x,
    });
  });

  const links = [];
  const linkKey = new Set();
  const addLink = (a, b, cls, strength) => {
    const k = [a.id, b.id].sort().join("|");
    if (a && b && a.id !== b.id && !linkKey.has(k)) {
      linkKey.add(k);
      links.push({ source: a.id, target: b.id, cls: cls || "link", strength: strength == null ? 0.45 : strength });
    }
  };
  const byId = {};
  nodes.forEach((n) => (byId[n.id] = n));

  // seed -> its ORDER / PARENT neighbours (same region)
  seedNodes.forEach((sn) => {
    const snum = regNum(sn.region_id);
    nodes.forEach((n) => {
      if (n.kind === "order" || n.kind === "parent") {
        const nn = regNum(n.source_region);
        if (snum && nn && snum === nn) addLink(sn, n, "link");
      }
    });
  });

  // ORDER reading-order chain (sorted by page, then vertical position)
  orderNodes.sort((a, b) =>
    ((Number(a.page) || 0) - (Number(b.page) || 0)) || (bboxY(a) - bboxY(b)));
  for (let i = 1; i < orderNodes.length; i++) addLink(orderNodes[i - 1], orderNodes[i], "chain", 0.9);

  // cross-graph / BFS regions -> same-page seed, else hub
  nodes.forEach((n) => {
    if (n.kind === "cross_graph" || n.kind === "bfs_crawler") {
      const sp = seedNodes.find((sn) => sn.page != null && n.page != null && Number(sn.page) === Number(n.page));
      addLink(sp || byId.hub, n, "dash");
    }
  });

  // any node that still has no link (e.g. a parent with no matching seed)
  // gets attached to the hub so it is never isolated
  const linked = new Set();
  links.forEach((l) => { linked.add(l.source); linked.add(l.target); });
  nodes.forEach((n) => {
    if (n.id !== byId.hub.id && !linked.has(n.id)) addLink(byId.hub, n, "dash");
  });

  const svg = d3.select(box).html("").append("svg")
    .attr("width", W).attr("height", H).attr("viewBox", [0, 0, W, H])
    .attr("class", "sem-force");
  const linkG = svg.append("g").attr("class", "g-links");
  const nodeG = svg.append("g").attr("class", "g-nodes");

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance(92).strength((l) => l.strength))
    .force("charge", d3.forceManyBody().strength(-280))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collide", d3.forceCollide().radius((d) => d.r + 20));

  const link = linkG.selectAll("line").data(links).join("line")
    .attr("stroke", (l) => l.cls === "chain" ? "#3fb950" : (l.cls === "dash" ? "#9b59b6" : "#4a5570"))
    .attr("stroke-width", (l) => l.cls === "chain" ? 2.4 : 1.2)
    .attr("stroke-dasharray", (l) => l.cls === "dash" ? "5 4" : null)
    .attr("opacity", (l) => l.cls === "chain" ? 0.9 : 0.55);

  const node = nodeG.selectAll("g").data(nodes).join("g")
    .call(d3.drag()
      .on("start", (ev, d) => { if (!ev.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on("end", (ev, d) => { if (!ev.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

  node.each(function (d) {
    const g = d3.select(this);
    g.append("circle").attr("class", "gforce-core")
      .attr("r", d.r)
      .attr("fill", d.kind === "hub" ? "#202a3d" : "#150f38")
      .attr("stroke", d.color).attr("stroke-width", d.kind === "hub" ? 2 : 2);
    if (d.kind === "hub") {
      g.append("text").attr("text-anchor", "middle").attr("dy", 4)
        .attr("fill", "#aab4c5").style("font-size", "13px").style("font-weight", "800").text("☰");
    } else {
      g.append("text").attr("class", "gforce-label")
        .attr("text-anchor", "middle").attr("dy", 4)
        .attr("fill", "#fff").style("font-size", "10px").style("font-weight", "700")
        .text(d.kind === "order" ? String((d.label || "").charAt(0)) : (d.label || "").slice(0, 6));
      g.append("text").attr("class", "gforce-sub")
        .attr("text-anchor", "middle").attr("dy", d.r + 14)
        .attr("fill", "#cdd6e3").style("font-size", "9px")
        .text((d.label || "").slice(0, 14));
    }
  });

  node.append("title").text((d) => {
    const t = (d.item && d.item.text) || d.text || "";
    return `${d.label}${d.page != null ? " · стр. " + (Number(d.page) + 1) : ""}\n${t}`;
  });

  node.on("click", (ev, d) => {
    if (d.kind === "hub") return;
    if (d.item) graphDetail(d.item);
    const r = d.region_id || d.source_region;
    if (r) graphNavigateRegion(r);
  });
  node.on("mouseover", (ev, d) => {
    const r = d.region_id || d.source_region;
    if (r) graphHighlightRegion(String(r).split("|").pop());
  });
  node.on("mouseout", () => graphClearHighlight());

  simulation.on("tick", () => {
    link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
    node.attr("transform", (d) => `translate(${d.x},${d.y})`);
  });

  svg.call(d3.zoom().scaleExtent([0.3, 4]).on("zoom", (ev) => {
    nodeG.attr("transform", ev.transform);
    linkG.attr("transform", ev.transform);
  }));

  state.structNodes = node;
  state.structSim = simulation;
}

function renderSemanticGraph(ctx) {
  const box = $("sem-graph");
  const ents = (ctx.entities || []).slice(0, 14);
  const comms = (ctx.communities || []).slice(0, 6);
  if (!ents.length && !comms.length) {
    box.innerHTML = '<p class="hint">Сущности не найдены в контексте.</p>';
    return;
  }
  if (!window.d3) {
    box.innerHTML = '<p class="hint">Библиотека D3 не загрузилась.</p>';
    return;
  }
  const d3 = window.d3;
  const W = Math.max(box.clientWidth || 500, 380);
  const H = Math.max(box.clientHeight || 480, 340);

  const nodes = [];
  const idSet = new Set();
  const addNode = (n) => { if (!idSet.has(n.id)) { idSet.add(n.id); nodes.push(n); } };
  addNode({ id: "q", kind: "question", label: "?", r: 30, color: "#4f8cff" });
  ents.forEach((e) => addNode({
    id: "e:" + (e.name || "?") + ":" + (e.type || ""), kind: "entity",
    label: e.name || "?", type: e.type || "", r: 17, color: "#7c5cff",
    region_id: e.region_id, description: e.description, score: e.score,
    related: (e.related || "").split(/[,;]/).map((s) => s.trim()).filter(Boolean),
  }));
  comms.forEach((c) => addNode({
    id: "c:" + (c.title || "?"), kind: "community",
    label: c.title || "?", r: 24, color: "#d29922",
    region_id: c.region_id, summary: c.summary, rating: c.rating,
  }));

  const links = [];
  const linkKey = new Set();
  const addLink = (a, b) => {
    const k = [a.id, b.id].sort().join("|");
    if (a.id !== b.id && !linkKey.has(k)) { linkKey.add(k); links.push({ source: a.id, target: b.id }); }
  };
  const q = nodes[0];
  nodes.filter((n) => n.kind === "entity").forEach((en) => addLink(q, en));
  // entity <-> community sharing a region
  nodes.filter((n) => n.kind === "entity").forEach((en) => {
    const er = en.region_id ? String(en.region_id).split("|").pop() : null;
    nodes.filter((n) => n.kind === "community").forEach((cn) => {
      const cr = cn.region_id ? String(cn.region_id).split("|").pop() : null;
      if (er && er === cr) addLink(en, cn);
    });
  });
  // entity <-> entity via the `related` attribute
  nodes.filter((n) => n.kind === "entity").forEach((en) => {
    (en.related || []).forEach((rn) => {
      const other = nodes.find((n) => n.kind === "entity" && n.label.toLowerCase() === rn.toLowerCase());
      if (other) addLink(en, other);
    });
  });

  const svg = d3.select(box).html("").append("svg")
    .attr("width", W).attr("height", H).attr("viewBox", [0, 0, W, H])
    .attr("class", "sem-force");
  const linkG = svg.append("g").attr("class", "g-links");
  const nodeG = svg.append("g").attr("class", "g-nodes");

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance(120).strength(0.5))
    .force("charge", d3.forceManyBody().strength(-340))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collide", d3.forceCollide().radius((d) => d.r + 24));

  const link = linkG.selectAll("line").data(links).join("line")
    .attr("stroke", "#4a5570").attr("stroke-width", 1.2).attr("opacity", 0.6);

  const node = nodeG.selectAll("g").data(nodes).join("g")
    .call(d3.drag()
      .on("start", (ev, d) => { if (!ev.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on("end", (ev, d) => { if (!ev.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

  node.each(function (d) {
    const g = d3.select(this);
    if (d.kind === "community") {
      g.append("rect").attr("class", "gforce-core")
        .attr("x", -58).attr("y", -16).attr("width", 116).attr("height", 32).attr("rx", 16)
        .attr("fill", "#2c1d04").attr("stroke", d.color).attr("stroke-width", 1.8);
      g.append("text").attr("class", "gforce-label")
        .attr("text-anchor", "middle").attr("dy", 4)
        .attr("fill", "#ffe9b3").style("font-size", "10px").style("font-weight", "700")
        .text("⚡ " + (d.label || "").slice(0, 18));
    } else {
      g.append("circle").attr("class", "gforce-core")
        .attr("r", d.r)
        .attr("fill", d.kind === "question" ? "#0d1e42" : "#150f38")
        .attr("stroke", d.color).attr("stroke-width", d.kind === "question" ? 3 : 1.8);
      if (d.kind === "question") {
        g.append("text").attr("text-anchor", "middle").attr("dy", 5)
          .attr("fill", "#fff").style("font-size", "17px").style("font-weight", "800").text("?");
      } else {
        g.append("text").attr("class", "gforce-label")
          .attr("text-anchor", "middle").attr("dy", 5)
          .attr("fill", "#fff").style("font-size", "11px").style("font-weight", "700")
          .text((d.label || "?").charAt(0));
        g.append("text").attr("class", "gforce-sub")
          .attr("text-anchor", "middle").attr("dy", d.r + 14)
          .attr("fill", "#cdd6e3").style("font-size", "9px")
          .text((d.label || "").slice(0, 14));
      }
    }
  });

  node.append("title").text((d) => {
    if (d.kind === "question") return "Вопрос";
    if (d.kind === "community") return `Сообщество: ${d.label} · rating ${d.rating || "?"}\n${d.summary || ""}`;
    return `${d.type}: ${d.label}\n${d.description || ""}`;
  });

  node.on("click", (ev, d) => {
    if (d.kind === "question") return;
    graphDetail({
      kind: d.kind, name: d.label, title: d.label, type: d.type,
      description: d.description, summary: d.summary, rating: d.rating,
      region_id: d.region_id, source: d.kind,
    });
    if (d.region_id) graphNavigateRegion(d.region_id);
  });
  node.on("mouseover", (ev, d) => { if (d.region_id) graphHighlightRegion(String(d.region_id).split("|").pop()); });
  node.on("mouseout", () => graphClearHighlight());

  simulation.on("tick", () => {
    link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
    node.attr("transform", (d) => `translate(${d.x},${d.y})`);
  });

  svg.call(d3.zoom().scaleExtent([0.3, 4]).on("zoom", (ev) => {
    nodeG.attr("transform", ev.transform);
    linkG.attr("transform", ev.transform);
  }));

  state.semNodes = node;
  state.semLinks = link;
  state.semSim = simulation;
}

/* ================= COMPARE ================= */

async function compare() {
  const file_hash = $("cmp-doc").value;
  const question = $("cmp-question").value.trim();
  if (!file_hash || !question) {
    $("cmp-status").textContent = "Выберите документ и введите вопрос.";
    return;
  }
  $("cmp-btn").disabled = true;
  $("cmp-status").textContent = "запуск всех стратегий…";
  animatePipeline();
  try {
    const data = await api("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_hash, question }),
    });
    renderCompare(data.results);
    $("cmp-status").textContent = "готово";
  } catch (e) {
    $("cmp-status").textContent = "ошибка: " + e.message;
  } finally {
    $("cmp-btn").disabled = false;
  }
}

function renderCompare(results) {
  const grid = $("cmp-grid");
  grid.innerHTML = "";
  const completed = results.filter((r) => r.status === "completed");
  const fastest = completed.length
    ? completed.reduce((m, r) => (r.metadata.total_ms < m.metadata.total_ms ? r : m), completed[0])
    : null;
  results.forEach((r) => {
    const src = r.context.sources || {};
    const card = document.createElement("div");
    card.className = "cmp-card" + (r === fastest ? " winner" : "");
    card.innerHTML = `
      <h4>${escapeHtml(r.strategy_label)}</h4>
      <p class="badge">${r.status === "completed" ? "ответ получен" : escapeHtml(r.status)}</p>
      <p class="cmp-desc">${escapeHtml(descOf(r.strategy))}</p>
      <p class="cmp-answer">${escapeHtml(r.answer || "—")}</p>
      <p class="cmp-meta">${timingHtml(r.metadata)}</p>
      <div class="cmp-src">
        <span class="min q">Q ${src.qdrant || 0}</span>
        <span class="min s">S ${src.structural || 0}</span>
        <span class="min m">M ${src.semantic || 0}</span>
      </div>`;
    grid.appendChild(card);
  });
}

$("cmp-btn").addEventListener("click", compare);

/* ================= UPLOAD ================= */

async function upload() {
  const input = $("upload-input");
  const file = input.files && input.files[0];
  if (!file) { $("upload-status").textContent = "Выберите PDF-файл."; return; }
  const fd = new FormData();
  fd.append("file", file);
  $("upload-btn").disabled = true;
  $("upload-status").textContent = "загрузка и обработка (1–3 минуты)…";
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    $("upload-status").textContent = r.status === "ok"
      ? `загружено: ${r.embeddings_computed ?? 0} эмбеддингов`
      : "ошибка: " + (r.message || r.status);
    setTimeout(loadFiles, 1500);
  } catch (e) {
    $("upload-status").textContent = "ошибка: " + e.message;
  } finally {
    $("upload-btn").disabled = false;
  }
}
$("upload-btn").addEventListener("click", upload);

bootstrap();
