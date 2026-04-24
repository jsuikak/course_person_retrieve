const statusGrid = document.getElementById("statusGrid");
const resultSummary = document.getElementById("resultSummary");
const queryPreview = document.getElementById("queryPreview");
const resultGrid = document.getElementById("resultGrid");
const statusDock = document.getElementById("statusDock");
const statusTrigger = document.getElementById("statusTrigger");

function setActiveSearchTab(tabName) {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const panes = {
    gallery: document.getElementById("tabPaneGallery"),
    video: document.getElementById("tabPaneVideo"),
  };

  for (const button of tabButtons) {
    const active = button.dataset.tab === tabName;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  }

  for (const [name, pane] of Object.entries(panes)) {
    if (!pane) continue;
    const active = name === tabName;
    pane.classList.toggle("is-active", active);
    pane.hidden = !active;
  }
}

function initSearchTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  for (const button of tabButtons) {
    button.addEventListener("click", () => {
      const targetTab = button.dataset.tab;
      if (!targetTab) return;
      setActiveSearchTab(targetTab);
    });
  }
  setActiveSearchTab("gallery");
}

function setStatusTriggerExpanded(expanded) {
  if (!statusTrigger) return;
  statusTrigger.setAttribute("aria-expanded", expanded ? "true" : "false");
}

function closeStatusDock() {
  if (!statusDock) return;
  statusDock.classList.remove("is-open");
  setStatusTriggerExpanded(false);
}

function initStatusDock() {
  if (!statusDock || !statusTrigger) return;

  statusDock.addEventListener("mouseenter", () => setStatusTriggerExpanded(true));
  statusDock.addEventListener("mouseleave", () => {
    if (!statusDock.classList.contains("is-open")) {
      setStatusTriggerExpanded(false);
    }
  });

  statusDock.addEventListener("focusin", () => setStatusTriggerExpanded(true));
  statusDock.addEventListener("focusout", (event) => {
    const nextTarget = event.relatedTarget;
    if (!nextTarget || !statusDock.contains(nextTarget)) {
      if (!statusDock.classList.contains("is-open")) {
        setStatusTriggerExpanded(false);
      }
    }
  });

  statusTrigger.addEventListener("click", (event) => {
    event.preventDefault();
    const willOpen = !statusDock.classList.contains("is-open");
    statusDock.classList.toggle("is-open", willOpen);
    setStatusTriggerExpanded(willOpen);
  });

  document.addEventListener("click", (event) => {
    const target = event.target;
    if (!target || !statusDock.contains(target)) {
      closeStatusDock();
    }
  });
}

function setMessage(id, text, isError = false) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.classList.remove("ok", "err");
  el.classList.add(isError ? "err" : "ok");
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function readFormAsObject(form) {
  const data = new FormData(form);
  const out = {};
  for (const [key, value] of data.entries()) {
    if (typeof value === "string") {
      const trimmed = value.trim();
      if (trimmed === "") continue;
      out[key] = trimmed;
    }
  }
  return out;
}

function renderStatus(data) {
  const items = [
    ["运行解释器", data?.runtime?.python_executable ?? "unknown"],
    [
      "运行依赖(ultralytics/uvicorn/torchreid)",
      `${data?.runtime?.ultralytics_importable ? "ok" : "missing"}/${data?.runtime?.uvicorn_importable ? "ok" : "missing"}/${data?.runtime?.torchreid_importable ? "ok" : "missing"}`,
    ],
    ["ArcFace 权重", data?.weights?.arcface?.exists ? "存在" : "缺失"],
    ["YOLO 默认权重", data?.weights?.yolo_default?.exists ? "存在" : "缺失"],
    ["图像索引(face/person)", `${data?.indexes?.image?.face_count ?? 0}/${data?.indexes?.image?.person_count ?? 0}`],
    ["视频索引(face/person)", `${data?.indexes?.video?.face_count ?? 0}/${data?.indexes?.video?.person_count ?? 0}`],
    ["默认设备", data?.default_device ?? "unknown"],
  ];
  statusGrid.innerHTML = items
    .map(([k, v]) => `<div class="status-item"><strong>${escapeHtml(k)}</strong>: ${escapeHtml(v)}</div>`)
    .join("");
}

async function refreshStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data?.detail || "状态查询失败");
  }
  renderStatus(data);
}

function renderResults(data) {
  queryPreview.innerHTML = "";
  resultGrid.innerHTML = "";

  const mode = data?.feature_mode || "-";
  const personModel = data?.person_model || "-";
  const libraryType = data?.library_type || "-";
  const indexName = data?.index_name || "-";
  const count = data?.result_count || 0;
  resultSummary.textContent = `mode=${mode} | person_model=${personModel} | library=${libraryType} | index=${indexName} | results=${count}`;

  if (data?.query_url) {
    queryPreview.innerHTML = `
      <strong>Query</strong>
      <img src="${escapeHtml(data.query_url)}" alt="query" />
    `;
  }

  const cards = (data?.results || []).map((item) => {
    const bbox = item?.bbox || {};
    const bboxText = `x=${bbox.x ?? 0}, y=${bbox.y ?? 0}, w=${bbox.w ?? 0}, h=${bbox.h ?? 0}`;
    return `
      <article class="result-card">
        <h3>Rank ${item.rank ?? "-"} | Score ${(item.score ?? 0).toFixed ? item.score.toFixed(4) : item.score}</h3>
        <p><strong>Source:</strong> ${escapeHtml(item.source_name || "")}</p>
        <p><strong>BBox:</strong> ${escapeHtml(bboxText)}</p>
        <p><strong>Frame:</strong> ${escapeHtml(item.frame_index ?? "-")}</p>
        ${item.annotated_url ? `<img src="${escapeHtml(item.annotated_url)}" alt="annotated" />` : ""}
        ${item.crop_url ? `<img src="${escapeHtml(item.crop_url)}" alt="crop" />` : ""}
      </article>
    `;
  });

  resultGrid.innerHTML = cards.join("");
}

async function submitRebuild(event) {
  event.preventDefault();
  setMessage("rebuildMessage", "正在构建索引...");
  try {
    const body = readFormAsObject(event.currentTarget);
    if (body.sample_fps) {
      body.sample_fps = Number(body.sample_fps);
    }
    const res = await fetch("/api/admin/rebuild-gallery-index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.detail || "构建失败");
    }
    setMessage(
      "rebuildMessage",
      `构建完成: ${data.feature_mode} / ${data.index_name} / total_items=${data.total_items}`,
    );
    await refreshStatus();
  } catch (error) {
    setMessage("rebuildMessage", error.message || "构建失败", true);
  }
}

async function submitGallerySearch(event) {
  event.preventDefault();
  setMessage("galleryMessage", "正在检索...");
  try {
    const formData = new FormData(event.currentTarget);
    const res = await fetch("/api/search/gallery", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.detail || "检索失败");
    }
    setMessage("galleryMessage", `检索完成: ${data.result_count} 条`);
    renderResults(data);
    await refreshStatus();
  } catch (error) {
    setMessage("galleryMessage", error.message || "检索失败", true);
  }
}

async function submitVideoSearch(event) {
  event.preventDefault();
  setMessage("videoMessage", "正在检索上传视频...");
  try {
    const formData = new FormData(event.currentTarget);
    const res = await fetch("/api/search/uploaded-video", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.detail || "视频检索失败");
    }
    setMessage("videoMessage", `检索完成: ${data.result_count} 条`);
    renderResults(data);
    await refreshStatus();
  } catch (error) {
    setMessage("videoMessage", error.message || "视频检索失败", true);
  }
}

async function clearOutputs() {
  try {
    const res = await fetch("/api/admin/clear-web-outputs", { method: "DELETE" });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.detail || "清理失败");
    }
    resultSummary.textContent = "已清理 Web 临时输出";
    queryPreview.innerHTML = "";
    resultGrid.innerHTML = "";
  } catch (error) {
    resultSummary.textContent = error.message || "清理失败";
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  initSearchTabs();
  initStatusDock();
  document.getElementById("refreshStatusBtn")?.addEventListener("click", () => {
    refreshStatus().catch((err) => {
      resultSummary.textContent = err.message || "状态查询失败";
    });
  });
  document.getElementById("rebuildForm")?.addEventListener("submit", submitRebuild);
  document.getElementById("gallerySearchForm")?.addEventListener("submit", submitGallerySearch);
  document.getElementById("videoSearchForm")?.addEventListener("submit", submitVideoSearch);
  document.getElementById("clearOutputsBtn")?.addEventListener("click", clearOutputs);

  try {
    await refreshStatus();
  } catch (error) {
    resultSummary.textContent = error.message || "状态查询失败";
  }
});
