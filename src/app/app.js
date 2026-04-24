const statusGrid = document.getElementById("statusGrid");
const resultSummary = document.getElementById("resultSummary");
const queryPreview = document.getElementById("queryPreview");
const resultGrid = document.getElementById("resultGrid");
const statusDock = document.getElementById("statusDock");
const statusTrigger = document.getElementById("statusTrigger");
let activeSearchTab = "gallery";
let runtimeOptions = null;
const indexStatusTimers = {};
const queryPreviewBindings = [
  ["galleryQuerySelect", "galleryQueryImagePreview"],
  ["videoQuerySelect", "videoQueryImagePreview"],
];

function setActiveSearchTab(tabName) {
  activeSearchTab = tabName;
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
  scheduleIndexStatusRefresh(tabName);
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

function normalizeSearchBody(form) {
  const body = readFormAsObject(form);
  if (body.topk) {
    body.topk = Number(body.topk);
  }
  if (body.sample_fps) {
    body.sample_fps = Number(body.sample_fps);
  }
  return body;
}

function normalizeRebuildBody(form) {
  const body = normalizeSearchBody(form);
  delete body.query_path;
  delete body.topk;
  return body;
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data?.detail || "请求失败");
  }
  return data;
}

function optionValue(item) {
  return typeof item === "string" ? item : item?.value || "";
}

function optionLabel(item) {
  if (typeof item === "string") return item;
  return item?.label || item?.value || "";
}

function encodePathForUrl(pathValue) {
  return pathValue
    .split("/")
    .filter(Boolean)
    .map((part) => encodeURIComponent(part))
    .join("/");
}

function queryPathToPreviewUrl(pathValue) {
  const value = String(pathValue || "").trim().replaceAll("\\", "/");
  if (!value) return "";
  if (value.startsWith("/data-runtime-static/")) return value;

  const runtimePrefix = "data_runtime/";
  if (!value.startsWith(runtimePrefix)) return "";

  const runtimeRelativePath = value.slice(runtimePrefix.length);
  if (!runtimeRelativePath) return "";
  return `/data-runtime-static/${encodePathForUrl(runtimeRelativePath)}`;
}

function fileNameFromPath(pathValue) {
  const value = String(pathValue || "").trim().replaceAll("\\", "/");
  const parts = value.split("/").filter(Boolean);
  return parts.length ? parts[parts.length - 1] : "";
}

function renderQueryPreviewPlaceholder(preview, text, isError = false) {
  preview.innerHTML = "";
  const placeholder = document.createElement("div");
  placeholder.className = "query-image-placeholder";
  placeholder.textContent = text;
  placeholder.classList.toggle("is-error", isError);
  preview.appendChild(placeholder);
}

function updateQueryImagePreview(selectId, previewId) {
  const select = document.getElementById(selectId);
  const preview = document.getElementById(previewId);
  if (!select || !preview) return;

  const selectedPath = select.value;
  const previewUrl = queryPathToPreviewUrl(selectedPath);
  if (!previewUrl) {
    renderQueryPreviewPlaceholder(preview, select.disabled ? "暂无查询图片" : "未选择查询图片");
    return;
  }

  preview.innerHTML = "";
  const img = document.createElement("img");
  const fileName = fileNameFromPath(selectedPath);
  img.src = previewUrl;
  img.alt = fileName ? `查询图片预览: ${fileName}` : "查询图片预览";
  img.addEventListener("error", () => {
    renderQueryPreviewPlaceholder(preview, "预览加载失败", true);
  });

  const caption = document.createElement("figcaption");
  caption.textContent = fileName || selectedPath;

  preview.append(img, caption);
}

function updateAllQueryImagePreviews() {
  for (const [selectId, previewId] of queryPreviewBindings) {
    updateQueryImagePreview(selectId, previewId);
  }
}

function initQueryImagePreviews() {
  for (const [selectId, previewId] of queryPreviewBindings) {
    const select = document.getElementById(selectId);
    if (!select) continue;
    select.addEventListener("change", () => updateQueryImagePreview(selectId, previewId));
    select.addEventListener("input", () => updateQueryImagePreview(selectId, previewId));
  }
  updateAllQueryImagePreviews();
}

function fillSelect(selectId, items, emptyText) {
  const select = document.getElementById(selectId);
  if (!select) return;

  const previousValue = select.value;
  select.innerHTML = "";

  if (!items?.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = emptyText;
    select.appendChild(option);
    select.disabled = true;
    return;
  }

  select.disabled = false;
  for (const item of items) {
    const value = optionValue(item);
    if (!value) continue;
    const option = document.createElement("option");
    option.value = value;
    option.textContent = optionLabel(item);
    select.appendChild(option);
  }

  if ([...select.options].some((option) => option.value === previousValue)) {
    select.value = previousValue;
  }
}

async function loadRuntimeOptions() {
  const res = await fetch("/api/runtime/options");
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data?.detail || "文件列表加载失败");
  }

  runtimeOptions = data;
  fillSelect("galleryQuerySelect", data.query_images, "data_runtime/query 下没有查询图片");
  fillSelect("videoQuerySelect", data.query_images, "data_runtime/query 下没有查询图片");
  fillSelect("galleryPathSelect", data.image_galleries, "data_runtime/gallery/images 下没有图库目录");
  fillSelect("videoPathSelect", data.video_galleries, "data_runtime/gallery/videos 下没有视频库目录");
  updateAllQueryImagePreviews();
  scheduleIndexStatusRefresh("gallery");
  scheduleIndexStatusRefresh("video");
  return data;
}

function getSearchForm(tabName) {
  if (tabName === "gallery") {
    return document.getElementById("gallerySearchForm");
  }
  if (tabName === "video") {
    return document.getElementById("videoSearchForm");
  }
  return null;
}

function getIndexStatusElement(tabName) {
  if (tabName === "gallery") {
    return document.getElementById("galleryIndexStatus");
  }
  if (tabName === "video") {
    return document.getElementById("videoIndexStatus");
  }
  return null;
}

function setIndexStatus(tabName, text, state = "unknown") {
  const el = getIndexStatusElement(tabName);
  if (!el) return;
  el.textContent = text;
  el.classList.remove("is-ready", "is-missing", "is-error", "is-unknown");
  el.classList.add(`is-${state}`);
}

function buildIndexStatusBody(tabName) {
  const form = getSearchForm(tabName);
  if (!form) return null;

  const body = {
    feature_mode: form.elements.feature_mode?.value || "face",
    person_model: form.elements.person_model?.value || "resnet",
  };

  const indexName = form.elements.index_name?.value?.trim();
  if (indexName) {
    body.index_name = indexName;
  }

  const galleryPath = form.elements.gallery_path?.value?.trim();
  if (!galleryPath) {
    const label = tabName === "gallery" ? "请选择图库路径" : "请选择视频库路径";
    setIndexStatus(tabName, `索引状态: ${label}`, "unknown");
    return null;
  }
  body.gallery_path = galleryPath;
  return body;
}

function renderIndexStatus(tabName, data) {
  const state = data?.exists ? "ready" : "missing";
  const stateText = data?.exists ? "已建立" : "未建立";
  const indexName = data?.index_name || "-";
  const libraryType = data?.library_type || "-";
  setIndexStatus(tabName, `索引状态: ${stateText} | ${libraryType} / ${indexName}`, state);
}

async function refreshIndexStatus(tabName) {
  const body = buildIndexStatusBody(tabName);
  if (!body) return;

  setIndexStatus(tabName, "索引状态: 检查中...", "unknown");
  try {
    const res = await fetch("/api/index/status", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.detail || "索引状态查询失败");
    }
    renderIndexStatus(tabName, data);
  } catch (error) {
    setIndexStatus(tabName, `索引状态: ${error.message || "查询失败"}`, "error");
  }
}

function scheduleIndexStatusRefresh(tabName = activeSearchTab) {
  clearTimeout(indexStatusTimers[tabName]);
  indexStatusTimers[tabName] = window.setTimeout(() => {
    refreshIndexStatus(tabName);
  }, 250);
}

function initIndexStatusWatchers() {
  for (const tabName of ["gallery", "video"]) {
    const form = getSearchForm(tabName);
    if (!form) continue;
    const watchedNames = ["query_path", "gallery_path", "feature_mode", "person_model", "index_name"];
    for (const name of watchedNames) {
      const field = form.elements[name];
      if (!field) continue;
      field.addEventListener("input", () => scheduleIndexStatusRefresh(tabName));
      field.addEventListener("change", () => scheduleIndexStatusRefresh(tabName));
    }
  }
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

async function buildIndex(tabName) {
  const form = getSearchForm(tabName);
  if (!form) return;
  const messageId = tabName === "gallery" ? "galleryMessage" : "videoMessage";
  const label = tabName === "gallery" ? "图库" : "视频库";
  setMessage(messageId, `正在构建${label}索引...`);
  try {
    const body = normalizeRebuildBody(form);
    const data = await postJson("/api/admin/rebuild-gallery-index", body);
    setMessage(
      messageId,
      `构建完成: ${data.feature_mode} / ${data.index_name} / total_items=${data.total_items}`,
    );
    renderIndexStatus(tabName, { ...data, exists: true });
    await refreshStatus();
  } catch (error) {
    setMessage(messageId, error.message || "构建失败", true);
    await refreshIndexStatus(tabName);
  }
}

async function buildGalleryIndex() {
  return buildIndex("gallery");
}

async function buildVideoIndex() {
  return buildIndex("video");
}

async function submitGallerySearch(event) {
  event.preventDefault();
  setMessage("galleryMessage", "正在检索...");
  try {
    const body = normalizeSearchBody(event.currentTarget);
    const data = await postJson("/api/search/gallery", body);
    setMessage("galleryMessage", `检索完成: ${data.result_count} 条`);
    renderResults(data);
    await refreshIndexStatus("gallery");
    await refreshStatus();
  } catch (error) {
    setMessage("galleryMessage", error.message || "检索失败", true);
  }
}

async function submitVideoSearch(event) {
  event.preventDefault();
  setMessage("videoMessage", "正在检索视频库...");
  try {
    const body = normalizeSearchBody(event.currentTarget);
    const data = await postJson("/api/search/gallery", body);
    setMessage("videoMessage", `检索完成: ${data.result_count} 条`);
    renderResults(data);
    await refreshIndexStatus("video");
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
    resultSummary.textContent = "已清理检索输出";
    queryPreview.innerHTML = "";
    resultGrid.innerHTML = "";
  } catch (error) {
    resultSummary.textContent = error.message || "清理失败";
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  initSearchTabs();
  initStatusDock();
  initIndexStatusWatchers();
  initQueryImagePreviews();
  document.getElementById("refreshStatusBtn")?.addEventListener("click", () => {
    refreshStatus().catch((err) => {
      resultSummary.textContent = err.message || "状态查询失败";
    });
  });
  document.getElementById("refreshOptionsBtn")?.addEventListener("click", () => {
    loadRuntimeOptions()
      .then(() => {
        resultSummary.textContent = "文件列表已刷新";
      })
      .catch((err) => {
        resultSummary.textContent = err.message || "文件列表刷新失败";
      });
  });
  document.getElementById("galleryBuildIndexBtn")?.addEventListener("click", buildGalleryIndex);
  document.getElementById("videoBuildIndexBtn")?.addEventListener("click", buildVideoIndex);
  document.getElementById("galleryRefreshIndexBtn")?.addEventListener("click", () => refreshIndexStatus("gallery"));
  document.getElementById("videoRefreshIndexBtn")?.addEventListener("click", () => refreshIndexStatus("video"));
  document.getElementById("gallerySearchForm")?.addEventListener("submit", submitGallerySearch);
  document.getElementById("videoSearchForm")?.addEventListener("submit", submitVideoSearch);
  document.getElementById("clearOutputsBtn")?.addEventListener("click", clearOutputs);

  try {
    await loadRuntimeOptions();
    await refreshStatus();
  } catch (error) {
    resultSummary.textContent = error.message || "初始化失败";
  }
});
