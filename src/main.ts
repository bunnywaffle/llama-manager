import {
  createIcons,
  Cpu,
  LayoutDashboard,
  Box,
  Sliders,
  Terminal,
  MessageSquare,
  DownloadCloud,
  HardDrive,
  Play,
  Square,
  RotateCcw,
  Download,
  RefreshCw,
  Search,
  CheckCircle,
  Copy,
  Trash2,
  FolderInput,
  Folder,
  Settings,
  Wrench,
  ChevronRight,
  ChevronLeft,
  Edit2,
  Plus,
  Check,
  X,
  Layers,
  Send,
  Bot,
  User,
  Sparkles,
  Monitor,
  Zap,
  Gauge,
  SlidersHorizontal,
  Boxes,
  Settings2,
  Brain,
} from "lucide";
import { marked } from "marked";
import hljs from "highlight.js";
import "highlight.js/styles/github-dark.css";
import {
  LlamaService,
  events,
  type ServerStatus,
  type ProgressReport,
  type ChatMessage,
  type ServerConfig,
  type AppSettings,
} from "./mygo";
import { showToast, showConfirm, initNativeContextMenu } from "./ui_native";
import {
  type ChatNode,
  type ChatSession,
  type Persona,
  getPersonas,
  saveCustomPersona,
  getPersonaById,
  loadAllSessions,
  saveSession,
  deleteSession,
  getActiveSessionId,
  setActiveSessionId,
  createNewSession,
  addMessageToSession,
  editMessageNode,
  regenerateAssistantNode,
  switchSiblingBranch,
  getActiveConversationPath,
  getSiblingInfo,
} from "./chat_tree";

function renderIcons() {
  createIcons({
    icons: {
      Cpu,
      LayoutDashboard,
      Box,
      Sliders,
      Terminal,
      MessageSquare,
      DownloadCloud,
      HardDrive,
      Play,
      Square,
      RotateCcw,
      Download,
      RefreshCw,
      Search,
      CheckCircle,
      Copy,
      Trash2,
      FolderInput,
      Folder,
      Settings,
      Wrench,
      ChevronRight,
      ChevronLeft,
      Edit2,
      Plus,
      Check,
      X,
      Layers,
      Send,
      Bot,
      User,
      Sparkles,
      Monitor,
      Zap,
      Gauge,
      SlidersHorizontal,
      Boxes,
      Settings2,
      Brain,
    },
  });
}

// Helpers
const $ = <T extends HTMLElement>(sel: string) => document.querySelector<T>(sel)!;
const $$ = (sel: string) => document.querySelectorAll<HTMLElement>(sel);

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function getExtensionForLang(lang: string): string {
  const map: Record<string, string> = {
    python: "py",
    py: "py",
    javascript: "js",
    js: "js",
    typescript: "ts",
    ts: "ts",
    golang: "go",
    go: "go",
    rust: "rs",
    rs: "rs",
    c: "c",
    cpp: "cpp",
    cplusplus: "cpp",
    csharp: "cs",
    cs: "cs",
    java: "java",
    html: "html",
    css: "css",
    scss: "scss",
    json: "json",
    yaml: "yaml",
    yml: "yml",
    xml: "xml",
    markdown: "md",
    md: "md",
    sql: "sql",
    shell: "sh",
    bash: "sh",
    sh: "sh",
    powershell: "ps1",
    ps1: "ps1",
    dockerfile: "dockerfile",
    php: "php",
    ruby: "rb",
    rb: "rb",
    lua: "lua",
    r: "r",
    swift: "swift",
    kotlin: "kt",
    kt: "kt",
    toml: "toml",
    ini: "ini",
    bat: "bat",
    cmd: "cmd",
  };
  return map[lang.toLowerCase()] || "txt";
}

function downloadSnippet(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

marked.setOptions({
  gfm: true,
  breaks: true,
});

marked.use({
  renderer: {
    code({ text, lang }) {
      const language = (lang || "").toLowerCase().trim();
      const cleanText = (text || "").replace(/^\s*[\r\n]+|[\r\n]+\s*$/g, "");
      let highlighted = "";
      if (language && hljs.getLanguage(language)) {
        try {
          highlighted = hljs.highlight(cleanText, { language }).value;
        } catch {
          highlighted = escapeHtml(cleanText);
        }
      } else {
        try {
          highlighted = hljs.highlightAuto(cleanText).value;
        } catch {
          highlighted = escapeHtml(cleanText);
        }
      }

      const displayLang = language || "code";
      const ext = getExtensionForLang(displayLang);
      const defaultFilename = `snippet.${ext}`;

      return `<div class="code-block-container" data-lang="${displayLang}"><div class="code-block-header"><span class="code-block-lang">${displayLang}</span><div class="code-block-actions"><button type="button" class="btn-code-copy" title="Copy code snippet"><i data-lucide="copy"></i><span>Copy</span></button><button type="button" class="btn-code-save" data-filename="${defaultFilename}" title="Save code to file"><i data-lucide="download"></i><span>Save</span></button></div></div><pre><code class="hljs ${language ? `language-${language}` : ""}">${highlighted}</code></pre></div>`;
    },
  },
});

function renderMarkdown(raw: string): string {
  if (!raw) return "";
  try {
    return marked.parse(raw, { async: false }) as string;
  } catch (err) {
    return escapeHtml(raw);
  }
}

let isServerRunning = false;
let currentServerHost = "127.0.0.1";
let currentServerPort = 8080;
let currentServerStatus: ServerStatus | null = null;
let chatHistory: ChatMessage[] = [];

interface AttachedImage {
  dataUrl: string;
  name: string;
  sizeFormatted: string;
}
let pendingChatImages: AttachedImage[] = [];

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  const kb = b / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
}

function isVisionActive(): boolean {
  if (isServerRunning) {
    return Boolean(currentServerStatus?.hasVision);
  }
  const mmVal = document.querySelector<HTMLInputElement>("#cfg-mmproj")?.value.trim();
  return Boolean(mmVal);
}

function updateVisionUI() {
  const active = isVisionActive();
  const attachBtn = document.querySelector<HTMLButtonElement>("#btn-chat-attach-image");
  const visionBadge = document.querySelector<HTMLElement>("#chat-vision-badge");
  const visionHint = document.querySelector<HTMLElement>("#chat-vision-hint");
  const chatInput = document.querySelector<HTMLTextAreaElement>("#chat-input");

  if (attachBtn) {
    attachBtn.style.display = active ? "inline-flex" : "none";
  }
  if (visionBadge) {
    visionBadge.style.display = active ? "inline-flex" : "none";
  }
  if (visionHint) {
    visionHint.style.display = active ? "inline-flex" : "none";
  }
  if (chatInput) {
    if (active) {
      chatInput.placeholder = "Send a message or ask about your image... (Enter to send, Shift+Enter for newline)";
    } else {
      chatInput.placeholder = "Send a message to your model... (Enter to send, Shift+Enter for newline)";
      if (pendingChatImages.length > 0) {
        pendingChatImages = [];
        renderChatImagePreviews();
      }
    }
  }
  renderIcons();
}

function renderChatImagePreviews() {
  const container = document.querySelector<HTMLElement>("#chat-image-previews");
  if (!container) return;

  if (pendingChatImages.length === 0) {
    container.style.display = "none";
    container.innerHTML = "";
    return;
  }

  container.style.display = "flex";
  container.innerHTML = "";

  pendingChatImages.forEach((img, idx) => {
    const card = document.createElement("div");
    card.className = "chat-image-thumb-card";

    const thumb = document.createElement("img");
    thumb.src = img.dataUrl;
    thumb.alt = img.name;
    thumb.className = "chat-image-thumb";

    const info = document.createElement("div");
    info.className = "chat-image-thumb-info";

    const nameSpan = document.createElement("span");
    nameSpan.className = "chat-image-thumb-name";
    nameSpan.textContent = img.name;

    const sizeSpan = document.createElement("span");
    sizeSpan.className = "chat-image-thumb-size";
    sizeSpan.textContent = img.sizeFormatted;

    info.appendChild(nameSpan);
    info.appendChild(sizeSpan);

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "chat-image-thumb-remove";
    removeBtn.title = "Remove image";
    removeBtn.innerHTML = `<i data-lucide="x"></i>`;
    removeBtn.addEventListener("click", () => {
      pendingChatImages.splice(idx, 1);
      renderChatImagePreviews();
    });

    card.appendChild(thumb);
    card.appendChild(info);
    card.appendChild(removeBtn);
    container.appendChild(card);
  });

  renderIcons();
}

function addChatImageFile(file: File) {
  if (!file.type.startsWith("image/")) {
    showToast("Please select a valid image file (PNG, JPEG, WebP, GIF, BMP).", "warning", "Invalid File");
    return;
  }
  if (!isVisionActive()) {
    showToast("Vision adaptor (--mmproj) is not attached. Attach a vision projector in Configuration to input images.", "warning", "Vision Adaptor Required");
    return;
  }
  if (pendingChatImages.length >= 4) {
    showToast("Maximum of 4 images can be attached per prompt.", "warning", "Limit Reached");
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target?.result as string;
    pendingChatImages.push({
      dataUrl,
      name: file.name,
      sizeFormatted: formatBytes(file.size),
    });
    renderChatImagePreviews();
  };
  reader.readAsDataURL(file);
}

function openImageModal(src: string) {
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.style.cssText = "display: flex; align-items: center; justify-content: center; z-index: 9999; background: rgba(0,0,0,0.85); cursor: zoom-out;";
  overlay.innerHTML = `
    <div style="position: relative; max-width: 90vw; max-height: 90vh;">
      <img src="${src}" style="max-width: 90vw; max-height: 90vh; border-radius: 8px; box-shadow: 0 10px 40px rgba(0,0,0,0.5); object-fit: contain;" />
      <button style="position: absolute; top: -14px; right: -14px; background: #ef4444; color: #fff; border: none; border-radius: 50%; width: 28px; height: 28px; cursor: pointer; font-size: 16px; font-weight: bold; display: flex; align-items: center; justify-content: center;">&times;</button>
    </div>
  `;
  overlay.addEventListener("click", () => overlay.remove());
  document.body.appendChild(overlay);
}

async function init() {
  renderIcons();
  setupTabs();
  setupSliders();
  setupEvents();
  setupChat();
  setupActions();
  setupInlineTest();
  initNativeContextMenu();
  setupSettings();
  setupTroubleshoot();

  await Promise.all([
    loadHardware(),
    loadBackendStatus(),
    loadModels(),
    loadConfig(),
    loadServerStatus(),
    loadLogs(),
    loadSettings(),
  ]);

  renderIcons();
}

// Tab navigation
function setupTabs() {
  $$(".nav-link").forEach((item) => {
    item.addEventListener("click", () => {
      const tabId = item.getAttribute("data-tab");
      if (!tabId) return;

      $$(".nav-link").forEach((i) => i.classList.remove("active"));
      item.classList.add("active");

      $$(".pane-view").forEach((p) => p.classList.remove("active"));
      const target = $(`#pane-${tabId}`);
      if (target) target.classList.add("active");

      const titles: Record<string, string> = {
        dashboard: "Overview",
        models: "Models",
        config: "Configuration",
        logs: "Console Logs",
        chat: "Playground",
        updates: "Releases",
        settings: "Settings",
      };
      $("#page-title").textContent = titles[tabId] || "llama.cpp";

      if (tabId === "updates") loadReleases();
      if (tabId === "models") loadModels();
      if (tabId === "settings") loadSettings();

      renderIcons();
    });
  });

  $("#btn-go-to-releases")?.addEventListener("click", () => {
    const relNav = document.querySelector<HTMLElement>('[data-tab="updates"]');
    relNav?.click();
  });
}


function setupSliders() {
  const gpuSlider = $<HTMLInputElement>("#cfg-gpu-layers");
  const gpuVal = $("#cfg-gpu-layers-val");
  gpuSlider.addEventListener("input", () => {
    gpuVal.textContent = gpuSlider.value;
  });

  const threadSlider = $<HTMLInputElement>("#cfg-threads");
  const threadVal = $("#cfg-threads-val");
  threadSlider.addEventListener("input", () => {
    threadVal.textContent = threadSlider.value;
  });
}

async function loadHardware() {
  try {
    const hw = await LlamaService.getHardware();
    const platEl = $("#hw-platform-tag");
    if (platEl) platEl.textContent = hw.platformDisplay || `${hw.os} (${hw.arch})`;

    $("#hw-gpu-name").textContent = hw.gpuName || "CPU Only";
    $("#hw-backend-tag").textContent = (hw.recommendedBackend || "cpu").toUpperCase();
    $("#hw-vram").textContent = hw.vram || "--";
    $("#dash-backend-val").textContent = `${(hw.recommendedBackend || "cpu").toUpperCase()} (${hw.gpuName || "CPU"})`;

    if (hw.threads) {
      const rec = Math.max(1, Math.min(16, Math.floor(hw.threads / 2)));
      $<HTMLInputElement>("#cfg-threads").value = String(rec);
      $("#cfg-threads-val").textContent = String(rec);
    }

    // Dynamic backend selection cards based on detected OS
    const isMac = hw.os === "darwin";
    const isLinux = hw.os === "linux";
    const cardMetal = document.getElementById("card-backend-metal");
    const cardCuda = document.getElementById("card-backend-cuda");
    const cardRocm = document.getElementById("card-backend-rocm");
    const cardVulkan = document.getElementById("card-backend-vulkan");

    if (cardMetal) cardMetal.style.display = isMac ? "flex" : "none";
    if (cardCuda) cardCuda.style.display = isMac ? "none" : "flex";
    if (cardRocm) cardRocm.style.display = isLinux || hw.hasRocm ? "flex" : "none";
    if (cardVulkan) cardVulkan.style.display = isMac ? "none" : "flex";
  } catch (err) {
    console.error("Hardware detection error:", err);
  }
}

async function loadBackendStatus() {
  try {
    const b = await LlamaService.getInstalledBackend();
    const alert = $("#backend-missing-alert");
    const badge = $("#backend-status-badge");
    const verText = $("#backend-version-text");
    const varText = $("#backend-variant-text");
    const uninstallBtn = document.querySelector<HTMLElement>("#btn-dash-uninstall-backend");

    if (b.installed) {
      alert.style.display = "none";
      badge.textContent = "Installed";
      badge.className = "tag-badge accent";
      verText.textContent = b.version || "Installed";
      varText.textContent = (b.variant || "cuda").toUpperCase();
      $("#dash-backend-val").textContent = `${(b.variant || "cuda").toUpperCase()} (${b.version || "Installed"})`;
      if (uninstallBtn) uninstallBtn.style.display = "inline-flex";
    } else {
      alert.style.display = "flex";
      badge.textContent = "Missing";
      badge.className = "tag-badge";
      verText.textContent = "None";
      varText.textContent = "Uninstalled";
      $("#dash-backend-val").textContent = "None (Uninstalled)";
      if (uninstallBtn) uninstallBtn.style.display = "none";
    }
  } catch (err) {
    console.error("Backend status error:", err);
  }
}

async function loadModels() {
  try {
    const models = await LlamaService.listModels();
    const quickSelect = $<HTMLSelectElement>("#quick-model-select");
    const tbody = $("#local-models-tbody");

    quickSelect.innerHTML = "";
    tbody.innerHTML = "";

    if (!models || models.length === 0) {
      quickSelect.innerHTML = `<option value="">(No models found in models/ folder)</option>`;
      tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-low); padding: 20px;">No models in models/ directory. Place .gguf files there or download below.</td></tr>`;
      return;
    }

    models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.path;
      opt.textContent = `${m.name} [${m.quant}] (${m.sizeFormatted})${m.isExternal ? " [Symlink]" : ""}`;
      quickSelect.appendChild(opt);

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight: 500; font-family: var(--font-mono); font-size: 12px;" title="${m.path}">
          ${m.name}
          ${m.isExternal ? '<span class="tag-badge accent" style="font-size: 10px; margin-left: 6px;" title="Symbolic link (0 extra disk space used)">Symlink</span>' : ''}
        </td>
        <td><span class="tag-badge">${m.quant}</span></td>
        <td style="font-family: var(--font-mono);">${m.sizeFormatted}</td>
        <td style="color: var(--text-low); font-size: 11.5px;">${m.modifiedAt}</td>
        <td style="text-align: right;">
          <div style="display: inline-flex; gap: 6px;">
            <button class="btn btn-primary btn-sm btn-run-model" data-path="${m.path}" title="Load and start server with this model">
              <i data-lucide="play"></i>
              <span>Run</span>
            </button>
            <button class="btn btn-secondary btn-sm btn-sel-model" data-path="${m.path}">Select</button>
            <button class="btn btn-danger btn-sm btn-del-model" data-name="${m.name}" data-path="${m.path}" data-external="${m.isExternal ? "true" : "false"}">
              ${m.isExternal ? "Unlink" : "Delete"}
            </button>
          </div>
        </td>
      `;
      tbody.appendChild(tr);
    });

    if (!quickSelect.dataset.listenerAttached) {
      quickSelect.dataset.listenerAttached = "true";
      quickSelect.addEventListener("change", async () => {
        const selectedPath = quickSelect.value;
        if (selectedPath) {
          await loadConfigForModel(selectedPath);
        } else {
          updateConfigModelBadge("");
        }
      });
    }

    $$(".btn-run-model").forEach((b) => {
      b.addEventListener("click", async () => {
        const p = b.getAttribute("data-path");
        if (p) {
          quickSelect.value = p;
          await loadConfigForModel(p);
          const nav = document.querySelector<HTMLElement>('[data-tab="dashboard"]');
          nav?.click();
          await startServerHandler();
        }
      });
    });

    $$(".btn-sel-model").forEach((b) => {
      b.addEventListener("click", async () => {
        const p = b.getAttribute("data-path");
        if (p) {
          quickSelect.value = p;
          await loadConfigForModel(p);
          const nav = document.querySelector<HTMLElement>('[data-tab="dashboard"]');
          nav?.click();
        }
      });
    });

    const defSelect = document.querySelector<HTMLSelectElement>("#settings-default-model-select");
    if (defSelect) {
      const cur = defSelect.value;
      defSelect.innerHTML = `<option value="">(Auto-select first available model)</option>`;
      models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.path;
        opt.textContent = `${m.name} [${m.quant}] (${m.sizeFormatted})${m.isExternal ? " [Linked]" : ""}`;
        defSelect.appendChild(opt);
      });
      if (cur) defSelect.value = cur;
    }

    try {
      const s = await LlamaService.getSettings();
      if (defSelect && s.defaultModel) {
        defSelect.value = s.defaultModel;
      }
      if (s.defaultModel && Array.from(quickSelect.options).some((o) => o.value === s.defaultModel)) {
        quickSelect.value = s.defaultModel;
      }
    } catch {}

    if (quickSelect.value) {
      await loadConfigForModel(quickSelect.value);
    }

    const routerSelect = document.querySelector<HTMLSelectElement>("#chat-router-model-select");
    if (routerSelect) {
      const cur = routerSelect.value;
      routerSelect.innerHTML = "";
      models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.name;
        opt.textContent = `${m.name} [${m.quant}]`;
        routerSelect.appendChild(opt);
      });
      if (cur) routerSelect.value = cur;
    }

    $$(".btn-del-model").forEach((b) => {
      b.addEventListener("click", async () => {
        const name = b.getAttribute("data-name");
        const path = b.getAttribute("data-path") || name;
        const isExternal = b.getAttribute("data-external") === "true";
        if (!name) return;

        if (isServerRunning) {
          showToast("Please stop the running server before deleting or unlinking models.", "warning", "Server Running");
          return;
        }

        const title = isExternal ? "Unlink Model" : "Delete Local Model";
        const msg = isExternal
          ? `Remove "${name}" from llama.cpp Manager? (Your original file on disk will NOT be deleted).`
          : `Are you sure you want to permanently delete "${name}" from disk?`;

        const confirmed = await showConfirm(title, msg, isExternal ? "Unlink Model" : "Delete Model", true);
        if (confirmed) {
          try {
            await LlamaService.deleteModel(path || name);
            showToast(isExternal ? `Unlinked model ${name}` : `Deleted model ${name}`, "info");
            await loadModels();
          } catch (err: any) {
            showToast(`Delete failed: ${err.message || err}`, "error", "Delete Failed");
          }
        }
      });
    });

    renderIcons();
  } catch (err) {
    console.error("Load models error:", err);
  }
}


function updateConfigModelBadge(modelPath?: string) {
  const badgeText = document.querySelector<HTMLElement>("#config-model-badge-text");
  const hintText = document.querySelector<HTMLElement>("#config-save-hint");
  if (!badgeText) return;

  if (modelPath) {
    const baseName = modelPath.split(/[/\\]/).pop()?.replace(/\.gguf$/i, "") || "Model";
    badgeText.textContent = `${baseName}.json`;
    if (hintText) {
      hintText.textContent = `Saves to ${baseName}.json alongside model.`;
    }
  } else {
    badgeText.textContent = "Global Default";
    if (hintText) {
      hintText.textContent = "Saves to default server configuration.";
    }
  }
}

function collectConfigFromUI(overrideModelPath?: string): ServerConfig {
  const quickSelect = $<HTMLSelectElement>("#quick-model-select");
  return {
    modelPath: overrideModelPath !== undefined ? overrideModelPath : (quickSelect.value || ""),
    host: $<HTMLInputElement>("#cfg-host").value || "127.0.0.1",
    port: Number($<HTMLInputElement>("#quick-port").value || 8080),
    gpuLayers: Number($<HTMLInputElement>("#cfg-gpu-layers").value || 99),
    contextSize: Number($<HTMLSelectElement>("#cfg-context").value || 4096),
    threads: Number($<HTMLInputElement>("#cfg-threads").value || 6),
    batchSize: Number($<HTMLInputElement>("#cfg-batch")?.value || 512),
    flashAttention: $<HTMLInputElement>("#cfg-flash-attn").checked,
    extraArgs: $<HTMLInputElement>("#cfg-extra-args").value,

    // Advanced Sampling
    temperature: Number($<HTMLInputElement>("#cfg-temp").value || 0.70),
    topP: Number($<HTMLInputElement>("#cfg-top-p").value || 0.90),
    topK: Number($<HTMLInputElement>("#cfg-top-k").value || 40),
    minP: Number($<HTMLInputElement>("#cfg-min-p").value || 0.05),
    typicalP: Number($<HTMLInputElement>("#cfg-typical").value || 1.00),
    repeatPenalty: Number($<HTMLInputElement>("#cfg-repeat-penalty").value || 1.10),
    repeatLastN: Number($<HTMLInputElement>("#cfg-repeat-last-n").value || 64),
    presencePenalty: Number($<HTMLInputElement>("#cfg-presence-penalty").value || 0.0),
    frequencyPenalty: Number($<HTMLInputElement>("#cfg-frequency-penalty").value || 0.0),
    maxPredict: Number($<HTMLInputElement>("#cfg-max-predict").value || -1),
    seed: Number($<HTMLInputElement>("#cfg-seed").value || -1),

    // Memory & KV Cache
    microBatchSize: Number($<HTMLInputElement>("#cfg-micro-batch").value || 512),
    cacheTypeK: $<HTMLSelectElement>("#cfg-cache-type-k").value || "f16",
    cacheTypeV: $<HTMLSelectElement>("#cfg-cache-type-v").value || "f16",
    mlock: $<HTMLInputElement>("#cfg-mlock").checked,
    noMmap: $<HTMLInputElement>("#cfg-no-mmap").checked,

    // Concurrency & Server
    parallelSlots: Number($<HTMLInputElement>("#cfg-parallel-slots").value || 1),
    contBatching: $<HTMLInputElement>("#cfg-cont-batching").checked,
    threadsHttp: Number($<HTMLInputElement>("#cfg-threads-http").value || 0),
    threadsBatch: Number($<HTMLInputElement>("#cfg-threads-batch").value || 0),
    embedding: $<HTMLInputElement>("#cfg-embedding").checked,
    metrics: $<HTMLInputElement>("#cfg-metrics").checked,

    // Multi-GPU & RoPE
    mainGpu: Number($<HTMLInputElement>("#cfg-main-gpu").value || 0),
    splitMode: $<HTMLSelectElement>("#cfg-split-mode").value || "none",
    ropeScaling: $<HTMLSelectElement>("#cfg-rope-scaling").value || "none",
    ropeScale: Number($<HTMLInputElement>("#cfg-rope-scale").value || 1.0),
    ropeFreqBase: Number($<HTMLInputElement>("#cfg-rope-freq-base").value || 0),

    // Router Mode
    routerMode: $<HTMLInputElement>("#cfg-router-mode")?.checked ?? false,
    modelsDir: $<HTMLInputElement>("#settings-models-dir")?.value || "",

    // Vision & Multimodal Projector
    mmprojPath: $<HTMLInputElement>("#cfg-mmproj")?.value || "",

    // Speculative Decoding & Drafter Model (MTP)
    draftModelPath: $<HTMLInputElement>("#cfg-draft-model")?.value || "",
    draftMaxTokens: Number($<HTMLInputElement>("#cfg-draft-tokens")?.value || 4),
    draftGpuLayers: Number($<HTMLInputElement>("#cfg-draft-layers")?.value || 99),
    draftSpecType: $<HTMLSelectElement>("#cfg-draft-type")?.value || "none",

    // Reasoning & Thinking Settings
    reasoningFormat: $<HTMLSelectElement>("#cfg-reasoning-format")?.value || "deepseek-legacy",
    reasoningEffort: $<HTMLSelectElement>("#cfg-reasoning-effort")?.value || "default",
    enableJinja: $<HTMLInputElement>("#cfg-enable-jinja")?.checked ?? true,
  };
}

function updateLinkedModelsBadge() {
  const meta = document.querySelector<HTMLElement>("#dash-linked-meta");
  if (!meta) return;
  const mm = document.querySelector<HTMLInputElement>("#cfg-mmproj")?.value.trim();
  const dm = document.querySelector<HTMLInputElement>("#cfg-draft-model")?.value.trim();

  if (mm && dm) {
    meta.textContent = "Vision + Drafter (MTP) active";
    meta.style.color = "var(--text-high)";
  } else if (mm) {
    meta.textContent = "Vision active";
    meta.style.color = "var(--text-high)";
  } else if (dm) {
    meta.textContent = "Drafter (MTP) active";
    meta.style.color = "var(--text-high)";
  } else {
    meta.textContent = "None active";
    meta.style.color = "var(--text-low)";
  }
  updateVisionUI();
}

function updateRouterModeUI(enabled: boolean) {
  const routerWrap = document.querySelector<HTMLElement>("#chat-router-model-wrap");
  if (routerWrap) {
    routerWrap.style.display = enabled ? "flex" : "none";
  }
}

function applyConfigToUI(cfg: ServerConfig) {
  if (cfg.host) {
    const hostEl = $<HTMLInputElement>("#cfg-host");
    if (hostEl) hostEl.value = cfg.host;
    currentServerHost = cfg.host;
  }
  if (cfg.port) {
    const portEl = $<HTMLInputElement>("#cfg-port");
    if (portEl) portEl.value = String(cfg.port);
    const qPort = $<HTMLInputElement>("#quick-port");
    if (qPort) qPort.value = String(cfg.port);
    currentServerPort = cfg.port;
  }
  if (cfg.gpuLayers !== undefined) {
    const gpuLayers = $<HTMLInputElement>("#cfg-gpu-layers");
    if (gpuLayers) gpuLayers.value = String(cfg.gpuLayers);
    const badge = $("#cfg-gpu-layers-val");
    if (badge) badge.textContent = String(cfg.gpuLayers);
  }
  if (cfg.contextSize) {
    const ctx = $<HTMLSelectElement>("#cfg-context");
    if (ctx) ctx.value = String(cfg.contextSize);
  }
  if (cfg.threads !== undefined) {
    const threads = $<HTMLInputElement>("#cfg-threads");
    if (threads) threads.value = String(cfg.threads);
    const tBadge = $("#cfg-threads-val");
    if (tBadge) tBadge.textContent = String(cfg.threads);
  }
  if (cfg.batchSize !== undefined) {
    const batchInput = $<HTMLInputElement>("#cfg-batch");
    if (batchInput) batchInput.value = String(cfg.batchSize);
  }
  if (cfg.flashAttention !== undefined) {
    const fa = $<HTMLInputElement>("#cfg-flash-attn");
    if (fa) fa.checked = cfg.flashAttention;
  }
  if (cfg.extraArgs !== undefined) {
    const extra = $<HTMLInputElement>("#cfg-extra-args");
    if (extra) extra.value = cfg.extraArgs;
  }

  const routerCheck = $<HTMLInputElement>("#cfg-router-mode");
  if (routerCheck && cfg.routerMode !== undefined) {
    routerCheck.checked = cfg.routerMode;
    updateRouterModeUI(cfg.routerMode);
  }

  // Advanced Sampling
  if (cfg.temperature !== undefined) {
    const el = $<HTMLInputElement>("#cfg-temp");
    if (el) el.value = String(cfg.temperature);
  }
  if (cfg.topP !== undefined) {
    const el = $<HTMLInputElement>("#cfg-top-p");
    if (el) el.value = String(cfg.topP);
  }
  if (cfg.topK !== undefined) {
    const el = $<HTMLInputElement>("#cfg-top-k");
    if (el) el.value = String(cfg.topK);
  }
  if (cfg.minP !== undefined) {
    const el = $<HTMLInputElement>("#cfg-min-p");
    if (el) el.value = String(cfg.minP);
  }
  if (cfg.typicalP !== undefined) {
    const el = $<HTMLInputElement>("#cfg-typical");
    if (el) el.value = String(cfg.typicalP);
  }
  if (cfg.repeatPenalty !== undefined) {
    const el = $<HTMLInputElement>("#cfg-repeat-penalty");
    if (el) el.value = String(cfg.repeatPenalty);
  }
  if (cfg.repeatLastN !== undefined) {
    const el = $<HTMLInputElement>("#cfg-repeat-last-n");
    if (el) el.value = String(cfg.repeatLastN);
  }
  if (cfg.presencePenalty !== undefined) {
    const el = $<HTMLInputElement>("#cfg-presence-penalty");
    if (el) el.value = String(cfg.presencePenalty);
  }
  if (cfg.frequencyPenalty !== undefined) {
    const el = $<HTMLInputElement>("#cfg-frequency-penalty");
    if (el) el.value = String(cfg.frequencyPenalty);
  }
  if (cfg.maxPredict !== undefined) {
    const el = $<HTMLInputElement>("#cfg-max-predict");
    if (el) el.value = String(cfg.maxPredict);
  }
  if (cfg.seed !== undefined) {
    const el = $<HTMLInputElement>("#cfg-seed");
    if (el) el.value = String(cfg.seed);
  }

  // Memory & KV Cache
  if (cfg.microBatchSize !== undefined) {
    const el = $<HTMLInputElement>("#cfg-micro-batch");
    if (el) el.value = String(cfg.microBatchSize);
  }
  if (cfg.cacheTypeK) {
    const el = $<HTMLSelectElement>("#cfg-cache-type-k");
    if (el) el.value = cfg.cacheTypeK;
  }
  if (cfg.cacheTypeV) {
    const el = $<HTMLSelectElement>("#cfg-cache-type-v");
    if (el) el.value = cfg.cacheTypeV;
  }
  if (cfg.mlock !== undefined) {
    const el = $<HTMLInputElement>("#cfg-mlock");
    if (el) el.checked = cfg.mlock;
  }
  if (cfg.noMmap !== undefined) {
    const el = $<HTMLInputElement>("#cfg-no-mmap");
    if (el) el.checked = cfg.noMmap;
  }

  // Concurrency & Server
  if (cfg.parallelSlots !== undefined) {
    const el = $<HTMLInputElement>("#cfg-parallel-slots");
    if (el) el.value = String(cfg.parallelSlots);
  }
  if (cfg.contBatching !== undefined) {
    const el = $<HTMLInputElement>("#cfg-cont-batching");
    if (el) el.checked = cfg.contBatching;
  }
  if (cfg.threadsHttp !== undefined) {
    const el = $<HTMLInputElement>("#cfg-threads-http");
    if (el) el.value = String(cfg.threadsHttp);
  }
  if (cfg.threadsBatch !== undefined) {
    const el = $<HTMLInputElement>("#cfg-threads-batch");
    if (el) el.value = String(cfg.threadsBatch);
  }
  if (cfg.embedding !== undefined) {
    const el = $<HTMLInputElement>("#cfg-embedding");
    if (el) el.checked = cfg.embedding;
  }
  if (cfg.metrics !== undefined) {
    const el = $<HTMLInputElement>("#cfg-metrics");
    if (el) el.checked = cfg.metrics;
  }

  // Multi-GPU & RoPE
  if (cfg.mainGpu !== undefined) {
    const el = $<HTMLInputElement>("#cfg-main-gpu");
    if (el) el.value = String(cfg.mainGpu);
  }
  if (cfg.splitMode) {
    const el = $<HTMLSelectElement>("#cfg-split-mode");
    if (el) el.value = cfg.splitMode;
  }
  if (cfg.ropeScaling) {
    const el = $<HTMLSelectElement>("#cfg-rope-scaling");
    if (el) el.value = cfg.ropeScaling;
  }
  if (cfg.ropeScale !== undefined) {
    const el = $<HTMLInputElement>("#cfg-rope-scale");
    if (el) el.value = String(cfg.ropeScale);
  }
  if (cfg.ropeFreqBase !== undefined) {
    const el = $<HTMLInputElement>("#cfg-rope-freq-base");
    if (el) el.value = String(cfg.ropeFreqBase);
  }

  // Vision Projector & Drafter Model (MTP)
  const mmprojInput = document.querySelector<HTMLInputElement>("#cfg-mmproj");
  if (mmprojInput && cfg.mmprojPath !== undefined) mmprojInput.value = cfg.mmprojPath;

  const draftInput = document.querySelector<HTMLInputElement>("#cfg-draft-model");
  if (draftInput && cfg.draftModelPath !== undefined) draftInput.value = cfg.draftModelPath;

  const draftTokens = document.querySelector<HTMLInputElement>("#cfg-draft-tokens");
  if (draftTokens && cfg.draftMaxTokens !== undefined) draftTokens.value = String(cfg.draftMaxTokens);

  const draftLayers = document.querySelector<HTMLInputElement>("#cfg-draft-layers");
  if (draftLayers && cfg.draftGpuLayers !== undefined) draftLayers.value = String(cfg.draftGpuLayers);

  const draftType = document.querySelector<HTMLSelectElement>("#cfg-draft-type");
  if (draftType && cfg.draftSpecType !== undefined) draftType.value = cfg.draftSpecType;

  // Reasoning & Thinking Settings
  const rFormat = document.querySelector<HTMLSelectElement>("#cfg-reasoning-format");
  if (rFormat && cfg.reasoningFormat) rFormat.value = cfg.reasoningFormat;

  const rEffort = document.querySelector<HTMLSelectElement>("#cfg-reasoning-effort");
  if (rEffort && cfg.reasoningEffort) rEffort.value = cfg.reasoningEffort;

  const rJinja = document.querySelector<HTMLInputElement>("#cfg-enable-jinja");
  if (rJinja && cfg.enableJinja !== undefined) rJinja.checked = cfg.enableJinja;

  updateLinkedModelsBadge();
  updateConfigModelBadge(cfg.modelPath);
}

async function loadConfigForModel(modelPath: string) {
  if (!modelPath) {
    updateConfigModelBadge("");
    return;
  }
  try {
    const cfg = await LlamaService.getModelConfig(modelPath);
    applyConfigToUI(cfg);

    // If mmprojPath wasn't saved in the config, try autodetecting an adjacent projector
    const mmprojInput = document.querySelector<HTMLInputElement>("#cfg-mmproj");
    if (mmprojInput && !mmprojInput.value.trim()) {
      try {
        const detected = await LlamaService.autoDetectMMProj(modelPath);
        if (detected) {
          mmprojInput.value = detected;
          updateLinkedModelsBadge();
          showToast(`Auto-linked vision projector: ${detected.split(/[/\\]/).pop()}`, "info", "Vision Projector Found");
        }
      } catch {}
    }
  } catch (err) {
    console.error("Failed to load model config for", modelPath, err);
  }
}

async function loadConfig() {
  try {
    const quickSelect = $<HTMLSelectElement>("#quick-model-select");
    const activeModel = quickSelect?.value;
    if (activeModel) {
      await loadConfigForModel(activeModel);
    } else {
      const cfg = await LlamaService.getServerConfig();
      applyConfigToUI(cfg);
    }

    const routerCheck = $<HTMLInputElement>("#cfg-router-mode");
    if (routerCheck && !routerCheck.dataset.listenerAttached) {
      routerCheck.dataset.listenerAttached = "true";
      routerCheck.addEventListener("change", () => {
        updateRouterModeUI(routerCheck.checked);
      });
    }
  } catch (err) {
    console.error("Load config error:", err);
  }
}


async function loadServerStatus() {
  try {
    const st = await LlamaService.getServerStatus();
    updateServerUI(st);
  } catch (err) {
    console.error("Server status error:", err);
  }
}

function updateServerUI(st: ServerStatus) {
  isServerRunning = st.running;
  currentServerStatus = st;
  if (st.host) currentServerHost = st.host;
  if (st.port) currentServerPort = st.port;
  updateVisionUI();
  const pill = $("#server-status-pill");
  const pillText = $("#server-status-text");
  const quickBtn = $("#btn-quick-toggle");
  const quickBtnText = $("#btn-quick-toggle-text");
  const dashStart = $("#btn-dash-start");
  const dashStop = $("#btn-dash-stop");
  const dashStatus = $("#dash-status-val");
  const dashUptime = $("#dash-uptime");
  const dashModel = $("#dash-model-val");
  const dashPort = $("#dash-port-val");

  if (st.running) {
    pill.className = "status-indicator online";
    pillText.textContent = `Running (${st.host}:${st.port})`;
    quickBtn.className = "btn btn-danger btn-sm";
    quickBtnText.textContent = "Stop";
    dashStart.style.display = "none";
    dashStop.style.display = "inline-flex";
    dashStatus.textContent = "Active";
    dashStatus.style.color = "var(--status-green)";
    dashUptime.textContent = `PID ${st.pid} · Uptime ${formatUptime(st.uptimeSeconds)}`;
    dashModel.textContent = st.activeModel || "Loaded";
    dashPort.textContent = `${st.host}:${st.port}`;
    const chatBadge = $("#chat-active-model-badge");
    if (chatBadge) {
      const modelName = st.activeModel ? st.activeModel.split("/").pop()!.split("\\").pop()! : "Active";
      chatBadge.innerHTML = `<i data-lucide="cpu"></i><span>${modelName}</span>`;
      renderIcons();
    }
  } else {
    pill.className = "status-indicator";
    pillText.textContent = st.errorMsg ? "Crashed / Error" : "Stopped";
    quickBtn.className = "btn btn-primary btn-sm";
    quickBtnText.textContent = "Start Server";
    dashStart.style.display = "inline-flex";
    dashStop.style.display = "none";
    if (st.errorMsg) {
      dashStatus.textContent = "Error";
      dashStatus.style.color = "var(--status-red)";
      dashUptime.textContent = st.errorMsg;
    } else {
      dashStatus.textContent = "Idle";
      dashStatus.style.color = "var(--text-high)";
      dashUptime.textContent = "Not running";
    }
    const chatBadge = $("#chat-active-model-badge");
    if (chatBadge) {
      chatBadge.innerHTML = `<i data-lucide="cpu"></i><span>Offline</span>`;
      renderIcons();
    }
  }
}


function formatUptime(sec: number) {
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

async function loadLogs() {
  try {
    const logs = await LlamaService.getRecentLogs();
    const term = $("#terminal-logs");
    if (logs && logs.length > 0) {
      term.textContent = logs.join("\n");
      term.scrollTop = term.scrollHeight;
    }
  } catch (err) {
    console.error("Load logs error:", err);
  }
}

function setupEvents() {
  events.serverLog.on((line) => {
    const term = $("#terminal-logs");
    term.textContent += (term.textContent ? "\n" : "") + line;
    term.scrollTop = term.scrollHeight;
  });

  events.serverStatus.on((st) => {
    updateServerUI(st);
  });

  events.downloadProgress.on((p: ProgressReport) => {
    const overlay = $("#progress-overlay");
    if (p.status === "completed") {
      overlay.classList.remove("active");
      loadBackendStatus();
      loadModels();
      return;
    }

    if (p.status === "downloading" || p.status === "extracting") {
      overlay.classList.add("active");
      $("#prog-task").textContent = p.status === "extracting" ? "Extracting..." : "Downloading...";
      $("#prog-item").textContent = p.itemName;
      $("#prog-pct").textContent = `${p.percent.toFixed(0)}%`;
      $("#prog-fill").style.width = `${p.percent}%`;
      $("#prog-speed").textContent = `${p.speedMBps.toFixed(1)} MB/s`;

      const mbRead = (p.bytesRead / (1024 * 1024)).toFixed(0);
      const mbTotal = (p.totalBytes / (1024 * 1024)).toFixed(0);
      $("#prog-bytes").textContent = `${mbRead} / ${mbTotal} MB`;
    }
  });
}

  // Start Server
async function startServerHandler() {
  const dashStart = document.querySelector<HTMLButtonElement>("#btn-dash-start");
  const quickBtn = document.querySelector<HTMLButtonElement>("#btn-quick-toggle");

  // 1. Verify backend installation
  try {
    const backendStatus = await LlamaService.getInstalledBackend();
    if (!backendStatus.installed) {
      showToast("Backend executable not found. Please install a backend variant in Overview or Settings first.", "warning", "Backend Missing");
      return;
    }
  } catch {
    // Continue to attempt start
  }

  // 2. Verify model selection or Router Mode
  const routerCheck = $<HTMLInputElement>("#cfg-router-mode");
  const routerEnabled = routerCheck ? routerCheck.checked : false;

  const quickSelect = $<HTMLSelectElement>("#quick-model-select");
  const modelPath = quickSelect.value;
  if (!routerEnabled && !modelPath) {
    showToast("No model selected. Import, download, or select a .gguf model in models/ first, or enable Router Mode in Configuration.", "warning", "Model Required");
    return;
  }

  const cfg = collectConfigFromUI(modelPath);

  if (dashStart) dashStart.disabled = true;
  if (quickBtn) quickBtn.disabled = true;

  try {
    await LlamaService.startServer(cfg);
    showToast(routerEnabled ? "llama-server started in Router Mode" : `llama-server started with model ${modelPath.split(/[/\\]/).pop()}`, "success", "Server Running");
  } catch (err: any) {
    showToast(`Launch error: ${err.message || err}`, "error", "Start Failed", 8000);
    loadLogs();
  } finally {
    if (dashStart) dashStart.disabled = false;
    if (quickBtn) quickBtn.disabled = false;
  }
}

async function stopServerHandler() {
  try {
    await LlamaService.stopServer();
    showToast("llama-server process stopped.", "info", "Server Stopped");
  } catch (err: any) {
    showToast(`Stop error: ${err.message || err}`, "error", "Stop Failed");
  }
}

function setupActions() {
  // GGUF File Import
  const handleImport = async () => {
    try {
      const imported = await LlamaService.importModelDialog();
      if (!imported) return; // user cancelled dialog

      await loadModels();
      const quickSelect = $<HTMLSelectElement>("#quick-model-select");
      quickSelect.value = imported.path;
      await loadConfigForModel(imported.path);
      showToast(`Imported model: ${imported.name} (${imported.sizeFormatted})`, "success", "Model Imported");
    } catch (err: any) {
      showToast(`Import failed: ${err.message || err}`, "error", "Import Error");
    }
  };

  $("#btn-import-gguf-dash")?.addEventListener("click", handleImport);
  $("#btn-import-gguf-models")?.addEventListener("click", handleImport);

  // Vision Projector Browse / Clear
  $("#btn-browse-mmproj")?.addEventListener("click", async () => {
    try {
      const selected = await LlamaService.selectFileDialog("Select Vision Projector (CLIP GGUF)", "Vision Projector GGUF", "gguf");
      if (selected) {
        const input = document.querySelector<HTMLInputElement>("#cfg-mmproj");
        if (input) input.value = selected;
        updateLinkedModelsBadge();
        showToast(`Linked vision projector: ${selected.split(/[/\\]/).pop()}`, "success", "Vision Linked");
      }
    } catch (err: any) {
      showToast(`Failed to select vision projector: ${err.message || err}`, "error", "Select Error");
    }
  });

  $("#btn-clear-mmproj")?.addEventListener("click", () => {
    const input = document.querySelector<HTMLInputElement>("#cfg-mmproj");
    if (input) input.value = "";
    updateLinkedModelsBadge();
    showToast("Cleared vision projector link.", "info");
  });

  // Drafter Model Browse / Clear
  $("#btn-browse-draft-model")?.addEventListener("click", async () => {
    try {
      const selected = await LlamaService.selectFileDialog("Select Drafter Model for Speculative Decoding", "Drafter Model GGUF", "gguf");
      if (selected) {
        const input = document.querySelector<HTMLInputElement>("#cfg-draft-model");
        if (input) input.value = selected;
        updateLinkedModelsBadge();
        showToast(`Linked drafter model: ${selected.split(/[/\\]/).pop()}`, "success", "Drafter Linked");
      }
    } catch (err: any) {
      showToast(`Failed to select drafter model: ${err.message || err}`, "error", "Select Error");
    }
  });

  $("#btn-clear-draft-model")?.addEventListener("click", () => {
    const input = document.querySelector<HTMLInputElement>("#cfg-draft-model");
    if (input) input.value = "";
    updateLinkedModelsBadge();
    showToast("Cleared drafter model link.", "info");
  });

  $("#btn-dash-start")?.addEventListener("click", startServerHandler);
  $("#btn-dash-stop")?.addEventListener("click", stopServerHandler);
  $("#btn-quick-toggle")?.addEventListener("click", () => {
    if (isServerRunning) stopServerHandler();
    else startServerHandler();
  });

  $("#btn-dash-restart")?.addEventListener("click", async () => {
    if (isServerRunning) await stopServerHandler();
    setTimeout(startServerHandler, 800);
  });

  $("#btn-refresh-models").addEventListener("click", loadModels);

  $("#btn-save-config").addEventListener("click", async () => {
    const cfg = collectConfigFromUI();
    try {
      if (cfg.modelPath) {
        await LlamaService.saveModelConfig(cfg.modelPath, cfg);
        const fileName = cfg.modelPath.split(/[/\\]/).pop()?.replace(/\.gguf$/i, "") + ".json";
        showToast(`Saved parameters to ${fileName}`, "success", "Model Config Saved");
      } else {
        await LlamaService.saveServerConfig(cfg);
        showToast("Global configuration saved successfully.", "success", "Settings Saved");
      }
    } catch (err: any) {
      showToast(`Failed to save configuration: ${err.message || err}`, "error", "Save Failed");
    }
  });

  $("#btn-clear-logs").addEventListener("click", () => {
    $("#terminal-logs").textContent = "";
  });
  $("#btn-copy-logs").addEventListener("click", () => {
    navigator.clipboard.writeText($("#terminal-logs").textContent || "");
  });

  // Curated downloads
  $$("[data-download-url]").forEach((b) => {
    b.addEventListener("click", async () => {
      const url = b.getAttribute("data-download-url");
      const filename = b.getAttribute("data-filename");
      if (!url || !filename) return;

      b.setAttribute("disabled", "true");
      try {
        await LlamaService.downloadModel(url, filename);
      } catch (err: any) {
        showToast(`Download failed: ${err.message || err}`, "error", "Download Error");
      } finally {
        b.removeAttribute("disabled");
      }
    });
  });

  // Hugging Face search
  $("#btn-hf-search").addEventListener("click", async () => {
    const repo = $<HTMLInputElement>("#hf-repo-input").value.trim();
    if (!repo) return;

    const btn = $<HTMLButtonElement>("#btn-hf-search");
    btn.disabled = true;
    try {
      const files = await LlamaService.fetchHFRepo(repo);
      const container = $("#hf-files-container");
      const tbody = $("#hf-files-tbody");
      tbody.innerHTML = "";

      if (!files || files.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-low); padding: 14px;">No .gguf files found in repository.</td></tr>`;
      } else {
        files.forEach((f) => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td style="font-family: var(--font-mono); font-size: 12px; font-weight: 500;">${f.filename}</td>
            <td><span class="tag-badge">${f.quant}</span></td>
            <td style="font-family: var(--font-mono);">${f.sizeFormatted}</td>
            <td style="text-align: right;">
              <button class="btn btn-secondary btn-sm btn-hf-dl" data-url="${f.downloadUrl}" data-file="${f.filename}">
                <i data-lucide="download"></i>
                <span>Get</span>
              </button>
            </td>
          `;
          tbody.appendChild(tr);
        });

        $$(".btn-hf-dl").forEach((dl) => {
          dl.addEventListener("click", async () => {
            const u = dl.getAttribute("data-url");
            const fn = dl.getAttribute("data-file");
            if (u && fn) {
              dl.setAttribute("disabled", "true");
              try {
                await LlamaService.downloadModel(u, fn);
              } catch (e: any) {
                showToast(`Download error: ${e.message || e}`, "error", "Download Error");
              } finally {
                dl.removeAttribute("disabled");
              }
            }
          });
        });
      }
      container.style.display = "block";
      renderIcons();
    } catch (err: any) {
      showToast(`Repository lookup failed: ${err.message || err}`, "error", "Hugging Face Error");
    } finally {
      btn.disabled = false;
    }
  });

  $("#btn-check-releases")?.addEventListener("click", loadReleases);
}

// GitHub releases
async function loadReleases() {
  const tbody = $("#releases-tbody");
  tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-low); padding: 18px;">Checking GitHub releases...</td></tr>`;

  try {
    const [releases, hw] = await Promise.all([
      LlamaService.checkReleases(),
      LlamaService.getHardware(),
    ]);
    tbody.innerHTML = "";

    if (!releases || releases.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-low);">No releases returned for ${hw.os} (${hw.arch}).</td></tr>`;
      return;
    }

    const isMac = hw.os === "darwin";
    const isLinux = hw.os === "linux";

    let variantOptionsHtml = "";
    if (isMac) {
      variantOptionsHtml = `
        <option value="metal" selected>Metal (Apple Silicon / Mac)</option>
        <option value="cpu">CPU (Standard)</option>
      `;
    } else if (isLinux) {
      variantOptionsHtml = `
        <option value="cuda" ${hw.recommendedBackend === "cuda" ? "selected" : ""}>CUDA (NVIDIA)</option>
        <option value="rocm" ${hw.recommendedBackend === "rocm" ? "selected" : ""}>ROCm (AMD)</option>
        <option value="vulkan" ${hw.recommendedBackend === "vulkan" ? "selected" : ""}>Vulkan (Universal)</option>
        <option value="cpu">CPU (Standard)</option>
      `;
    } else {
      // Windows
      variantOptionsHtml = `
        <option value="cuda" ${hw.recommendedBackend === "cuda" ? "selected" : ""}>CUDA 12.4 (NVIDIA)</option>
        <option value="vulkan" ${hw.recommendedBackend === "vulkan" ? "selected" : ""}>Vulkan (AMD / Universal)</option>
        <option value="cpu">CPU (Standard)</option>
      `;
    }

    releases.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight: 600; font-family: var(--font-mono); font-size: 12.5px;">${r.tag}</td>
        <td style="color: var(--text-medium); font-size: 12px;">${r.name || r.tag}</td>
        <td style="color: var(--text-low); font-size: 11.5px;">${new Date(r.publishedAt).toLocaleDateString()}</td>
        <td>
          <select class="form-select" style="width: 170px; padding: 4px 8px; font-size: 11.5px;" id="rel-var-${r.tag}">
            ${variantOptionsHtml}
          </select>
        </td>
        <td style="text-align: right;">
          <button class="btn btn-secondary btn-sm btn-install-backend" data-tag="${r.tag}">
            <i data-lucide="download"></i>
            <span>${r.isCurrent ? "Re-Install" : "Install"}</span>
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    $$(".btn-install-backend").forEach((b) => {
      b.addEventListener("click", async () => {
        const tag = b.getAttribute("data-tag");
        if (!tag) return;
        const sel = document.querySelector<HTMLSelectElement>(`#rel-var-${tag}`);
        const v = sel?.value || "cuda";

        b.setAttribute("disabled", "true");
        try {
          await LlamaService.downloadBackend(tag, v);
          showToast(`llama.cpp ${tag} (${v.toUpperCase()}) installed successfully. Ready to run!`, "success", "Backend Ready");
          loadBackendStatus();
        } catch (e: any) {
          showToast(`Installation failed: ${e.message || e}`, "error", "Install Failed");
        } finally {
          b.removeAttribute("disabled");
        }
      });
    });

    renderIcons();
  } catch (err: any) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--status-red);">Failed to fetch releases: ${err.message || err}</td></tr>`;
  }
}

// Inline Test Prompt on Dashboard
function setupInlineTest() {
  const input = $<HTMLInputElement>("#dash-test-input");
  const btn = $<HTMLButtonElement>("#btn-dash-test-send");
  const output = $("#dash-test-output");

  btn.addEventListener("click", async () => {
    const prompt = input.value.trim();
    if (!prompt) return;

    if (!isServerRunning) {
      showToast("Please start the server first to run a test prompt.", "warning", "Server Idle");
      return;
    }

    btn.disabled = true;
    output.style.display = "block";
    output.textContent = "Processing prompt...";

    try {
      const res = await LlamaService.sendChat([{ role: "user", content: prompt }], 0.7, "");
      output.textContent = res;
    } catch (e: any) {
      output.textContent = `Error: ${e.message || e}`;
    } finally {
      btn.disabled = false;
    }
  });
}

// Settings Management
async function loadSettings() {
  try {
    const s = await LlamaService.getSettings();
    const modelsInput = document.querySelector<HTMLInputElement>("#settings-models-dir");
    if (modelsInput) modelsInput.value = s.modelsDir || "";

    const appDataInput = document.querySelector<HTMLInputElement>("#settings-appdata-dir");
    if (appDataInput) appDataInput.value = s.appDataDir || "";

    const backendTag = document.querySelector<HTMLElement>("#settings-current-backend-tag");
    if (backendTag) {
      backendTag.textContent = (s.activeBackend || "cuda").toUpperCase();
    }

    const defSelect = document.querySelector<HTMLSelectElement>("#settings-default-model-select");
    if (defSelect && s.defaultModel) {
      defSelect.value = s.defaultModel;
    }

    await renderBackendManager();
  } catch (err) {
    console.error("Load settings error:", err);
  }
}

async function renderBackendManager() {
  const container = document.querySelector<HTMLElement>("#backend-manager-list");
  if (!container) return;

  try {
    const list = await LlamaService.getInstalledBackends();
    container.innerHTML = "";

    if (!list || list.length === 0) {
      container.innerHTML = `<div style="text-align: center; color: var(--text-low); padding: 16px;">No backends available for your system architecture.</div>`;
      return;
    }

    list.forEach((b) => {
      const card = document.createElement("div");
      card.className = `backend-manager-card ${b.isActive ? "active" : ""}`;

      let badgeHtml = "";
      if (b.isActive) {
        badgeHtml = `<span class="tag-badge accent">CURRENT ACTIVE</span>`;
      } else if (b.installed) {
        badgeHtml = `<span class="tag-badge" style="color: #4ade80; border-color: rgba(74, 222, 128, 0.3);">INSTALLED</span>`;
      } else {
        badgeHtml = `<span class="tag-badge" style="color: var(--text-low);">NOT INSTALLED</span>`;
      }

      let metaText = "";
      if (b.installed) {
        metaText = `Version: ${b.version || "Installed"} • Executable: ${b.executablePath}`;
      } else {
        metaText = "Hardware accelerator runtime package ready for download and setup.";
      }

      let actionsHtml = "";
      if (b.isActive) {
        actionsHtml = `
          <button class="btn btn-secondary btn-sm btn-uninstall-backend-card" data-variant="${b.variant}" title="Uninstall active backend">
            <i data-lucide="trash-2"></i>
            <span>Uninstall</span>
          </button>
        `;
      } else if (b.installed) {
        actionsHtml = `
          <button class="btn btn-primary btn-sm btn-activate-backend-card" data-variant="${b.variant}" title="Switch active runtime to ${b.name}">
            <i data-lucide="check"></i>
            <span>Set Active</span>
          </button>
          <button class="btn btn-secondary btn-sm btn-uninstall-backend-card" data-variant="${b.variant}" title="Uninstall backend">
            <i data-lucide="trash-2"></i>
            <span>Uninstall</span>
          </button>
        `;
      } else {
        actionsHtml = `
          <button class="btn btn-secondary btn-sm btn-install-backend-card" data-variant="${b.variant}" title="Download and install ${b.name}">
            <i data-lucide="download"></i>
            <span>Install</span>
          </button>
        `;
      }

      card.innerHTML = `
        <div class="backend-manager-main">
          <div class="backend-manager-title-row">
            <span class="backend-manager-name">${b.name}</span>
            ${badgeHtml}
          </div>
          <div class="backend-manager-meta">${metaText}</div>
        </div>
        <div class="backend-manager-actions">
          ${actionsHtml}
        </div>
      `;

      container.appendChild(card);
    });

    // Wire activate buttons
    container.querySelectorAll<HTMLButtonElement>(".btn-activate-backend-card").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const variant = btn.dataset.variant;
        if (!variant) return;

        if (isServerRunning) {
          const ok = await showConfirm(
            "Switch Active Backend",
            "The llama-server is currently running. Switching backend will stop the server. Continue?",
            "Stop & Switch",
            true
          );
          if (!ok) return;
          await LlamaService.stopServer();
        }

        btn.disabled = true;
        try {
          await LlamaService.setActiveBackend(variant);
          showToast(`Active backend switched to ${variant.toUpperCase()} successfully.`, "success", "Backend Activated");
          await loadBackendStatus();
          await loadSettings();
        } catch (err: any) {
          showToast(`Failed to activate backend: ${err.message || err}`, "error", "Activation Error");
        } finally {
          btn.disabled = false;
        }
      });
    });

    // Wire uninstall buttons
    container.querySelectorAll<HTMLButtonElement>(".btn-uninstall-backend-card").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const variant = btn.dataset.variant;
        if (!variant) return;

        const ok = await showConfirm(
          "Uninstall Backend",
          `Are you sure you want to uninstall the ${variant.toUpperCase()} backend? All binary files and libraries for this backend will be deleted from disk.`,
          "Uninstall",
          true
        );
        if (!ok) return;

        if (isServerRunning) {
          await LlamaService.stopServer();
        }

        btn.disabled = true;
        try {
          await LlamaService.uninstallBackend(variant);
          showToast(`Backend ${variant.toUpperCase()} uninstalled cleanly.`, "success", "Backend Uninstalled");
          await loadBackendStatus();
          await loadSettings();
        } catch (err: any) {
          showToast(`Failed to uninstall backend: ${err.message || err}`, "error", "Uninstall Error");
        } finally {
          btn.disabled = false;
        }
      });
    });

    // Wire install buttons
    container.querySelectorAll<HTMLButtonElement>(".btn-install-backend-card").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const variant = btn.dataset.variant;
        if (!variant) return;

        btn.disabled = true;
        try {
          showToast(`Downloading and installing ${variant.toUpperCase()} backend...`, "info", "Downloading Backend");
          await LlamaService.switchBackend(variant);
          showToast(`Backend ${variant.toUpperCase()} installed and activated!`, "success", "Backend Installed");
          await loadBackendStatus();
          await loadSettings();
        } catch (err: any) {
          showToast(`Failed to install backend: ${err.message || err}`, "error", "Installation Error");
        } finally {
          btn.disabled = false;
        }
      });
    });

    renderIcons();
  } catch (err) {
    console.error("renderBackendManager error:", err);
  }
}

function setupSettings() {
  // Browse Models Dir
  $("#btn-browse-models-dir")?.addEventListener("click", async () => {
    try {
      const selected = await LlamaService.selectFolderDialog("Select Models Folder");
      if (selected) {
        await LlamaService.setModelsDir(selected);
        const modelsInput = document.querySelector<HTMLInputElement>("#settings-models-dir");
        if (modelsInput) modelsInput.value = selected;
        await loadModels();
        showToast(`Models directory set to ${selected}`, "success", "Settings Updated");
      }
    } catch (err: any) {
      showToast(`Failed to set models directory: ${err.message || err}`, "error", "Settings Error");
    }
  });

  // Open Models Dir in Explorer
  $("#btn-open-models-dir")?.addEventListener("click", async () => {
    const dir = document.querySelector<HTMLInputElement>("#settings-models-dir")?.value || "";
    await LlamaService.openFolderInExplorer(dir);
  });

  // Browse App Data Dir
  $("#btn-browse-appdata-dir")?.addEventListener("click", async () => {
    try {
      const selected = await LlamaService.selectFolderDialog("Select Application Data Folder");
      if (selected) {
        await LlamaService.setAppDataDir(selected);
        const appDataInput = document.querySelector<HTMLInputElement>("#settings-appdata-dir");
        if (appDataInput) appDataInput.value = selected;
        await loadBackendStatus();
        showToast(`Application data directory set to ${selected}`, "success", "Settings Updated");
      }
    } catch (err: any) {
      showToast(`Failed to set app data directory: ${err.message || err}`, "error", "Settings Error");
    }
  });

  // Open App Data Dir in Explorer
  $("#btn-open-appdata-dir")?.addEventListener("click", async () => {
    const dir = document.querySelector<HTMLInputElement>("#settings-appdata-dir")?.value || "";
    await LlamaService.openFolderInExplorer(dir);
  });

  // Default model change
  $("#settings-default-model-select")?.addEventListener("change", async () => {
    const defSelect = document.querySelector<HTMLSelectElement>("#settings-default-model-select");
    if (!defSelect) return;
    const s = await LlamaService.getSettings();
    s.defaultModel = defSelect.value;
    await LlamaService.saveSettings(s);
  });

  // Uninstall All Backends
  $("#btn-uninstall-all-backends")?.addEventListener("click", async () => {
    const ok = await showConfirm(
      "Uninstall All Backends",
      "Are you sure you want to uninstall ALL llama.cpp backends and wipe the bin folder? You will need to install a backend before running models again.",
      "Uninstall All",
      true
    );
    if (!ok) return;

    if (isServerRunning) {
      await LlamaService.stopServer();
    }

    const btn = document.querySelector<HTMLButtonElement>("#btn-uninstall-all-backends");
    if (btn) btn.disabled = true;
    try {
      await LlamaService.uninstallAllBackends();
      showToast("All backends uninstalled and bin folder cleaned.", "success", "Backends Removed");
      await loadBackendStatus();
      await loadSettings();
    } catch (err: any) {
      showToast(`Failed to uninstall all backends: ${err.message || err}`, "error", "Uninstall Error");
    } finally {
      if (btn) btn.disabled = false;
    }
  });

  // Overview quick action buttons
  $("#btn-dash-manage-backend")?.addEventListener("click", () => {
    const nav = document.querySelector<HTMLElement>('[data-tab="settings"]');
    nav?.click();
  });

  $("#btn-dash-uninstall-backend")?.addEventListener("click", async () => {
    const b = await LlamaService.getInstalledBackend();
    if (!b.installed) return;
    const variantName = (b.variant || "current").toUpperCase();
    const ok = await showConfirm(
      "Uninstall Active Backend",
      `Are you sure you want to uninstall the ${variantName} backend?`,
      "Uninstall",
      true
    );
    if (!ok) return;

    if (isServerRunning) {
      await LlamaService.stopServer();
    }

    try {
      await LlamaService.uninstallBackend(b.variant);
      showToast(`Backend ${variantName} uninstalled cleanly.`, "success", "Backend Uninstalled");
      await loadBackendStatus();
      await loadSettings();
    } catch (err: any) {
      showToast(`Failed to uninstall backend: ${err.message || err}`, "error", "Uninstall Error");
    }
  });
}

// Backend Troubleshooting & Diagnostics
function setupTroubleshoot() {
  const modal = document.querySelector<HTMLElement>("#modal-troubleshoot");
  const btnClose = document.querySelector<HTMLElement>("#btn-close-troubleshoot-modal");
  const btnRerun = document.querySelector<HTMLElement>("#btn-rerun-diagnostics");
  const btnRepair = document.querySelector<HTMLElement>("#btn-repair-backend");
  const btnKill = document.querySelector<HTMLElement>("#btn-kill-zombies");
  const list = document.querySelector<HTMLElement>("#diag-items-list");
  const summary = document.querySelector<HTMLElement>("#diag-summary-text");

  function openModal() {
    if (modal) modal.classList.add("open");
    runScan();
  }

  function closeModal() {
    if (modal) modal.classList.remove("open");
  }

  document.querySelectorAll(".btn-open-troubleshoot").forEach((b) => {
    b.addEventListener("click", openModal);
  });

  btnClose?.addEventListener("click", closeModal);

  async function runScan() {
    if (!list || !summary) return;
    summary.textContent = "Scanning backend executable, runtime DLLs, ports, and permissions...";
    list.innerHTML = `<div style="text-align: center; color: var(--text-low); padding: 18px;">Running diagnostics...</div>`;

    try {
      const report = await LlamaService.runDiagnostics();
      list.innerHTML = "";

      if (report.healthy && report.issuesCount === 0) {
        summary.innerHTML = `<span style="color: #4ade80; font-weight: 600;">All diagnostic checks passed.</span> Backend is healthy and ready.`;
      } else {
        summary.innerHTML = `<span style="color: #f87171; font-weight: 600;">Found ${report.issuesCount} issue(s).</span> Review items below or click 'Auto-Repair Backend'.`;
      }

      report.items.forEach((it) => {
        const itemDiv = document.createElement("div");
        itemDiv.className = `diag-item status-${it.status}`;

        const badgeClass = it.status === "ok" ? "ok" : it.status === "warning" ? "warning" : "error";
        const badgeText = it.status === "ok" ? "PASSED" : it.status === "warning" ? "WARNING" : "ERROR";

        itemDiv.innerHTML = `
          <div class="diag-item-content">
            <div class="diag-item-header">
              <span class="diag-item-title">${it.title}</span>
              <span class="diag-badge ${badgeClass}">${badgeText}</span>
            </div>
            <div class="diag-item-msg">${it.message}</div>
          </div>
        `;
        list.appendChild(itemDiv);
      });
    } catch (err: any) {
      summary.textContent = `Diagnostic scan failed: ${err.message || err}`;
    }
  }

  btnRerun?.addEventListener("click", runScan);

  btnKill?.addEventListener("click", async () => {
    try {
      await LlamaService.killZombieServers();
      showToast("Sent termination signal to lingering llama-server processes.", "info", "Process Terminated");
      await runScan();
      await loadServerStatus();
    } catch (err: any) {
      showToast(`Error: ${err.message || err}`, "error", "Kill Failed");
    }
  });

  btnRepair?.addEventListener("click", async () => {
    if (isServerRunning) {
      await LlamaService.stopServer();
      showToast("Stopped current server to proceed with repair.", "info");
    }

    try {
      const btn = btnRepair as HTMLButtonElement;
      btn.disabled = true;
      await LlamaService.repairBackend("reinstall");
      showToast("Backend re-download and repair completed successfully.", "success", "Repair Complete");
      await loadBackendStatus();
      await runScan();
    } catch (err: any) {
      showToast(`Auto-repair failed: ${err.message || err}`, "error", "Repair Error");
    } finally {
      if (btnRepair) (btnRepair as HTMLButtonElement).disabled = false;
    }
  });
}

// Chat Playground with Branching Tree and Persona Switching
function setupChat() {
  const form = $<HTMLFormElement>("#chat-form");
  const input = $<HTMLTextAreaElement>("#chat-input");
  const container = $("#chat-messages");
  const threadsContainer = $("#chat-threads-list");
  const personaSelect = $<HTMLSelectElement>("#chat-persona-select");
  const reasoningSelect = $<HTMLSelectElement>("#chat-reasoning-select");
  const sessionTitle = document.querySelector<HTMLElement>("#chat-session-title");
  const btnNewThread = $("#btn-new-thread");
  const btnClearChat = $("#btn-clear-chat");
  const btnNewPersona = $("#btn-new-persona");
  const modalPersona = $("#modal-persona");
  const btnCloseModal = $("#btn-close-persona-modal");
  const btnCancelPersona = $("#btn-cancel-persona");
  const btnSavePersona = $("#btn-save-persona");

  // Auto-resize chat input textarea
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  });

  // Enter to send, Shift+Enter for new line
  input.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });

  // Code block copy & save actions via event delegation
  container.addEventListener("click", async (e) => {
    const target = e.target as HTMLElement;

    // 1. Copy snippet
    const copyBtn = target.closest<HTMLButtonElement>(".btn-code-copy");
    if (copyBtn) {
      const codeContainer = copyBtn.closest<HTMLElement>(".code-block-container");
      const codeEl = codeContainer?.querySelector("code");
      if (codeEl) {
        const codeText = codeEl.textContent || "";
        try {
          await navigator.clipboard.writeText(codeText);
          const span = copyBtn.querySelector("span");
          if (span) span.textContent = "Copied!";
          copyBtn.classList.add("copied");
          setTimeout(() => {
            if (span) span.textContent = "Copy";
            copyBtn.classList.remove("copied");
          }, 1500);
        } catch (err: any) {
          showToast(`Copy failed: ${err.message || err}`, "error");
        }
      }
      return;
    }

    // 2. Save snippet to file
    const saveBtn = target.closest<HTMLButtonElement>(".btn-code-save");
    if (saveBtn) {
      const codeContainer = saveBtn.closest<HTMLElement>(".code-block-container");
      const codeEl = codeContainer?.querySelector("code");
      if (codeEl) {
        const codeText = codeEl.textContent || "";
        const lang = codeContainer?.dataset.lang || "code";
        const ext = getExtensionForLang(lang);
        const defaultFilename = saveBtn.dataset.filename || `snippet.${ext}`;

        try {
          const savedPath = await LlamaService.saveSnippetDialog(defaultFilename, codeText);
          if (savedPath) {
            showToast(`Snippet saved to ${savedPath.split(/[/\\]/).pop()}`, "success", "Code Saved");
          }
        } catch (err) {
          // Fallback to browser blob download
          downloadSnippet(defaultFilename, codeText);
          showToast(`Downloaded ${defaultFilename}`, "success", "Code Saved");
        }
      }
      return;
    }
  });

  let sessions = loadAllSessions();
  let currentSessionId = getActiveSessionId();
  let currentSession: ChatSession;

  if (sessions.length === 0 || !currentSessionId) {
    currentSession = createNewSession("default");
    sessions = [currentSession];
    currentSessionId = currentSession.id;
  } else {
    const found = sessions.find((s) => s.id === currentSessionId);
    if (found) {
      currentSession = found;
    } else {
      currentSession = sessions[0]!;
      currentSessionId = currentSession.id;
      setActiveSessionId(currentSessionId);
    }
  }

  // Populate persona selector
  function renderPersonaOptions() {
    personaSelect.innerHTML = "";
    const personas = getPersonas();
    personas.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = `${p.name} ${p.isCustom ? "(Custom)" : ""}`;
      personaSelect.appendChild(opt);
    });
    personaSelect.value = currentSession.personaId || "default";
  }

  personaSelect.addEventListener("change", () => {
    currentSession.personaId = personaSelect.value;
    saveSession(currentSession);
  });

  reasoningSelect?.addEventListener("change", () => {
    currentSession.reasoningEffort = reasoningSelect.value;
    saveSession(currentSession);
  });

  // Modal handlers
  function openPersonaModal() {
    $<HTMLInputElement>("#persona-name-input").value = "";
    $<HTMLInputElement>("#persona-desc-input").value = "";
    $<HTMLTextAreaElement>("#persona-prompt-input").value = "";
    modalPersona.classList.add("open");
  }

  function closePersonaModal() {
    modalPersona.classList.remove("open");
  }

  btnNewPersona.addEventListener("click", openPersonaModal);
  btnCloseModal.addEventListener("click", closePersonaModal);
  btnCancelPersona.addEventListener("click", closePersonaModal);

  btnSavePersona.addEventListener("click", () => {
    const name = $<HTMLInputElement>("#persona-name-input").value.trim();
    const tagline = $<HTMLInputElement>("#persona-desc-input").value.trim();
    const systemPrompt = $<HTMLTextAreaElement>("#persona-prompt-input").value.trim();

    if (!name || !systemPrompt) {
      showToast("Please provide both a Persona Name and a System Prompt.", "warning", "Validation Error");
      return;
    }

    const newPersona: Persona = {
      id: "persona_" + Date.now(),
      name,
      tagline: tagline || "Custom Assistant Persona",
      systemPrompt,
      isCustom: true,
    };

    saveCustomPersona(newPersona);
    renderPersonaOptions();
    personaSelect.value = newPersona.id;
    currentSession.personaId = newPersona.id;
    saveSession(currentSession);
    closePersonaModal();
  });

  // Render session list
  function renderSessions() {
    threadsContainer.innerHTML = "";
    sessions = loadAllSessions();
    if (sessions.length === 0) {
      currentSession = createNewSession();
      sessions = [currentSession];
      currentSessionId = currentSession.id;
    }

    sessions.forEach((s) => {
      const item = document.createElement("div");
      item.className = `thread-item ${s.id === currentSessionId ? "active" : ""}`;

      const icon = document.createElement("i");
      icon.setAttribute("data-lucide", "message-square");
      icon.className = "thread-icon";

      const titleSpan = document.createElement("span");
      titleSpan.className = "thread-title";
      titleSpan.textContent = s.title || "Untitled Chat";

      const delBtn = document.createElement("button");
      delBtn.className = "thread-del-btn";
      delBtn.title = "Delete conversation";
      delBtn.innerHTML = `<i data-lucide="trash-2"></i>`;

      delBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const confirmed = await showConfirm(
          "Delete Thread",
          "Are you sure you want to delete this conversation thread and its history?",
          "Delete Thread",
          true
        );
        if (confirmed) {
          deleteSession(s.id);
          sessions = loadAllSessions();
          currentSessionId = getActiveSessionId();
          if (!currentSessionId && sessions.length > 0 && sessions[0]) {
            currentSessionId = sessions[0].id;
            setActiveSessionId(currentSessionId);
          } else if (sessions.length === 0) {
            const fresh = createNewSession();
            currentSessionId = fresh.id;
            sessions = [fresh];
          }
          currentSession = sessions.find((x) => x.id === currentSessionId)!;
          renderSessions();
          renderPersonaOptions();
          renderStage();
          showToast("Conversation thread deleted.", "info");
        }
      });

      item.appendChild(icon);
      item.appendChild(titleSpan);
      item.appendChild(delBtn);

      item.addEventListener("click", () => {
        if (currentSessionId === s.id) return;
        currentSessionId = s.id;
        setActiveSessionId(s.id);
        currentSession = s;
        renderSessions();
        renderPersonaOptions();
        renderStage();
      });

      threadsContainer.appendChild(item);
    });

    renderIcons();
  }

  // Create new thread
  btnNewThread.addEventListener("click", () => {
    currentSession = createNewSession(personaSelect.value || "default", reasoningSelect ? reasoningSelect.value : "default");
    currentSessionId = currentSession.id;
    renderSessions();
    renderPersonaOptions();
    renderStage();
  });

  // Clear current thread messages
  btnClearChat.addEventListener("click", async () => {
    const confirmed = await showConfirm(
      "Clear Conversation",
      "Clear all message history in the active thread?",
      "Clear All",
      true
    );
    if (confirmed) {
      currentSession.nodes = {};
      currentSession.rootNodeId = null;
      currentSession.activeNodeId = null;
      currentSession.title = "New Conversation";
      saveSession(currentSession);
      renderSessions();
      renderStage();
      showToast("Conversation history cleared.", "info");
    }
  });

  // Helper to extract <think> / <thought> blocks
  function parseReasoning(content: string): { reasoning?: string; answer: string } {
    const thinkRegex = /<(think|thought)>([\s\S]*?)(?:<\/\1>|$)/i;
    const match = content.match(thinkRegex);
    if (!match) {
      return { answer: content };
    }

    const reasoning = match && match[2] ? match[2].trim() : "";
    const answer = content.replace(/<(think|thought)>[\s\S]*?(?:<\/\1>|$)/gi, "").trim();

    return {
      reasoning: reasoning.length > 0 ? reasoning : undefined,
      answer,
    };
  }

  // Shared renderer for bubble content (works for static and live streaming)
  function renderBubbleContent(bubbleEl: HTMLElement, content: string, isStreaming = false) {
    const parsed = parseReasoning(content);
    if (parsed.reasoning) {
      let details = bubbleEl.querySelector<HTMLDetailsElement>(".reasoning-accordion");
      let body = bubbleEl.querySelector<HTMLElement>(".reasoning-body");
      let answerDiv = bubbleEl.querySelector<HTMLElement>(".chat-answer-body");

      if (!details) {
        bubbleEl.innerHTML = "";
        details = document.createElement("details");
        details.className = "reasoning-accordion";

        const summary = document.createElement("summary");
        summary.className = "reasoning-summary";
        summary.innerHTML = `
          <div class="reasoning-summary-left">
            <i data-lucide="brain"></i>
            <span>Thought Process</span>
          </div>
          <i data-lucide="chevron-right" class="reasoning-chevron"></i>
        `;

        body = document.createElement("div");
        body.className = "reasoning-body";

        details.appendChild(summary);
        details.appendChild(body);
        bubbleEl.appendChild(details);
        renderIcons();
      }

      if (body) {
        body.textContent = parsed.reasoning;
      }

      if (!answerDiv) {
        answerDiv = document.createElement("div");
        answerDiv.className = "chat-answer-body";
        bubbleEl.appendChild(answerDiv);
      }

      if (parsed.answer) {
        answerDiv.className = "chat-answer-body";
        answerDiv.innerHTML = renderMarkdown(parsed.answer);
        renderIcons();
      } else {
        if (isStreaming) {
          answerDiv.className = "chat-answer-body chat-answer-empty";
          answerDiv.textContent = "Thinking...";
        } else {
          answerDiv.className = "chat-answer-body chat-answer-empty";
          answerDiv.textContent = "(Reasoning complete)";
        }
      }
    } else {
      const details = bubbleEl.querySelector(".reasoning-accordion");
      if (details) {
        bubbleEl.innerHTML = "";
      }
      if (content) {
        bubbleEl.innerHTML = renderMarkdown(content);
        renderIcons();
      } else if (isStreaming) {
        bubbleEl.textContent = "Thinking...";
      } else {
        bubbleEl.textContent = "";
      }
    }
  }

  // Render stage (chat history tree path)
  function renderStage() {
    if (sessionTitle) {
      sessionTitle.textContent = currentSession.title || "Chat Playground";
    }
    if (reasoningSelect) {
      reasoningSelect.value = currentSession.reasoningEffort || "default";
    }
    container.innerHTML = "";

    const path = getActiveConversationPath(currentSession);

    if (path.length === 0) {
      const emptyDiv = document.createElement("div");
      emptyDiv.className = "chat-empty-state";
      emptyDiv.innerHTML = `
        <div class="chat-empty-icon"><i data-lucide="sparkles"></i></div>
        <div class="chat-empty-title">Playground Ready</div>
        <div class="chat-empty-desc">Send a prompt below to interact with your loaded model in real time. Branch conversations anytime with inline edit or regenerate.</div>
      `;
      container.appendChild(emptyDiv);
      renderIcons();
      return;
    }

    path.forEach((node) => {
      const wrapper = document.createElement("div");
      wrapper.className = `chat-entry-wrapper ${node.role}`;
      wrapper.id = `chat-node-${node.id}`;

      // Content bubble (without "You" / "Assistant" text header)
      const bubble = document.createElement("div");
      bubble.className = `chat-entry ${node.role}`;

      if (node.role === "assistant") {
        renderBubbleContent(bubble, node.content, false);
      } else {
        let userHtml = "";
        if (node.images && node.images.length > 0) {
          userHtml += `<div class="chat-entry-images">`;
          node.images.forEach((imgUrl) => {
            userHtml += `<div class="chat-entry-image-wrap" title="Click to view full image"><img src="${imgUrl}" alt="Attached image" class="chat-entry-img" /></div>`;
          });
          userHtml += `</div>`;
        }
        if (node.content) {
          userHtml += renderMarkdown(node.content);
        }
        bubble.innerHTML = userHtml;

        bubble.querySelectorAll<HTMLImageElement>(".chat-entry-img").forEach((img) => {
          img.addEventListener("click", () => {
            openImageModal(img.src);
          });
        });

        renderIcons();
      }

      wrapper.appendChild(bubble);

      // Meta bar with sibling pagination & action buttons
      const metaBar = document.createElement("div");
      metaBar.className = "entry-meta-bar";

      // Branch pagination pill (< 1 / 2 >)
      const siblingInfo = getSiblingInfo(currentSession, node.id);
      if (siblingInfo.total > 1) {
        const branchNav = document.createElement("div");
        branchNav.className = "branch-nav";

        const btnPrev = document.createElement("button");
        btnPrev.className = "branch-btn";
        btnPrev.disabled = siblingInfo.index <= 1;
        btnPrev.title = "Previous variation";
        btnPrev.innerHTML = `<i data-lucide="chevron-left"></i>`;
        btnPrev.addEventListener("click", () => {
          if (siblingInfo.index > 1) {
            const targetSibling = siblingInfo.siblings[siblingInfo.index - 2];
            if (targetSibling) {
              switchSiblingBranch(currentSession, targetSibling.id);
              renderStage();
            }
          }
        });

        const pillText = document.createElement("span");
        pillText.className = "branch-counter";
        pillText.textContent = `${siblingInfo.index} / ${siblingInfo.total}`;

        const btnNext = document.createElement("button");
        btnNext.className = "branch-btn";
        btnNext.disabled = siblingInfo.index >= siblingInfo.total;
        btnNext.title = "Next variation";
        btnNext.innerHTML = `<i data-lucide="chevron-right"></i>`;
        btnNext.addEventListener("click", () => {
          if (siblingInfo.index < siblingInfo.total) {
            const targetSibling = siblingInfo.siblings[siblingInfo.index];
            if (targetSibling) {
              switchSiblingBranch(currentSession, targetSibling.id);
              renderStage();
            }
          }
        });

        branchNav.appendChild(btnPrev);
        branchNav.appendChild(pillText);
        branchNav.appendChild(btnNext);
        metaBar.appendChild(branchNav);
      } else {
        const spacer = document.createElement("span");
        metaBar.appendChild(spacer);
      }

      // Actions toolbar
      const actionsDiv = document.createElement("div");
      actionsDiv.className = "entry-actions";

      // Copy button
      const copyBtn = document.createElement("button");
      copyBtn.className = "entry-action-btn";
      copyBtn.title = "Copy message";
      copyBtn.innerHTML = `<i data-lucide="copy"></i><span>Copy</span>`;
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(node.content);
        copyBtn.innerHTML = `<i data-lucide="check"></i><span>Copied</span>`;
        renderIcons();
        setTimeout(() => {
          copyBtn.innerHTML = `<i data-lucide="copy"></i><span>Copy</span>`;
          renderIcons();
        }, 1500);
      });
      actionsDiv.appendChild(copyBtn);

      // User node: Edit button
      if (node.role === "user") {
        const editBtn = document.createElement("button");
        editBtn.className = "entry-action-btn";
        editBtn.title = "Edit message & branch";
        editBtn.innerHTML = `<i data-lucide="edit-2"></i><span>Edit</span>`;
        editBtn.addEventListener("click", () => {
          showInlineEdit(wrapper, bubble, metaBar, node);
        });
        actionsDiv.appendChild(editBtn);
      }

      // Assistant node: Regenerate button
      if (node.role === "assistant") {
        const regenBtn = document.createElement("button");
        regenBtn.className = "entry-action-btn";
        regenBtn.title = "Regenerate alternative response";
        regenBtn.innerHTML = `<i data-lucide="rotate-ccw"></i><span>Regenerate</span>`;
        regenBtn.addEventListener("click", () => {
          executeRegenerate(node);
        });
        actionsDiv.appendChild(regenBtn);
      }

      metaBar.appendChild(actionsDiv);
      wrapper.appendChild(metaBar);
      container.appendChild(wrapper);
    });

    container.scrollTop = container.scrollHeight;
    renderIcons();
  }

  // Inline edit handler for user messages
  function showInlineEdit(wrapper: HTMLElement, bubble: HTMLElement, metaBar: HTMLElement, node: ChatNode) {
    const header = wrapper.querySelector<HTMLElement>(".entry-header");
    if (header) header.style.display = "none";
    bubble.style.display = "none";
    metaBar.style.display = "none";

    const editForm = document.createElement("form");
    editForm.className = "inline-edit-form";

    const textarea = document.createElement("textarea");
    textarea.className = "inline-edit-textarea";
    textarea.value = node.content;

    textarea.addEventListener("input", () => {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 250)}px`;
    });

    textarea.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        editForm.dispatchEvent(new Event("submit", { cancelable: true }));
      }
    });

    const actions = document.createElement("div");
    actions.className = "inline-edit-actions";

    const btnCancel = document.createElement("button");
    btnCancel.type = "button";
    btnCancel.className = "btn btn-secondary btn-sm";
    btnCancel.textContent = "Cancel";
    btnCancel.addEventListener("click", () => {
      editForm.remove();
      if (header) header.style.display = "inline-flex";
      bubble.style.display = "block";
      metaBar.style.display = "flex";
    });

    const btnSave = document.createElement("button");
    btnSave.type = "submit";
    btnSave.className = "btn btn-primary btn-sm";
    btnSave.textContent = "Save & Branch";

    actions.appendChild(btnCancel);
    actions.appendChild(btnSave);
    editForm.appendChild(textarea);
    editForm.appendChild(actions);

    editForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const updatedText = textarea.value.trim();
      if (!updatedText) return;

      editForm.remove();
      editMessageNode(currentSession, node.id, updatedText, node.images);
      renderSessions();
      renderStage();

      await triggerAssistantGeneration();
    });

    wrapper.insertBefore(editForm, metaBar);
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
  }

  // Regenerate assistant response
  async function executeRegenerate(assistantNode: ChatNode) {
    if (!isServerRunning) {
      showToast("Start the server on the Overview tab before generating responses.", "warning", "Server Stopped");
      return;
    }
    regenerateAssistantNode(currentSession, assistantNode.id);
    renderStage();
    await triggerAssistantGeneration();
  }

  // Trigger inference generation for the current active branch with live streaming
  async function triggerAssistantGeneration() {
    if (!isServerRunning) {
      showToast("Start the server on the Overview tab before testing in the playground.", "warning", "Server Stopped");
      return;
    }

    const sendBtn = document.querySelector<HTMLButtonElement>("#btn-chat-send");
    if (sendBtn) sendBtn.disabled = true;

    const path = getActiveConversationPath(currentSession);
    let activeNode = path.length > 0 ? path[path.length - 1] : undefined;

    let targetNode: ChatNode;
    if (!activeNode || activeNode.role === "user") {
      targetNode = addMessageToSession(currentSession, "assistant", "");
      renderStage();
    } else {
      targetNode = activeNode;
      targetNode.content = "";
      saveSession(currentSession);
      renderStage();
    }

    const bubbleEl = document.querySelector<HTMLElement>(`#chat-node-${targetNode.id} .chat-entry`);
    if (bubbleEl) {
      renderBubbleContent(bubbleEl, "", true);
    }

    const messagesToSend = getActiveConversationPath(currentSession)
      .filter((n) => n.id !== targetNode.id)
      .map((n) => {
        if (n.role === "user" && n.images && n.images.length > 0) {
          const parts: any[] = [];
          if (n.content && n.content.trim()) {
            parts.push({ type: "text", text: n.content.trim() });
          } else {
            parts.push({ type: "text", text: "Describe this image in detail." });
          }
          n.images.forEach((img) => {
            parts.push({
              type: "image_url",
              image_url: { url: img },
            });
          });
          return { role: n.role, content: parts };
        }
        return { role: n.role, content: n.content };
      });

    // Inject persona system prompt
    const persona = getPersonaById(currentSession.personaId);
    if (persona && persona.systemPrompt) {
      messagesToSend.unshift({ role: "system" as any, content: persona.systemPrompt });
    }

    try {
      const tempVal = Number($<HTMLInputElement>("#cfg-temp")?.value || 0.7);
      const routerSelect = document.querySelector<HTMLSelectElement>("#chat-router-model-select");
      const routerCheck = document.querySelector<HTMLInputElement>("#cfg-router-mode");
      const modelArg = routerCheck && routerCheck.checked && routerSelect && routerSelect.value ? routerSelect.value : "";
      const reasoningEffort = reasoningSelect ? reasoningSelect.value : (currentSession.reasoningEffort || "default");

      // Build payload for llama-server
      const reqBody: any = {
        messages: messagesToSend,
        temperature: tempVal,
        stream: true,
      };
      if (modelArg) {
        reqBody.model = modelArg;
      }
      if (reasoningEffort && reasoningEffort !== "default" && reasoningEffort !== "auto") {
        if (reasoningEffort === "off" || reasoningEffort === "none") {
          reqBody.chat_template_kwargs = { enable_thinking: false };
        } else {
          reqBody.reasoning_effort = reasoningEffort;
          reqBody.chat_template_kwargs = { reasoning_effort: reasoningEffort };
        }
      }

      const fetchHost = (!currentServerHost || currentServerHost === "0.0.0.0") ? "127.0.0.1" : currentServerHost;
      const streamUrl = `http://${fetchHost}:${currentServerPort}/v1/chat/completions`;

      let streamedSuccessfully = false;
      let reasoningAccumulated = "";
      let contentAccumulated = "";
      let accumulatedText = "";

      try {
        const resp = await fetch(streamUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(reqBody),
        });

        if (resp.ok && resp.body) {
          streamedSuccessfully = true;
          const reader = resp.body.getReader();
          const decoder = new TextDecoder("utf-8");
          let sseBuffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            sseBuffer += decoder.decode(value, { stream: true });
            const lines = sseBuffer.split("\n");
            sseBuffer = lines.pop() || "";

            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed || trimmed.startsWith(":")) continue;
              if (trimmed === "data: [DONE]") {
                break;
              }
              if (trimmed.startsWith("data:")) {
                const jsonStr = trimmed.slice(5).trim();
                try {
                  const chunk = JSON.parse(jsonStr);
                  const delta = chunk.choices?.[0]?.delta;
                  if (delta) {
                    if (delta.reasoning_content) {
                      reasoningAccumulated += delta.reasoning_content;
                    }
                    if (delta.content) {
                      contentAccumulated += delta.content;
                    }

                    if (reasoningAccumulated && !contentAccumulated.includes("<think>")) {
                      accumulatedText = `<think>\n${reasoningAccumulated}\n</think>\n${contentAccumulated}`;
                    } else {
                      accumulatedText = contentAccumulated;
                    }

                    targetNode.content = accumulatedText;
                    if (bubbleEl) {
                      renderBubbleContent(bubbleEl, accumulatedText, true);
                    }
                    container.scrollTop = container.scrollHeight;
                  }
                } catch {
                  // Ignore partial chunk parse error
                }
              }
            }
          }

          if (sseBuffer.trim().startsWith("data:") && sseBuffer.trim() !== "data: [DONE]") {
            try {
              const chunk = JSON.parse(sseBuffer.trim().slice(5).trim());
              const delta = chunk.choices?.[0]?.delta;
              if (delta) {
                if (delta.reasoning_content) reasoningAccumulated += delta.reasoning_content;
                if (delta.content) contentAccumulated += delta.content;
                if (reasoningAccumulated && !contentAccumulated.includes("<think>")) {
                  accumulatedText = `<think>\n${reasoningAccumulated}\n</think>\n${contentAccumulated}`;
                } else {
                  accumulatedText = contentAccumulated;
                }
                targetNode.content = accumulatedText;
              }
            } catch {}
          }
        }
      } catch (fetchErr) {
        console.warn("Direct stream fetch failed, falling back to non-streaming RPC:", fetchErr);
        streamedSuccessfully = false;
      }

      // If streaming didn't succeed, fallback to non-streaming RPC
      if (!streamedSuccessfully) {
        const reply = modelArg
          ? await LlamaService.sendChatWithModel(modelArg, messagesToSend, tempVal, reasoningEffort)
          : await LlamaService.sendChat(messagesToSend, tempVal, reasoningEffort);
        targetNode.content = reply;
      }

      saveSession(currentSession);
      renderSessions();
      renderStage();
    } catch (err: any) {
      targetNode.content = `Error: ${err.message || err}`;
      saveSession(currentSession);
      renderStage();
    } finally {
      if (sendBtn) sendBtn.disabled = false;
    }
  }

  // Image attachment button & file picker
  const attachBtn = document.querySelector<HTMLButtonElement>("#btn-chat-attach-image");
  const fileInput = document.querySelector<HTMLInputElement>("#chat-file-input");

  attachBtn?.addEventListener("click", () => {
    if (!isVisionActive()) {
      showToast("Vision adaptor (--mmproj) is not attached. Attach a vision projector in Configuration to input images.", "warning", "Vision Adaptor Required");
      return;
    }
    fileInput?.click();
  });

  fileInput?.addEventListener("change", () => {
    if (fileInput.files) {
      Array.from(fileInput.files).forEach(addChatImageFile);
      fileInput.value = "";
    }
  });

  // Clipboard paste for images
  input.addEventListener("paste", (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (!item) continue;
      if (item.type.indexOf("image") !== -1) {
        if (!isVisionActive()) {
          showToast("Vision adaptor (--mmproj) is not attached. Attach a vision projector in Configuration to input images.", "warning", "Vision Adaptor Required");
          return;
        }
        const file = item.getAsFile();
        if (file) {
          addChatImageFile(file);
          e.preventDefault();
        }
      }
    }
  });

  // Drag and drop images into chat
  const chatInputWrapper = document.querySelector<HTMLElement>("#chat-input-wrapper");
  const chatLog = document.querySelector<HTMLElement>("#chat-messages");

  [chatInputWrapper, chatLog].forEach((target) => {
    if (!target) return;
    target.addEventListener("dragover", (e) => {
      e.preventDefault();
      if (isVisionActive()) {
        chatInputWrapper?.classList.add("chat-drop-active");
      }
    });

    target.addEventListener("dragleave", () => {
      chatInputWrapper?.classList.remove("chat-drop-active");
    });

    target.addEventListener("drop", (e) => {
      e.preventDefault();
      chatInputWrapper?.classList.remove("chat-drop-active");
      if (!isVisionActive()) {
        showToast("Vision adaptor (--mmproj) is not attached. Attach a vision projector in Configuration to input images.", "warning", "Vision Adaptor Required");
        return;
      }
      if (e.dataTransfer?.files) {
        Array.from(e.dataTransfer.files).forEach((file) => {
          if (file.type.startsWith("image/")) {
            addChatImageFile(file);
          }
        });
      }
    });
  });

  // Handle new message submit from bottom bar
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    const images = pendingChatImages.map((img) => img.dataUrl);

    if (!text && images.length === 0) return;

    if (!isServerRunning) {
      showToast("Start the server on the Overview tab before testing in the playground.", "warning", "Server Stopped");
      return;
    }

    input.value = "";
    input.style.height = "auto";
    pendingChatImages = [];
    renderChatImagePreviews();

    addMessageToSession(currentSession, "user", text, images);
    renderSessions();
    renderStage();

    await triggerAssistantGeneration();
  });

  // Initial load
  renderPersonaOptions();
  renderSessions();
  renderStage();
  updateVisionUI();
}


window.addEventListener("DOMContentLoaded", init);
