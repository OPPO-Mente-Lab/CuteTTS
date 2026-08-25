/*
 * Copyright 2026 OPPO and Fudan University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

"use strict";

// Resolve every HTTP/WebSocket endpoint from the script location.  This keeps
// the demo working both at http://host:7860/ and behind gateways that expose it
// under a prefix such as /proxy/7860/.
const applicationBaseUrl = new URL(
  "../",
  document.currentScript?.src || new URL("static/app.js", window.location.href),
);

function appUrl(path) {
  return new URL(String(path).replace(/^\/+/, ""), applicationBaseUrl);
}

const translations = {
  en: {
    connecting: "Connecting to the model…",
    modelLabel: "Checkpoint",
    deviceLabel: "Backend",
    advancedTitle: "Advanced settings",
    advancedHint: "Sampling parameters",
    cfgStrength: "CFG strength",
    diffusionSteps: "Diffusion steps",
    sway: "Sway",
    seed: "Seed",
    modeTitle: "Choose a mode",
    modeHint: "Both modes decode audio patch by patch.",
    referenceMode: "Voice Clone",
    ttsMode: "Text to Speech",
    referenceTitle: "Add a reference",
    referenceHint: "Audio · mono or stereo · 2–30 seconds",
    dropPrimary: "Drop a reference audio file here",
    dropSecondary: "or click to choose a file",
    chooseFile: "Choose file",
    removeReference: "Remove reference",
    textTitle: "Write your text",
    textPlaceholder: "Type what you want to hear…",
    generate: "Generate speech",
    generating: "Generating…",
    waitingModel: "Waiting for the model to be ready.",
    modelReady: "Model ready",
    modelBusy: "Model is generating audio…",
    modelError: "Model failed to load",
    disconnected: "Server unavailable",
    phaseStarting: "Starting the inference service…",
    phaseUnloading: "Unloading the previous runtime…",
    phaseLoadingModel: "Loading the speech model…",
    phaseLoadingSpeaker: "Loading the speaker encoder…",
    phaseCompiling: "Preparing optimized inference…",
    phaseWarmingTts: "Warming up standard TTS…",
    phaseWarmingReference: "Warming up reference synthesis…",
    enterText: "Enter some text to continue.",
    addReference: "Upload a reference audio file to continue.",
    uploading: "Uploading and validating the reference…",
    uploadFailed: "The reference could not be uploaded.",
    invalidAudio: "Choose a supported audio file.",
    fileTooLarge: "The reference must be smaller than 50 MiB.",
    readyToGenerate: "Ready to generate.",
    resultEyebrow: "Output",
    resultTitle: "Generated audio",
    live: "Live",
    complete: "Complete",
    waitingAudio: "Waiting for first audio…",
    streamingAudio: "Receiving decoded audio patches…",
    audioReady: "Audio is ready",
    latency: "Latency",
    waitingMetric: "Waiting…",
    audioLength: "Audio length",
    generationTime: "Generation time",
    rtf: "RTF",
    footer: "Runs locally. Uploaded references and results expire automatically.",
    generationFailed: "Generation failed.",
    connectionClosed: "The generation connection closed unexpectedly.",
    audioUnavailable: "This browser cannot play streamed audio, but generation can continue.",
    patches: "patches",
    playAudio: "Play audio",
    pauseAudio: "Pause audio",
    seekAudio: "Seek audio",
    downloadAudio: "Download audio",
    runtimeReloadFailed: "The runtime could not be switched.",
    invalidSettings: "Check the advanced sampling settings.",
  },
  zh: {
    connecting: "正在连接模型…",
    modelLabel: "模型权重",
    deviceLabel: "计算后端",
    advancedTitle: "高级设置",
    advancedHint: "采样参数",
    cfgStrength: "CFG 强度",
    diffusionSteps: "扩散步数",
    sway: "Sway 系数",
    seed: "随机种子",
    modeTitle: "选择生成模式",
    modeHint: "两种模式均逐 patch 解码音频。",
    referenceMode: "音色克隆",
    ttsMode: "文本转语音",
    referenceTitle: "添加参考音频",
    referenceHint: "音频 · 单声道或双声道 · 2–30 秒",
    dropPrimary: "拖动参考音频到这里",
    dropSecondary: "或点击进入文件系统选择",
    chooseFile: "选择文件",
    removeReference: "移除参考音频",
    textTitle: "输入生成文本",
    textPlaceholder: "输入你想听到的内容…",
    generate: "生成语音",
    generating: "正在生成…",
    waitingModel: "等待模型和预处理准备完成。",
    modelReady: "模型已就绪",
    modelBusy: "模型正在生成音频…",
    modelError: "模型加载失败",
    disconnected: "无法连接服务器",
    phaseStarting: "正在启动推理服务…",
    phaseUnloading: "正在卸载上一个运行时…",
    phaseLoadingModel: "正在加载语音模型…",
    phaseLoadingSpeaker: "正在加载音色编码器…",
    phaseCompiling: "正在准备优化推理…",
    phaseWarmingTts: "正在预热普通 TTS…",
    phaseWarmingReference: "正在预热参考音色合成…",
    enterText: "请输入生成文本。",
    addReference: "请先上传参考音频。",
    uploading: "正在上传并校验参考音频…",
    uploadFailed: "参考音频上传失败。",
    invalidAudio: "请选择后端支持的音频文件。",
    fileTooLarge: "参考音频必须小于 50 MiB。",
    readyToGenerate: "可以开始生成。",
    resultEyebrow: "生成结果",
    resultTitle: "生成音频",
    live: "实时",
    complete: "已完成",
    waitingAudio: "正在等待首包音频…",
    streamingAudio: "正在接收逐 patch 解码音频…",
    audioReady: "音频已生成",
    latency: "延迟",
    waitingMetric: "等待中…",
    audioLength: "音频长度",
    generationTime: "生成时间",
    rtf: "RTF",
    footer: "所有处理均在本地完成，上传音频和生成结果将自动过期。",
    generationFailed: "生成失败。",
    connectionClosed: "生成连接意外断开。",
    audioUnavailable: "当前浏览器不能播放流式音频，但可以继续完成生成。",
    patches: "个 patch",
    playAudio: "播放音频",
    pauseAudio: "暂停音频",
    seekAudio: "拖动音频进度",
    downloadAudio: "下载音频",
    runtimeReloadFailed: "无法切换模型运行时。",
    invalidSettings: "请检查高级采样参数。",
  },
};

const elements = {
  languageToggle: document.querySelector("#language-toggle"),
  languageMenu: document.querySelector("#language-menu"),
  languageOptions: [...document.querySelectorAll("[data-locale]")],
  statusDot: document.querySelector("#status-dot"),
  statusText: document.querySelector("#status-text"),
  runtimeDetail: document.querySelector("#runtime-detail"),
  modelChip: document.querySelector("#model-chip"),
  modelSelect: document.querySelector("#model-select"),
  deviceSelect: document.querySelector("#device-select"),
  modeButtons: [...document.querySelectorAll(".mode-button")],
  referenceSection: document.querySelector("#reference-section"),
  referenceInput: document.querySelector("#reference-input"),
  dropZone: document.querySelector("#drop-zone"),
  referenceFile: document.querySelector("#reference-file"),
  fileName: document.querySelector("#file-name"),
  fileMeta: document.querySelector("#file-meta"),
  removeReference: document.querySelector("#remove-reference"),
  textStep: document.querySelector("#text-step"),
  textInput: document.querySelector("#text-input"),
  characterCount: document.querySelector("#character-count"),
  cfgInput: document.querySelector("#cfg-input"),
  stepsInput: document.querySelector("#steps-input"),
  swayField: document.querySelector("#sway-field"),
  swayInput: document.querySelector("#sway-input"),
  seedInput: document.querySelector("#seed-input"),
  formError: document.querySelector("#form-error"),
  generateButton: document.querySelector("#generate-button"),
  buttonReason: document.querySelector("#button-reason"),
  resultCard: document.querySelector("#result-card"),
  liveBadge: document.querySelector("#live-badge"),
  waveformCanvas: document.querySelector("#waveform-canvas"),
  streamPlayButton: document.querySelector("#stream-play-button"),
  streamProgress: document.querySelector("#stream-progress"),
  streamPosition: document.querySelector("#stream-position"),
  streamStatus: document.querySelector("#stream-status"),
  streamDuration: document.querySelector("#stream-duration"),
  patchStatus: document.querySelector("#patch-status"),
  playerDownload: document.querySelector("#player-download"),
  metricsGrid: document.querySelector("#metrics-grid"),
  metricLatency: document.querySelector("#metric-latency"),
  metricDuration: document.querySelector("#metric-duration"),
  metricGeneration: document.querySelector("#metric-generation"),
  metricRtf: document.querySelector("#metric-rtf"),
};

const state = {
  locale: "en",
  mode: "reference",
  service: null,
  statusFetched: false,
  reference: null,
  uploading: false,
  generating: false,
  runtimeReloading: false,
  runtimeReloadTimer: null,
  runtimeOptionsSignature: "",
  defaultsModelId: null,
  socket: null,
  sampleRate: 24000,
  totalSamples: 0,
  patchCount: 0,
  pcmChunks: [],
  waveformPeaks: [],
  waveformPeak: 0,
  waveformPeakSamples: 0,
  waveformScale: 0.08,
  completed: false,
  audioContext: null,
  audioGain: null,
  isPlaying: false,
  playbackSample: 0,
  playbackStartSample: 0,
  playbackStartedAt: 0,
  scheduledUntil: 0,
  scheduledSources: new Set(),
  seeking: false,
  resumeAfterSeek: false,
};

function t(key) {
  return translations[state.locale][key] || translations.en[key] || key;
}

function setLanguage(locale) {
  state.locale = locale;
  document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
  elements.languageOptions.forEach((option) => {
    option.setAttribute("aria-checked", String(option.dataset.locale === locale));
  });
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.placeholder = t(node.dataset.i18nPlaceholder);
  });
  document.querySelectorAll("[data-i18n-aria]").forEach((node) => {
    node.setAttribute("aria-label", t(node.dataset.i18nAria));
  });
  renderStatus();
  updateGenerateState();
  if (state.completed) {
    elements.liveBadge.lastElementChild.textContent = t("complete");
    elements.streamStatus.textContent = t("audioReady");
  } else if (state.generating) {
    elements.streamStatus.textContent = state.patchCount ? t("streamingAudio") : t("waitingAudio");
    if (!state.patchCount) elements.metricLatency.textContent = t("waitingMetric");
  }
  renderPlaybackButton();
}

function phaseLabel(phase) {
  const labels = {
    starting: "phaseStarting",
    unloading: "phaseUnloading",
    loading_model: "phaseLoadingModel",
    loading_speaker: "phaseLoadingSpeaker",
    compiling: "phaseCompiling",
    warming_tts: "phaseWarmingTts",
    warming_reference: "phaseWarmingReference",
  };
  return t(labels[phase] || "connecting");
}

function replaceOptions(select, options) {
  const fragment = document.createDocumentFragment();
  options.forEach((option) => {
    const node = document.createElement("option");
    node.value = option.id;
    node.textContent = option.label;
    fragment.appendChild(node);
  });
  select.replaceChildren(fragment);
}

function applyModelDefaults() {
  const service = state.service;
  const modelId = service?.selected_model;
  const defaults = service?.defaults;
  if (!modelId || !defaults || state.defaultsModelId === modelId) return;
  state.defaultsModelId = modelId;
  elements.cfgInput.value = String(defaults.cfg_strength ?? 2);
  elements.stepsInput.value = String(defaults.diffusion_steps ?? 10);
  elements.swayInput.value = String(defaults.diffusion_sway_coefficient ?? -0.8);
  elements.seedInput.value = String(defaults.seed ?? 42);
  const swaySupported = defaults.sway_supported !== false;
  elements.swayField.hidden = !swaySupported;
  elements.swayInput.disabled = !swaySupported;
  const allowed = defaults.allowed_diffusion_steps;
  if (Array.isArray(allowed)) {
    elements.stepsInput.min = String(Math.min(...allowed));
    elements.stepsInput.max = String(Math.max(...allowed));
    elements.stepsInput.dataset.allowed = allowed.join(",");
  } else {
    elements.stepsInput.min = "1";
    elements.stepsInput.removeAttribute("max");
    delete elements.stepsInput.dataset.allowed;
  }
}

function syncRuntimeControls() {
  const service = state.service;
  const models = service?.models || [];
  const devices = service?.devices || [];
  const signature = JSON.stringify([
    models.map(({ id, label }) => [id, label]),
    devices.map(({ id, label }) => [id, label]),
  ]);
  if (signature !== state.runtimeOptionsSignature) {
    state.runtimeOptionsSignature = signature;
    replaceOptions(elements.modelSelect, models);
    replaceOptions(elements.deviceSelect, devices);
  }
  if (!state.runtimeReloading) {
    if (service?.selected_model) elements.modelSelect.value = service.selected_model;
    if (service?.requested_device) elements.deviceSelect.value = service.requested_device;
  }
  applyModelDefaults();
  const locked = Boolean(
    state.generating
    || state.runtimeReloading
    || service?.loading
    || service?.busy,
  );
  elements.modelSelect.disabled = locked || models.length === 0;
  elements.deviceSelect.disabled = locked || devices.length === 0;
}

function samplingParameters() {
  const cfgStrength = Number(elements.cfgInput.value);
  const diffusionSteps = Number(elements.stepsInput.value);
  const seed = Number(elements.seedInput.value);
  const swaySupported = state.service?.defaults?.sway_supported !== false;
  const sway = swaySupported ? Number(elements.swayInput.value) : 0;
  const allowed = String(elements.stepsInput.dataset.allowed || "")
    .split(",")
    .filter(Boolean)
    .map(Number);
  const valid = Number.isFinite(cfgStrength)
    && cfgStrength >= 0
    && Number.isInteger(diffusionSteps)
    && diffusionSteps >= 1
    && (!allowed.length || allowed.includes(diffusionSteps))
    && Number.isInteger(seed)
    && Number.isFinite(sway)
    && (!swaySupported || (sway >= -1 && sway <= 1.751938));
  return { cfgStrength, diffusionSteps, sway, seed, valid };
}

function scheduleRuntimeReload() {
  if (!elements.modelSelect.value || !elements.deviceSelect.value) return;
  window.clearTimeout(state.runtimeReloadTimer);
  state.runtimeReloading = true;
  syncRuntimeControls();
  updateGenerateState();
  state.runtimeReloadTimer = window.setTimeout(async () => {
    clearError();
    try {
      const response = await fetch(appUrl("api/runtime/load"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: elements.modelSelect.value,
          device: elements.deviceSelect.value,
        }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || t("runtimeReloadFailed"));
    } catch (error) {
      state.runtimeReloading = false;
      showError(error instanceof Error ? error.message : t("runtimeReloadFailed"));
    }
    await refreshStatus();
  }, 250);
}

function renderStatus() {
  const service = state.service;
  elements.statusDot.className = "status-dot";
  if (!service) {
    elements.statusDot.classList.add(state.statusFetched ? "error" : "loading");
    elements.statusText.textContent = state.statusFetched ? t("disconnected") : t("connecting");
    elements.runtimeDetail.textContent = "—";
    return;
  }
  elements.modelChip.textContent = service.model_label || service.model || "CuteTTS";
  const runtimeDevice = service.requested_device === "auto" && service.resolved_device
    ? `auto → ${service.resolved_device}`
    : (service.resolved_device || service.requested_device || "—");
  elements.runtimeDetail.textContent = `${runtimeDevice} · ${Number(service.sample_rate || 24000).toLocaleString()} Hz`;
  if (service.error || service.phase === "error") {
    elements.statusDot.classList.add("error");
    elements.statusText.textContent = t("modelError");
    elements.runtimeDetail.textContent = service.error || elements.runtimeDetail.textContent;
  } else if (service.busy || service.phase === "busy" || state.generating) {
    elements.statusDot.classList.add("busy");
    elements.statusText.textContent = t("modelBusy");
  } else if (service.ready) {
    elements.statusDot.classList.add("ready");
    elements.statusText.textContent = t("modelReady");
  } else {
    elements.statusDot.classList.add("loading");
    elements.statusText.textContent = phaseLabel(service.phase);
  }
}

async function refreshStatus() {
  try {
    const response = await fetch(appUrl("api/status"), { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.service = await response.json();
    if (
      state.runtimeReloading
      && (
        state.service.loading
        || state.service.phase === "error"
        || (
          state.service.ready
          && state.service.selected_model === elements.modelSelect.value
          && state.service.requested_device === elements.deviceSelect.value
        )
      )
    ) state.runtimeReloading = false;
  } catch (_) {
    state.service = null;
  }
  state.statusFetched = true;
  syncRuntimeControls();
  renderStatus();
  updateGenerateState();
}

function updateGenerateState() {
  const ready = Boolean(state.service?.ready) && !state.service?.busy;
  const hasText = Boolean(elements.textInput.value.trim());
  const hasReference = state.mode === "tts" || Boolean(state.reference);
  const sampling = samplingParameters();
  const enabled = ready
    && hasText
    && hasReference
    && sampling.valid
    && !state.uploading
    && !state.generating
    && !state.runtimeReloading;
  elements.generateButton.disabled = !enabled;
  elements.generateButton.classList.toggle("generating", state.generating);

  if (state.generating) elements.buttonReason.textContent = "";
  else if (state.uploading) elements.buttonReason.textContent = t("uploading");
  else if (!ready || state.runtimeReloading) elements.buttonReason.textContent = t("waitingModel");
  else if (!hasReference) elements.buttonReason.textContent = t("addReference");
  else if (!hasText) elements.buttonReason.textContent = t("enterText");
  else if (!sampling.valid) elements.buttonReason.textContent = t("invalidSettings");
  else elements.buttonReason.textContent = t("readyToGenerate");
  syncRuntimeControls();
}

function showError(message) {
  elements.formError.textContent = message;
  elements.formError.hidden = false;
}

function clearError() {
  elements.formError.hidden = true;
  elements.formError.textContent = "";
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}

function formatClock(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}

function formatSeconds(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  return `${seconds.toFixed(2)} s`;
}

async function deleteReference(referenceId) {
  if (!referenceId) return;
  try {
    await fetch(appUrl(`api/references/${encodeURIComponent(referenceId)}`), { method: "DELETE" });
  } catch (_) {
    // Temporary files expire on the server, so failed cleanup is non-fatal.
  }
}

async function uploadReference(file) {
  clearError();
  if (!file || file.size <= 0) {
    showError(t("invalidAudio"));
    return;
  }
  if (file.size > 50 * 1024 * 1024) {
    showError(t("fileTooLarge"));
    return;
  }

  state.uploading = true;
  elements.dropZone.classList.add("uploading");
  updateGenerateState();
  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const response = await fetch(appUrl("api/references"), { method: "POST", body: form });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || t("uploadFailed"));
    const previous = state.reference;
    state.reference = body;
    elements.fileName.textContent = body.original_name || file.name;
    elements.fileMeta.textContent = `${formatBytes(body.size_bytes || file.size)} · ${Number(body.duration_seconds).toFixed(1)} s · ${Number(body.sample_rate).toLocaleString()} Hz`;
    elements.dropZone.hidden = true;
    elements.referenceFile.hidden = false;
    if (previous?.reference_id) deleteReference(previous.reference_id);
  } catch (error) {
    showError(error instanceof Error ? error.message : t("uploadFailed"));
  } finally {
    state.uploading = false;
    elements.dropZone.classList.remove("uploading");
    elements.referenceInput.value = "";
    updateGenerateState();
  }
}

async function removeReference() {
  const current = state.reference;
  state.reference = null;
  elements.referenceFile.hidden = true;
  elements.dropZone.hidden = false;
  elements.fileName.textContent = "";
  elements.fileMeta.textContent = "";
  updateGenerateState();
  if (current?.reference_id) await deleteReference(current.reference_id);
}

function setMode(mode) {
  if (state.generating || !["reference", "tts"].includes(mode)) return;
  state.mode = mode;
  elements.modeButtons.forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-checked", String(active));
  });
  elements.referenceSection.hidden = mode === "tts";
  elements.textStep.textContent = mode === "tts" ? "02" : "03";
  clearError();
  updateGenerateState();
}

async function ensureAudioContext() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return false;
  if (!state.audioContext || state.audioContext.state === "closed") {
    state.audioContext = new AudioContextClass();
    state.audioGain = state.audioContext.createGain();
    state.audioGain.connect(state.audioContext.destination);
  }
  if (state.audioContext.state === "suspended") await state.audioContext.resume();
  return true;
}

function renderPlaybackButton() {
  elements.streamPlayButton.classList.toggle("playing", state.isPlaying);
  elements.streamPlayButton.setAttribute(
    "aria-label",
    t(state.isPlaying ? "pauseAudio" : "playAudio"),
  );
  elements.streamPlayButton.disabled = state.totalSamples === 0 && !state.generating;
}

function currentPlaybackSample() {
  const context = state.audioContext;
  if (!state.isPlaying || !context) return state.playbackSample;
  const elapsed = Math.max(0, context.currentTime - state.playbackStartedAt);
  return Math.min(
    state.totalSamples,
    state.playbackStartSample + Math.round(elapsed * state.sampleRate),
  );
}

function stopScheduledAudio() {
  state.scheduledSources.forEach((source) => {
    try { source.stop(); } catch (_) { /* already stopped */ }
  });
  state.scheduledSources.clear();
  state.scheduledUntil = state.audioContext?.currentTime || 0;
}

function scheduleSamples(samples, startAt) {
  const context = state.audioContext;
  if (!context || !state.audioGain || samples.length === 0) return startAt;
  const audioBuffer = context.createBuffer(1, samples.length, state.sampleRate);
  audioBuffer.copyToChannel(samples, 0);
  const source = context.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(state.audioGain);
  source.start(startAt);
  state.scheduledSources.add(source);
  source.addEventListener(
    "ended",
    () => state.scheduledSources.delete(source),
    { once: true },
  );
  return startAt + audioBuffer.duration;
}

function scheduleFromSample(targetSample) {
  const context = state.audioContext;
  if (!context || !state.isPlaying || targetSample >= state.totalSamples) return;
  const startAt = context.currentTime + 0.025;
  let cursorTime = startAt;
  for (const chunk of state.pcmChunks) {
    const chunkEnd = chunk.start + chunk.samples.length;
    if (chunkEnd <= targetSample) continue;
    const offset = Math.max(0, targetSample - chunk.start);
    cursorTime = scheduleSamples(chunk.samples.subarray(offset), cursorTime);
  }
  state.playbackSample = targetSample;
  state.playbackStartSample = targetSample;
  state.playbackStartedAt = startAt;
  state.scheduledUntil = cursorTime;
}

async function startPlayback(targetSample = state.playbackSample) {
  const available = await ensureAudioContext().catch(() => false);
  if (!available) {
    elements.streamStatus.textContent = t("audioUnavailable");
    return;
  }
  stopScheduledAudio();
  let target = Math.max(0, Math.min(Math.round(targetSample), state.totalSamples));
  if (state.completed && state.totalSamples > 0 && target >= state.totalSamples) target = 0;
  state.playbackSample = target;
  state.isPlaying = true;
  scheduleFromSample(target);
  renderPlaybackButton();
}

function pausePlayback() {
  state.playbackSample = currentPlaybackSample();
  state.isPlaying = false;
  stopScheduledAudio();
  renderPlaybackButton();
  updatePlayerProgress();
}

function float32LittleEndian(buffer) {
  const view = new DataView(buffer);
  const samples = new Float32Array(Math.floor(buffer.byteLength / 4));
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = view.getFloat32(index * 4, true);
  }
  return samples;
}

function appendWaveformSamples(samples) {
  const peakBlockSamples = Math.max(1, Math.round(state.sampleRate / 100));
  for (let index = 0; index < samples.length; index += 1) {
    state.waveformPeak = Math.max(state.waveformPeak, Math.abs(samples[index]));
    state.waveformPeakSamples += 1;
    if (state.waveformPeakSamples >= peakBlockSamples) {
      const peak = Math.min(1, state.waveformPeak);
      state.waveformPeaks.push(peak);
      state.waveformScale = Math.max(state.waveformScale, peak);
      state.waveformPeak = 0;
      state.waveformPeakSamples = 0;
    }
  }
}

function drawWaveform(playedSample) {
  const canvas = elements.waveformCanvas;
  const bounds = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(bounds.width));
  const height = Math.max(1, Math.round(bounds.height));
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const pixelWidth = Math.round(width * dpr);
  const pixelHeight = Math.round(height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);

  const sourcePeaks = state.waveformPeaks.length
    ? state.waveformPeaks
    : Array.from({ length: 48 }, (_, index) => 0.045 + 0.018 * Math.sin(index * 0.72) ** 2);
  const barStep = 5;
  const barWidth = 2.5;
  const barCount = Math.max(1, Math.floor(width / barStep));
  const playedRatio = state.totalSamples > 0 ? playedSample / state.totalSamples : 0;
  const scale = Math.max(0.08, state.waveformScale);

  for (let bar = 0; bar < barCount; bar += 1) {
    const start = Math.floor((bar * sourcePeaks.length) / barCount);
    const end = Math.max(start + 1, Math.floor(((bar + 1) * sourcePeaks.length) / barCount));
    let peak = 0;
    for (let index = start; index < Math.min(end, sourcePeaks.length); index += 1) {
      peak = Math.max(peak, sourcePeaks[index]);
    }
    const normalized = Math.min(1, Math.pow(peak / scale, 0.58));
    const barHeight = Math.max(3, normalized * (height - 8));
    const x = bar * barStep + (barStep - barWidth) / 2;
    const y = (height - barHeight) / 2;
    context.fillStyle = (bar + 0.5) / barCount <= playedRatio
      ? "#64d2ff"
      : "rgba(255, 255, 255, 0.25)";
    context.beginPath();
    if (typeof context.roundRect === "function") {
      context.roundRect(x, y, barWidth, barHeight, barWidth / 2);
      context.fill();
    } else {
      context.fillRect(x, y, barWidth, barHeight);
    }
  }
}

function playPcmPatch(buffer) {
  const samples = float32LittleEndian(buffer);
  if (!samples.length) return;
  const oldTotal = state.totalSamples;
  state.pcmChunks.push({ start: oldTotal, samples });
  appendWaveformSamples(samples);
  state.totalSamples += samples.length;
  state.patchCount += 1;
  elements.streamDuration.textContent = formatClock(state.totalSamples / state.sampleRate);
  elements.streamProgress.max = String(state.totalSamples / state.sampleRate);
  elements.streamProgress.disabled = false;
  elements.streamStatus.textContent = t("streamingAudio");
  elements.patchStatus.textContent = `${state.patchCount} ${t("patches")}`;
  renderPlaybackButton();

  const context = state.audioContext;
  if (!context || !state.audioGain || !state.isPlaying) return;
  if (oldTotal === 0 || state.scheduledUntil <= context.currentTime + 0.01) {
    stopScheduledAudio();
    state.playbackSample = oldTotal;
    scheduleFromSample(oldTotal);
  } else {
    state.scheduledUntil = scheduleSamples(samples, state.scheduledUntil);
  }
}

function resetResult() {
  stopScheduledAudio();
  state.isPlaying = false;
  state.playbackSample = 0;
  state.playbackStartSample = 0;
  state.playbackStartedAt = 0;
  state.totalSamples = 0;
  state.patchCount = 0;
  state.pcmChunks = [];
  state.waveformPeaks = [];
  state.waveformPeak = 0;
  state.waveformPeakSamples = 0;
  state.waveformScale = 0.08;
  state.completed = false;
  elements.resultCard.hidden = false;
  elements.liveBadge.classList.remove("complete");
  elements.liveBadge.lastElementChild.textContent = t("live");
  elements.streamStatus.textContent = t("waitingAudio");
  elements.patchStatus.textContent = "";
  elements.streamPosition.textContent = "0:00";
  elements.streamDuration.textContent = "0:00";
  elements.streamProgress.min = "0";
  elements.streamProgress.max = "0";
  elements.streamProgress.value = "0";
  elements.streamProgress.disabled = true;
  elements.metricLatency.textContent = t("waitingMetric");
  elements.metricDuration.textContent = "—";
  elements.metricGeneration.textContent = "—";
  elements.metricRtf.textContent = "—";
  elements.metricsGrid.hidden = false;
  elements.playerDownload.hidden = true;
  elements.playerDownload.removeAttribute("href");
  renderPlaybackButton();
}

function completeResult(message) {
  const metrics = message.metrics || {};
  state.completed = true;
  elements.liveBadge.classList.add("complete");
  elements.liveBadge.lastElementChild.textContent = t("complete");
  elements.streamStatus.textContent = t("audioReady");
  elements.metricLatency.textContent = formatSeconds(Number(metrics.ttfa_seconds));
  elements.metricDuration.textContent = formatSeconds(Number(metrics.audio_duration_seconds));
  elements.metricGeneration.textContent = formatSeconds(Number(metrics.generation_seconds));
  elements.metricRtf.textContent = Number(metrics.rtf).toFixed(3);
  elements.metricsGrid.hidden = false;
  if (message.download_url) {
    const url = appUrl(message.download_url).href;
    elements.playerDownload.href = url;
    elements.playerDownload.download = `cutetts-${message.job_id || "generated"}.wav`;
    elements.playerDownload.hidden = false;
  }
}

function updatePlayerProgress() {
  const sample = currentPlaybackSample();
  if (!state.seeking) {
    elements.streamProgress.value = String(sample / state.sampleRate);
  }
  elements.streamPosition.textContent = formatClock(sample / state.sampleRate);
  const maximum = Number(elements.streamProgress.max) || 0;
  const ratio = maximum > 0 ? Math.min(100, (Number(elements.streamProgress.value) / maximum) * 100) : 0;
  elements.streamProgress.style.setProperty("--played", `${ratio}%`);
  drawWaveform(sample);

  const context = state.audioContext;
  if (
    state.completed
    && state.isPlaying
    && sample >= state.totalSamples
    && context
    && context.currentTime >= state.scheduledUntil - 0.01
  ) {
    state.playbackSample = state.totalSamples;
    state.isPlaying = false;
    renderPlaybackButton();
  }
}

function playbackAnimationFrame() {
  updatePlayerProgress();
  window.requestAnimationFrame(playbackAnimationFrame);
}

async function generate() {
  if (elements.generateButton.disabled) return;
  clearError();
  resetResult();
  state.generating = true;
  updateGenerateState();
  renderStatus();

  const audioReady = await ensureAudioContext().catch(() => false);
  if (!audioReady) {
    elements.streamStatus.textContent = t("audioUnavailable");
  } else {
    state.isPlaying = true;
    state.playbackSample = 0;
    renderPlaybackButton();
  }

  const websocketUrl = appUrl("api/generate");
  websocketUrl.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(websocketUrl);
  state.socket = socket;
  socket.binaryType = "arraybuffer";
  let receivedTerminalMessage = false;
  const sampling = samplingParameters();

  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({
      mode: state.mode,
      text: elements.textInput.value.trim(),
      reference_id: state.mode === "reference" ? state.reference?.reference_id : null,
      cfg_strength: sampling.cfgStrength,
      diffusion_steps: sampling.diffusionSteps,
      diffusion_sway_coefficient: sampling.sway,
      seed: sampling.seed,
    }));
  });

  socket.addEventListener("message", (event) => {
    if (event.data instanceof ArrayBuffer) {
      playPcmPatch(event.data);
      return;
    }
    let message;
    try { message = JSON.parse(event.data); } catch (_) { return; }
    if (message.type === "start") {
      state.sampleRate = Number(message.sample_rate) || 24000;
    } else if (message.type === "latency") {
      elements.metricLatency.textContent = formatSeconds(Number(message.ttfa_seconds));
    } else if (message.type === "complete") {
      receivedTerminalMessage = true;
      completeResult(message);
    } else if (message.type === "error") {
      receivedTerminalMessage = true;
      state.completed = true;
      showError(message.message || t("generationFailed"));
    }
  });

  socket.addEventListener("error", () => {
    if (!receivedTerminalMessage) showError(t("generationFailed"));
  });

  socket.addEventListener("close", () => {
    if (!receivedTerminalMessage) {
      state.completed = true;
      showError(t("connectionClosed"));
    }
    state.generating = false;
    state.socket = null;
    updateGenerateState();
    refreshStatus();
  });
}

function setLanguageMenu(open) {
  elements.languageMenu.hidden = !open;
  elements.languageToggle.setAttribute("aria-expanded", String(open));
}

elements.languageToggle.addEventListener("click", (event) => {
  event.stopPropagation();
  setLanguageMenu(elements.languageMenu.hidden);
});
elements.languageOptions.forEach((option) => option.addEventListener("click", () => {
  setLanguage(option.dataset.locale);
  setLanguageMenu(false);
}));
document.addEventListener("click", (event) => {
  if (!elements.languageMenu.hidden && !event.target.closest(".language-selector")) {
    setLanguageMenu(false);
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") setLanguageMenu(false);
});
elements.modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
elements.modelSelect.addEventListener("change", scheduleRuntimeReload);
elements.deviceSelect.addEventListener("change", scheduleRuntimeReload);
[elements.cfgInput, elements.stepsInput, elements.swayInput, elements.seedInput]
  .forEach((input) => input.addEventListener("input", () => {
    clearError();
    updateGenerateState();
  }));
elements.textInput.addEventListener("input", () => {
  elements.characterCount.textContent = String(elements.textInput.value.length);
  clearError();
  updateGenerateState();
});
elements.referenceInput.addEventListener("change", () => {
  const [file] = elements.referenceInput.files;
  if (file) uploadReference(file);
});
elements.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.referenceInput.click();
  }
});
["dragenter", "dragover"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  if (!state.uploading) elements.dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove("dragging");
}));
elements.dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) uploadReference(file);
});
elements.removeReference.addEventListener("click", removeReference);
elements.generateButton.addEventListener("click", generate);
elements.streamPlayButton.addEventListener("click", () => {
  if (state.isPlaying) pausePlayback();
  else startPlayback();
});
elements.streamProgress.addEventListener("pointerdown", () => {
  state.resumeAfterSeek = state.isPlaying;
  state.seeking = true;
  if (state.isPlaying) pausePlayback();
});
elements.streamProgress.addEventListener("input", () => {
  if (!state.seeking) {
    state.resumeAfterSeek = state.isPlaying;
    state.seeking = true;
    if (state.isPlaying) pausePlayback();
  }
  const seconds = Number(elements.streamProgress.value) || 0;
  state.playbackSample = Math.max(
    0,
    Math.min(state.totalSamples, Math.round(seconds * state.sampleRate)),
  );
  elements.streamPosition.textContent = formatClock(state.playbackSample / state.sampleRate);
});
function finishSeeking() {
  if (!state.seeking) return;
  state.seeking = false;
  const resume = state.resumeAfterSeek;
  state.resumeAfterSeek = false;
  if (resume) startPlayback(state.playbackSample);
  else updatePlayerProgress();
}

elements.streamProgress.addEventListener("change", finishSeeking);
elements.streamProgress.addEventListener("pointerup", finishSeeking);
elements.streamProgress.addEventListener("pointercancel", finishSeeking);

setLanguage("en");
refreshStatus();
setInterval(refreshStatus, 1000);
window.requestAnimationFrame(playbackAnimationFrame);
