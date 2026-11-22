/* ==========================================================================
   DJI Manual Agent — Web UI
   原生 JS，无框架、无 CDN。与 FastAPI 同源，通过 SSE 接收 Agent 事件。
   ========================================================================== */

(() => {
  "use strict";

  const STORAGE = {
    sessions: "dji-agent-sessions-v1",
    prefs: "dji-agent-prefs-v1",
    theme: "dji-agent-theme",
  };
  const DEFAULT_DEMO_TASK = "Mavic 4 Pro 智能飞行电池低温下怎么保养？";
  const MAX_HISTORY = 12;

  const state = {
    sessions: [],
    currentId: null,
    tools: [],
    health: null,
    config: null,
    busy: false,
    abort: null,
    live: null,
    lastResult: null,
    prefs: {
      theme: "dark",
      showInspector: true,
      showCitations: true,
    },
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const app = $("#app");
  const chatEl = $("#chat");
  const messagesEl = $("#messages");
  const welcomeEl = $("#welcome");
  const inputEl = $("#input");
  const sendBtn = $("#sendBtn");
  const stopBtn = $("#stopBtn");
  const sessionListEl = $("#sessionList");
  const toolListEl = $("#toolList");
  const inspectorBody = $("#inspectorBody");
  const inspectorSub = $("#inspectorSub");
  const settingsModal = $("#settingsModal");
  const backdrop = $("#backdrop");
  const toastEl = $("#toast");

  /* ------------------------------------------------------------------ */
  /* Utilities                                                          */
  /* ------------------------------------------------------------------ */

  function uid() {
    try {
      if (window.crypto && crypto.randomUUID) return crypto.randomUUID().slice(0, 8);
    } catch (e) {}
    return Math.random().toString(36).slice(2, 10);
  }

  function nowIso() {
    return new Date().toISOString();
  }

  const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

  function escapeHtml(value) {
    return String(value === undefined || value === null ? "" : value).replace(/[&<>"']/g, (ch) => HTML_ESCAPES[ch]);
  }

  function truncate(value, max = 120) {
    const text = String(value === undefined || value === null ? "" : value);
    return text.length > max ? text.slice(0, max) + "…" : text;
  }

  function formatNumber(value) {
    const number = Number(value || 0);
    return Number.isFinite(number) ? number.toLocaleString("en-US") : String(value);
  }

  function formatRelative(iso) {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    const diff = Date.now() - date.getTime();
    if (diff < 60 * 1000) return "刚刚";
    if (diff < 60 * 60 * 1000) return `${Math.floor(diff / 60000)} 分钟前`;
    if (diff < 24 * 60 * 60 * 1000) return `${Math.floor(diff / 3600000)} 小时前`;
    return `${date.getMonth() + 1}/${date.getDate()}`;
  }

  function citationLabel(citation) {
    const parts = [citation.title || citation.doc_id || citation.chunk_id || citation.ref];
    if (citation.version && !String(parts[0]).includes(citation.version)) {
      parts.push(citation.version);
    }
    if (citation.page !== undefined && citation.page !== null && citation.page !== "" && String(citation.page) !== "0") {
      parts.push(`第${citation.page}页`);
    }
    if (citation.section) parts.push(citation.section);
    return parts.filter(Boolean).join(" · ");
  }

  function copyText(text) {
    if (!text) return;
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(
        () => showToast("已复制"),
        () => fallbackCopy(text)
      );
    } else {
      fallbackCopy(text);
    }
  }

  function fallbackCopy(text) {
    try {
      const area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
      showToast("已复制");
    } catch (e) {
      showToast("复制失败", "error");
    }
  }

  function downloadText(filename, text) {
    const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  let toastTimer = null;
  function showToast(message, type = "") {
    if (!toastEl) return;
    toastEl.textContent = message;
    toastEl.className = "toast" + (type ? ` ${type}` : "");
    toastEl.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toastEl.hidden = true;
    }, 2600);
  }

  function scrollBottom(force = false) {
    if (!chatEl) return;
    const distance = chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight;
    if (force) {
      if (typeof chatEl.scrollTo === "function") {
        chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "auto" });
      } else {
        chatEl.scrollTop = chatEl.scrollHeight;
      }
    } else if (distance < 280) {
      chatEl.scrollTop = chatEl.scrollHeight;
    }
  }

  function isTypingTarget(target) {
    if (!target) return false;
    const tag = target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
  }

  /* ------------------------------------------------------------------ */
  /* Markdown (safe, dependency-free)                                   */
  /* ------------------------------------------------------------------ */

  function renderInline(text) {
    const inlineCodes = [];
    let value = String(text || "");
    value = value.replace(/`([^`]+)`/g, (_match, code) => {
      inlineCodes.push(code);
      return `@@IC${inlineCodes.length - 1}@@`;
    });
    value = value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    value = value.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    value = value.replace(
      /\[([^\]]+)\]\((https?:[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    value = value.replace(
      /(?<![A-Za-z0-9\[［])(S\d+(?:\s*[、,，]?\s*S\d+)+)(?![A-Za-z0-9\]］])/gi,
      (_match, group) =>
        (group.match(/S\d+/gi) || [])
          .map((ref) => `[${ref.toUpperCase()}]`)
          .join("")
    );
    value = value.replace(
      /[\[［](S\d+(?:\s*[,，、]\s*S\d+)*)[\]］]/g,
      (_match, group) =>
        group
          .split(/[,，、]/)
          .map((part) => {
            const ref = part.trim().toUpperCase();
            return `<span class="cite" role="button" tabindex="0" data-ref="${ref}">${ref}</span>`;
          })
          .join(" ")
    );
    value = value.replace(/@@IC(\d+)@@/g, (_match, index) => `<code>${inlineCodes[Number(index)]}</code>`);
    return value;
  }

  function renderMarkdown(input) {
    if (!input) return "";
    const codeBlocks = [];
    let text = escapeHtml(input);
    text = text.replace(/```[^\n]*\n?([\s\S]*?)```/g, (_match, code) => {
      codeBlocks.push(code.replace(/\n$/, ""));
      return `\n@@CODE${codeBlocks.length - 1}@@\n`;
    });

    const lines = text.split(/\r?\n/);
    const output = [];
    let paragraph = [];
    let listType = null;

    const flushParagraph = () => {
      if (!paragraph.length) return;
      output.push(`<p>${renderInline(paragraph.join("<br>"))}</p>`);
      paragraph = [];
    };
    const closeList = () => {
      if (!listType) return;
      output.push(`</${listType}>`);
      listType = null;
    };

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line) {
        flushParagraph();
        closeList();
        continue;
      }
      const codeMatch = line.match(/^@@CODE(\d+)@@$/);
      if (codeMatch) {
        flushParagraph();
        closeList();
        output.push(`<pre><code>${codeBlocks[Number(codeMatch[1])]}</code></pre>`);
        continue;
      }
      const heading = line.match(/^(#{1,4})\s+(.+)$/);
      if (heading) {
        flushParagraph();
        closeList();
        const level = Math.min(6, heading[1].length + 1);
        output.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        continue;
      }
      const unordered = line.match(/^[-*]\s+(.+)$/);
      if (unordered) {
        flushParagraph();
        if (listType !== "ul") {
          closeList();
          output.push("<ul>");
          listType = "ul";
        }
        output.push(`<li>${renderInline(unordered[1])}</li>`);
        continue;
      }
      const ordered = line.match(/^\d+\.\s+(.+)$/);
      if (ordered) {
        flushParagraph();
        if (listType !== "ol") {
          closeList();
          output.push("<ol>");
          listType = "ol";
        }
        output.push(`<li>${renderInline(ordered[1])}</li>`);
        continue;
      }
      if (/^>\s?/.test(line)) {
        flushParagraph();
        closeList();
        output.push(`<blockquote>${renderInline(line.replace(/^>\s?/, ""))}</blockquote>`);
        continue;
      }
      if (/^[-*_]{3,}$/.test(line)) {
        flushParagraph();
        closeList();
        output.push("<hr>");
        continue;
      }
      paragraph.push(line);
    }
    flushParagraph();
    closeList();

    let html = output.join("\n");
    html = html.replace(/@@CODE(\d+)@@/g, (_match, index) => `<pre><code>${codeBlocks[Number(index)]}</code></pre>`);
    return html;
  }

  function parseThought(text) {
    if (!text) return "";
    try {
      const parsed = JSON.parse(text);
      return String(parsed.thought || parsed.final_answer || "").trim();
    } catch (e) {
      return String(text).trim();
    }
  }

  function statusInfo(status) {
    const map = {
      ok: "ok",
      max_steps: "max steps",
      max_tool_calls: "max tools",
      max_llm_calls: "max llm",
      timeout: "timeout",
      error: "error",
      no_answer: "no answer",
      stopped: "stopped",
    };
    const label = map[status] || status || "unknown";
    const cls = status === "ok" ? "ok" : status === "error" ? "error" : "warn";
    return { label, cls };
  }

  /* ------------------------------------------------------------------ */
  /* Sessions                                                           */
  /* ------------------------------------------------------------------ */

  function loadSessions() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORAGE.sessions) || "[]");
      state.sessions = Array.isArray(raw)
        ? raw.filter((s) => s && s.id && Array.isArray(s.messages)).slice(0, 30)
        : [];
    } catch (e) {
      state.sessions = [];
    }
    state.sessions.forEach((session) => {
      session.title = session.title || "新会话";
      session.updatedAt = session.updatedAt || session.createdAt || nowIso();
    });
    state.sessions.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
    state.currentId = state.sessions.length ? state.sessions[0].id : null;
  }

  function saveSessions() {
    try {
      localStorage.setItem(STORAGE.sessions, JSON.stringify(state.sessions.slice(0, 30)));
    } catch (e) {
      showToast("本地存储空间不足，旧会话可能未保存", "error");
    }
  }

  function currentSession() {
    return state.sessions.find((session) => session.id === state.currentId) || null;
  }

  function ensureSession() {
    return currentSession() || createSession();
  }

  function createSession() {
    const session = {
      id: `web-${uid()}`,
      title: "新会话",
      createdAt: nowIso(),
      updatedAt: nowIso(),
      messages: [],
    };
    state.sessions.unshift(session);
    state.sessions = state.sessions.slice(0, 30);
    state.currentId = session.id;
    saveSessions();
    renderSessions();
    renderMessages();
    updateTitleUI();
    return session;
  }

  function selectSession(id) {
    if (!state.sessions.some((session) => session.id === id)) return;
    state.currentId = id;
    renderSessions();
    renderMessages();
    updateTitleUI();
    closeOverlays();
    const session = currentSession();
    const lastWithResult = session ? [...session.messages].reverse().find((m) => m.role === "assistant" && m.result) : null;
    if (lastWithResult) setInspector(lastWithResult.result);
    inputEl.focus();
  }

  function deleteSession(id) {
    const index = state.sessions.findIndex((session) => session.id === id);
    if (index < 0) return;
    state.sessions.splice(index, 1);
    if (state.currentId === id) {
      state.currentId = state.sessions.length ? state.sessions[0].id : null;
      if (!state.currentId) {
        createSession();
        return;
      }
    }
    saveSessions();
    renderSessions();
    renderMessages();
    updateTitleUI();
    showToast("已删除会话");
  }

  function clearAllSessions() {
    if (!window.confirm("清空所有本地会话？该操作不可撤销。")) return;
    state.sessions = [];
    state.currentId = null;
    createSession();
    closeSettings();
    showToast("已清空本地会话");
  }

  function updateSessionTitle(session, task) {
    if (!session) return;
    if (!session.title || session.title === "新会话") {
      session.title = truncate(task.replace(/\s+/g, " "), 30);
    }
    session.updatedAt = nowIso();
  }

  function updateTitleUI() {
    const session = currentSession();
    const title = session ? session.title || "新会话" : "新会话";
    $("#chatTitle").textContent = title;
    $("#sessionPill").textContent = session ? session.id.replace(/^web-/, "").slice(0, 8) : "web";
    document.title = `${title} · DJI 手册 Agent`;
  }

  function renderSessions() {
    if (!sessionListEl) return;
    sessionListEl.innerHTML = "";
    $("#emptySessions").hidden = state.sessions.length > 0;

    state.sessions.forEach((session) => {
      const item = document.createElement("li");
      item.className = "session-wrap";
      const active = session.id === state.currentId ? " active" : "";
      item.innerHTML = `
        <button class="session-item${active}" data-id="${escapeHtml(session.id)}" type="button" title="${escapeHtml(session.title)}">
          <span class="session-icon">◈</span>
          <span class="session-meta">
            <span class="session-name">${escapeHtml(session.title || "新会话")}</span>
            <span class="session-time">${escapeHtml(formatRelative(session.updatedAt))}</span>
          </span>
        </button>
        <button class="session-del" data-del="${escapeHtml(session.id)}" type="button" aria-label="删除会话" title="删除">×</button>`;
      sessionListEl.appendChild(item);
    });
  }

  /* ------------------------------------------------------------------ */
  /* Message rendering                                                  */
  /* ------------------------------------------------------------------ */

  function citationChipHtml(citation) {
    const ref = citation.ref || "S?";
    const label = citationLabel(citation);
    const chunk = citation.chunk_id && citation.chunk_id !== label ? `chunk ${truncate(citation.chunk_id, 64)}` : "";
    return `<div class="citation-chip" data-ref="${escapeHtml(ref)}" role="button" tabindex="0" title="${escapeHtml(label)}">
      <span class="ref">${escapeHtml(ref)}</span>
      <span class="cite-label"><strong>${escapeHtml(truncate(label, 110))}</strong>${chunk ? `<small>${escapeHtml(chunk)}</small>` : ""}</span>
    </div>`;
  }

  function citationsHtml(citations) {
    if (!Array.isArray(citations) || !citations.length) return "";
    return `<div class="citations">${citations.map(citationChipHtml).join("")}</div>`;
  }

  function metaHtml(result) {
    const status = statusInfo(result.status);
    const usage = result.usage || {};
    const steps = Array.isArray(result.steps) ? result.steps : [];
    const toolCalls = usage.tool_calls !== undefined ? usage.tool_calls : steps.filter((s) => s.tool_call).length;
    const parts = [
      `<span class="status-pill ${status.cls}">${escapeHtml(status.label)}</span>`,
      `<span>${steps.length} 步</span>`,
      `<span>${toolCalls} 次工具</span>`,
      `<span>${((Number(result.duration_ms) || 0) / 1000).toFixed(1)}s</span>`,
    ];
    if (usage.total_tokens) parts.push(`<span>${formatNumber(usage.total_tokens)} tokens</span>`);
    if (result.forced_final) parts.push("<span>forced final</span>");
    const grounding = result.metadata && result.metadata.grounding;
    if (grounding && grounding.checked_sentences) {
      const pct = Math.round((grounding.score || 0) * 100);
      const coverage = Math.round((grounding.coverage || 0) * 100);
      const healthy = grounding.passed && coverage >= 90;
      parts.push(
        `<span class="status-pill ${healthy ? "ok" : "warn"}" title="逐句引用支持度 / 引用覆盖率">引用校验 ${pct}%${coverage < 100 ? ` · 覆盖 ${coverage}%` : ""}</span>`
      );
    }
    const claims = result.metadata && result.metadata.claim_verification;
    if (claims && (claims.issues || []).length) {
      parts.push(
        `<span class="status-pill warn" title="第二模型逐句语义复核">语义复核 ${claims.issues.length} 条</span>`
      );
    }
    return `<div class="msg-meta">${parts.join('<span class="meta-sep">·</span>')}</div>`;
  }

  function actionsHtml() {
    return `<div class="msg-actions">
      <button class="action-btn" type="button" data-action="copy">⧉ 复制</button>
      <button class="action-btn" type="button" data-action="trace">⌘ 轨迹</button>
      <button class="action-btn" type="button" data-action="download">↓ 导出</button>
    </div>`;
  }

  function stepHtml(step, maxObservation = 900) {
    const call = step.tool_call;
    const result = step.result;
    let status = "";
    if (result) status = result.ok ? '<span class="step-ok">✓ success</span>' : '<span class="step-fail">✕ failed</span>';
    const args = call ? JSON.stringify(call.arguments || {}, null, 2) : "";
    let observation = "";
    if (result) {
      if (result.content) observation = escapeHtml(truncate(result.content, maxObservation));
      else if (result.error) observation = escapeHtml(result.error);
    }
    return `<div class="step">
      <div class="step-head"><span class="step-no">${escapeHtml(step.index || "")}</span>${call ? `<span class="step-tool">${escapeHtml(call.name)}</span>` : ""}${status}</div>
      ${step.thought ? `<div class="step-thought">${escapeHtml(truncate(step.thought, 280))}</div>` : ""}
      ${args ? `<pre class="step-args">${escapeHtml(args)}</pre>` : ""}
      ${observation ? `<div class="step-observation">${observation}</div>` : ""}
      ${step.error ? `<div class="step-fail">${escapeHtml(step.error)}</div>` : ""}
    </div>`;
  }

  function stepsDetails(steps) {
    if (!Array.isArray(steps) || !steps.length) return "";
    return `<details class="trace-details"><summary>执行轨迹 · ${steps.length} 步</summary><div class="steps">${steps.map((step) => stepHtml(step, 900)).join("")}</div></details>`;
  }

  function finalizeAssistant(el, result, task) {
    if (!el) return;
    el.classList.remove("streaming");
    el.classList.toggle("error", result.status === "error");
    const body = $(".msg-body", el);
    if (!body) return;
    const answer = result.answer || "";
    body.innerHTML = `
      <div class="answer markdown">${renderMarkdown(answer) || "<p>（未返回内容）</p>"}</div>
      ${state.prefs.showCitations ? citationsHtml(result.citations) : ""}
      ${metaHtml(result)}
      ${actionsHtml()}
      ${stepsDetails(result.steps)}`;

    const copyBtn = $('[data-action="copy"]', body);
    if (copyBtn) copyBtn.addEventListener("click", () => copyText(answer));
    const traceBtn = $('[data-action="trace"]', body);
    if (traceBtn) {
      traceBtn.addEventListener("click", () => {
        setInspector(result);
        openInspector();
      });
    }
    const downloadBtn = $('[data-action="download"]', body);
    if (downloadBtn) downloadBtn.addEventListener("click", () => downloadResult(result, task));
    $$(".citation-chip", body).forEach((chip) => {
      chip.addEventListener("click", () => flash(chip));
    });
    el._result = result;
  }

  function messageElement(message, lastTask) {
    if (message.role === "user") {
      const el = document.createElement("div");
      el.className = "message user";
      el.innerHTML = `<div class="avatar">你</div><div class="msg-body"></div>`;
      $(".msg-body", el).textContent = message.content || "";
      return el;
    }
    const el = document.createElement("div");
    el.className = "message assistant";
    el.innerHTML = `<div class="avatar">AI</div><div class="msg-body"></div>`;
    if (message.result) {
      finalizeAssistant(el, message.result, lastTask);
    } else {
      $(".msg-body", el).innerHTML = `<div class="answer markdown">${renderMarkdown(message.content || "")}</div>`;
    }
    return el;
  }

  function renderMessages() {
    if (!messagesEl) return;
    const session = currentSession();
    messagesEl.innerHTML = "";
    const hasMessages = !!(session && session.messages.length);
    welcomeEl.hidden = hasMessages;
    if (!hasMessages) {
      updateTitleUI();
      return;
    }
    let lastTask = "";
    session.messages.forEach((message) => {
      if (message.role === "user") lastTask = message.content || "";
      messagesEl.appendChild(messageElement(message, lastTask));
    });
    updateTitleUI();
    scrollBottom(true);
    requestAnimationFrame(() => scrollBottom(true));
  }

  function appendUserMessage(text) {
    const el = document.createElement("div");
    el.className = "message user";
    el.innerHTML = `<div class="avatar">你</div><div class="msg-body"></div>`;
    $(".msg-body", el).textContent = text;
    messagesEl.appendChild(el);
    welcomeEl.hidden = true;
    scrollBottom(true);
  }

  function createLiveAssistant(task, sessionId) {
    const el = document.createElement("div");
    el.className = "message assistant streaming";
    el.innerHTML = `
      <div class="avatar">AI</div>
      <div class="msg-body">
        <div class="thinking"><span class="dots"><i></i><i></i><i></i></span><span class="thinking-text">Agent 正在分析…</span></div>
        <div class="live-trace"></div>
        <div class="answer markdown"></div>
      </div>`;
    messagesEl.appendChild(el);
    scrollBottom(true);
    return {
      el,
      traceEl: $(".live-trace", el),
      thinkingEl: $(".thinking", el),
      task,
      sessionId,
      assistantAt: nowIso(),
      finished: false,
      rows: [],
      lastToolRow: null,
      pendingAnswer: "",
      endPayload: null,
      result: null,
      streamError: null,
    };
  }

  function addLiveRow(live, kind, html) {
    if (!live || live.finished) return null;
    if (live.thinkingEl) live.thinkingEl.style.display = "none";
    const row = document.createElement("div");
    row.className = `live-row ${kind}`;
    row.innerHTML = `<span class="live-dot"></span><div class="live-content">${html}</div>`;
    live.traceEl.appendChild(row);
    live.rows.push(row);
    if (live.rows.length > 12) {
      const old = live.rows.shift();
      if (old) old.remove();
    }
    scrollBottom();
    return row;
  }

  function formatArgs(args) {
    if (!args || typeof args !== "object") return "";
    const entries = Object.entries(args);
    if (!entries.length) return "";
    return entries
      .map(([key, value]) => {
        let text;
        try {
          text = typeof value === "string" ? value : JSON.stringify(value);
        } catch (e) {
          text = String(value);
        }
        return `${key}=${truncate(text, 46)}`;
      })
      .join("  ");
  }

  /* ------------------------------------------------------------------ */
  /* Streaming                                                          */
  /* ------------------------------------------------------------------ */

  async function consumeSSE(body, onEvent) {
    const reader = body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    const processBlock = (block) => {
      let event = "message";
      const dataLines = [];
      block.split(/\r?\n/).forEach((line) => {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      });
      if (!dataLines.length) return;
      let payload;
      try {
        payload = JSON.parse(dataLines.join("\n"));
      } catch (e) {
        return;
      }
      onEvent(event, payload);
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let index;
      while ((index = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, index);
        buffer = buffer.slice(index + 2);
        processBlock(block);
      }
    }
    buffer += decoder.decode().replace(/\r\n/g, "\n");
    if (buffer.trim()) processBlock(buffer);
  }

  function normalizeEventPayload(type, payload) {
    // 兼容旧版服务端：data 里是 {type, step, payload, timestamp} 事件包
    if (payload && payload.type === type && payload.payload && typeof payload.payload === "object") {
      return payload.payload;
    }
    return payload;
  }

  function handleStreamEvent(type, payload, live) {
    if (!live) return;
    switch (type) {
      case "run_start":
        addLiveRow(live, "active", "正在分析任务并规划检索…");
        break;
      case "llm_response": {
        const thought = parseThought(payload.text || "");
        if (thought) {
          addLiveRow(live, "active", `思考：${escapeHtml(truncate(thought, 260))}`);
        } else if (Array.isArray(payload.tool_calls) && payload.tool_calls.length) {
          addLiveRow(live, "active", "模型决定调用工具…");
        }
        break;
      }
      case "tool_call": {
        const args = formatArgs(payload.arguments);
        const row = addLiveRow(
          live,
          "tool",
          `调用 <code>${escapeHtml(payload.name || "tool")}</code>${args ? ` <span class="live-args">${escapeHtml(args)}</span>` : ""}`
        );
        live.lastToolRow = row;
        break;
      }
      case "tool_result": {
        const row = live.lastToolRow;
        if (row) {
          const content = document.createElement("div");
          content.className = "live-obs";
          content.textContent = truncate(payload.content || payload.error || "", 300);
          $(".live-content", row).appendChild(content);
          if (!payload.ok) row.classList.add("error");
        }
        break;
      }
      case "warning":
        addLiveRow(live, "warn", escapeHtml(payload.message || "警告"));
        break;
      case "budget":
        addLiveRow(live, "warn", `触发停止条件：${escapeHtml(payload.reason || "budget")}`);
        break;
      case "final":
        live.pendingAnswer = payload.answer || "";
        break;
      case "run_end":
        live.endPayload = payload;
        break;
      case "result":
        finalizeRun(live, payload);
        break;
      case "error":
        live.streamError = payload.error || "运行出错";
        addLiveRow(live, "error", escapeHtml(live.streamError));
        break;
      default:
        break;
    }
  }

  async function streamTask(task, history, live) {
    const controller = new AbortController();
    state.abort = controller;
    try {
      const response = await fetch("/v1/agent/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task, session_id: live.sessionId, history }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      if (!response.body) throw new Error("当前浏览器不支持流式响应");
      await consumeSSE(response.body, (type, payload) =>
        handleStreamEvent(type, normalizeEventPayload(type, payload), live)
      );
      if (!live.finished) {
        if (live.endPayload && live.endPayload.answer) {
          finalizeRun(live, resultFromEnd(live));
        } else {
          throw new Error(live.streamError || "流式响应未返回结果");
        }
      }
    } catch (error) {
      if (error.name === "AbortError") {
        abortLive(live);
        return;
      }
      try {
        await runSync(task, history, live);
      } catch (syncError) {
        failLive(live, `${error.message || error}；同步回退失败：${syncError.message || syncError}`);
      }
    }
  }

  async function runSync(task, history, live) {
    const response = await fetch("/v1/agent/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, session_id: live.sessionId, history }),
    });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch (e) {}
      throw new Error(detail);
    }
    const result = await response.json();
    finalizeRun(live, result);
  }

  function resultFromEnd(live) {
    const payload = live.endPayload || {};
    return {
      task: live.task,
      answer: payload.answer || "",
      status: payload.status || "ok",
      citations: [],
      steps: [],
      usage: { tool_calls: payload.tool_calls || 0, total_tokens: 0 },
      trace_id: "",
      duration_ms: payload.duration_ms || 0,
      warnings: payload.warnings || [],
      forced_final: !!payload.forced_final,
      metadata: {},
    };
  }

  function compactResult(result) {
    return {
      task: result.task || "",
      answer: result.answer || "",
      status: result.status || "ok",
      citations: result.citations || [],
      usage: result.usage || {},
      trace_id: result.trace_id || "",
      duration_ms: result.duration_ms || 0,
      warnings: result.warnings || [],
      forced_final: !!result.forced_final,
      steps: (result.steps || []).map((step) => ({
        index: step.index,
        thought: step.thought || "",
        tool_call: step.tool_call || null,
        result: step.result
          ? {
              ok: !!step.result.ok,
              content: truncate(step.result.content || "", 1500),
              error: step.result.error || null,
              citations: step.result.citations || [],
            }
          : null,
      })),
      metadata: {
        provider: (result.metadata && result.metadata.provider) || "",
        model: (result.metadata && result.metadata.model) || "",
        kb: (result.metadata && result.metadata.kb) || "",
        grounding: (result.metadata && result.metadata.grounding) || null,
        claim_verification: (result.metadata && result.metadata.claim_verification) || null,
        evidence_profile: (result.metadata && result.metadata.evidence_profile) || null,
      },
    };
  }

  function finalizeRun(live, result) {
    if (!live || live.finished) return;
    live.finished = true;
    live.result = result;
    finalizeAssistant(live.el, result, live.task);

    const session = state.sessions.find((item) => item.id === live.sessionId);
    if (session) {
      session.messages.push({
        role: "assistant",
        content: result.answer || "",
        at: live.assistantAt,
        result: compactResult(result),
      });
      session.updatedAt = nowIso();
      saveSessions();
      renderSessions();
      if (state.currentId === session.id) updateTitleUI();
    }
    setInspector(result);
    state.lastResult = result;
    scrollBottom();
    requestAnimationFrame(() => scrollBottom(true));
    window.setTimeout(() => scrollBottom(true), 120);
  }

  function abortLive(live) {
    if (!live || live.finished) return;
    addLiveRow(live, "warn", "已停止生成");
    live.finished = true;
    const text = live.pendingAnswer || "（已停止）";
    const body = $(".msg-body", live.el);
    if (body) {
      body.innerHTML = `<div class="answer markdown">${renderMarkdown(text)}</div><div class="msg-meta"><span class="status-pill warn">stopped</span></div>`;
    }
    const session = state.sessions.find((item) => item.id === live.sessionId);
    if (session) {
      session.messages.push({ role: "assistant", content: text, at: live.assistantAt });
      session.updatedAt = nowIso();
      saveSessions();
      renderSessions();
    }
  }

  function failLive(live, message) {
    if (!live || live.finished) return;
    addLiveRow(live, "error", escapeHtml(message));
    live.finished = true;
    live.el.classList.add("error");
    const body = $(".msg-body", live.el);
    if (body) {
      body.innerHTML = `<div class="answer markdown"><p>运行失败：${escapeHtml(message)}</p></div><div class="msg-meta"><span class="status-pill error">error</span></div>`;
    }
    const session = state.sessions.find((item) => item.id === live.sessionId);
    if (session) {
      session.messages.push({ role: "assistant", content: `运行失败：${message}`, at: live.assistantAt });
      session.updatedAt = nowIso();
      saveSessions();
      renderSessions();
    }
  }

  async function sendTask(task) {
    task = String(task || "").trim();
    if (!task) return;
    if (state.busy) {
      showToast("Agent 正在运行，请稍候");
      return;
    }
    const session = ensureSession();
    const history = session.messages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .slice(-MAX_HISTORY)
      .map((message) => ({ role: message.role, content: message.content }));

    session.messages.push({ role: "user", content: task, at: nowIso() });
    updateSessionTitle(session, task);
    saveSessions();
    renderSessions();
    appendUserMessage(task);
    inputEl.value = "";
    autoGrowInput();

    const live = createLiveAssistant(task, session.id);
    state.live = live;
    state.busy = true;
    updateBusyUI();
    try {
      await streamTask(task, history, live);
      if (!live.finished) failLive(live, "未收到运行结果");
    } catch (error) {
      failLive(live, error && error.message ? error.message : String(error));
    } finally {
      state.busy = false;
      state.abort = null;
      state.live = null;
      updateBusyUI();
      scrollBottom();
    }
  }

  function updateBusyUI() {
    sendBtn.disabled = state.busy;
    stopBtn.hidden = !state.busy;
    inputEl.setAttribute("placeholder", state.busy ? "Agent 正在检索与推理…" : "问点什么，例如：固件升级失败怎么办？");
  }

  function stopRun() {
    if (state.abort) state.abort.abort();
  }

  /* ------------------------------------------------------------------ */
  /* Inspector                                                          */
  /* ------------------------------------------------------------------ */

  function setInspector(result) {
    state.lastResult = result || null;
    if (!result) {
      inspectorSub.textContent = "每一步决策与观察";
      inspectorBody.innerHTML = `<div class="inspector-empty"><div class="empty-graphic">⌘</div><p>发送一条消息后，这里会显示 Agent 的思考、工具调用、观察与引用。</p></div>`;
      return;
    }
    const status = statusInfo(result.status);
    const usage = result.usage || {};
    const steps = Array.isArray(result.steps) ? result.steps : [];
    const citations = Array.isArray(result.citations) ? result.citations : [];
    const warnings = Array.isArray(result.warnings) ? result.warnings : [];
    const toolCalls = usage.tool_calls !== undefined ? usage.tool_calls : steps.filter((step) => step.tool_call).length;
    const summary = [
      ["状态", `<span class="status-pill ${status.cls}">${escapeHtml(status.label)}</span>`],
      ["耗时", `${((Number(result.duration_ms) || 0) / 1000).toFixed(2)}s`],
      ["步数", String(steps.length)],
      ["工具", String(toolCalls)],
      ["Tokens", usage.total_tokens ? formatNumber(usage.total_tokens) : "—"],
      ["引用", String(citations.length)],
    ]
      .map(([key, value]) => `<div class="summary-card"><span>${key}</span><strong>${value}</strong></div>`)
      .join("");

    const warningsHtml = warnings.length
      ? `<div class="inspector-section-title">警告</div><div class="inspector-warnings">${warnings.map((warning) => `• ${escapeHtml(warning)}`).join("<br>")}</div>`
      : "";
    const grounding = result.metadata && result.metadata.grounding;
    const groundingHealthy = grounding
      ? grounding.passed && (grounding.coverage || 0) >= 0.9
      : false;
    const claims = result.metadata && result.metadata.claim_verification;
    const claimsHtml = claims
      ? `<div class="inspector-section-title">语义复核</div>
         <div class="grounding-card ${(claims.issues || []).length ? "warn" : "ok"}">
           <div class="grounding-score">
             <strong>${claims.checked || 0}</strong>
             <span>条 claim · ${(claims.issues || []).length} 个问题${claims.error ? ` · ${escapeHtml(claims.error)}` : ""}</span>
           </div>
           ${
             (claims.issues || []).length
               ? `<ul>${claims.issues
                   .slice(0, 5)
                   .map(
                     (issue) =>
                       `<li><b>${escapeHtml(issue.verdict)}</b> ${escapeHtml(issue.claim)}${issue.reason ? ` — ${escapeHtml(issue.reason)}` : ""}</li>`
                   )
                   .join("")}</ul>`
               : '<p class="empty-hint">未发现明显语义问题。</p>'
           }
         </div>`
      : "";
    const groundingHtml = grounding
      ? `<div class="inspector-section-title">引用校验</div>
         <div class="grounding-card ${groundingHealthy ? "ok" : "warn"}">
           <div class="grounding-score">
             <strong>${Math.round((grounding.score || 0) * 100)}%</strong>
             <span>支持度 · ${Math.round((grounding.coverage || 0) * 100)}% 引用覆盖 · ${grounding.supported_sentences}/${grounding.cited_sentences || 0} 句有支持 · ${grounding.uncited_sentences || 0} 句无引用</span>
           </div>
           ${
             (grounding.issues || []).length
               ? `<ul>${grounding.issues
                   .slice(0, 5)
                   .map((issue) => `<li><b>${escapeHtml(issue.kind)}</b> ${escapeHtml(issue.detail)}</li>`)
                   .join("")}</ul>`
               : '<p class="empty-hint">未发现明显引用问题。</p>'
           }
         </div>`
      : "";
    const stepsHtml = steps.length
      ? steps.map((step) => stepHtml(step, 1600)).join("")
      : '<p class="empty-hint">没有工具调用。</p>';
    const citationsHtmlValue = citations.length
      ? citations.map(citationChipHtml).join("")
      : '<p class="empty-hint">没有引用。</p>';

    inspectorSub.textContent =
      (result.metadata && result.metadata.provider ? `${result.metadata.provider} · ${result.metadata.model || ""}` : "") ||
      result.trace_id ||
      "每一步决策与观察";

    inspectorBody.innerHTML = `
      <div class="inspector-summary">${summary}</div>
      <div class="inspector-task">${escapeHtml(result.task || "")}</div>
      ${warningsHtml}
      ${groundingHtml}
      ${claimsHtml}
      <div class="inspector-section-title">执行步骤</div>
      <div class="inspector-steps">${stepsHtml}</div>
      <div class="inspector-section-title">引用</div>
      <div class="inspector-citations">${citationsHtmlValue}</div>
      <details class="raw-json"><summary>原始 JSON</summary><pre>${escapeHtml(JSON.stringify(result, null, 2))}</pre></details>`;
    $$(".citation-chip", inspectorBody).forEach((chip) => chip.addEventListener("click", () => flash(chip)));
  }

  function openInspector() {
    if (window.innerWidth > 1180) {
      app.classList.remove("inspector-collapsed");
      state.prefs.showInspector = true;
      savePrefs();
    } else {
      app.classList.add("inspector-open");
    }
    updateBackdrop();
  }

  function closeInspector() {
    if (window.innerWidth > 1180) {
      app.classList.add("inspector-collapsed");
      state.prefs.showInspector = false;
      savePrefs();
    } else {
      app.classList.remove("inspector-open");
    }
    updateBackdrop();
  }

  function toggleInspector() {
    const isOpen =
      window.innerWidth > 1180
        ? !app.classList.contains("inspector-collapsed")
        : app.classList.contains("inspector-open");
    if (isOpen) closeInspector();
    else openInspector();
  }

  function flash(element) {
    if (!element) return;
    element.classList.add("flash");
    setTimeout(() => element.classList.remove("flash"), 1200);
    try {
      element.scrollIntoView({ block: "nearest", behavior: "smooth" });
    } catch (e) {}
  }

  function scrollToCitation(scope, ref) {
    const chip = scope ? $(`.citation-chip[data-ref="${ref}"]`, scope) : null;
    if (chip) {
      flash(chip);
      return;
    }
    const inspectorChip = $(`.citation-chip[data-ref="${ref}"]`, inspectorBody);
    if (inspectorChip) {
      flash(inspectorChip);
      openInspector();
    }
  }

  /* ------------------------------------------------------------------ */
  /* Settings / health / tools                                          */
  /* ------------------------------------------------------------------ */

  function openSettings() {
    settingsModal.hidden = false;
    $("#themeSelect").value = state.prefs.theme;
    $("#inspectorDefault").checked = !!state.prefs.showInspector;
    $("#citationsDefault").checked = !!state.prefs.showCitations;
    renderServerConfig();
  }

  function closeSettings() {
    settingsModal.hidden = true;
  }

  function renderServerConfig() {
    const container = $("#serverConfig");
    const config = state.config;
    if (!config) {
      container.innerHTML = '<p class="settings-note">配置加载失败，请检查服务是否运行。</p>';
      return;
    }
    const runtime = config.runtime || {};
    const rows = [
      ["Agent", (config.agent && config.agent.name) || "—"],
      ["LLM", `${(config.llm && config.llm.provider) || "—"} / ${(config.llm && config.llm.model) || "—"}`],
      ["Base URL", (config.llm && config.llm.base_url) || "—"],
      ["工具协议", (config.llm && config.llm.tool_mode) || "—"],
      ["API Key", config.llm && config.llm.api_key ? "已配置" : "未配置"],
      ["知识库", `${(config.knowledge && config.knowledge.backend) || "—"} · ${(config.knowledge && config.knowledge.collection) || "—"}`],
      ["Chroma", (config.knowledge && config.knowledge.chroma_path) || "—"],
      ["JSONL", (config.knowledge && config.knowledge.jsonl_path) || "—"],
      ["检索", `top_k=${(config.knowledge && config.knowledge.top_k) ?? "—"} · max_terms=${(config.knowledge && config.knowledge.max_terms) ?? "—"}`],
      ["预算", `${(config.agent && config.agent.max_steps) ?? "—"} 步 / ${(config.agent && config.agent.max_tool_calls) ?? "—"} 工具 / ${(config.agent && config.agent.max_seconds) ?? "—"}s`],
      ["记忆", config.memory && config.memory.enabled ? config.memory.db_path : "关闭"],
      ["追踪", config.trace && config.trace.enabled ? config.trace.dir : "关闭"],
      ["配置", runtime.config_path || config.config_path || "(默认值)"],
    ];
    container.innerHTML = rows
      .map(
        ([key, value]) =>
          `<div class="config-key">${escapeHtml(key)}</div><div class="config-value">${escapeHtml(value)}</div>`
      )
      .join("");
  }

  function renderHealth(ok, errorMessage) {
    const dot = $("#healthDot");
    const label = $("#healthLabel");
    const sub = $("#healthSub");
    const chip = $("#modelChip");
    dot.className = "health-dot";
    if (!ok) {
      dot.classList.add("error");
      label.textContent = "服务未连接";
      sub.textContent = "请启动 python -m agent serve";
      chip.textContent = "offline";
      return;
    }
    const health = state.health || {};
    const llm = health.llm || {};
    const knowledge = health.knowledge || {};
    const countValue =
      knowledge.count !== undefined && knowledge.count !== null
        ? knowledge.count
        : (knowledge.primary && knowledge.primary.count) ||
          (knowledge.fallback && knowledge.fallback.count) ||
          null;
    const count = countValue !== null ? `${countValue} 条片段` : "知识库就绪";
    if (llm.provider === "offline") {
      dot.classList.add("offline");
      label.textContent = "离线策略";
      sub.textContent = `${count} · 无 LLM`;
      chip.textContent = "offline · policy";
    } else if (llm.healthy) {
      dot.classList.add("ok");
      label.textContent = "LLM 在线";
      sub.textContent = `${llm.provider} · ${truncate(llm.model || "", 26)}`;
      chip.textContent = `${llm.provider} · ${truncate(llm.model || "", 28)}`;
    } else {
      dot.classList.add("error");
      label.textContent = "LLM 不可用";
      sub.textContent = truncate(llm.error || "可继续使用离线策略", 42);
      chip.textContent = "LLM offline";
    }
  }

  async function refreshHealth() {
    try {
      const response = await fetch("/healthz", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.health = await response.json();
      renderHealth(true);
    } catch (error) {
      renderHealth(false, error.message);
    }
  }

  function renderTools() {
    $("#toolCount").textContent = String(state.tools.length);
    if (!state.tools.length) {
      toolListEl.innerHTML = '<li class="empty-hint">暂无可用工具</li>';
      return;
    }
    toolListEl.innerHTML = state.tools
      .map((tool) => {
        const risk = tool.risk || "safe";
        return `<li title="${escapeHtml(tool.description || "")}">
          <span class="risk-dot ${escapeHtml(risk)}"></span>
          <code>${escapeHtml(tool.name)}</code>
        </li>`;
      })
      .join("");
  }

  async function loadTools() {
    try {
      const response = await fetch("/v1/tools");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.tools = await response.json();
      renderTools();
    } catch (error) {
      toolListEl.innerHTML = '<li class="empty-hint">工具加载失败</li>';
    }
  }

  async function loadConfig(show = false) {
    try {
      const response = await fetch("/v1/config", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.config = await response.json();
      renderServerConfig();
      if (show) showToast("配置已刷新");
    } catch (error) {
      if (show) showToast("配置加载失败", "error");
    }
  }

  /* ------------------------------------------------------------------ */
  /* Preferences / theme                                                */
  /* ------------------------------------------------------------------ */

  function loadPrefs() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE.prefs) || "{}");
      state.prefs = Object.assign({}, state.prefs, saved || {});
    } catch (e) {}
    const savedTheme = localStorage.getItem(STORAGE.theme);
    if (savedTheme === "light" || savedTheme === "dark") state.prefs.theme = savedTheme;
    applyTheme();
    applyInspectorPreference();
  }

  function savePrefs() {
    try {
      localStorage.setItem(STORAGE.prefs, JSON.stringify(state.prefs));
      localStorage.setItem(STORAGE.theme, state.prefs.theme);
    } catch (e) {}
  }

  function applyTheme() {
    document.documentElement.dataset.theme = state.prefs.theme === "light" ? "light" : "dark";
  }

  function applyInspectorPreference() {
    if (window.innerWidth > 1180) {
      app.classList.toggle("inspector-collapsed", !state.prefs.showInspector);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Overlays                                                           */
  /* ------------------------------------------------------------------ */

  function updateBackdrop() {
    if (!backdrop) return;
    const need =
      (window.innerWidth <= 820 && app.classList.contains("sidebar-open")) ||
      (window.innerWidth <= 1180 && app.classList.contains("inspector-open"));
    backdrop.hidden = !need;
  }

  function closeOverlays() {
    app.classList.remove("sidebar-open", "inspector-open");
    updateBackdrop();
  }

  function openSidebar() {
    app.classList.add("sidebar-open");
    updateBackdrop();
  }

  /* ------------------------------------------------------------------ */
  /* Download / export                                                  */
  /* ------------------------------------------------------------------ */

  function downloadResult(result, task) {
    const lines = [`# ${task || "DJI 手册 Agent 回答"}`, "", result.answer || "", ""];
    if (Array.isArray(result.citations) && result.citations.length) {
      lines.push("## 引用", "");
      result.citations.forEach((citation) => {
        lines.push(`- [${citation.ref}] ${citationLabel(citation)}`);
      });
      lines.push("");
    }
    if (Array.isArray(result.steps) && result.steps.length) {
      lines.push("## 执行轨迹", "");
      result.steps.forEach((step) => {
        const tool = step.tool_call ? ` · ${step.tool_call.name}(${JSON.stringify(step.tool_call.arguments || {})})` : "";
        lines.push(`### Step ${step.index}${tool}`);
        if (step.thought) lines.push("", step.thought);
        if (step.result && step.result.content) {
          lines.push("", "<details><summary>observation</summary>", "", "```", step.result.content, "```", "", "</details>");
        }
        lines.push("");
      });
    }
    downloadText(`dji-agent-${Date.now()}.md`, lines.join("\n"));
    showToast("已导出 Markdown");
  }

  /* ------------------------------------------------------------------ */
  /* Events                                                             */
  /* ------------------------------------------------------------------ */

  function autoGrowInput() {
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
  }

  function bindEvents() {
    sendBtn.addEventListener("click", () => sendTask(inputEl.value));
    stopBtn.addEventListener("click", stopRun);
    inputEl.addEventListener("input", autoGrowInput);
    inputEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendTask(inputEl.value);
      }
    });

    $("#examples").addEventListener("click", (event) => {
      const button = event.target.closest(".example");
      if (button) sendTask(button.dataset.q || "");
    });
    $("#newChatBtn").addEventListener("click", () => {
      createSession();
      closeOverlays();
      inputEl.focus();
    });
    $("#clearSessionsBtn").addEventListener("click", clearAllSessions);
    sessionListEl.addEventListener("click", (event) => {
      const deleteButton = event.target.closest("[data-del]");
      if (deleteButton) {
        event.stopPropagation();
        deleteSession(deleteButton.dataset.del);
        return;
      }
      const item = event.target.closest(".session-item");
      if (item) selectSession(item.dataset.id);
    });

    $("#healthBtn").addEventListener("click", openSettings);
    $("#settingsBtn").addEventListener("click", openSettings);
    $("#closeSettingsBtn").addEventListener("click", closeSettings);
    settingsModal.addEventListener("click", (event) => {
      if (event.target === settingsModal) closeSettings();
    });
    $("#refreshConfigBtn").addEventListener("click", () => loadConfig(true));
    $("#clearAllSessionsBtn").addEventListener("click", clearAllSessions);

    $("#themeSelect").addEventListener("change", (event) => {
      state.prefs.theme = event.target.value === "light" ? "light" : "dark";
      applyTheme();
      savePrefs();
    });
    $("#inspectorDefault").addEventListener("change", (event) => {
      state.prefs.showInspector = event.target.checked;
      savePrefs();
      applyInspectorPreference();
    });
    $("#citationsDefault").addEventListener("change", (event) => {
      state.prefs.showCitations = event.target.checked;
      savePrefs();
      if (!state.busy) renderMessages();
    });

    $("#inspectorBtn").addEventListener("click", toggleInspector);
    $("#closeInspectorBtn").addEventListener("click", closeInspector);
    $("#menuBtn").addEventListener("click", openSidebar);
    $("#closeSidebar").addEventListener("click", () => {
      app.classList.remove("sidebar-open");
      updateBackdrop();
    });
    backdrop.addEventListener("click", closeOverlays);

    messagesEl.addEventListener("click", (event) => {
      const cite = event.target.closest(".cite");
      if (cite) {
        scrollToCitation(cite.closest(".msg-body"), cite.dataset.ref);
        return;
      }
      const chip = event.target.closest(".citation-chip");
      if (chip) flash(chip);
    });

    inspectorBody.addEventListener("click", (event) => {
      const chip = event.target.closest(".citation-chip");
      if (chip) flash(chip);
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        if (!settingsModal.hidden) closeSettings();
        closeOverlays();
        return;
      }
      if (isTypingTarget(event.target)) return;
      if (event.key === "n" || event.key === "N") {
        event.preventDefault();
        createSession();
        inputEl.focus();
      } else if (event.key === "/") {
        event.preventDefault();
        inputEl.focus();
      } else if (event.key === "i" || event.key === "I") {
        event.preventDefault();
        toggleInspector();
      }
    });

    window.addEventListener("resize", () => {
      if (window.innerWidth > 1180) app.classList.remove("inspector-open");
      if (window.innerWidth > 820) app.classList.remove("sidebar-open");
      if (window.innerWidth > 1180) {
        app.classList.toggle("inspector-collapsed", !state.prefs.showInspector);
      }
      updateBackdrop();
    });
  }

  /* ------------------------------------------------------------------ */
  /* Init                                                               */
  /* ------------------------------------------------------------------ */

  async function init() {
    loadPrefs();
    loadSessions();
    bindEvents();
    if (!state.sessions.length) {
      createSession();
    } else {
      state.currentId = state.sessions[0].id;
    }
    renderSessions();
    renderMessages();
    updateTitleUI();
    autoGrowInput();
    updateBusyUI();
    updateBackdrop();

    // 健康/配置/工具并行加载，不阻塞首屏与 ?demo=1 的自动提问
    refreshHealth();
    loadTools();
    loadConfig();
    window.setInterval(refreshHealth, 30000);

    const params = new URLSearchParams(window.location.search);
    const queryParam = params.get("q");
    const demoParam = params.get("demo");
    const initialTask = queryParam || (demoParam ? DEFAULT_DEMO_TASK : "");
    if (initialTask) {
      window.setTimeout(() => sendTask(initialTask), 450);
    } else {
      inputEl.focus();
    }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
