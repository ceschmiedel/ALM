// ALM Federation Console — vanilla JS SPA, no build step, no dependencies.
// Talks to the REST API mounted alongside this static bundle (alm/api/app.py).

// ---------------------------------------------------------------------------
// state & storage
// ---------------------------------------------------------------------------

const store = {
  get apiBase() {
    return localStorage.getItem("alm_api_base") || window.location.origin;
  },
  set apiBase(v) {
    localStorage.setItem("alm_api_base", v);
  },
  get token() {
    return localStorage.getItem("alm_token") || "";
  },
  set token(v) {
    localStorage.setItem("alm_token", v);
  },
  get theme() {
    return localStorage.getItem("alm_theme") || "dark";
  },
  set theme(v) {
    localStorage.setItem("alm_theme", v);
  },
};

const TIER_META = {
  micro_slm: { label: "Micro-SLM", size: "< 1B", role: "Roteamento e classificação", accent: "info" },
  slm: { label: "SLM", size: "1B – 4B", role: "Especialista de domínio padrão", accent: "indigo" },
  small: { label: "Small", size: "4B – 8B", role: "Raciocínio mais longo, ainda em escopo", accent: "success" },
  orchestrator: { label: "Orchestrator", size: "frontier", role: "Planejamento, arbitragem difícil, fallback", accent: "warning" },
};

const METRIC_META = {
  "domain accuracy": { fmt: (v) => v.toFixed(3), higherBetter: true },
  "cost per query": { fmt: (v) => `$${v.toFixed(6)}`, higherBetter: false },
  "latency p95 (ms)": { fmt: (v) => Math.round(v).toLocaleString("pt-BR"), higherBetter: false },
  "fallback rate": { fmt: (v) => v.toFixed(3), higherBetter: null },
  consistency: { fmt: (v) => v.toFixed(3), higherBetter: true },
  auditability: { fmt: (v) => v.toFixed(3), higherBetter: true },
};

// ---------------------------------------------------------------------------
// API client
// ---------------------------------------------------------------------------

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function api(path, { method = "GET", body, timeoutMs = 30000 } = {}) {
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (store.token) headers.Authorization = `Bearer ${store.token}`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(`${store.apiBase}${path}`, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timer);
    throw new ApiError(err.name === "AbortError" ? "tempo esgotado ao contatar a API" : "não foi possível contatar a API", 0);
  }
  clearTimeout(timer);

  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const detail = (data && (data.detail || data.message)) || response.statusText;
    throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail), response.status);
  }
  return data;
}

async function apiUpload(path, formData, { timeoutMs = 5 * 60 * 1000 } = {}) {
  const headers = {};
  if (store.token) headers.Authorization = `Bearer ${store.token}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(`${store.apiBase}${path}`, {
      method: "POST",
      headers,
      body: formData,
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timer);
    throw new ApiError(err.name === "AbortError" ? "tempo esgotado no upload" : "não foi possível contatar a API", 0);
  }
  clearTimeout(timer);
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const detail = (data && (data.detail || data.message)) || response.statusText;
    throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail), response.status);
  }
  return data;
}

const Api = {
  health: () => api("/v1/health"),
  stats: () => api("/v1/stats"),
  sessions: (limit = 8) => api(`/v1/sessions?limit=${limit}`),
  experts: () => api("/v1/experts"),
  expertDetail: (id) => api(`/v1/experts/${encodeURIComponent(id)}`),
  createExpert: (payload) => api("/v1/experts", { method: "POST", body: payload }),
  updateExpert: (id, payload) => api(`/v1/experts/${encodeURIComponent(id)}`, { method: "PATCH", body: payload }),
  deleteExpert: (id) => api(`/v1/experts/${encodeURIComponent(id)}`, { method: "DELETE" }),
  models: () => api("/v1/models"),
  createModel: (payload) => api("/v1/models", { method: "POST", body: payload }),
  deleteModel: (id) => api(`/v1/models/${encodeURIComponent(id)}`, { method: "DELETE" }),
  setModelDefault: (id) => api(`/v1/models/${encodeURIComponent(id)}/default`, { method: "POST" }),
  modelsHealth: () => api("/v1/models/health"),
  modelBackends: () => api("/v1/models/backends"),
  probeModel: (payload) => api("/v1/models/probe", { method: "POST", body: payload, timeoutMs: 60000 }),
  ollamaTags: () => api("/v1/ollama/tags"),
  ollamaRunning: () => api("/v1/ollama/running"),
  packs: () => api("/v1/packs"),
  graphStats: () => api("/v1/graph/stats"),
  cmragStats: () => api("/v1/cmrag/stats"),
  documents: (domain = "") => api(`/v1/cmrag/documents${domain ? `?domain=${encodeURIComponent(domain)}` : ""}`),
  deleteDocument: (id) => api(`/v1/cmrag/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  upload: (formData) => apiUpload("/v1/cmrag/upload", formData),
  ask: (payload) => api("/v1/ask", { method: "POST", body: payload, timeoutMs: 10 * 60 * 1000 }),
  evalDatasets: () => api("/v1/eval/datasets"),
  evalRuns: (limit = 25) => api(`/v1/eval/runs?limit=${limit}`),
  evalRunDetail: (id) => api(`/v1/eval/runs/${encodeURIComponent(id)}`),
  runEval: (payload) => api("/v1/eval/run", { method: "POST", body: payload, timeoutMs: 10 * 60 * 1000 }),
};

// ---------------------------------------------------------------------------
// small DOM helpers
// ---------------------------------------------------------------------------

function h(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtBytes(n) {
  if (!n) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function fmtRelativeTime(iso) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const diffSec = Math.round((Date.now() - then) / 1000);
  const abs = Math.abs(diffSec);
  const units = [
    ["ano", 31536000],
    ["mês", 2592000],
    ["dia", 86400],
    ["hora", 3600],
    ["min", 60],
  ];
  for (const [name, secs] of units) {
    if (abs >= secs) {
      const v = Math.floor(abs / secs);
      return diffSec >= 0 ? `há ${v} ${name}${v > 1 ? "s" : ""}` : `em ${v} ${name}${v > 1 ? "s" : ""}`;
    }
  }
  return "agora";
}

function iconSvg(name) {
  const icons = {
    experts: '<path d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z"/><path d="M5 21a7 7 0 0 1 14 0"/>',
    domains: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    corpus: '<path d="M4 5.5C4 4.7 7.6 4 12 4s8 .7 8 1.5v13c0 .8-3.6 1.5-8 1.5s-8-.7-8-1.5v-13Z"/><path d="M4 5.5C4 6.3 7.6 7 12 7s8-.7 8-1.5"/>',
    shield: '<path d="M12 3l7 3v6c0 4.5-3 7.7-7 9-4-1.3-7-4.5-7-9V6l7-3Z"/>',
    routing: '<circle cx="6" cy="6" r="2.4"/><circle cx="18" cy="6" r="2.4"/><circle cx="12" cy="18" r="2.4"/><path d="M8 7.3 12 15.7M16 7.3 12 15.7"/>',
    warning: '<path d="M12 3.5 21 19H3l9-15.5Z"/><path d="M12 9.5v4.2M12 16.7h.01"/>',
    check: '<path d="M20 6 9 17l-5-5"/>',
    cross: '<path d="M18 6 6 18M6 6l12 12"/>',
    cpu: '<rect x="7" y="7" width="10" height="10" rx="1.4"/><path d="M9 4v3M15 4v3M9 17v3M15 17v3M4 9h3M4 15h3M17 9h3M17 15h3"/>',
    play: '<path d="M7 5.5v13l11-6.5-11-6.5Z"/>',
    inbox: '<path d="M4 12h4l2 3h4l2-3h4"/><path d="M4 12 6 5h12l2 7v6a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 4 18v-6Z"/>',
    trash: '<path d="M4 7h16M9 7V5.5A1.5 1.5 0 0 1 10.5 4h3A1.5 1.5 0 0 1 15 5.5V7m2 0-.7 12.1a2 2 0 0 1-2 1.9H9.7a2 2 0 0 1-2-1.9L7 7"/>',
    refresh: '<path d="M20 11A8 8 0 1 0 6.3 17.3M20 11V5m0 6h-6"/>',
    plug: '<path d="M9 3v4M15 3v4M8 7h8l1 4a5 5 0 0 1-5 6h0a5 5 0 0 1-5-6l1-4Z"/><path d="M12 17v4"/>',
    zap: '<path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z"/>',
    chart: '<path d="M4 19V9m6 10V4m6 15v-7m6 7V11"/>',
    chat: '<path d="M20 15.5a2.5 2.5 0 0 1-2.5 2.5H8l-4 3V6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5Z"/><path d="M8.5 10.5h7M8.5 13.5h4"/>',
    upload: '<path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5"/><path d="M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15"/>',
  };
  return `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${icons[name] || ""}</svg>`;
}

// ---------------------------------------------------------------------------
// toasts
// ---------------------------------------------------------------------------

function toast(message, kind = "info") {
  const el = h(`<div class="toast ${kind}"><div>${escapeHtml(message)}</div></div>`);
  document.getElementById("toasts").appendChild(el);
  setTimeout(() => {
    el.style.transition = "opacity 0.2s, transform 0.2s";
    el.style.opacity = "0";
    el.style.transform = "translateX(16px)";
    setTimeout(() => el.remove(), 220);
  }, 4200);
}

// ---------------------------------------------------------------------------
// charts (inline SVG helpers — no charting library)
// ---------------------------------------------------------------------------

function sparklineSvg(values, { width = 120, height = 32, stroke = "var(--accent)" } = {}) {
  if (!values.length) return `<svg width="${width}" height="${height}"></svg>`;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const step = width / Math.max(values.length - 1, 1);
  const points = values.map((v, i) => `${(i * step).toFixed(1)},${(height - ((v - min) / span) * (height - 4) - 2).toFixed(1)}`).join(" ");
  const last = values[values.length - 1];
  const lastY = (height - ((last - min) / span) * (height - 4) - 2).toFixed(1);
  return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <polyline points="${points}" fill="none" stroke="${stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="${((values.length - 1) * step).toFixed(1)}" cy="${lastY}" r="2.4" fill="${stroke}"/>
  </svg>`;
}

function ringSvg(fraction, { size = 46, thickness = 5, color = "var(--accent)" } = {}) {
  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(1, fraction));
  return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="var(--bg-3)" stroke-width="${thickness}"/>
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="${color}" stroke-width="${thickness}"
      stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - clamped)}"
      transform="rotate(-90 ${size / 2} ${size / 2})" style="transition: stroke-dashoffset 0.6s var(--ease)"/>
  </svg>`;
}

// ---------------------------------------------------------------------------
// router / shell
// ---------------------------------------------------------------------------

const routes = {
  overview: { title: "Visão geral", subtitle: "Saúde da federação e inventário em tempo real", render: renderOverview },
  chat: { title: "Conversar", subtitle: "O roteador escolhe os agentes; a resposta vem com a trilha de decisão", render: renderChat },
  agents: { title: "Agentes", subtitle: "Crie especialistas e atribua o modelo que cada um usa", render: renderAgents },
  data: { title: "Dados", subtitle: "Carregue planilhas e CSVs para o corpus de um domínio", render: renderData },
  orchestration: { title: "Orquestração", subtitle: "Modelos Ollama, atribuição por camada e fallback hospedado", render: renderOrchestration },
  performance: { title: "Performance", subtitle: "Avaliações da federação contra a baseline monolítica", render: renderPerformance },
};

let currentRoute = "overview";
let healthPollTimer = null;

function navigate(route) {
  if (!routes[route]) route = "overview";
  currentRoute = route;
  window.location.hash = `#/${route}`;
  document.querySelectorAll(".nav-item").forEach((btn) => {
    const active = btn.dataset.route === route;
    btn.classList.toggle("active", active);
    if (active) btn.setAttribute("aria-current", "page");
    else btn.removeAttribute("aria-current");
  });
  document.getElementById("pageTitle").textContent = routes[route].title;
  document.getElementById("pageSubtitle").textContent = routes[route].subtitle;
  const content = document.getElementById("content");
  content.innerHTML = "";
  document.getElementById("topbarActions").innerHTML = "";
  routes[route].render(content);
}

function setContent(el, html) {
  el.innerHTML = html;
}

function loadingBlock(label = "Carregando…") {
  return `<div class="empty-state"><div class="spinner"></div><p>${escapeHtml(label)}</p></div>`;
}

function errorBlock(message, { retry } = {}) {
  const el = h(`
    <div class="empty-state">
      ${iconSvg("warning")}
      <p class="title">Não foi possível carregar os dados</p>
      <p class="muted">${escapeHtml(message)}</p>
      ${retry ? `<button class="btn btn-ghost btn-sm" id="retryBtn">${iconSvg("refresh")} Tentar novamente</button>` : ""}
    </div>`);
  if (retry) el.querySelector("#retryBtn").addEventListener("click", retry);
  return el;
}

function emptyBlock(icon, title, subtitle) {
  return `<div class="empty-state">${iconSvg(icon)}<p class="title">${escapeHtml(title)}</p><p class="muted">${subtitle}</p></div>`;
}

// -- connection status ------------------------------------------------------

async function pollHealth() {
  const dot = document.getElementById("connDot");
  const label = document.getElementById("connLabel");
  try {
    const health = await Api.health();
    dot.className = "conn-dot ok";
    label.textContent = `online · ${health.tenant}`;
    return health;
  } catch (err) {
    dot.className = "conn-dot bad";
    label.textContent = "sem conexão";
    return null;
  }
}

// ---------------------------------------------------------------------------
// view: overview
// ---------------------------------------------------------------------------

async function renderOverview(root) {
  setContent(root, loadingBlock("Consultando a saúde da federação…"));
  document.getElementById("topbarActions").innerHTML = `<button class="btn btn-ghost" id="refreshOverview">${iconSvg("refresh")} Atualizar</button>`;

  let health, sessions;
  try {
    [health, sessions] = await Promise.all([Api.health(), Api.sessions(6).catch(() => [])]);
  } catch (err) {
    root.innerHTML = "";
    root.appendChild(errorBlock(err.message, { retry: () => renderOverview(root) }));
    return;
  }

  const warningsHtml = health.warnings.length
    ? `<div class="card">
        <div class="card-header"><h3>Avisos de degradação</h3><span class="badge badge-warning dot">${health.warnings.length}</span></div>
        <div class="card-body" style="display:flex; flex-direction:column; gap:10px;">
          ${health.warnings.map((w) => `<div class="alert alert-warning">${iconSvg("warning")}<div class="alert-body">${escapeHtml(w)}</div></div>`).join("")}
        </div>
      </div>`
    : `<div class="alert alert-success">${iconSvg("check")}<div class="alert-body"><strong>Tudo operacional.</strong> Nenhum aviso de degradação no momento.</div></div>`;

  const sessionsHtml = sessions.length
    ? `<div class="list">${sessions
        .map(
          (s) => `<div class="list-row">
            <div class="list-row-main">
              <div class="swatch">${escapeHtml((s.strategy || "?").slice(0, 2))}</div>
              <div style="min-width:0">
                <div class="list-row-title truncate" style="max-width:360px">${escapeHtml(s.intent_text || "(sem texto)")}</div>
                <div class="list-row-sub">${escapeHtml(s.strategy || "—")} · ${fmtRelativeTime(s.created_at)}</div>
              </div>
            </div>
            ${statusBadge(s.status, s.used_fallback)}
          </div>`
        )
        .join("")}</div>`
    : emptyBlock("inbox", "Nenhuma sessão ainda", "Rode <code>alm ask</code> ou use a API para ver o histórico aqui.");

  root.innerHTML = `
    <div class="grid grid-stats" id="statCards"></div>
    <div class="grid grid-main-side" style="margin-top:16px">
      <div class="card">
        <div class="card-header"><h3>Sessões recentes</h3><p class="muted">${sessions.length} exibidas</p></div>
        <div class="card-body tight">${sessionsHtml}</div>
      </div>
      <div style="display:flex; flex-direction:column; gap:16px">
        ${warningsHtml}
        <div class="card card-pad">
          <div class="section-title">Topologia de serving</div>
          <div class="kv"><span class="k">Modelos registrados</span><span class="v">${health.models.models}</span></div>
          <div class="kv"><span class="k">Bases compartilhadas</span><span class="v">${health.models.shared_bases}</span></div>
          <div class="kv"><span class="k">Adaptadores LoRA</span><span class="v">${health.models.adapters}</span></div>
          <div class="kv"><span class="k">Cargas evitadas</span><span class="v">${health.models.loads_saved}</span></div>
        </div>
      </div>
    </div>`;

  const cards = [
    { label: "Especialistas", value: health.experts, icon: "experts", accent: "indigo", foot: `${health.domains.length} domínio(s)` },
    { label: "Domínios", value: health.domains.length, icon: "domains", accent: "info", foot: health.domains.join(", ") || "—" },
    { label: "Chunks no corpus", value: health.corpus.chunks ?? 0, icon: "corpus", accent: "success", foot: `${health.corpus.documents ?? 0} documento(s)` },
    {
      label: "Governança",
      value: health.governance_enabled ? "Ativa" : "Desligada",
      icon: "shield",
      accent: health.governance_enabled ? "success" : "warning",
      foot: "IBAC por agente",
    },
    {
      label: "Roteamento semântico",
      value: health.semantic_routing ? "Sim" : "Léxico",
      icon: "routing",
      accent: health.semantic_routing ? "success" : "warning",
      foot: "embeddings do Context Graph",
    },
    {
      label: "Fallback disponível",
      value: health.fallback_available ? "Sim" : "Não",
      icon: "zap",
      accent: health.fallback_available ? "success" : "danger",
      foot: health.orchestrator_configured ? "orchestrator configurado" : "sem orchestrator",
    },
  ];
  document.getElementById("statCards").innerHTML = cards.map(statCard).join("");
  document.getElementById("refreshOverview").addEventListener("click", () => renderOverview(root));
}

function statCard({ label, value, icon, accent, foot }) {
  return `<div class="stat-card">
    <div class="stat-head">
      <span class="stat-label">${escapeHtml(label)}</span>
      <span class="stat-icon accent-${accent}">${iconSvg(icon)}</span>
    </div>
    <span class="stat-value">${escapeHtml(String(value))}</span>
    <span class="stat-foot truncate">${escapeHtml(String(foot))}</span>
  </div>`;
}

function statusBadge(status, usedFallback) {
  const map = {
    success: ["badge-success", "sucesso"],
    denied: ["badge-danger", "negado"],
    error: ["badge-danger", "erro"],
  };
  const [cls, label] = map[status] || ["badge-neutral", status || "—"];
  return `<span class="badge ${cls} dot">${label}${usedFallback ? " · fallback" : ""}</span>`;
}

// ---------------------------------------------------------------------------
// view: chat
// ---------------------------------------------------------------------------

const chatState = { messages: [], lastResult: null };

async function renderChat(root) {
  let experts = [];
  let health = null;
  try {
    [experts, health] = await Promise.all([Api.experts(), Api.health().catch(() => null)]);
  } catch (err) {
    root.innerHTML = "";
    root.appendChild(errorBlock(err.message, { retry: () => renderChat(root) }));
    return;
  }

  document.getElementById("topbarActions").innerHTML = `<button class="btn btn-ghost" id="clearChat">${iconSvg("trash")} Limpar conversa</button>`;

  root.innerHTML = `
    <div class="chat-shell">
      <div class="chat-panel">
        <div class="chat-log" id="chatLog"></div>
        <div class="chat-composer">
          <textarea id="chatInput" rows="1" placeholder="Pergunte algo sobre os dados carregados… (Enter envia, Shift+Enter quebra linha)"></textarea>
          <button class="btn btn-primary" id="chatSend">${iconSvg("play")} Enviar</button>
        </div>
      </div>

      <div style="display:flex; flex-direction:column; gap:16px">
        <div class="card card-pad">
          <div class="section-title">Como funciona</div>
          <p class="muted" style="font-size:12.5px; line-height:1.6">
            Você não escolhe o agente. O roteador classifica a pergunta, seleciona os
            especialistas que declararam capacidade sobre ela e, se nenhum tiver confiança
            suficiente, escala para o orchestrator. A resposta mostra qual caminho foi tomado.
          </p>
        </div>

        <div class="card">
          <div class="card-header"><h3>Agentes disponíveis</h3><span class="badge badge-neutral">${experts.length}</span></div>
          <div class="card-body tight" id="chatExperts"></div>
        </div>

        <div class="card" id="lastTraceCard" hidden>
          <div class="card-header"><h3>Trilha da última resposta</h3></div>
          <div class="card-body tight" id="lastTrace"></div>
        </div>
      </div>
    </div>`;

  document.getElementById("chatExperts").innerHTML = experts.length
    ? `<div class="list">${experts
        .map(
          (e) => `<div class="list-row">
            <div class="list-row-main">
              <div class="swatch">${escapeHtml(e.id.slice(0, 2))}</div>
              <div style="min-width:0">
                <div class="list-row-title truncate" style="max-width:180px" title="${escapeHtml(e.id)}">${escapeHtml(e.id)}</div>
                <div class="list-row-sub truncate" style="max-width:180px">${escapeHtml(e.domain)}</div>
              </div>
            </div>
            <span class="badge ${e.calibrated ? "badge-success dot" : "badge-neutral"}" title="${
              e.calibrated ? "confiança calibrada" : "confiança não calibrada"
            }">${e.calibrated ? "cal." : "—"}</span>
          </div>`
        )
        .join("")}</div>`
    : emptyBlock("experts", "Nenhum agente ainda", "Crie um agente na aba <strong>Agentes</strong>.");

  const logEl = document.getElementById("chatLog");
  const inputEl = document.getElementById("chatInput");

  function paintLog() {
    if (!chatState.messages.length) {
      logEl.innerHTML = `<div class="empty-state" style="margin:auto">
        ${iconSvg("chat")}
        <p class="title">Faça uma pergunta</p>
        <p class="muted">${
          health && health.corpus && health.corpus.chunks
            ? `${health.corpus.chunks} trechos indexados em ${escapeHtml((health.domains || []).join(", ") || "nenhum domínio")}.`
            : "Nenhum dado indexado ainda — carregue uma planilha na aba Dados."
        }</p>
      </div>`;
      return;
    }
    logEl.innerHTML = chatState.messages.map(renderMessage).join("");
    logEl.scrollTop = logEl.scrollHeight;
  }

  async function send() {
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = "";
    inputEl.style.height = "auto";
    chatState.messages.push({ role: "user", text });
    chatState.messages.push({ role: "agent", pending: true });
    paintLog();
    document.getElementById("chatSend").disabled = true;

    try {
      const result = await Api.ask({ intent: text, include_trace: true });
      chatState.messages.pop();
      chatState.messages.push({
        role: "agent",
        text: result.answer || "(resposta vazia)",
        status: result.status,
        confidence: result.confidence,
        experts: result.experts || [],
        usedFallback: result.used_fallback,
        citations: result.citations || [],
        metrics: result.metrics || {},
        error: result.error,
      });
      chatState.lastResult = result;
      paintTrace(result);
    } catch (err) {
      chatState.messages.pop();
      chatState.messages.push({ role: "agent", error: err.message, status: "error" });
    } finally {
      document.getElementById("chatSend").disabled = false;
      paintLog();
    }
  }

  function paintTrace(result) {
    const card = document.getElementById("lastTraceCard");
    const body = document.getElementById("lastTrace");
    const events = (result.trace && result.trace.events) || [];
    if (!events.length) {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    body.innerHTML = events
      .map(
        (e) => `<div class="trace-line">
          <span class="trace-layer">${escapeHtml(e.layer || "?")}</span>
          <span style="min-width:0">${escapeHtml(e.summary || e.category || "")}${
            e.expert_id ? ` <span class="muted">· ${escapeHtml(e.expert_id)}</span>` : ""
          }</span>
        </div>`
      )
      .join("");
  }

  document.getElementById("chatSend").addEventListener("click", send);
  inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });
  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
  });
  document.getElementById("clearChat").addEventListener("click", () => {
    chatState.messages = [];
    chatState.lastResult = null;
    document.getElementById("lastTraceCard").hidden = true;
    paintLog();
  });

  paintLog();
  if (chatState.lastResult) paintTrace(chatState.lastResult);
  inputEl.focus();
}

function renderMessage(message) {
  if (message.role === "user") {
    return `<div class="msg user">
      <div class="msg-avatar">EU</div>
      <div class="msg-body">${escapeHtml(message.text)}</div>
    </div>`;
  }
  if (message.pending) {
    return `<div class="msg agent">
      <div class="msg-avatar">${iconSvg("routing")}</div>
      <div class="msg-body"><span class="typing"><span></span><span></span><span></span></span></div>
    </div>`;
  }
  if (message.error) {
    return `<div class="msg agent error">
      <div class="msg-avatar">!</div>
      <div class="msg-body">${escapeHtml(message.error)}</div>
    </div>`;
  }

  const badges = [];
  if (message.usedFallback) {
    badges.push(`<span class="badge badge-warning dot">fallback · orchestrator</span>`);
  }
  (message.experts || []).forEach((e) => badges.push(`<span class="badge badge-accent">${escapeHtml(e)}</span>`));
  if (typeof message.confidence === "number") {
    const cls = message.confidence >= 0.7 ? "badge-success" : message.confidence >= 0.4 ? "badge-warning" : "badge-danger";
    badges.push(`<span class="badge ${cls}">confiança ${message.confidence.toFixed(2)}</span>`);
  }
  if (message.metrics && message.metrics.wall_ms) {
    badges.push(`<span class="badge badge-neutral">${Math.round(message.metrics.wall_ms)} ms</span>`);
  }

  const citations = (message.citations || []).slice(0, 5);
  const citationsHtml = citations.length
    ? `<div class="msg-citations">Fontes:<ol>${citations
        .map(
          (c) => `<li>${escapeHtml(c.title || c.document_id || "documento")}${
            c.domain ? ` <span class="muted">(${escapeHtml(c.domain)})</span>` : ""
          }</li>`
        )
        .join("")}</ol></div>`
    : "";

  return `<div class="msg agent">
    <div class="msg-avatar">${iconSvg("routing")}</div>
    <div>
      <div class="msg-body">${escapeHtml(message.text)}${citationsHtml}</div>
      <div class="msg-meta">${badges.join("")}</div>
    </div>
  </div>`;
}

// ---------------------------------------------------------------------------
// view: agents
// ---------------------------------------------------------------------------

async function renderAgents(root) {
  setContent(root, loadingBlock("Carregando agentes…"));
  document.getElementById("topbarActions").innerHTML = `<button class="btn btn-primary" id="newAgent">${iconSvg("experts")} Novo agente</button>`;

  let experts, models, graph;
  try {
    [experts, models, graph] = await Promise.all([Api.experts(), Api.models(), Api.graphStats()]);
  } catch (err) {
    root.innerHTML = "";
    root.appendChild(errorBlock(err.message, { retry: () => renderAgents(root) }));
    return;
  }

  root.innerHTML = experts.length
    ? `<div class="agent-grid" id="agentGrid"></div>`
    : emptyBlock("experts", "Nenhum agente registrado", "Crie o primeiro agente para o roteador ter a quem delegar.");

  if (experts.length) {
    document.getElementById("agentGrid").innerHTML = experts
      .map((e) => {
        const tierMeta = TIER_META[e.tier] || { label: e.tier, accent: "neutral" };
        return `<div class="agent-card">
          <div class="agent-card-head">
            <div class="swatch">${escapeHtml(e.id.slice(0, 2))}</div>
            <div style="min-width:0; flex:1">
              <div class="agent-card-title truncate">${escapeHtml(e.label || e.id)}</div>
              <div class="agent-card-sub">${escapeHtml(e.domain)} · ${escapeHtml(e.id)}</div>
            </div>
            <span class="badge accent-${tierMeta.accent}">${escapeHtml(tierMeta.label)}</span>
          </div>
          <div class="agent-caps">
            ${(e.capabilities || []).map((c) => `<span class="badge badge-neutral">${escapeHtml(c)}</span>`).join("") || `<span class="muted" style="font-size:12px">sem capacidades</span>`}
          </div>
          <div class="kv"><span class="k">Modelo</span><span class="v">${escapeHtml(e.model || "—")}</span></div>
          <div class="kv"><span class="k">Recupera de</span><span class="v">${escapeHtml((e.retrieval_domains || []).join(", ") || "—")}</span></div>
          <div class="agent-card-foot">
            ${e.calibrated ? `<span class="badge badge-success dot">calibrado</span>` : `<span class="badge badge-warning">não calibrado</span>`}
            <div class="flex gap-2">
              <button class="btn btn-ghost btn-sm edit-agent" data-id="${escapeHtml(e.id)}">Editar</button>
              <button class="btn btn-danger-ghost btn-sm del-agent" data-id="${escapeHtml(e.id)}">${iconSvg("trash")}</button>
            </div>
          </div>
        </div>`;
      })
      .join("");

    root.querySelectorAll(".del-agent").forEach((btn) =>
      btn.addEventListener("click", async () => {
        if (!confirm(`Remover o agente "${btn.dataset.id}"? As capacidades dele saem do grafo de roteamento.`)) return;
        try {
          await Api.deleteExpert(btn.dataset.id);
          toast(`Agente ${btn.dataset.id} removido`, "success");
          renderAgents(root);
        } catch (err) {
          toast(err.message, "error");
        }
      })
    );
    root.querySelectorAll(".edit-agent").forEach((btn) =>
      btn.addEventListener("click", () => openAgentEditor(root, { models, graph, expertId: btn.dataset.id }))
    );
  }

  document.getElementById("newAgent").addEventListener("click", () => openAgentEditor(root, { models, graph }));
}

async function openAgentEditor(root, { models, graph, expertId = "" }) {
  const backdrop = document.getElementById("drawerBackdrop");
  const body = document.getElementById("drawerBody");
  const editing = Boolean(expertId);
  document.getElementById("drawerTitle").textContent = editing ? `Editar ${expertId}` : "Novo agente";
  body.innerHTML = loadingBlock();
  backdrop.hidden = false;

  let spec = null;
  let ollama = { reachable: false, models: [] };
  try {
    const [loadedSpec, tags] = await Promise.all([
      editing ? Api.expertDetail(expertId) : Promise.resolve(null),
      // The registry is often empty on a fresh install, so offering only
      // registered models would leave the picker blank. Ollama's library is
      // the other half of the answer to "which model should this agent use".
      Api.ollamaTags().catch(() => ({ reachable: false, models: [] })),
    ]);
    spec = loadedSpec;
    ollama = tags;
  } catch (err) {
    body.innerHTML = "";
    body.appendChild(errorBlock(err.message));
    return;
  }

  const registeredNames = new Set(models.models.map((m) => m.model_name).filter(Boolean));
  const unregisteredOllama = (ollama.models || []).filter((m) => !registeredNames.has(m.name));

  const registeredGroup = models.models.length
    ? `<optgroup label="Registrados na ALM">${models.models
        .map(
          (m) =>
            `<option value="${escapeHtml(m.model_id)}" ${spec && spec.model === m.model_id ? "selected" : ""}>${escapeHtml(m.model_id)} — ${escapeHtml(TIER_META[m.tier]?.label || m.tier)} · ${escapeHtml(m.backend)}</option>`
        )
        .join("")}</optgroup>`
    : "";

  // Picking one of these registers it on save — the operator should not have to
  // visit another screen first just to make a model selectable here.
  const ollamaGroup = unregisteredOllama.length
    ? `<optgroup label="Do Ollama (registra ao salvar)">${unregisteredOllama
        .map(
          (m) =>
            `<option value="ollama:${escapeHtml(m.name)}">${escapeHtml(m.name)} — ${escapeHtml(m.parameter_size || "?")} · ${fmtBytes(m.size_bytes)}</option>`
        )
        .join("")}</optgroup>`
    : "";

  const modelOptions = registeredGroup + ollamaGroup;
  const noModelsHint = !models.models.length && !unregisteredOllama.length
    ? `<span class="hint">Nenhum modelo disponível. Inicie o Ollama (<code>ollama serve</code>) ou registre um em <strong>Orquestração</strong>.</span>`
    : "";

  const domains = graph.domains || [];
  const capabilities = spec ? spec.capabilities : [];

  body.innerHTML = `
    <div class="field-row">
      <label class="field">
        <span class="field-label">ID do agente</span>
        <input type="text" id="agId" placeholder="ex.: vendas-expert" value="${escapeHtml(spec ? spec.id : "")}" ${editing ? "disabled" : ""} />
      </label>
      <label class="field">
        <span class="field-label">Domínio</span>
        <input list="agDomainList" type="text" id="agDomain" placeholder="ex.: vendas" value="${escapeHtml(spec ? spec.domain : "")}" ${editing ? "disabled" : ""} />
        <datalist id="agDomainList">${domains.map((d) => `<option value="${escapeHtml(d)}">`).join("")}</datalist>
      </label>
    </div>
    <label class="field">
      <span class="field-label">Nome de exibição</span>
      <input type="text" id="agLabel" placeholder="ex.: Especialista de Vendas" value="${escapeHtml(spec ? spec.label : "")}" />
    </label>
    <label class="field">
      <span class="field-label">Descrição</span>
      <input type="text" id="agDescription" placeholder="O que este agente resolve" value="${escapeHtml(spec ? spec.description : "")}" />
    </label>
    <div class="field-row">
      <label class="field">
        <span class="field-label">Modelo</span>
        <select id="agModel">
          <option value="">— nenhum (usa o padrão da camada) —</option>
          ${modelOptions}
        </select>
        ${noModelsHint}
      </label>
      <label class="field">
        <span class="field-label">Camada</span>
        <select id="agTier">
          ${Object.entries(TIER_META)
            .map(([k, v]) => `<option value="${k}" ${spec && spec.tier === k ? "selected" : ""}>${v.label} · ${v.size}</option>`)
            .join("")}
        </select>
      </label>
    </div>
    <label class="field">
      <span class="field-label">Instruções de sistema (opcional)</span>
      <textarea id="agSystem" rows="3" placeholder="Como este especialista deve responder">${escapeHtml(spec ? spec.prompt?.system || "" : "")}</textarea>
    </label>

    <div class="divider"></div>
    <div class="flex-between" style="margin-bottom:10px">
      <div class="section-title" style="margin:0">Capacidades</div>
      <button class="btn btn-ghost btn-sm" id="agAddCap">+ Adicionar</button>
    </div>
    <p class="hint" style="margin-bottom:12px">
      É contra isto que o roteador compara a pergunta. Exemplos valem mais que a descrição —
      são o sinal mais forte que o matcher tem.
    </p>
    <div id="agCaps"></div>

    <div class="flex gap-2" style="margin-top:16px">
      <button class="btn btn-ghost" id="agCancel" style="flex:1; justify-content:center">Cancelar</button>
      <button class="btn btn-primary" id="agSave" style="flex:2; justify-content:center">${editing ? "Salvar alterações" : "Criar agente"}</button>
    </div>
  `;

  const capsEl = document.getElementById("agCaps");

  function capBlock(cap = {}, index) {
    return `<div class="cap-editor" data-cap="${index}">
      <div class="cap-editor-head">
        <span>Capacidade ${index + 1}</span>
        <button class="btn btn-danger-ghost btn-sm remove-cap" data-index="${index}">remover</button>
      </div>
      <label class="field">
        <span class="field-label">ID</span>
        <input type="text" class="cap-id" placeholder="ex.: analisar_vendas" value="${escapeHtml(cap.id || "")}" />
      </label>
      <label class="field">
        <span class="field-label">Descrição</span>
        <input type="text" class="cap-description" placeholder="O que ela faz" value="${escapeHtml(cap.description || "")}" />
      </label>
      <label class="field">
        <span class="field-label">Exemplos de pergunta (um por linha)</span>
        <textarea class="cap-examples" rows="2" placeholder="Qual produto teve maior receita?">${escapeHtml((cap.examples || []).join("\n"))}</textarea>
      </label>
      <label class="field" style="margin-bottom:0">
        <span class="field-label">Palavras-chave (separadas por vírgula)</span>
        <input type="text" class="cap-keywords" placeholder="receita, margem, vendas" value="${escapeHtml((cap.keywords || []).join(", "))}" />
      </label>
    </div>`;
  }

  let capList = capabilities.length ? capabilities.map((c) => ({ ...c })) : [{}];

  function paintCaps() {
    capsEl.innerHTML = capList.map((c, i) => capBlock(c, i)).join("");
    capsEl.querySelectorAll(".remove-cap").forEach((btn) =>
      btn.addEventListener("click", () => {
        if (capList.length === 1) {
          toast("Um agente precisa de pelo menos uma capacidade", "error");
          return;
        }
        readCaps();
        capList.splice(Number(btn.dataset.index), 1);
        paintCaps();
      })
    );
  }

  function readCaps() {
    capList = Array.from(capsEl.querySelectorAll(".cap-editor")).map((el) => ({
      id: el.querySelector(".cap-id").value.trim(),
      description: el.querySelector(".cap-description").value.trim(),
      examples: el.querySelector(".cap-examples").value.split("\n").map((s) => s.trim()).filter(Boolean),
      keywords: el.querySelector(".cap-keywords").value.split(",").map((s) => s.trim()).filter(Boolean),
    }));
    return capList;
  }

  paintCaps();
  document.getElementById("agAddCap").addEventListener("click", () => {
    readCaps();
    capList.push({});
    paintCaps();
  });
  document.getElementById("agCancel").addEventListener("click", () => (backdrop.hidden = true));

  // Picking a model settles the tier: the two disagreeing is not a choice the
  // operator meant to make, it is a mistake waiting to misroute the agent.
  document.getElementById("agModel").addEventListener("change", (event) => {
    const chosen = models.models.find((m) => m.model_id === event.target.value);
    if (chosen) document.getElementById("agTier").value = chosen.tier;
  });

  /** Register an Ollama pick on the fly, returning the model id to bind. */
  async function resolveModelId(selected) {
    if (!selected.startsWith("ollama:")) return selected;
    const modelName = selected.slice("ollama:".length);
    const tier = document.getElementById("agTier").value;
    const modelId = `${tier}-${modelName.replace(/[:/.]/g, "-")}`;
    const already = models.models.some((m) => m.model_id === modelId);
    if (!already) {
      await Api.createModel({
        model_id: modelId,
        tier,
        backend: "ollama",
        model_name: modelName,
        description: `Registrado ao criar um agente em ${new Date().toLocaleString("pt-BR")}`,
      });
    }
    return modelId;
  }

  document.getElementById("agSave").addEventListener("click", async () => {
    const caps = readCaps().filter((c) => c.id);
    if (!caps.length) {
      toast("Informe pelo menos uma capacidade com ID", "error");
      return;
    }
    const saveBtn = document.getElementById("agSave");
    saveBtn.disabled = true;
    try {
      const payload = {
        label: document.getElementById("agLabel").value.trim(),
        description: document.getElementById("agDescription").value.trim(),
        model: await resolveModelId(document.getElementById("agModel").value),
        tier: document.getElementById("agTier").value,
        system_prompt: document.getElementById("agSystem").value.trim(),
        capabilities: caps,
      };
      if (editing) {
        await Api.updateExpert(expertId, payload);
        toast(`${expertId} atualizado`, "success");
      } else {
        const id = document.getElementById("agId").value.trim();
        const domain = document.getElementById("agDomain").value.trim();
        if (!id || !domain) {
          toast("ID e domínio são obrigatórios", "error");
          saveBtn.disabled = false;
          return;
        }
        await Api.createExpert({ ...payload, id, domain, create_domain: true });
        toast(`Agente ${id} criado`, "success");
      }
      backdrop.hidden = true;
      renderAgents(root);
    } catch (err) {
      toast(err.message, "error");
    } finally {
      saveBtn.disabled = false;
    }
  });
}

// ---------------------------------------------------------------------------
// view: data
// ---------------------------------------------------------------------------

async function renderData(root) {
  setContent(root, loadingBlock("Carregando corpus…"));
  document.getElementById("topbarActions").innerHTML = `<button class="btn btn-ghost" id="refreshData">${iconSvg("refresh")} Atualizar</button>`;

  let graph, documents, stats;
  try {
    [graph, documents, stats] = await Promise.all([Api.graphStats(), Api.documents(), Api.cmragStats()]);
  } catch (err) {
    root.innerHTML = "";
    root.appendChild(errorBlock(err.message, { retry: () => renderData(root) }));
    return;
  }

  const domains = graph.domains || [];

  root.innerHTML = `
    <div class="grid grid-main-side">
      <div style="display:flex; flex-direction:column; gap:16px">
        <div class="card">
          <div class="card-header">
            <div><h3>Carregar planilha ou CSV</h3><p class="muted">Cada aba vira um documento; as linhas viram trechos recuperáveis com o cabeçalho junto</p></div>
          </div>
          <div class="card-body">
            <div class="field-row">
              <label class="field">
                <span class="field-label">Domínio de destino</span>
                <input list="dataDomainList" type="text" id="uploadDomain" placeholder="ex.: financeiro" value="${escapeHtml(domains[0] || "")}" />
                <datalist id="dataDomainList">${domains.map((d) => `<option value="${escapeHtml(d)}">`).join("")}</datalist>
              </label>
              <label class="field">
                <span class="field-label">Sensibilidade</span>
                <select id="uploadSensitivity">
                  <option value="public">public</option>
                  <option value="internal" selected>internal</option>
                  <option value="confidential">confidential</option>
                  <option value="restricted">restricted</option>
                </select>
              </label>
            </div>
            <div class="dropzone" id="dropzone">
              ${iconSvg("corpus")}
              <div class="dz-title">Arraste um arquivo aqui ou clique para escolher</div>
              <div class="dz-sub">.csv, .tsv, .xlsx — até 25 MB</div>
              <input type="file" id="fileInput" accept=".csv,.tsv,.txt,.xlsx,.xlsm" hidden />
            </div>
            <div id="uploadResult" style="margin-top:14px"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header"><h3>Documentos no corpus</h3><span class="badge badge-neutral">${documents.length}</span></div>
          <div class="card-body tight" id="docList"></div>
        </div>
      </div>

      <div class="card card-pad" id="corpusStats"></div>
    </div>`;

  function paintStats(current) {
    document.getElementById("corpusStats").innerHTML = `
      <div class="section-title">Corpus</div>
      <div class="kv"><span class="k">Documentos</span><span class="v">${current.documents ?? 0}</span></div>
      <div class="kv"><span class="k">Trechos indexados</span><span class="v">${current.chunks ?? 0}</span></div>
      <div class="divider"></div>
      <div class="section-title">Por domínio</div>
      ${Object.entries(current.chunks_by_domain || {})
        .map(([d, n]) => `<div class="kv"><span class="k">${escapeHtml(d)}</span><span class="v">${n}</span></div>`)
        .join("") || `<p class="muted" style="font-size:12.5px">nada indexado ainda</p>`}`;
  }

  paintStats(stats);
  renderDocList(document.getElementById("docList"), documents, root);

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    if (event.dataTransfer.files.length) upload(event.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) upload(fileInput.files[0]);
  });

  async function upload(file) {
    const domain = document.getElementById("uploadDomain").value.trim();
    if (!domain) {
      toast("Informe o domínio de destino", "error");
      return;
    }
    const resultEl = document.getElementById("uploadResult");
    resultEl.innerHTML = loadingBlock(`Lendo e indexando ${file.name}…`);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("domain", domain);
    formData.append("sensitivity", document.getElementById("uploadSensitivity").value);

    try {
      const result = await Api.upload(formData);
      const sheets = result.sheets || [];
      resultEl.innerHTML = `<div class="alert alert-success">${iconSvg("check")}<div class="alert-body">
        <strong>${escapeHtml(result.filename)}</strong> indexado em <strong>${escapeHtml(result.domain)}</strong>:
        ${sheets.length} documento(s), ${result.chunks} trecho(s).
        <ul style="margin:6px 0 0; padding-left:18px">
          ${sheets
            .map(
              (s) =>
                `<li>${escapeHtml(s.sheet)} — ${s.rows} linha(s), ${(s.columns || []).length} coluna(s)${s.skipped ? " <em>(inalterado)</em>" : ""}</li>`
            )
            .join("")}
        </ul>
      </div></div>`;
      toast("Arquivo indexado", "success");
      const [freshDocs, freshStats] = await Promise.all([Api.documents(), Api.cmragStats()]);
      renderDocList(document.getElementById("docList"), freshDocs, root);
      paintStats(freshStats);
      // The count badge sits in the card header, outside the repainted body.
      const badge = document.querySelector("#docList")?.closest(".card")?.querySelector(".badge");
      if (badge) badge.textContent = freshDocs.length;
    } catch (err) {
      resultEl.innerHTML = `<div class="alert alert-danger">${iconSvg("cross")}<div class="alert-body">${escapeHtml(err.message)}</div></div>`;
      toast(err.message, "error");
    } finally {
      fileInput.value = "";
    }
  }

  document.getElementById("refreshData").addEventListener("click", () => renderData(root));
}

function renderDocList(container, documents, root) {
  if (!documents.length) {
    container.innerHTML = emptyBlock("corpus", "Corpus vazio", "Carregue uma planilha ou instale um pack de domínio.");
    return;
  }
  container.innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>Documento</th><th>Domínio</th><th>Sensibilidade</th><th></th></tr></thead>
    <tbody>${documents
      .map(
        (d) => `<tr>
          <td><div class="truncate" style="max-width:280px" title="${escapeHtml(d.uri || d.title)}">${escapeHtml(d.title)}</div></td>
          <td>${escapeHtml(d.domain)}</td>
          <td><span class="badge badge-neutral">${escapeHtml(d.sensitivity)}</span></td>
          <td><button class="btn btn-danger-ghost btn-sm del-doc" data-id="${escapeHtml(d.document_id)}">${iconSvg("trash")}</button></td>
        </tr>`
      )
      .join("")}</tbody>
  </table></div>`;

  container.querySelectorAll(".del-doc").forEach((btn) =>
    btn.addEventListener("click", async () => {
      if (!confirm("Remover este documento e todos os seus trechos do índice?")) return;
      try {
        const result = await Api.deleteDocument(btn.dataset.id);
        toast(`Documento removido (${result.chunks_removed} trechos)`, "success");
        renderData(root);
      } catch (err) {
        toast(err.message, "error");
      }
    })
  );
}

// ---------------------------------------------------------------------------
// view: orchestration
// ---------------------------------------------------------------------------

async function renderOrchestration(root) {
  setContent(root, loadingBlock("Consultando o daemon Ollama e o registro de modelos…"));
  document.getElementById("topbarActions").innerHTML = `<button class="btn btn-ghost" id="refreshOrch">${iconSvg("refresh")} Atualizar</button>`;

  let tags, running, models, health;
  try {
    [tags, running, models, health] = await Promise.all([
      Api.ollamaTags(),
      Api.ollamaRunning().catch(() => ({ reachable: false, models: [] })),
      Api.models(),
      Api.health(),
    ]);
  } catch (err) {
    root.innerHTML = "";
    root.appendChild(errorBlock(err.message, { retry: () => renderOrchestration(root) }));
    return;
  }

  const runningNames = new Set((running.models || []).map((m) => m.name));
  const registeredNames = new Set(models.models.map((m) => m.model_name || m.served_name));

  root.innerHTML = `
    <div class="grid grid-main-side">
      <div style="display:flex; flex-direction:column; gap:16px">
        <div class="card">
          <div class="card-header">
            <div>
              <h3>Biblioteca Ollama</h3>
              <p class="muted">${tags.base_url}</p>
            </div>
            ${tags.reachable
              ? `<span class="badge badge-success dot">daemon online</span>`
              : `<span class="badge badge-danger dot">indisponível</span>`}
          </div>
          <div class="card-body tight" id="ollamaLibrary"></div>
        </div>

        <div class="card">
          <div class="card-header"><h3>Modelos registrados na federação</h3><span class="badge badge-neutral">${models.models.length}</span></div>
          <div class="card-body tight" id="registeredModels"></div>
        </div>
      </div>

      <div style="display:flex; flex-direction:column; gap:16px">
        <div class="card">
          <div class="card-header"><h3>Atribuir modelo por camada</h3></div>
          <div class="card-body" id="tierAssign"></div>
        </div>
        <div class="card">
          <div class="card-header">
            <div>
              <h3>Fallback hospedado</h3>
              <p class="muted">Modelo grande via API, usado só quando o roteador escala</p>
            </div>
          </div>
          <div class="card-body" id="hostedFallback"></div>
        </div>

        <div class="card card-pad">
          <div class="section-title">Topologia de serving</div>
          <div id="servingTopology"></div>
        </div>
      </div>
    </div>`;

  // -- Ollama library -------------------------------------------------------
  const libEl = document.getElementById("ollamaLibrary");
  if (!tags.reachable) {
    libEl.innerHTML = `<div class="alert alert-warning">${iconSvg("warning")}<div class="alert-body">
      <strong>Não foi possível contatar o daemon Ollama.</strong> ${escapeHtml(tags.error || "")}<br/>
      Rode <code>ollama serve</code> na máquina que hospeda a API, ou ajuste <code>ALM_OLLAMA_BASE_URL</code>.
    </div></div>`;
  } else if (!tags.models.length) {
    libEl.innerHTML = emptyBlock("inbox", "Nenhum modelo baixado", "Rode <code>ollama pull llama3.2</code> (ou outro modelo) para começar.");
  } else {
    libEl.innerHTML = `<div class="list">${tags.models
      .map((m) => {
        const isRunning = runningNames.has(m.name);
        const isRegistered = registeredNames.has(m.name);
        return `<div class="list-row">
          <div class="list-row-main">
            <div class="swatch">${escapeHtml(m.name.slice(0, 2))}</div>
            <div style="min-width:0">
              <div class="list-row-title truncate" style="max-width:220px">${escapeHtml(m.name)}</div>
              <div class="list-row-sub">${escapeHtml(m.parameter_size || "?")} · ${escapeHtml(m.quantization || "?")} · ${fmtBytes(m.size_bytes)}</div>
            </div>
          </div>
          <div class="flex gap-2">
            ${isRunning ? `<span class="badge badge-info dot">em memória</span>` : ""}
            ${isRegistered ? `<span class="badge badge-success dot">atribuído</span>` : `<button class="btn btn-ghost btn-sm assign-btn" data-model="${escapeHtml(m.name)}">Atribuir</button>`}
          </div>
        </div>`;
      })
      .join("")}</div>`;
    libEl.querySelectorAll(".assign-btn").forEach((btn) =>
      btn.addEventListener("click", () => {
        document.getElementById("tierOllamaModel").value = btn.dataset.model;
        syncModelIdSuggestion();
        document.getElementById("tierAssign").scrollIntoView({ behavior: "smooth", block: "center" });
      })
    );
  }

  // -- registered models table ----------------------------------------------
  const regEl = document.getElementById("registeredModels");
  const placeholderCount = models.models.filter((m) => m.backend === "heuristic").length;

  regEl.innerHTML = models.models.length
    ? `${placeholderCount
        ? `<div class="alert alert-warning" style="margin:12px 0">${iconSvg("warning")}<div class="alert-body">
            <strong>${placeholderCount} modelo(s) no backend <code>heuristic</code></strong> — respondedores
            extrativos do pack demo, não modelos de linguagem. Enquanto servirem uma camada, as respostas
            saem truncadas. Atribua um modelo Ollama à camada ou remova-os.
          </div></div>`
        : ""}
      <div class="table-wrap"><table>
        <thead><tr><th>ID</th><th>Camada</th><th>Backend</th><th>Nome servido</th><th class="num">Custo /1k</th><th></th></tr></thead>
        <tbody>${models.models
          .map((m) => {
            const meta = TIER_META[m.tier] || { label: m.tier, accent: "neutral" };
            return `<tr>
              <td class="mono">${escapeHtml(m.model_id)}${
                m.tier_default
                  ? ` <span class="badge badge-success" title="É este que a camada usa">ativo</span>`
                  : ""
              }</td>
              <td><span class="badge accent-${meta.accent}">${escapeHtml(meta.label)}</span></td>
              <td><span class="badge ${m.backend === "heuristic" ? "badge-warning" : "badge-neutral"}">${escapeHtml(m.backend)}</span></td>
              <td class="mono truncate" style="max-width:160px">${escapeHtml(m.model_name || m.adapter || "—")}</td>
              <td class="num mono">$${(m.cost_per_1k_input || 0).toFixed(4)}</td>
              <td class="flex gap-2">
                ${m.tier_default ? "" : `<button class="btn btn-ghost btn-sm use-model" data-id="${escapeHtml(m.model_id)}" title="Usar este na camada">usar</button>`}
                <button class="btn btn-danger-ghost btn-sm del-model" data-id="${escapeHtml(m.model_id)}" title="Remover">${iconSvg("trash")}</button>
              </td>
            </tr>`;
          })
          .join("")}</tbody>
      </table></div>`
    : emptyBlock("cpu", "Nenhum modelo registrado", "Atribua um modelo Ollama a uma camada ao lado para começar.");

  regEl.querySelectorAll(".use-model").forEach((btn) =>
    btn.addEventListener("click", async () => {
      try {
        await Api.setModelDefault(btn.dataset.id);
        toast(`${btn.dataset.id} agora serve a sua camada`, "success");
        renderOrchestration(root);
      } catch (err) {
        toast(err.message, "error");
      }
    })
  );
  regEl.querySelectorAll(".del-model").forEach((btn) =>
    btn.addEventListener("click", async () => {
      if (!confirm(`Remover o modelo "${btn.dataset.id}" do registro?`)) return;
      try {
        await Api.deleteModel(btn.dataset.id);
        toast(`Modelo ${btn.dataset.id} removido`, "success");
        renderOrchestration(root);
      } catch (err) {
        toast(err.message, "error");
      }
    })
  );

  // -- tier assignment form ---------------------------------------------------
  const tierEl = document.getElementById("tierAssign");
  const currentByTier = {};
  models.models.forEach((m) => {
    (currentByTier[m.tier] ||= []).push(m);
  });

  const modelOptions = (tags.models || [])
    .map(
      (m) =>
        `<option value="${escapeHtml(m.name)}">${escapeHtml(m.name)} — ${escapeHtml(m.parameter_size || "?")} · ${fmtBytes(m.size_bytes)}</option>`
    )
    .join("");

  tierEl.innerHTML = `
    <div class="field">
      <span class="field-label">Camada</span>
      <select id="tierSelect">
        ${Object.entries(TIER_META).map(([k, v]) => `<option value="${k}">${v.label} · ${v.size}</option>`).join("")}
      </select>
      <span class="hint" id="tierHint"></span>
    </div>
    <div class="field">
      <span class="field-label">Modelo Ollama</span>
      <select id="tierOllamaModel">
        ${modelOptions || `<option value="">nenhum modelo disponível no daemon</option>`}
      </select>
      ${tags.reachable ? "" : `<span class="hint">Daemon indisponível — inicie o Ollama para listar os modelos.</span>`}
    </div>
    <div class="field-row">
      <label class="field">
        <span class="field-label">ID do modelo na ALM</span>
        <input type="text" id="tierModelId" placeholder="ex.: slm-llama3.2" />
      </label>
      <label class="field">
        <span class="field-label">Janela de contexto</span>
        <input type="number" id="tierContext" value="8192" min="512" step="512" />
      </label>
    </div>
    <div class="divider"></div>
    <div id="tierCurrent"></div>
    <button class="btn btn-primary" id="tierSubmit" style="width:100%; justify-content:center">${iconSvg("plug")} Atribuir modelo à camada</button>
  `;

  function renderTierCurrent() {
    const tier = document.getElementById("tierSelect").value;
    document.getElementById("tierHint").textContent = TIER_META[tier].role;
    const list = currentByTier[tier] || [];
    document.getElementById("tierCurrent").innerHTML = list.length
      ? `<div class="section-title">Atualmente nesta camada</div>` +
        list.map((m) => `<div class="kv"><span class="k mono">${escapeHtml(m.model_id)}</span><span class="v">${escapeHtml(m.model_name || m.adapter)}</span></div>`).join("")
      : `<div class="alert alert-info">${iconSvg("warning")}<div class="alert-body">Nenhum modelo nesta camada ainda — o ${health.fallback_available ? "roteador" : "sistema"} usará o que estiver disponível.</div></div>`;
  }

  function syncModelIdSuggestion() {
    const tier = document.getElementById("tierSelect").value;
    const model = document.getElementById("tierOllamaModel").value.trim();
    if (!model) return;
    const slug = model.replace(/[:/]/g, "-");
    document.getElementById("tierModelId").value = `${tier}-${slug}`;
  }

  document.getElementById("tierSelect").addEventListener("change", () => {
    renderTierCurrent();
    syncModelIdSuggestion();
  });
  document.getElementById("tierOllamaModel").addEventListener("change", syncModelIdSuggestion);
  renderTierCurrent();
  syncModelIdSuggestion();

  document.getElementById("tierSubmit").addEventListener("click", async () => {
    const tier = document.getElementById("tierSelect").value;
    const modelName = document.getElementById("tierOllamaModel").value.trim();
    const modelId = document.getElementById("tierModelId").value.trim();
    const contextWindow = parseInt(document.getElementById("tierContext").value, 10) || 8192;
    if (!modelName || !modelId) {
      toast("Informe o modelo Ollama e o ID do modelo na ALM", "error");
      return;
    }
    const submitBtn = document.getElementById("tierSubmit");
    submitBtn.disabled = true;
    try {
      await Api.createModel({
        model_id: modelId,
        tier,
        backend: "ollama",
        model_name: modelName,
        context_window: contextWindow,
        // Without this the tier keeps resolving to whatever id sorts first —
        // typically a pack's heuristic placeholder — and the assignment would
        // look like it worked while changing nothing.
        tier_default: true,
        description: `Atribuído via console em ${new Date().toLocaleString("pt-BR")}`,
      });
      toast(`${modelId} agora serve a camada ${TIER_META[tier].label}`, "success");
      renderOrchestration(root);
    } catch (err) {
      toast(err.message, "error");
    } finally {
      submitBtn.disabled = false;
    }
  });

  // -- serving topology -------------------------------------------------------
  const topoEl = document.getElementById("servingTopology");
  const groups = Object.entries(models.serving.adapter_groups || {});
  const parts = [];
  parts.push(`<div class="kv"><span class="k">Total de modelos</span><span class="v">${models.serving.models}</span></div>`);
  parts.push(`<div class="kv"><span class="k">Bases compartilhadas</span><span class="v">${models.serving.shared_bases}</span></div>`);
  parts.push(`<div class="kv"><span class="k">Adaptadores LoRA</span><span class="v">${models.serving.adapters}</span></div>`);
  parts.push(`<div class="kv"><span class="k">Cargas de GPU evitadas</span><span class="v">${models.serving.loads_saved}</span></div>`);
  if (groups.length) {
    parts.push('<div class="divider"></div><div class="section-title">Base → adaptadores</div>');
    groups.forEach(([base, adapters]) => {
      parts.push(`<div style="margin-bottom:10px">
        <div class="mono" style="font-size:12.5px; font-weight:700">${escapeHtml(base)}</div>
        <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:6px">
          ${adapters.map((a) => `<span class="badge badge-accent">${escapeHtml(a)}</span>`).join("")}
        </div>
      </div>`);
    });
  }
  topoEl.innerHTML = parts.join("");

  renderHostedFallback(document.getElementById("hostedFallback"), models, root);

  document.getElementById("refreshOrch").addEventListener("click", () => renderOrchestration(root));
}

const HOSTED_PRESETS = {
  openai: { label: "OpenAI", models: ["gpt-4o", "gpt-4o-mini", "o3-mini"], keyHint: "sk-…" },
  anthropic: { label: "Anthropic", models: ["claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5"], keyHint: "sk-ant-…" },
  google: { label: "Google Gemini", models: ["gemini-2.0-flash", "gemini-1.5-pro"], keyHint: "AIza…" },
  openai_compat: { label: "Compatível com OpenAI", models: [], keyHint: "opcional" },
};

function renderHostedFallback(container, models, root) {
  const existing = models.models.filter((m) => m.tier === "orchestrator");

  container.innerHTML = `
    ${existing.length
      ? `<div class="section-title">Orchestrator atual</div>
         ${existing
           .map(
             (m) => `<div class="flex-between" style="margin-bottom:8px">
               <div style="min-width:0">
                 <div class="mono truncate" style="font-size:12.5px; font-weight:650">${escapeHtml(m.model_id)}</div>
                 <div class="list-row-sub">${escapeHtml(m.backend)} · ${escapeHtml(m.model_name || "—")}</div>
               </div>
               <button class="btn btn-ghost btn-sm probe-existing" data-id="${escapeHtml(m.model_id)}">Testar</button>
             </div>`
           )
           .join("")}
         <div class="divider"></div>`
      : `<div class="alert alert-warning" style="margin-bottom:14px">${iconSvg("warning")}<div class="alert-body">
          Sem modelo de orchestrator, o fallback não existe: requisições que o roteador não consegue classificar falham em vez de escalar.
        </div></div>`}

    <div class="field">
      <span class="field-label">Provedor</span>
      <select id="hostedBackend">
        ${Object.entries(HOSTED_PRESETS).map(([k, v]) => `<option value="${k}">${escapeHtml(v.label)}</option>`).join("")}
      </select>
    </div>
    <div class="field">
      <span class="field-label">Modelo</span>
      <select id="hostedModel"></select>
    </div>
    <div class="field" id="hostedCustomWrap" hidden>
      <span class="field-label">Nome do modelo</span>
      <input type="text" id="hostedCustomModel" placeholder="ex.: meu-modelo" />
    </div>
    <div class="field" id="hostedEndpointWrap" hidden>
      <span class="field-label">Endpoint</span>
      <input type="text" id="hostedEndpoint" placeholder="https://meu-servidor/v1" />
    </div>
    <div class="field">
      <span class="field-label">API token</span>
      <input type="password" id="hostedKey" placeholder="" />
      <span class="hint">Fica no servidor, junto do registro de modelos. A API nunca devolve a chave.</span>
    </div>
    <div class="flex gap-2">
      <button class="btn btn-ghost" id="hostedProbe" style="flex:1; justify-content:center">${iconSvg("zap")} Testar</button>
      <button class="btn btn-primary" id="hostedSave" style="flex:1; justify-content:center">${iconSvg("plug")} Salvar</button>
    </div>
    <div id="hostedResult" style="margin-top:12px"></div>
  `;

  const backendEl = container.querySelector("#hostedBackend");
  const modelEl = container.querySelector("#hostedModel");

  function syncPreset() {
    const preset = HOSTED_PRESETS[backendEl.value];
    const custom = !preset.models.length;
    modelEl.innerHTML = preset.models.map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");
    modelEl.parentElement.hidden = custom;
    container.querySelector("#hostedCustomWrap").hidden = !custom;
    container.querySelector("#hostedEndpointWrap").hidden = backendEl.value !== "openai_compat";
    container.querySelector("#hostedKey").placeholder = preset.keyHint;
  }
  backendEl.addEventListener("change", syncPreset);
  syncPreset();

  function readForm() {
    const backend = backendEl.value;
    const custom = !HOSTED_PRESETS[backend].models.length;
    return {
      backend,
      model_name: custom ? container.querySelector("#hostedCustomModel").value.trim() : modelEl.value,
      endpoint: container.querySelector("#hostedEndpoint").value.trim(),
      api_key: container.querySelector("#hostedKey").value.trim(),
    };
  }

  function showResult(result) {
    const el = container.querySelector("#hostedResult");
    el.innerHTML = result.ok
      ? `<div class="alert alert-success">${iconSvg("check")}<div class="alert-body"><strong>Respondeu em ${Math.round(result.latency_ms)} ms.</strong> ${escapeHtml((result.reply || "").slice(0, 80))}</div></div>`
      : `<div class="alert alert-danger">${iconSvg("cross")}<div class="alert-body"><strong>Falhou.</strong> ${escapeHtml(result.error || "sem resposta")}</div></div>`;
  }

  container.querySelectorAll(".probe-existing").forEach((btn) =>
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        showResult(await Api.probeModel({ model_id: btn.dataset.id }));
      } catch (err) {
        toast(err.message, "error");
      } finally {
        btn.disabled = false;
      }
    })
  );

  container.querySelector("#hostedProbe").addEventListener("click", async () => {
    const form = readForm();
    if (!form.model_name) {
      toast("Informe o modelo", "error");
      return;
    }
    const btn = container.querySelector("#hostedProbe");
    btn.disabled = true;
    try {
      showResult(await Api.probeModel(form));
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  });

  container.querySelector("#hostedSave").addEventListener("click", async () => {
    const form = readForm();
    if (!form.model_name) {
      toast("Informe o modelo", "error");
      return;
    }
    const btn = container.querySelector("#hostedSave");
    btn.disabled = true;
    try {
      await Api.createModel({
        model_id: `orchestrator-${form.model_name.replace(/[:/.]/g, "-")}`,
        tier: "orchestrator",
        backend: form.backend,
        model_name: form.model_name,
        endpoint: form.endpoint,
        api_key: form.api_key,
        tier_default: true,
        description: `Fallback hospedado configurado pelo console em ${new Date().toLocaleString("pt-BR")}`,
      });
      toast("Modelo de fallback registrado", "success");
      renderOrchestration(root);
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  });
}

// ---------------------------------------------------------------------------
// view: performance
// ---------------------------------------------------------------------------

async function renderPerformance(root) {
  setContent(root, loadingBlock("Carregando histórico de avaliações…"));
  document.getElementById("topbarActions").innerHTML = "";

  let runs, datasets;
  try {
    [runs, datasets] = await Promise.all([Api.evalRuns(30), Api.evalDatasets().catch(() => [])]);
  } catch (err) {
    root.innerHTML = "";
    root.appendChild(errorBlock(err.message, { retry: () => renderPerformance(root) }));
    return;
  }

  root.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div><h3>Nova avaliação</h3><p class="muted">Roda o conjunto de teste contra a federação e, opcionalmente, contra a baseline monolítica</p></div>
      </div>
      <div class="card-body" id="evalForm"></div>
    </div>

    <div id="evalResult" style="margin-top:16px"></div>

    <div class="card" style="margin-top:16px">
      <div class="card-header"><h3>Histórico de execuções</h3><span class="badge badge-neutral">${runs.length}</span></div>
      <div class="card-body tight" id="runsHistory"></div>
    </div>
  `;

  // -- new evaluation form ----------------------------------------------------
  const formEl = document.getElementById("evalForm");
  const validPacks = datasets.filter((d) => !d.error && d.eval_files && d.eval_files.length);
  formEl.innerHTML = `
    <div class="field-row-3">
      <label class="field">
        <span class="field-label">Pack</span>
        <select id="evalPack">
          ${validPacks.length ? validPacks.map((p) => `<option value="${escapeHtml(p.pack)}">${escapeHtml(p.pack)} (${p.eval_files.length} arquivo(s))</option>`).join("") : `<option value="">nenhum pack com dataset encontrado</option>`}
        </select>
      </label>
      <label class="field">
        <span class="field-label">Domínio (opcional)</span>
        <select id="evalDomain"><option value="">todos</option></select>
      </label>
      <label class="field">
        <span class="field-label">Repetições</span>
        <input type="number" id="evalRepeats" value="1" min="1" max="5" />
      </label>
    </div>
    <div class="flex-between">
      <label class="checkbox-row">
        <input type="checkbox" id="evalCompare" checked />
        Comparar com a baseline monolítica
      </label>
      <button class="btn btn-primary" id="evalRun">${iconSvg("play")} Rodar avaliação</button>
    </div>
  `;

  function updateDomainOptions() {
    const pack = validPacks.find((p) => p.pack === document.getElementById("evalPack").value);
    const domainSelect = document.getElementById("evalDomain");
    const domains = pack ? pack.domains : [];
    domainSelect.innerHTML = `<option value="">todos</option>` + domains.map((d) => `<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`).join("");
  }
  if (validPacks.length) {
    document.getElementById("evalPack").addEventListener("change", updateDomainOptions);
    updateDomainOptions();
  }

  document.getElementById("evalRun").addEventListener("click", async () => {
    const packPath = document.getElementById("evalPack").value;
    if (!packPath) {
      toast("Nenhum pack disponível para avaliar", "error");
      return;
    }
    const domain = document.getElementById("evalDomain").value;
    const repeats = parseInt(document.getElementById("evalRepeats").value, 10) || 1;
    const compare = document.getElementById("evalCompare").checked;

    const runBtn = document.getElementById("evalRun");
    runBtn.disabled = true;
    const resultEl = document.getElementById("evalResult");
    resultEl.innerHTML = `<div class="card card-pad">${loadingBlock("Rodando casos de teste — isso pode levar alguns minutos…")}</div>`;

    try {
      const result = await Api.runEval({ pack: packPath, domain, compare, repeats });
      renderEvalResult(resultEl, result);
      toast("Avaliação concluída", "success");
      Api.evalRuns(30).then((fresh) => renderRunsHistory(document.getElementById("runsHistory"), fresh));
    } catch (err) {
      resultEl.innerHTML = "";
      resultEl.appendChild(errorBlock(err.message));
      toast(`Falha na avaliação: ${err.message}`, "error");
    } finally {
      runBtn.disabled = false;
    }
  });

  // -- history ------------------------------------------------------------
  renderRunsHistory(document.getElementById("runsHistory"), runs);
}

function renderEvalResult(container, result) {
  const fed = result.federation;
  const comparison = result.comparison;

  let verdictHtml = "";
  if (comparison) {
    const verdictIcon = { federation: "check", baseline: "cross", inconclusive: "warning" }[comparison.verdict] || "warning";
    verdictHtml = `<div class="verdict-banner ${comparison.verdict}">
      <div class="verdict-icon">${iconSvg(verdictIcon)}</div>
      <div>
        <div class="verdict-title">Veredito: ${escapeHtml(comparison.verdict)}</div>
        <div class="verdict-text">${escapeHtml(comparison.recommendation)}</div>
      </div>
    </div>`;
  }

  let barsHtml = "";
  if (comparison) {
    const fedRow = comparison.federation ? metricsRow(comparison.federation) : {};
    const baseRow = comparison.baseline ? metricsRow(comparison.baseline) : {};
    barsHtml = `<div class="bar-compare">
      ${Object.entries(fedRow)
        .map(([metric, fedVal]) => {
          const baseVal = baseRow[metric] ?? 0;
          const meta = METRIC_META[metric] || { fmt: (v) => v.toFixed(3) };
          const max = Math.max(fedVal, baseVal, 0.0001);
          const win = comparison.wins ? comparison.wins[metric] : null;
          return `<div>
            <div class="bar-metric-name">
              <span>${escapeHtml(metric)} ${win === true ? '<span class="win">✓ federação vence</span>' : win === false ? '<span class="lose">✗ baseline vence</span>' : ""}</span>
              <span class="mono">${meta.fmt(fedVal)} <span class="muted">vs</span> ${meta.fmt(baseVal)}</span>
            </div>
            <div class="bar-track"><div class="bar-fill federation" style="width:${(fedVal / max) * 100}%"></div></div>
            <div class="bar-track" style="margin-top:4px"><div class="bar-fill baseline" style="width:${(baseVal / max) * 100}%"></div></div>
          </div>`;
        })
        .join("")}
      <div class="bar-legend">
        <span><span class="swatch-dot" style="background:var(--accent)"></span>Federação</span>
        <span><span class="swatch-dot" style="background:var(--text-3)"></span>Baseline monolítica</span>
      </div>
    </div>`;
  } else {
    barsHtml = `<div class="grid grid-stats">${Object.entries(metricsRow(fed))
      .map(([metric, v]) => {
        const meta = METRIC_META[metric] || { fmt: (x) => x.toFixed(3) };
        return statCard({ label: metric, value: meta.fmt(v), icon: "chart", accent: "indigo", foot: `${fed.cases} casos` });
      })
      .join("")}</div>`;
  }

  container.innerHTML = `
    <div style="display:flex; flex-direction:column; gap:16px">
      ${verdictHtml}
      <div class="card card-pad">
        <div class="flex-between" style="margin-bottom:14px">
          <div class="section-title" style="margin:0">Resultado · ${fed.cases} caso(s)</div>
          <span class="badge badge-neutral mono">${escapeHtml(result.run_id)}</span>
        </div>
        ${barsHtml}
      </div>
    </div>`;
}

function metricsRow(metrics) {
  return {
    "domain accuracy": metrics.domain_accuracy,
    "cost per query": metrics.cost_per_query,
    "latency p95 (ms)": metrics.latency_p95_ms,
    "fallback rate": metrics.fallback_rate,
    consistency: metrics.consistency,
    auditability: metrics.auditability,
  };
}

function renderRunsHistory(container, runs) {
  if (!runs.length) {
    container.innerHTML = emptyBlock("chart", "Nenhuma execução ainda", "Rode uma avaliação acima para ver o histórico aqui.");
    return;
  }
  container.innerHTML = `<div class="table-wrap"><table>
    <thead><tr>
      <th>Execução</th><th>Domínio</th><th>Braço</th><th class="num">Casos</th>
      <th class="num">Acurácia</th><th class="num">Custo/consulta</th><th class="num">Latência p95</th><th>Quando</th>
    </tr></thead>
    <tbody>${runs
      .map(
        (r) => `<tr class="clickable" data-run="${escapeHtml(r.run_id)}">
          <td class="mono">${escapeHtml(r.run_id)}</td>
          <td>${escapeHtml(r.domain || "todos")}</td>
          <td><span class="badge ${r.arm === "federation" ? "badge-accent" : "badge-neutral"}">${escapeHtml(r.arm)}</span></td>
          <td class="num mono">${r.cases}</td>
          <td class="num mono">${(r.metrics.domain_accuracy ?? 0).toFixed(3)}</td>
          <td class="num mono">$${(r.metrics.cost_per_query ?? 0).toFixed(6)}</td>
          <td class="num mono">${Math.round(r.metrics.latency_p95_ms ?? 0).toLocaleString("pt-BR")} ms</td>
          <td class="muted">${fmtRelativeTime(r.created_at)}</td>
        </tr>`
      )
      .join("")}</tbody>
  </table></div>`;

  container.querySelectorAll("tr[data-run]").forEach((row) =>
    row.addEventListener("click", () => openRunDrawer(row.dataset.run))
  );
}

async function openRunDrawer(runId) {
  const backdrop = document.getElementById("drawerBackdrop");
  const body = document.getElementById("drawerBody");
  document.getElementById("drawerTitle").textContent = `Execução ${runId}`;
  body.innerHTML = loadingBlock("Carregando casos…");
  backdrop.hidden = false;

  try {
    const detail = await Api.evalRunDetail(runId);
    const rows = detail.cases
      .map(
        (c) => `<tr>
          <td class="mono truncate" style="max-width:120px" title="${escapeHtml(c.case_id)}">${escapeHtml(c.case_id)}</td>
          <td>${c.correct ? `<span class="badge badge-success dot">ok</span>` : `<span class="badge badge-danger dot">falhou</span>`}</td>
          <td class="num mono">${c.score.toFixed(2)}</td>
          <td class="num mono">${Math.round(c.latency_ms)} ms</td>
          <td class="num mono">$${c.cost_usd.toFixed(6)}</td>
          <td>${c.used_fallback ? `<span class="badge badge-warning">fallback</span>` : "—"}</td>
          <td>${c.trace_complete ? iconSvg("check") : iconSvg("cross")}</td>
        </tr>`
      )
      .join("");

    body.innerHTML = `
      <div class="kv"><span class="k">Dataset</span><span class="v mono">${escapeHtml(detail.dataset)}</span></div>
      <div class="kv"><span class="k">Domínio</span><span class="v">${escapeHtml(detail.domain || "todos")}</span></div>
      <div class="kv"><span class="k">Pack</span><span class="v">${escapeHtml(detail.pack || "—")}</span></div>
      <div class="kv"><span class="k">Casos</span><span class="v">${detail.case_count}</span></div>
      <div class="divider"></div>
      <div class="section-title">Resultados por caso</div>
      <div class="table-wrap"><table>
        <thead><tr><th>Caso</th><th>Status</th><th class="num">Score</th><th class="num">Latência</th><th class="num">Custo</th><th>Fallback</th><th>Trace</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>`;
  } catch (err) {
    body.innerHTML = "";
    body.appendChild(errorBlock(err.message));
  }
}

// ---------------------------------------------------------------------------
// settings modal
// ---------------------------------------------------------------------------

function initSettings() {
  const backdrop = document.getElementById("settingsBackdrop");
  const open = () => {
    document.getElementById("apiBaseInput").value = store.apiBase;
    document.getElementById("apiTokenInput").value = store.token;
    backdrop.hidden = false;
  };
  const close = () => {
    backdrop.hidden = true;
  };
  document.getElementById("settingsToggle").addEventListener("click", open);
  document.getElementById("settingsClose").addEventListener("click", close);
  document.getElementById("settingsCancel").addEventListener("click", close);
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) close();
  });
  document.getElementById("settingsSave").addEventListener("click", async () => {
    store.apiBase = document.getElementById("apiBaseInput").value.trim().replace(/\/$/, "") || window.location.origin;
    store.token = document.getElementById("apiTokenInput").value.trim();
    close();
    toast("Configurações salvas", "success");
    await pollHealth();
    navigate(currentRoute);
  });
}

function initDrawer() {
  const backdrop = document.getElementById("drawerBackdrop");
  document.getElementById("drawerClose").addEventListener("click", () => (backdrop.hidden = true));
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) backdrop.hidden = true;
  });
}

// ---------------------------------------------------------------------------
// theme
// ---------------------------------------------------------------------------

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  store.theme = theme;
}

function initTheme() {
  applyTheme(store.theme);
  document.getElementById("themeToggle").addEventListener("click", () => {
    applyTheme(store.theme === "dark" ? "light" : "dark");
  });
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

function initNav() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => navigate(btn.dataset.route));
  });
}

async function init() {
  initTheme();
  initNav();
  initSettings();
  initDrawer();

  const startRoute = (window.location.hash || "").replace(/^#\//, "") || "overview";
  navigate(routes[startRoute] ? startRoute : "overview");

  await pollHealth();
  healthPollTimer = setInterval(pollHealth, 15000);
}

init();
