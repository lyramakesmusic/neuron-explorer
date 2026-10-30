import { createMap } from "./map3d.js?v=78";
import { createScatter } from "./scatter.js?v=39";
import { t, tp, initI18n, applyTranslations, getLang } from "./i18n.js?v=1";

const $ = s => document.querySelector(s);
const api = async (p, body) => {
  const r = await fetch(p, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!r.ok) throw new Error(`${p.split("?")[0]} ${r.status}: ${(await r.text()).slice(0, 160)}`);
  return r.json();
};
initI18n($("#lang"));
addEventListener("unhandledrejection", e => { const el = document.querySelector("#status"); if (el) el.textContent = String(e.reason?.message || e.reason); });
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtDose = d => (d > 0 ? "+" : d < 0 ? "−" : "") + Math.abs(d);

let agentSelecting = false, agentMarked = false;  // Claude's selections and marks (see agentUi); declared early, openNeuron reads them on load
const state = { layer: null, units: {}, unitMap: {}, focus: null, unit: null, view: "map", chat: { mode: "add", dir: "writes", dose: { add: 0.8, mul: 2 } } };
const ADD_STEPS = [...Array.from({ length: 30 }, (_, i) => +((i + 1) * 0.05).toFixed(2)), 1.75, 2, 2.5, 3, 4, 5];
const MUL_STEPS = [...Array.from({ length: 9 }, (_, i) => i * 0.5), 5, 6, 8, 12, 16];

const index = await api("/api/index");
const arch = index.arch;
const tip = $("#tip");

// ---------- map ----------
const map = createMap($("#map"), $("#layer-labels"), arch, {
  onPick: (layer, n) => openNeuron(layer, n, "neuron", true, false),
  onPickHead: (layer, head) => openHead(layer, head),
  onPickLayer: l => { if (l !== state.layer) { layerSel.value = l; setLayer(l); } },
  onLayerMode: l => {
    $("#layer-back").classList.toggle("on", l !== null);
    $("#layer-back").textContent = l !== null ? t("map.allLayers", { l }) : "";
    if (l !== null && l !== state.layer) setLayer(l);
  },
  onHover: (p, x, y) => {
    if (!p) return hideTip();
    if (p.type === "head") {
      if (hoverKey !== `h/${p.layer}/${p.head}`) map.setHoverConnections(null);
      const key = `h/${p.layer}/${p.head}`, head = `L${p.layer} h${p.head}`;
      const body = () => {
        const h = hoverCache.get(key);
        const side = (sg, list) => list.slice(0, 3).map(c => `<div class="tl"><span class="sgn">${sg}</span><span class="faint">L${c.layer} N${c.n}</span>  ${esc(state.unitMap[c.layer]?.get(c.n)?.name || (c.lens || []).slice(0, 4).map(t => t.trim() || t).join("  "))}</div>`).join("");
        return h ? side("←", h.reads) + side("→", h.writes) : "";
      };
      showTipHTML(x, y, head, body());
      clearTimeout(hoverTimer);
      hoverKey = key;
      hoverTimer = setTimeout(async () => {
        if (!hoverCache.has(key)) hoverCache.set(key, await api(`/api/head?layer=${p.layer}&head=${p.head}`));
        if (hoverKey !== key) return;
        const h = hoverCache.get(key);
        if (!(state.focus && (state.focus.kind === "neuron" || state.focus.kind === "head")))
          map.setHoverHead(p.layer, p.head, headRoleLinks(h));
        showTipHTML(x, y, head, body());
      }, 120);
      return;
    }
    if (p.type === "block") {
      if (hoverKey) { hoverKey = null; map.setHoverConnections(null); }
      const full = p.full;
      return showTip(x, y, `L${p.layer}`, full ? t("map.fullAttention", { q: arch.attn_heads, kv: arch.kv_heads }) : t("map.linearAttention", { k: arch.linear_k_heads, v: arch.linear_v_heads }));
    }
    const u = state.unitMap[p.layer]?.get(p.n), key = `${p.layer}/${p.n}`;
    const head = `L${p.layer} N${p.n}${u ? `  ${t("map.mass", { z: u.z.toFixed(1) })}` : ""}`;
    const body = () => {
      const h = hoverCache.get(key), rows = [];
      if (u?.name) rows.push(`<div class="tname">${esc(u.name)}</div>`);
      if (u) rows.push(`<div class="twords">${esc(u.words.join("  "))}</div>`);
      if (h && h.steer) rows.push(`<div class="tl"><span class="sgn up">+</span>${esc(h.steer.join("  "))}</div><div class="tl"><span class="sgn down">−</span>${esc(h.steer_neg.join("  "))}</div>`);
      if (h) rows.push(`<div class="tl faintline"><span class="sgn up">+</span>${esc(h.lens.join("  "))}</div><div class="tl faintline"><span class="sgn down">−</span>${esc(h.lens_neg.join("  "))}</div>`);
      const nh = hoverCache.get(`nh/${key}`);
      if (nh) {
        const top = roleHeads(nh), into = top.filter(k => k.role === "o").slice(0, 3), from = top.filter(k => k.role !== "o").slice(0, 4);
        if (into.length) rows.push(`<div class="tl faintline"><span class="sgn">←</span>${into.map(k => `L${k.layer} h${k.head}`).join("  ")}</div>`);
        if (from.length) rows.push(`<div class="tl faintline"><span class="sgn">→</span>${from.map(k => `${k.role} L${k.layer} h${k.head}`).join("  ")}</div>`);
      }
      return rows.join("");
    };
    showTipHTML(x, y, head, body());
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(async () => {
      const [hv, nh] = await Promise.all([hoverCache.get(key) || api(`/api/hover?layer=${p.layer}&n=${p.n}`),
        hoverCache.get(`nh/${key}`) || api(`/api/neuron/heads?layer=${p.layer}&n=${p.n}`)]);
      hoverCache.set(key, hv); hoverCache.set(`nh/${key}`, nh);
      if (hoverKey !== key) return;
      // with a neuron or head open, hover only shows the tooltip; its links and web stay on screen
      if (!(state.focus && (state.focus.kind === "neuron" || state.focus.kind === "head"))) map.setHoverConnections({ layer: p.layer, n: p.n }, hoverCache.get(key).con);
      showTipHTML(x, y, head, body());
    }, 120);
    hoverKey = key;
  },
});
// a neuron's heads over all four roles (/api/neuron/heads roles), strongest first, each head once
const roleHeads = r => {
  const seen = new Set();
  return ["q", "k", "v", "o"].flatMap(role => (r.roles?.[role] || []).map(x => ({ ...x, role }))).filter(x => x.z >= 3)
    .sort((a, b) => b.z - a.z).filter(x => !seen.has(`${x.layer}/${x.head}`) && seen.add(`${x.layer}/${x.head}`));
};
function showTip(x, y, head, body) { showTipHTML(x, y, head, esc(body)); }
function showTipHTML(x, y, head, html) {
  tip.innerHTML = `<div class="lab">${esc(head)}</div>${html}`;
  tip.style.display = "block";
  const r = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(x + 14, innerWidth - r.width - 8)}px`;
  tip.style.top = `${Math.min(y + 14, innerHeight - r.height - 8)}px`;
}
function hideTip() {
  tip.style.display = "none";
  if (hoverKey) { hoverKey = null; clearTimeout(hoverTimer); map.setHoverConnections(null); }
}
let hoverTimer, hoverKey = null;
const hoverCache = new Map();
// tip lives in the embed view section; lift it to body so the map can use it too
document.body.appendChild(tip);

// ---------- sheet layout ----------
const layouts = {};
let wantLayout = null;
async function setLayout(kind) {
  wantLayout = kind;
  document.querySelectorAll("#layout-mode button").forEach(b => b.classList.toggle("on", b.dataset.k === kind));
  if (kind === "index") { map.setLayout(null); return; }
  if (!layouts[kind]) {
    const r = await api(`/api/layout?kind=${kind}`);
    if (wantLayout !== kind) return;
    if (!r.available) { document.querySelectorAll("#layout-mode button").forEach(b => b.classList.toggle("on", b.dataset.k === "index")); map.setLayout(null); return; }
    const bin = atob(r.cell), a = new Int16Array(bin.length / 2);
    for (let i = 0; i < a.length; i++) a[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    layouts[kind] = a;
  }
  map.setLayout(layouts[kind]);
}
document.querySelectorAll("#layout-mode button").forEach(b => b.onclick = () => setLayout(b.dataset.k));
setLayout("writes");

// connection web of the focused neuron: downstream cone forward, upstream cone backward
let webOn = true, webKey = null;
async function refreshWeb() {
  document.querySelectorAll("#web-mode button").forEach(b => b.classList.toggle("on", (b.dataset.w === "on") === webOn));
  const f = state.focus, first = Math.max(1, Math.min(24, Math.round(Number($("#web-z").value) || 5)));
  const k = Number(new URLSearchParams(location.search).get("webk")) || 2;
  const hops = Number(new URLSearchParams(location.search).get("webhops")) || 0;  // hops out each way; 0 = all the way
  if (!webOn || !f || f.kind !== "neuron") { webKey = null; state.webSum = null; map.setWeb(null); renderList(); return; }
  const key = `${f.layer}/${f.n}/${first}/${k}/${hops}`;
  if (key === webKey) return;
  webKey = key;
  const r = await fetch(`/api/web?layer=${f.layer}&n=${f.n}&first=${first}&k=${k}&hops=${hops}`);
  if (!r.ok || webKey !== key) return;
  state.webTitle = { k, n: r.headers.get("X-Web-Total") };
  $("#web-z").title = t("web.title", state.webTitle);
  map.setWeb(new Int16Array(await r.arrayBuffer()));
  const sum = await api(`/api/web/summary?layer=${f.layer}&n=${f.n}&first=${first}&k=${k}&hops=${hops}&top=24`);
  if (webKey !== key) return;
  state.webSum = { ...sum, layer: f.layer, n: f.n, links: r.headers.get("X-Web-Total") };
  renderList();
}
document.querySelectorAll("#web-mode button").forEach(b => b.onclick = () => { webOn = b.dataset.w === "on"; webKey = null; refreshWeb(); });
$("#web-z").onchange = () => refreshWeb();

// ---------- scatter ----------
const scatter = createScatter($("#scatter"), {
  onPick: i => { const d = scatter.data; if (d.kind[i] === "neuron") openNeuron(d.layer, d.neuron[i]); },
  onHover: async (i, x, y) => {
    if (i < 0) { scatterHover = null; return hideTip(); }
    const d = scatter.data, layer = d.layer, key = `${layer}/${d.row[i]}`;
    const head = `${d.kind[i] === "baseline" ? t("scatter.baseline") : (d.kind[i] === "random" ? t("scatter.random") + " " : "N") + d.neuron[i]} · ${d.prompt[i]} · ${fmtDose(d.dose[i])}`;
    scatterHover = key;
    if (rolloutCache.has(key)) return showTip(x, y, head, rolloutCache.get(key));
    showTip(x, y, head, "…");
    const r = await api(`/api/rollout?layer=${layer}&i=${d.row[i]}`);
    rolloutCache.set(key, r.text.slice(0, 500));
    if (scatterHover === key && state.view === "embed") showTip(x, y, head, rolloutCache.get(key));
  },
});
let scatterHover = null;
const rolloutCache = new Map();
document.querySelectorAll("#color-by button").forEach(b => b.onclick = () => {
  document.querySelectorAll("#color-by button").forEach(x => x.classList.toggle("on", x === b));
  scatter.setColorBy(b.dataset.c);
});

// ---------- views / layer ----------
document.querySelectorAll("#views button").forEach(b => b.onclick = () => setView(b.dataset.view));
function setView(v) {
  state.view = v;
  document.querySelectorAll("#views button").forEach(x => x.classList.toggle("on", x.dataset.view === v));
  $("#map-view").classList.toggle("on", v === "map");
  $("#embed-view").classList.toggle("on", v === "embed");
  if (v === "embed") loadEmbeddings();
}
const layerSel = $("#layer");
for (let l = 0; l < arch.n_layers; l++) {
  const o = document.createElement("option"); o.value = l;
  o.textContent = `L${l}${index.layers[l] ? "" : " ·"}`;
  layerSel.appendChild(o);
}
layerSel.onchange = () => { setLayer(+layerSel.value); map.enterLayer(+layerSel.value); };

async function setLayer(l) {
  state.layer = l; layerSel.value = l;
  if (scatter.data && scatter.data.layer !== l) { scatter.setData(null); hideTip(); }
  try { localStorage.setItem("ne.layer", l); } catch {}
  if (!state.units[l]) {
    if (!lensResults) $("#rows").innerHTML = `<li class="head">…</li>`;
    const us = index.layers[l] ? await api(`/api/units?layer=${l}`) : [];
    state.units[l] = us;
    state.unitMap[l] = new Map(us.filter(u => u.kind === "neuron").map(u => [u.n, u]));
    if (state.layer !== l) return;
  }
  if (state.units[l].length) map.setUnits(l, state.units[l]);
  renderList();
  if (state.view === "embed") loadEmbeddings();
}
async function loadEmbeddings() {
  const l = state.layer;
  $("#embed-empty").textContent = "…";
  const d = index.layers[l] ? await api(`/api/embeddings?layer=${l}`) : {};
  if (state.layer !== l) return;
  $("#embed-empty").textContent = d.x ? "" : "—";
  scatter.setData(d.x ? { ...d, layer: l } : null);
  scatter.select(state.layer, state.focus?.layer === state.layer ? state.focus.n : null);
}

// ---------- left panel: net (the open neuron's web, then the list) | run (text through the model) ----------
let leftTab = "net";
document.querySelectorAll("#ltabs button[data-lt]").forEach(b => b.onclick = () => showLeft(b.dataset.lt));
function showLeft(t) {
  if (t === leftTab) return;
  leftTab = t;
  document.querySelectorAll("#ltabs button[data-lt]").forEach(b => b.classList.toggle("on", b.dataset.lt === t));
  $("#lt-net").classList.toggle("on", t === "net"); $("#lt-run").classList.toggle("on", t === "run");
  $(`#ltabs button[data-lt="${t}"]`).classList.remove("fresh");
  syncLeftMap();
}
// a tab whose contents changed while hidden (Claude ran text, or searched) gets a dot until opened
function markLeftFresh(t) { if (t !== leftTab) $(`#ltabs button[data-lt="${t}"]`).classList.add("fresh"); }
// the map shows what the open tab shows: lens hits and the open neuron's web on net, what fired on run
function syncLeftMap() {
  lightHeads(state.inf.pos);
  if (leftTab === "run" && state.inf.text) { inferView(); showTokenPaths(state.inf.pos); return; }
  map.setMarks(leftTab === "net" && lensResults?.results ? lensResults.results : []);
  if (String(webKey).startsWith("tok/")) { webKey = null; refreshWeb(); }
}

// ---------- lens search ----------
let lensTimer, lensResults = null, fireResults = null;
$("#lens-q").addEventListener("input", e => {
  if (e.isTrusted) showLeft(state.qMode === "fires" ? "run" : "net");
  clearTimeout(lensTimer);
  lensTimer = setTimeout(runLensSearch, state.qMode === "fires" ? 700 : 250);
});
state.qMode = "lens";
document.querySelectorAll("#q-mode button").forEach(b => b.onclick = e => {
  state.qMode = b.dataset.q;
  document.querySelectorAll("#q-mode button").forEach(x => x.classList.toggle("on", x === b));
  syncPlaceholders();
  if (e.isTrusted) {
    showLeft(state.qMode === "fires" ? "run" : "net");
    if (state.qMode === "fires" && document.activeElement !== $("#inf-text")) $("#inf-text").focus();
  }
  runLensSearch();
});
// the paste box in the list drives the same search as the header box
$("#inf-text").addEventListener("input", () => {
  if (state.qMode !== "fires") $('#q-mode button[data-q="fires"]').click();  // typing here means fires mode
  $("#lens-q").value = $("#inf-text").value; $("#lens-q").dispatchEvent(new Event("input"));
});
$("#lens-q").addEventListener("input", () => { if (state.qMode === "fires" && document.activeElement === $("#lens-q")) $("#inf-text").value = $("#lens-q").value; });
function syncPlaceholders() {
  $("#lens-q").placeholder = state.qMode === "fires" ? t("header.searchText") : t("header.searchLens");
  $("#c-input").placeholder = state.chat.dir === "reads" ? t("chat.textToRead") : t("chat.who");
}
// fires: run text through the model once, then filter what fired without re-running
state.inf = { text: null, pos: -1, sign: "both" };
let qGen = 0;
const stillWanted = (g, mode, q) => g === qGen && state.qMode === mode && $("#lens-q").value.trim() === q;
async function runInfer(text, g) {
  markLeftFresh("run");
  if (text !== state.inf.text) {
    $("#run-rows").innerHTML = `<li class="head">${t("results.running")}</li>`;
    const r = await api("/api/infer", { text });
    if (!stillWanted(g, "fires", text)) return;
    if (!r.ready) { $("#run-rows").innerHTML = `<li class="head">${esc(localizeStatus(state.statusText) || t("results.measuring"))}</li>`; return; }
    state.inf = { ...state.inf, text, pos: -1, tokens: r.tokens };
    $("#inf-toks").innerHTML = r.tokens.map((t, i) => `<button data-i="${r.pos[i]}">${esc(t)}</button>`).join("");
  }
  await inferView(g);
}
// hovering a token previews it: its strongest neurons marked on the map, and the direct links among them as the web;
// clicking pins it; leaving the tokens goes back to the pinned token (or none) and the open neuron's web
let tokHover = null, tokTimer;
async function showTokenPaths(pos) {
  if (pos < 0) { webKey = null; return refreshWeb(); }
  if (leftTab !== "run") return;
  const r = await fetch(`/api/infer/paths?pos=${pos}&min=${$("#inf-min").value || 1}`);
  if (!r.ok || (tokHover ?? state.inf.pos) !== pos) return;
  webKey = `tok/${pos}`;
  map.setWeb(new Int16Array(await r.arrayBuffer()));
}
function previewToken(pos) {
  const keep = state.inf.pos;
  state.inf.pos = pos; inferView(); state.inf.pos = keep;
  showTokenPaths(pos); lightHeads(pos);
}
// on run, the head lines light by how much each head writes at the token (pinned or hovered), and with a head open
// the token chips shade by where that head reads from there
let headsGen = 0;
async function lightHeads(pos) {
  const g = ++headsGen, f = state.focus;
  const on = leftTab === "run" && state.inf.text && pos >= 0;
  const r = on ? await api(`/api/infer/heads?pos=${pos}`) : null;
  if (g !== headsGen) return;
  if (r?.layers) map.setHeadLevels(r.layers.map(row => row.map(v => Math.max(0, (v - 0.15) / 0.6))));
  else if (f?.kind === "neuron" && state.nheads?.layer === f.layer && state.nheads.n === f.n) {
    const lv = [];
    for (const [l, zs] of Object.entries(state.nheads.layers)) lv[+l] = zs.map(z => Math.max(0, Math.min(1, (z - 2.5) / 5)));
    map.setHeadLevels(lv);
  } else map.setHeadLevels(null);
  shadeTokens(on ? pos : -1);
}
let shadeGen = 0;
async function shadeTokens(pos) {
  const g = ++shadeGen, f = state.focus, btns = [...document.querySelectorAll("#inf-toks button")];
  const r = pos >= 0 && f?.kind === "head" ? await api(`/api/head/row?layer=${f.layer}&head=${f.n}&pos=${pos}`) : null;
  if (g !== shadeGen) return;
  btns.forEach((b, i) => {
    const w = r?.units?.[i];
    b.style.background = w == null || b.classList.contains("on") ? "" :
      `rgba(${w >= 0 ? "255,181,71" : "255,79,154"},${Math.min(0.85, Math.abs(w) * 1.6).toFixed(3)})`;
    b.style.color = w != null && Math.abs(w) > 0.3 && !b.classList.contains("on") ? "#000" : "";
  });
}
$("#inf-toks").addEventListener("pointerover", e => {
  const b = e.target.closest("button"); if (!b || tokHover === +b.dataset.i) return;
  tokHover = +b.dataset.i; clearTimeout(tokTimer);
  tokTimer = setTimeout(() => previewToken(tokHover), 90);
});
$("#inf-toks").addEventListener("pointerleave", () => {
  clearTimeout(tokTimer); tokHover = null;
  inferView(); showTokenPaths(state.inf.pos); lightHeads(state.inf.pos);
});
$("#inf-toks").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  state.inf.pos = state.inf.pos === +b.dataset.i ? -1 : +b.dataset.i;
  document.querySelectorAll("#inf-toks button").forEach(x => x.classList.toggle("on", +x.dataset.i === state.inf.pos));
  inferView(); showTokenPaths(state.inf.pos); lightHeads(state.inf.pos);
});
document.querySelectorAll("#inf-sign button").forEach(b => b.onclick = () => {
  state.inf.sign = b.dataset.s;
  document.querySelectorAll("#inf-sign button").forEach(x => x.classList.toggle("on", x === b));
  inferView();
});
let infTimer;
for (const id of ["#inf-lo", "#inf-hi", "#inf-min", "#inf-q"]) $(id).addEventListener("input", () => { clearTimeout(infTimer); infTimer = setTimeout(inferView, 200); });
let viewGen = 0;
async function inferView(g = qGen) {
  const f = state.inf, mine = ++viewGen;
  if (!f.text) return;
  const qs = new URLSearchParams({ pos: f.pos, sign: f.sign, lo: $("#inf-lo").value || 0, hi: $("#inf-hi").value || 99,
    min: $("#inf-min").value || 0, q: $("#inf-q").value, limit: 300 });
  const r = await api(`/api/infer/view?${qs}`);
  if (mine !== viewGen || g !== qGen || state.qMode !== "fires") return;
  const marks = [];
  if (r.map) {
    const bin = atob(r.map), u16 = new Uint16Array(bin.length / 2);
    for (let i = 0; i < u16.length; i++) u16[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    const N = r.shape[1];
    for (let i = 0; i < u16.length; i++) {
      if (!u16[i]) continue;
      const v = halfToFloat(u16[i]);
      if (v) marks.push({ layer: Math.floor(i / N), n: i % N, sign: v < 0 ? "-" : "+", w: Math.min(1, 0.25 + Math.abs(v) / 4) });
    }
  }
  if (leftTab === "run") map.setMarks(marks.length > 40000 ? marks.sort((a, b) => b.w - a.w).slice(0, 40000) : marks);
  fireResults = { ...r, fires: true, ready: true };
  renderResults(fireResults, $("#run-rows"));
}
function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}

async function runLensSearch() {
  const q = $("#lens-q").value.trim(), g = ++qGen;
  if (q.length < 2) {
    // an emptied box clears both: lens hits, and the last run with its token chips, pinned token and token web
    lensResults = null; fireResults = null;
    state.inf = { ...state.inf, text: null, pos: -1, tokens: null };
    $("#inf-toks").innerHTML = ""; $("#run-rows").innerHTML = "";
    syncLeftMap(); renderList(); return;
  }
  if (state.qMode === "fires") return runInfer(q, g);
  const r = await api(`/api/search?q=${encodeURIComponent(q)}&limit=400`);
  if (!stillWanted(g, "lens", q)) return;
  lensResults = r;
  if (leftTab === "net") map.setMarks(r.results || []); else markLeftFresh("net");
  if ($("#list").classList.contains("hidden")) setListHidden(false);
  renderList();
}

// ---------- list ----------
$("#filter").oninput = renderList;
$("#show-random").onchange = renderList;
// what the open neuron's web says: lens tokens pooled over its cone (see /api/web/summary); click a token to search it
function webSummaryHTML() {
  const w = state.webSum, f = state.focus;
  if (!w || !f || f.kind !== "neuron" || w.layer !== f.layer || w.n !== f.n) return "";
  const chips = list => list.length ? list.map(x => `<span class="chip" data-q="${esc(x.tok)}" title="${esc(t("web.chipTitle", { n: x.n, w: x.w }))}">${esc(x.tok)}<i>${x.n}</i></span>`).join("")
    : `<span class="faint">${t("web.nothing")}</span>`;
  const src = w.src === "steer" ? t("web.srcSteer") : t("web.srcLogit");
  return `<div class="head">${t("web.head", { layer: w.layer, n: w.n, links: w.links, src })}</div>
    <div class="wsum"><div class="lab"><span class="sw up"></span>${t("web.fedBy", { n: w.up.neurons })}</div><div class="chips">${chips(w.up.fed_by)}</div></div>
    <div class="wsum"><div class="lab"><span class="sw down"></span>${t("web.heldBy")}</div><div class="chips">${chips(w.up.held_by || [])}</div></div>
    <div class="wsum"><div class="lab"><span class="sw up"></span>${t("web.toward", { n: w.down.neurons })}</div><div class="chips">${chips(w.down.toward)}</div></div>
    <div class="wsum"><div class="lab"><span class="sw down"></span>${t("web.away")}</div><div class="chips">${chips(w.down.away)}</div></div>`;
}
// the open neuron's heads by role: later heads whose query, key or value reads it, and heads whose output it reads
function nheadsHTML() {
  const H = state.nheads, f = state.focus;
  if (!H?.roles || !f || f.kind !== "neuron" || H.layer !== f.layer || H.n !== f.n) return "";
  const name = x => `L${x.layer} h${x.heads.length > 1 ? `${x.heads[0]}–${x.heads.at(-1)}` : x.head}`;
  const row = r => `<div class="lrow"><span class="sgn">${r}</span><div class="chips">${(H.roles[r] || []).filter(x => x.z >= 2).map(x =>
    `<span class="chip" data-l="${x.layer}" data-h="${x.head}">${name(x)}<i>${x.z}</i></span>`).join("")}</div></div>`;
  return `<div class="wsum"><div class="lab">${t("lens.heads")}</div>${["q", "k", "v", "o"].map(row).join("")}</div>`;
}
// the open neuron's own lenses (what it writes), above its web (what its cone does)
function lensHTML() {
  const L = state.lens, f = state.focus;
  if (!L || !f || f.kind !== "neuron" || L.layer !== f.layer || L.n !== f.n) return "";
  const chips = ts => ts.slice(0, 12).map(t => `<span class="chip" data-q="${esc(t.trim() || t)}">${esc(t)}</span>`).join("");
  const pair = (up, down) => `<div class="lrow"><span class="sgn up">+</span><div class="chips">${chips(up)}</div></div>
    <div class="lrow"><span class="sgn down">−</span><div class="chips">${chips(down)}</div></div>`;
  return (L.steer_up ? `<div class="wsum"><div class="lab">${t("lens.steer", { up: L.kl_up, down: L.kl_down })}</div>${pair(L.steer_up, L.steer_down)}</div>` : "")
    + `<div class="wsum"><div class="lab">${t("lens.logit")}</div>${pair(L.up, L.down)}</div>`;
}
// a head's strongest neurons in each role, as map links
const ROLE_COL = { q: "#73ff8c", k: "#b399ff", v: "#66d9ff", o: "#66d9ff" };
const headRoleLinks = h => [["q", h.q], ["k", h.k], ["v", h.reads], ["o", h.writes]]
  .flatMap(([role, list]) => (list || []).slice(0, 5).map(c => ({ ...c, role, side: role === "o" ? "down" : "up" })));
// the open head by role: the neurons feeding its query, key and value (earlier) and reading its output (later), each
// with its own lens tokens
function headHTML() {
  const H = state.headInfo, f = state.focus;
  if (!H || !f || f.kind !== "head" || H.layer !== f.layer || H.head !== f.n) return "";
  const line = c => { const nm = state.unitMap[c.layer]?.get(c.n)?.name;
    return `<div class="nline" data-nl="${c.layer}" data-nn="${c.n}"><span class="id">L${c.layer} N${c.n}</span>`
      + `<span class="w">${nm ? `<b>${esc(nm)}</b> ` : ""}${esc((c.lens || []).slice(0, 4).map(t => t.trim() || t).join("  "))}</span><span class="z">${c.z.toFixed(0)}</span></div>`; };
  const role = (r, list) => `<div class="wsum"><div class="lab"><span class="sw" style="background:${ROLE_COL[r]}"></span>${r}</div>${(list || []).slice(0, 10).map(line).join("")}</div>`;
  return role("q", H.q) + role("k", H.k) + role("v", H.reads) + role("o", H.writes);
}
function renderList() {
  $("#net").innerHTML = headHTML() || lensHTML() + webSummaryHTML() + nheadsHTML();
  if (lensResults) return renderResults(lensResults, $("#rows"));
  const q = $("#filter").value.trim().toLowerCase(), showR = $("#show-random").checked;
  const units = (state.units[state.layer] || []).filter(u => (showR || u.kind === "neuron") &&
    (!q || u.words.some(w => w.includes(q)) || u.name.toLowerCase().includes(q) || String(u.n) === q));
  const rnd = (state.units[state.layer] || []).filter(u => u.kind === "random").map(u => u.z);
  const floor = rnd.length ? Math.max(...rnd) : -Infinity;
  const ol = $("#rows");
  ol.innerHTML = units.slice(0, 600).map(u => `<li data-n="${u.n}" data-kind="${u.kind}" class="${u.kind}${u.z <= floor ? " below" : ""}${state.focus && state.focus.n === u.n && state.focus.layer === state.layer && u.kind === "neuron" ? " sel" : ""}">
    <span class="n">${u.kind === "random" ? "r" : "N"}${u.n}</span><span class="z">${u.z.toFixed(1)}</span>
    <span class="w">${u.name ? `<b>${esc(u.name)}</b> ` : ""}${esc(u.words.join(" "))}</span></li>`).join("");
}
// lens hits (net tab) or what fired (run tab)
function renderResults(r, ol) {
  if (!r.ready) { ol.innerHTML = `<li class="head">${esc(localizeStatus(state.statusText) || t("results.indexing"))}</li>`; return; }
  const rows = (r.results || []).map(x => `<li class="res${x.swept === false ? " unswept" : ""}" data-l="${x.layer}" data-n="${x.n}" data-kind="neuron">
    <span class="n">L${x.layer} N${x.n}</span><span class="sg ${x.sign === "+" ? "up" : "down"}">${r.fires ? `${x.sign === "+" ? "" : "−"}${x.score.toFixed(1)}×` : `${x.sign === "+" ? "+" : "−"}${x.rank + 1}`}</span>
    <span class="w">${r.fires ? `<span class="tk">"${esc(x.token)}"</span>` : ""}${x.name ? `<b>${esc(x.name)}</b> ` : ""}${esc(x.lens.join(" "))}</span></li>`).join("");
  const head = r.fires ? t("results.pass", { total: r.total ?? 0 }) : t("results.summary", { total: r.total ?? 0, tokens: r.tokens ?? 0 });
  ol.innerHTML = `<li class="head">${head}</li>` + rows;
}
// a head chip marks its head on the map while hovered
$("#net").addEventListener("mouseover", e => { const c = e.target.closest(".chip[data-h]"); map.hoverHead(c ? +c.dataset.l : null, c ? +c.dataset.h : null); });
$("#net").addEventListener("mouseleave", () => map.hoverHead(null));
// a web-summary token searches the lens for it
$("#net").addEventListener("click", e => {
  const nl = e.target.closest(".nline"); if (nl) return goToNeuron(+nl.dataset.nl, +nl.dataset.nn, "neuron");
  const chip = e.target.closest(".chip"); if (!chip) return;
  if (chip.dataset.h) return openHead(+chip.dataset.l, +chip.dataset.h);

  if (state.qMode !== "lens") $('#q-mode button[data-q="lens"]').click();
  $("#lens-q").value = chip.dataset.q; $("#lens-q").dispatchEvent(new Event("input"));
});
for (const ol of [$("#rows"), $("#run-rows")]) {
  ol.addEventListener("click", e => {
    const li = e.target.closest("li"); if (!li || !li.dataset.n) return;
    goToNeuron(li.dataset.l ? +li.dataset.l : state.layer, +li.dataset.n, li.dataset.kind);
  });
  ol.addEventListener("pointerover", e => {
    const li = e.target.closest("li"); if (li && li.dataset.kind === "neuron") map.hover(li.dataset.l ? +li.dataset.l : state.layer, +li.dataset.n);
  });
  ol.addEventListener("pointerleave", () => map.hover(null));
}

// ---------- jump / routing ----------
$("#jump").addEventListener("keydown", e => {
  if (e.key !== "Enter") return;
  const m = e.target.value.match(/L?\s*(\d+)\D+N?\s*(\d+)/i);
  if (m) openNeuron(+m[1], +m[2]);
});
addEventListener("hashchange", fromHash);
// what's open is kept in this browser (a reload reopens it) and never in the address bar, where every neuron would
// become a browser-history entry; a link with #L19/N7176 (a neuron) or #L19/h2 (an attention head) opens that
function saveFocus() {
  const f = state.focus;
  try { localStorage.setItem("explorer.focus", JSON.stringify(f && (f.kind === "neuron" || f.kind === "head") ? f : null)); } catch {}
}
function restoreFocus() {
  let f = null; try { f = JSON.parse(localStorage.getItem("explorer.focus") || "null"); } catch {}
  if (!f) return;
  if (f.kind === "head") openHead(f.layer, f.n); else openNeuron(f.layer, f.n, "neuron", false);
}
function fromHash() {
  const m = location.hash.match(/^#L(\d+)\/(N|h)(\d+)/);
  if (!m) return;
  const kind = m[2] === "h" ? "head" : "neuron";
  history.replaceState(null, "", location.pathname + location.search);  // the link is used up; later picks are remembered
  if (isFocus(+m[1], +m[3], kind)) return;
  if (kind === "head") openHead(+m[1], +m[3]); else openNeuron(+m[1], +m[3], "neuron", true);
}

// ---------- panel collapse ----------
function updateInsets() {
  const listShown = !$("#list").classList.contains("hidden");
  const focusShown = focusEl.classList.contains("on") && !focusEl.classList.contains("collapsed");
  map.setInsets(listShown ? $("#list").getBoundingClientRect().width + 24 : 0, focusShown ? focusEl.getBoundingClientRect().width + 24 : 0);
}
$("#layer-back").onclick = () => map.exitLayer();
// panel widths follow the window (vw-based), so window resizes and browser zoom must re-measure them
addEventListener("resize", () => updateInsets());
function setListHidden(h) {
  document.body.classList.toggle("list-hidden", h);
  $("#list").classList.toggle("hidden", h); $("#list-tab").classList.toggle("on", h); updateInsets();
}
function setFocusCollapsed(c) {
  focusEl.classList.toggle("collapsed", c);
  syncEdge(); updateInsets();
}
// the right edge handle: brings back a collapsed panel, or opens Claude when no panel is up
function syncEdge() {
  const on = focusEl.classList.contains("on"), col = focusEl.classList.contains("collapsed"), f = state.focus;
  const t = $("#focus-tab");
  t.classList.toggle("on", !on || col);
  t.textContent = on && col && f && f.kind !== "head" ? `‹ ${f.kind === "random" ? "r" : "N"}${f.n}` : "‹ claude";
}
$("#list-hide").onclick = () => setListHidden(true);
$("#list-tab").onclick = () => setListHidden(false);
$("#f-collapse").onclick = () => setFocusCollapsed(true);
$("#focus-tab").onclick = () => focusEl.classList.contains("on") ? setFocusCollapsed(false) : openClaude();

// ---------- focus panel ----------
const focusEl = $("#focus");
// closing the neuron keeps the panel up when Claude is the open tab
$("#f-close").onclick = () => { flushName(); if (running) running.abort(); state.focus = null; $("#list").classList.remove("has-focus");
  map.selectHead(null); state.headInfo = null; state.nheads = null; $("#f-name").style.display = ""; lightHeads(state.inf.pos);
  if (curTab() === "claude") syncTabs(); else focusEl.classList.remove("on");
  setFocusCollapsed(false); map.select(null); refreshWeb(); renderAgentActions(); scatter.select(null, null); saveFocus(); renderList(); };
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => setTab(b.dataset.tab));
const tabBtn = t => $(`#tabs button[data-tab="${t}"]`);
const curTab = () => $("#tabs button.on")?.dataset.tab;
function setTab(t) {
  document.querySelectorAll("#tabs button[data-tab]").forEach(x => x.classList.toggle("on", x.dataset.tab === t));
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("on", x.id === `tab-${t}`));
  focusEl.classList.toggle("on-claude", t === "claude");
  tabBtn(t).classList.remove("fresh");
  if (t === "claude") { const th = $("#a-thread"); th.scrollTop = th.scrollHeight; }
}
// a tab that changed while another was showing gets a dot until opened
function markFresh(t) { if (curTab() !== t) tabBtn(t).classList.add("fresh"); }
// neuron tabs show while a neuron is open (Chat for real neurons only); Claude always
function syncTabs() {
  const f = state.focus?.kind === "head" ? null : state.focus;
  for (const t of ["examples", "details"]) tabBtn(t).style.display = f ? "" : "none";
  tabBtn("chat").style.display = f && f.kind === "neuron" ? "" : "none";
  if (tabBtn(curTab()).style.display === "none") setTab(f ? "examples" : "claude");
}
function openClaude() {
  focusEl.classList.add("on"); syncTabs(); setTab("claude"); setFocusCollapsed(false); $("#a-input").focus();
}

async function openNeuron(layer, n, kind = "neuron", push = true, fly = true) {
  if (!agentSelecting) clearAgentMarks();
  flushName();
  if (running) running.abort();
  state.focus = { layer, n, kind }; state.unit = null;
  map.selectHead(null); state.headInfo = null; state.nheads = null; $("#f-name").style.display = ""; lightHeads(state.inf.pos);
  if (map.layerMode !== null && map.layerMode !== layer) map.enterLayer(layer);
  if (push && kind === "neuron") saveFocus();
  focusEl.classList.add("on"); $("#list").classList.add("has-focus");
  setFocusCollapsed(false);
  $("#f-id").textContent = kind === "random" ? t("focus.idRandom", { layer, n }) : `L${layer} N${n}`;
  $("#f-rank").textContent = "";
  setNameField(state.unitMap[layer]?.get(n)?.name || "");
  $("#f-name").disabled = kind !== "neuron";
  $("#tab-examples").innerHTML = ""; $("#tab-details").innerHTML = "";
  state.thread = []; renderThread(); $("#c-reads").innerHTML = ""; $("#c-lens").innerHTML = "";
  syncTabs();
  if (layer !== state.layer) await setLayer(layer);
  if (!isFocus(layer, n, kind)) return;
  setNameField(state.unitMap[layer]?.get(n)?.name || "");
  renderList();
  if (kind === "neuron") { map.select(layer, n, fly); scatter.select(layer, n); loadNeuronHeads(layer, n); }
  refreshWeb(); renderAgentActions();
  const u = await api(`/api/unit?layer=${layer}&n=${n}&kind=${kind}`);
  if (!isFocus(layer, n, kind)) return;
  state.unit = u;
  setNameField(u.name || "");
  if (u.stats) $("#f-rank").textContent = t("focus.rank", { rank: u.rank ?? "—", n: u.n_ranked, mass: (u.stats.mass ?? u.stats.z).toFixed(1), nullPct: (u.stats.null_pct * 100).toFixed(0) });
  renderExamples(u);
  renderDetails(u);
  const hasRollouts = !!(u.rollouts && u.rollouts.length);
  document.querySelector('#tabs button[data-tab="examples"]').style.display = hasRollouts ? "" : "none";
  if (!hasRollouts && document.querySelector("#tabs button.on")?.dataset.tab === "examples") setTab("details");
  if (kind === "neuron") loadLive(layer, n);
}
// the heads linked to the open neuron light up on their slabs (see /api/neuron/heads)
async function loadNeuronHeads(layer, n) {
  state.nheads = null; lightHeads(tokHover ?? state.inf.pos);
  const r = await api(`/api/neuron/heads?layer=${layer}&n=${n}`);
  if (!isFocus(layer, n, "neuron")) return;
  state.nheads = { ...r, layer, n }; lightHeads(tokHover ?? state.inf.pos); renderList(); renderConnections();
  const want = Object.values(r.roles || {}).flat().filter(x => x.z >= 2 && !hoverCache.has(`h/${x.layer}/${x.head}`));
  await Promise.all(want.map(async x => hoverCache.set(`h/${x.layer}/${x.head}`, await api(`/api/head?layer=${x.layer}&head=${x.head}`))));
  if (isFocus(layer, n, "neuron")) renderConnections();
  map.setNeuronHeads(roleHeads(r).slice(0, 8));
}

// a head opens in the list panel: its id up top, what it reads and writes in net, and on run the text shaded by
// where it reads from; the neuron panel steps aside unless Claude is showing
async function openHead(layer, head) {
  if (!agentSelecting) clearAgentMarks();
  flushName();
  if (running) running.abort();
  state.focus = { layer, n: head, kind: "head" }; state.unit = null;
  saveFocus();
  map.select(null); scatter.select(null, null); map.selectHead(layer, head);
  $("#list").classList.add("has-focus");
  if ($("#list").classList.contains("hidden")) setListHidden(false);
  $("#f-id").textContent = `L${layer} h${head}`; $("#f-rank").textContent = "";
  $("#f-name").style.display = "none";
  if (curTab() === "claude") syncTabs(); else { focusEl.classList.remove("on"); syncTabs(); }
  syncEdge(); updateInsets();
  refreshWeb(); renderAgentActions();
  shadeTokens(tokHover ?? state.inf.pos);
  lightHeads(tokHover ?? state.inf.pos);
  const r = await api(`/api/head?layer=${layer}&head=${head}`);
  if (!isFocus(layer, head, "head")) return;
  state.headInfo = r; hoverCache.set(`h/${layer}/${head}`, r);
  map.setHeadLinks(layer, head, headRoleLinks(r));
  renderList();
}

const isFocus = (layer, n, kind) => state.focus && state.focus.layer === layer && state.focus.n === n && state.focus.kind === kind;

function renderExamples(u) {
  if (!u.rollouts || !u.rollouts.length) { $("#tab-examples").innerHTML = `<div class="slot">—</div>`; return; }
  const prompts = Object.keys(u.prompts).filter(p => u.rollouts.some(r => r.prompt === p));
  const count = {};
  for (const r of u.rollouts) count[r.dose] = (count[r.dose] || 0) + 1;
  // doses shared by several prompts get columns; a prompt's one-off doses span its empty cells
  const doses = Object.keys(count).map(Number).filter(d => count[d] > 1).sort((a, b) => a - b);
  const relOf = d => u.rollouts.find(r => r.dose === d).rel_push.toFixed(2);
  const cell = r => `<div class="cell${r.finish === "length" ? " capped" : ""}">${esc(r.text)}</div>`;
  let h = `<div class="grid" style="grid-template-columns: 66px repeat(${1 + doses.length}, minmax(0,1fr))"><div class="h"></div><div class="h">0</div>`;
  for (const d of doses) h += `<div class="h ${d > 0 ? "up" : "down"}">${fmtDose(d)}<span class="dim"> · ${relOf(d)}</span></div>`;
  for (const p of prompts) {
    const b = u.baselines.find(r => r.prompt === p);
    h += `<div class="h">${p}</div><div class="cell base${b?.finish === "length" ? " capped" : ""}">${esc(b?.text ?? "")}</div>`;
    const mine = u.rollouts.filter(r => r.prompt === p);
    const shared = doses.map(d => mine.find(r => r.dose === d));
    const extra = mine.filter(r => !doses.includes(r.dose));
    if (shared.some(Boolean)) h += shared.map(r => r ? cell(r) : "<div></div>").join("");
    else if (extra.length) {
      const span = Math.max(1, Math.floor(doses.length / extra.length));
      h += extra.map((r, i) => `<div class="cell${r.finish === "length" ? " capped" : ""}" style="grid-column: span ${i === extra.length - 1 ? doses.length - span * (extra.length - 1) : span}"><span class="h ${r.dose > 0 ? "up" : "down"}">${fmtDose(r.dose)} · ${r.rel_push.toFixed(2)}</span>  ${esc(r.text)}</div>`).join("");
    } else h += doses.map(() => "<div></div>").join("");
  }
  $("#tab-examples").innerHTML = h + "</div>";
}
$("#tab-examples").addEventListener("click", e => { const c = e.target.closest(".cell"); if (c) c.classList.toggle("open"); });

function renderDetails(u) {
  const s = u.stats;
  let h = "";
  if (s) {
    h += `<div class="lab">${t("details.words")}</div><div class="crumbs">` + s.words.map(([w, c, z]) =>
      `<span class="tok" title="${esc(t("details.rollouts", { n: c, z: z.toFixed(1), prompts: (s.per_prompt[w] || []).join(" ") }))}">${esc(w)}</span>`).join("") + `</div>`;
    h += `<table class="dtable"><tr><th class="num">${t("details.dose")}</th><th class="num">${t("details.push")}</th><th class="num">${t("details.distinct")}</th><th class="num">${t("details.capped")}</th><th class="num">${t("details.logprob")}</th></tr>` +
      Object.entries(s.dose).map(([d, v]) => `<tr><td class="num">${d}</td><td class="num">${v.rel_push.toFixed(2)}</td><td class="num">${v.distinct4.toFixed(2)}</td><td class="num">${(v.capped * 100).toFixed(0)}%</td><td class="num">${v.logprob.toFixed(2)}</td></tr>`).join("") + `</table>`;
  }
  h += `<div id="d-lens" class="slot">—</div><div class="lab" style="margin-top:18px">${t("details.connections")}</div><div id="d-con" class="slot">—</div>`;
  $("#tab-details").innerHTML = h;
}

async function loadLive(layer, n) {
  $("#d-lens") && ($("#d-lens").textContent = "…");
  $("#d-con") && ($("#d-con").textContent = "…");
  const conP = api(`/api/connections?layer=${layer}&n=${n}&k=24`);
  let lens;
  try { lens = await api(`/api/lens?layer=${layer}&n=${n}`); }
  catch (err) { if (isFocus(layer, n, "neuron") && $("#d-lens")) $("#d-lens").textContent = err.message; }
  if (!isFocus(layer, n, "neuron")) return;
  conP.then(con => {
    if (!isFocus(layer, n, "neuron")) return;
    map.setConnections(con); state.con = con; renderConnections();
  }, err => { if (isFocus(layer, n, "neuron") && $("#d-con")) $("#d-con").textContent = err.message; });
  if (!lens) return;
  state.lens = { ...lens, layer, n }; renderList(); renderLensDetails();
}

// the open neuron's lenses in Details and above the chat; rendered from state so a language change can redraw it
function renderLensDetails() {
  const L = state.lens, f = state.focus;
  if (!L || !f || f.kind !== "neuron" || L.layer !== f.layer || L.n !== f.n) return;
  const tok = x => `<span class="tok">${esc(x)}</span>`;
  const steer = L.steer_up ? `<div class="lab">${t("lens.steerShort")}</div><div class="lens"><div class="up"><div class="lab">${t("lens.push", { kl: L.kl_up })}</div>${L.steer_up.map(tok).join("")}</div><div class="down"><div class="lab">${t("lens.pull", { kl: L.kl_down })}</div>${L.steer_down.map(tok).join("")}</div></div><div class="lab" style="margin-top:14px">${t("lens.logit")}</div>` : "";
  const el = $("#d-lens");
  if (el) {
    el.classList.remove("slot");
    el.innerHTML = `${steer}<div class="lens"><div class="up"><div class="lab">+</div>${L.up.map(tok).join("")}</div><div class="down"><div class="lab">−</div>${L.down.map(tok).join("")}</div></div>`;
  }
  const words = state.unitMap[L.layer]?.get(L.n)?.words || [];
  $("#c-lens").innerHTML = (words.length ? `<div class="words">${esc(words.join("  "))}</div>` : "") +
    (L.steer_up ? `<div><span class="sgn up">+</span>${esc(L.steer_up.slice(0, 10).join("  "))}</div><div><span class="sgn down">−</span>${esc(L.steer_down.slice(0, 10).join("  "))}</div>` : "") +
    `<div class="faintline"><span class="sgn up">+</span>${esc(L.up.slice(0, 10).join("  "))}</div><div class="faintline"><span class="sgn down">−</span>${esc(L.down.slice(0, 10).join("  "))}</div>`;
}

// sign = sign(g * u): the sign of the reading neuron's activation that the writing neuron's direction produces on its
// own. Reversing the writer reverses g and u together, so the sign holds for either direction of the writer.
function renderConnections() {
  const el = $("#d-con") || $("#d-con-wrap"); if (!el || !state.con) return;
  const tok = t => `<span class="tok">${esc(t)}</span>`;
  const row = c => {
    const pole = c.sign > 0 ? 1 : -1, toks = (pole > 0 ? c.lens : c.lens_neg) || [];
    const logit = (pole > 0 ? c.logit : c.logit_neg) || [];
    const sweep = c.name || (c.words || []).join(" ");
    return `<tr data-l="${c.layer}" data-n="${c.n}" class="con"><td class="pole ${pole > 0 ? "up" : "down"}">${pole > 0 ? "↑" : "↓"}</td>
      <td>${toks.slice(0, 4).map(tok).join("")}${logit.length ? `<div class="logit">${logit.slice(0, 4).map(tok).join("")}</div>` : ""}${sweep ? `<div class="dim">${esc(sweep)}</div>` : ""}</td>
      <td class="num dim">${c.z.toFixed(0)}</td><td class="mono faint">L${c.layer}·${c.n}</td></tr>`;
  };
  el.outerHTML = `<div id="d-con-wrap">
    <div class="lens"><div><div class="lab">${t("details.writesInto")}</div><table class="dtable">${state.con.down.slice(0, 14).map(row).join("")}</table></div>
    <div><div class="lab">${t("details.drivenBy")}</div><table class="dtable">${state.con.up.slice(0, 14).map(row).join("")}</table></div></div>
    ${headTablesHTML()}</div>`;
}
// the open neuron's heads in Details: later heads reading it (q / k / v) and heads it reads (o), each with what the
// head writes (its o tokens) over what it carries (v tokens, faint)
function headTablesHTML() {
  const H = state.nheads, f = state.focus;
  if (!H?.roles || !f || f.kind !== "neuron" || H.layer !== f.layer || H.n !== f.n) return "";
  const tok = t => `<span class="tok">${esc(t.tok)}</span>`;
  const row = x => {
    const info = hoverCache.get(`h/${x.layer}/${x.head}`);
    const id = `L${x.layer}·h${x.heads.length > 1 ? `${x.heads[0]}–${x.heads.at(-1)}` : x.head}`;
    return `<tr class="con" data-hl="${x.layer}" data-hh="${x.head}"><td class="pole mono dim">${x.role}</td>
      <td>${info ? info.write_toks.slice(0, 4).map(tok).join("") + `<div class="logit">${info.read_toks.slice(0, 4).map(tok).join("")}</div>` : ""}</td>
      <td class="num dim">${x.z.toFixed(1)}</td><td class="mono faint">${id}</td></tr>`;
  };
  const rows = roles => roles.flatMap(r => (H.roles[r] || []).map(x => ({ ...x, role: r }))).filter(x => x.z >= 2).sort((a, b) => b.z - a.z);
  return `<div class="lens"><div><div class="lab">${t("details.headsReading")}</div><table class="dtable">${rows(["q", "k", "v"]).slice(0, 10).map(row).join("")}</table></div>
    <div><div class="lab">${t("details.headsFeeding")}</div><table class="dtable">${rows(["o"]).slice(0, 10).map(row).join("")}</table></div></div>`;
}
// opening a neuron from a list or link also centres the camera on it
function goToNeuron(layer, n, kind = "neuron") {
  openNeuron(layer, n, kind, true, false);
  if (kind === "neuron") map.focusNeuron(layer, n);
}
$("#tab-details").addEventListener("click", e => {
  const th = e.target.closest("tr[data-hl]"); if (th) return openHead(+th.dataset.hl, +th.dataset.hh);
  const tr = e.target.closest("tr[data-l]"); if (tr) goToNeuron(+tr.dataset.l, +tr.dataset.n);
});
$("#tab-details").addEventListener("mouseover", e => { const th = e.target.closest("tr[data-hl]"); map.hoverHead(th ? +th.dataset.hl : null, th ? +th.dataset.hh : null); });
$("#tab-details").addEventListener("mouseleave", () => map.hoverHead(null));

// name autosave
let nameTimer, pendingName = null;
async function saveName(p) {
  await api("/api/name", { layer: p.layer, n: p.n, name: p.name });
  const u = state.unitMap[p.layer]?.get(p.n); if (u) u.name = p.name.trim();
  if (p.layer === state.layer && state.units[p.layer]?.length) map.setUnits(p.layer, state.units[p.layer]);
  renderList();
  if (isFocus(p.layer, p.n, "neuron")) { $("#f-name").classList.add("saved"); setTimeout(() => $("#f-name").classList.remove("saved"), 900); }
}
function flushName() {
  clearTimeout(nameTimer);
  if (pendingName) { const p = pendingName; pendingName = null; saveName(p); }
}
// the name field wraps long names: a one-line textarea that grows to fit; Enter finishes editing instead of a newline
function sizeNameField() { const t = $("#f-name"); t.style.height = "auto"; t.style.height = `${t.scrollHeight}px`; }
function setNameField(v) { $("#f-name").value = v; sizeNameField(); }
$("#f-name").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); e.target.blur(); } });
$("#f-name").addEventListener("input", () => { const t = $("#f-name"); if (t.value.includes("\n")) t.value = t.value.replace(/\n+/g, " "); sizeNameField(); });
addEventListener("resize", sizeNameField);
$("#f-name").addEventListener("input", () => {
  const f = state.focus; if (!f || f.kind !== "neuron") return;
  pendingName = { layer: f.layer, n: f.n, name: $("#f-name").value };
  clearTimeout(nameTimer);
  nameTimer = setTimeout(flushName, 500);
});

// ---------- chat ----------
const doseEl = $("#c-dose");
function showDose() { doseEl.value = state.chat.mode === "mul" ? `×${state.chat.dose.mul}` : state.chat.dose.add; }
doseEl.addEventListener("change", () => {
  const v = parseFloat(doseEl.value.replace(/[×x*]/g, ""));
  if (Number.isFinite(v) && v >= 0) state.chat.dose[state.chat.mode] = v;
  showDose();
});
document.querySelectorAll("#c-mode button").forEach(b => b.onclick = () => {
  state.chat.mode = b.dataset.m;
  document.querySelectorAll("#c-mode button").forEach(x => x.classList.toggle("on", x === b)); showDose();
});
document.querySelectorAll("#c-dir button").forEach(b => b.onclick = () => {
  state.chat.dir = b.dataset.d;
  document.querySelectorAll("#c-dir button").forEach(x => x.classList.toggle("on", x === b));
  $("#tab-chat").classList.toggle("reads", b.dataset.d === "reads");
  syncPlaceholders();
});
function step(dir) {
  const steps = state.chat.mode === "mul" ? MUL_STEPS : ADD_STEPS, key = state.chat.mode;
  // step to the next value in the ladder above or below the current dose (typed values land between steps)
  const v = state.chat.dose[key];
  const next = dir > 0 ? steps.find(x => x > v + 1e-9) : [...steps].reverse().find(x => x < v - 1e-9);
  if (next !== undefined) state.chat.dose[key] = next;
  showDose();
}
$("#c-minus").onclick = () => step(-1); $("#c-plus").onclick = () => step(1);
showDose();

let running = null;
state.thread = [];
const ROWS = ["up", "zero", "down"];
function turnLabels(t) {
  return t.mode === "mul" ? [`▲ ×${t.dose}`, "×1", `▼ ×−${t.dose}`] : [`▲ +${t.dose}`, "0", `▼ −${t.dose}`];
}
function renderThread() {
  const el = $("#c-thread");
  el.innerHTML = state.thread.map((turn, i) => `<div class="turn" data-i="${i}">
    <div class="user">${esc(turn.user)}<span class="recipe">${turn.mode === "mul" ? "×" : t("chat.add")} ${turn.dose}</span></div>
    <div class="cols">${ROWS.map((c, r) => `<div class="col ${c}${turn.live && !turn.done[r] ? " live" : ""}"><span class="lab">${turnLabels(turn)[r]}</span>${esc(turn.replies[r])}</div>`).join("")}
    ${turn.error ? `<div class="err">${esc(turn.error)}</div>` : ""}</div></div>`).join("");
  el.scrollTop = el.scrollHeight;
}
$("#c-clear").onclick = () => { if (running) running.abort(); state.thread = []; renderThread(); };
let reading = false;

$("#c-form").addEventListener("submit", async e => {
  e.preventDefault();
  if (running) { running.abort(); return; }
  const f = state.focus; if (!f) return;
  const text = $("#c-input").value.trim() || $("#c-input").placeholder;
  if (state.chat.dir === "reads") return runReads(f, text);
  const dose = state.chat.mode === "mul" ? state.chat.dose.mul : state.chat.dose.add;
  const history = state.thread.filter(t => !t.error).map(t => ({ user: t.user, replies: t.replies }));
  const turn = { user: text, replies: ["", "", ""], done: [false, false, false], dose, mode: state.chat.mode, live: true };
  state.thread.push(turn); $("#c-input").value = ""; renderThread();
  const cols = [...$("#c-thread").lastElementChild.querySelectorAll(".col")];
  const ctl = new AbortController(), thread = state.thread;
  running = ctl; $("#c-go").textContent = t("chat.stop");
  try {
    const res = await fetch("/api/steer", { method: "POST", headers: { "Content-Type": "application/json" }, signal: ctl.signal,
      body: JSON.stringify({ layer: f.layer, n: f.n, prompt: text, dose, mode: state.chat.mode, history }) });
    if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 160)}`);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, k); buf = buf.slice(k + 2);
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.error) turn.error = ev.error;
        if (ev.row == null) continue;
        turn.replies[ev.row] += ev.t;
        cols[ev.row].lastChild.nodeType === 3 ? (cols[ev.row].lastChild.textContent += ev.t) : cols[ev.row].append(ev.t);
        const th = $("#c-thread"); if (th.scrollHeight - th.scrollTop - th.clientHeight < 80) th.scrollTop = th.scrollHeight;
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") turn.error = String(err);
  } finally {
    turn.live = false; turn.done = [true, true, true];
    if (running === ctl) { running = null; $("#c-go").textContent = t("chat.run"); }
    if (state.thread === thread) renderThread();
  }
});

async function runReads(f, text) {
  if (reading) return;
  reading = true; $("#c-go").textContent = "…";
  let r;
  try { r = await api("/api/reads", { layer: f.layer, n: f.n, text, chat: true }); }
  catch (err) { $("#c-reads").textContent = err.message; return; }
  finally { reading = false; $("#c-go").textContent = t("chat.run"); }
  if (!isFocus(f.layer, f.n, "neuron")) return;
  const m = Math.max(1e-6, ...r.units.map(u => Math.abs(u.act)));
  $("#c-reads").innerHTML = r.units.map(u => {
    const a = u.act / m, c = a >= 0 ? `rgba(255,181,71,${a * 0.85})` : `rgba(255,79,154,${-a * 0.85})`;
    return `<span style="background:${c}" title="${u.pieces.map(v => v.toFixed(3)).join(" · ")}">${esc(u.text)}</span>`;
  }).join("");
}

// ---------- status ----------
// keeps polling (slowly once ready), so a server restart or a one-off error message clears itself
async function pollStatus() {
  try {
    const s = await api("/api/status");
    state.statusText = s.status;
    $("#status").textContent = localizeStatus(s.status);
    if (s.status === "ready" && lensResults && !lensResults.ready) runLensSearch();
    setTimeout(pollStatus, s.status === "ready" ? 8000 : 2000);
  } catch {
    state.statusText = "server unreachable";
    $("#status").textContent = t("status.unreachable");
    setTimeout(pollStatus, 2000);
  }
}
pollStatus();

// server status strings are stable English identifiers; keep the API language-neutral and translate here
function localizeStatus(s) {
  if (!s || s === "ready") return "";
  if (s === "idle") return t("status.idle");
  if (s === "loading") return t("status.loading");
  if (s === "no model") return t("status.noModel");
  if (s === "server unreachable") return t("status.unreachable");
  let m = s.match(/^indexing lens (\d+)\/(\d+)$/);
  if (m) return t("status.indexing", { i: m[1], n: m[2] });
  m = s.match(/^measuring neuron ranges (\d+)k\/(\d+)k tokens$/);
  if (m) return t("status.measuring", { seen: m[1], tokens: m[2] });
  return s;
}
window.__ne = { map, state, scatter };
// ---------- start ----------
const have = Object.keys(index.layers).map(Number);
let saved = null; try { const v = localStorage.getItem("ne.layer"); if (v !== null) saved = +v; } catch {}
const first = have.includes(saved) ? saved : have.includes(22) ? 22 : (have[0] ?? 0);
await setLayer(first);
if (location.hash) fromHash(); else restoreFocus();

// ---------- claude: the Claude tab, one claude -p session per conversation ----------
const agent = { chat: null, kind: "claude", busy: false, sending: 0, box: null, seg: null, segText: "", reply: "", waiters: [] };
const agentEl = $("#tab-claude");
function mdInline(t) {
  return esc(t).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/\bL(\d{1,2})[ \/]N(\d{1,4})\b/g, (m, l, n) => `<a class="nref" data-l="${l}" data-n="${n}">${m}</a>`);
}
function md(src) {
  const out = []; let list = false;
  for (const line of src.split("\n")) {
    const h = line.match(/^#{1,4}\s+(.*)/), li = line.match(/^\s*[-*]\s+(.*)/);
    if (li) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${mdInline(li[1])}</li>`); continue; }
    if (list) { out.push("</ul>"); list = false; }
    if (h) out.push(`<h3>${mdInline(h[1])}</h3>`);
    else if (line.trim()) out.push(`<p>${mdInline(line)}</p>`);
  }
  if (list) out.push("</ul>");
  return out.join("");
}
const EMPTY = () => `<div class="a-empty dim">${t("agent.empty")}</div>`;
// the conversation persists in this browser: chat id (= claude session, resumed server side) and the rendered thread
const AGENT_KEY = "explorer.agent";
function saveAgent() {
  try { localStorage.setItem(AGENT_KEY, JSON.stringify({ chat: agent.chat, kind: agent.kind, html: $("#a-thread").innerHTML })); } catch {}
}
// which agent the chat runs on: the tab is named after it, and the dropdown picks it for the next new chat
const AGENT_LABELS = { claude: "Claude", codex: "Codex", dsh: "DSH" };
function setAgentKind(kind) {
  agent.kind = Object.hasOwn(AGENT_LABELS, kind) ? kind : "claude";
  $("#a-agent").value = agent.kind;
  $('#tabs button[data-tab="claude"]').textContent = AGENT_LABELS[agent.kind];
  $("#a-input").placeholder = t(`agent.ask.${agent.kind}`);
  $("#a-input").title = t(`agent.inputTitle.${agent.kind}`);
}
function agentError(r) {
  if (r && r.code) { const s = t(`agent.err.${r.code}`); if (s !== `agent.err.${r.code}`) return s; }
  return String((r && r.error) || r || "");
}
document.addEventListener("i18nchange", () => { if (agent && agent.kind) setAgentKind(agent.kind); });
$("#a-agent").addEventListener("change", e => {
  const kind = e.target.value;
  try { localStorage.setItem("explorer.agentKind", kind); } catch {}
  if (agent.chat || $("#a-thread .a-msg")) { pastStash = null; $("#a-past").classList.remove("on"); resetAgent(); clearAgentMarks(); }
  setAgentKind(kind);
});
function loadAgent() {
  try {
    const v = JSON.parse(localStorage.getItem(AGENT_KEY) || "null"); if (!v) return;
    agent.chat = v.chat; $("#a-thread").innerHTML = v.html || EMPTY(); setAgentKind(v.kind);
    $("#a-thread").querySelectorAll("#a-usename, .a-msg.queued, .a-busy, .a-qctl").forEach(b => b.remove());
    $("#a-thread").scrollTop = $("#a-thread").scrollHeight;
    addTurnControls();
    relabelAgentThread();
  } catch {}
}
// quick actions above the input, for the open neuron
function renderAgentActions() {
  const f = state.focus;
  const ref = f && f.kind === "head" ? `L${f.layer} h${f.n}` : f ? `L${f.layer} N${f.n}` : "";
  $("#a-actions").innerHTML = f && f.kind === "neuron" ? `<button data-a="explain">${t("agent.explain", { ref })}</button><button data-a="web">${t("agent.web")}</button>`
    : f && f.kind === "head" ? `<button data-a="explain">${t("agent.explain", { ref })}</button>` : "";
}
$("#a-actions").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b || agent.busy) return;
  if (b.dataset.a === "explain") explainFocus();
  else if (b.dataset.a === "web") { const f = state.focus; agentTurn(`Look at the cone of L${f.layer} N${f.n} (what it reads from and writes into, a few hops out) and tell me briefly what that circuit seems to do. Highlight the neurons you mention.`, t("agent.webShown", { ref: `L${f.layer} N${f.n}` })); }
});
function resetAgent() {
  closeEvents(); if (agent.busy) finishTurn();
  agent.chat = null; $("#a-thread").innerHTML = EMPTY();
  let pref = "claude"; try { pref = localStorage.getItem("explorer.agentKind") || "claude"; } catch {}
  setAgentKind(pref);
  saveAgent();
}
agentEl.addEventListener("click", e => {
  const a = e.target.closest("a.nref"); if (a) goToNeuron(+a.dataset.l, +a.dataset.n);
});
// what Claude does to the screen (see agent_mcp.py)
// Claude's labels and highlights last until its next reply, or until the person opens a neuron themselves
function clearAgentMarks() {
  map.setNotes([]);
  if (agentMarked) { agentMarked = false; syncLeftMap(); }
}
async function agentUi(e) {
  if (e.action === "head") {
    agentSelecting = true;
    try { await openHead(e.layer, e.head); } finally { agentSelecting = false; }
  } else if (e.action === "select") {  // visibly go there
    agentSelecting = true;
    try { await openNeuron(e.layer, e.n, "neuron", true, false); } finally { agentSelecting = false; }
    map.flyToNeuron(e.layer, e.n, 0.3);
  }
  else if (e.action === "highlight") { agentMarked = true; map.setMarks((e.neurons || []).map(x => ({ layer: x.layer, n: x.n, sign: x.sign || "+", w: 1 }))); }
  else if (e.action === "rename") {
    const u = state.unitMap[e.layer]?.get(e.n); if (u) u.name = e.name.trim();
    if (isFocus(e.layer, e.n, "neuron")) setNameField(e.name.trim());
    if (e.layer === state.layer && state.units[e.layer]?.length) map.setUnits(e.layer, state.units[e.layer]);
    renderList();
  }
  else if (e.action === "annotate") map.setNotes(e.items || []);
  else if (e.action === "camera") {
    if (map.layerMode !== null) map.exitLayer();
    if (e.mode === "home") map.home();
    else if (e.mode === "layer") map.viewLayer(e.layer);
    else if (e.mode === "neuron") map.flyToNeuron(e.layer, e.n, e.closeness || 0.22);
  } else if (e.action === "tab") { if (state.focus && tabBtn(e.tab)) { if (curTab() === "claude") markFresh(e.tab); else setTab(e.tab); } }
  else if (e.action === "search") {
    if (state.qMode !== "lens") $('#q-mode button[data-q="lens"]').click();
    $("#lens-q").value = e.query; $("#lens-q").dispatchEvent(new Event("input"));
  } else if (e.action === "text") showText(e.text, e.token);
}
// put text in the fires box, run it, and pin the first token containing `token` (or the strongest one)
async function showText(text, token) {
  if (state.qMode !== "fires") $('#q-mode button[data-q="fires"]').click();
  $("#inf-text").value = text; $("#lens-q").value = text;
  clearTimeout(lensTimer);
  const g = ++qGen;
  await runInfer(text, g);
  if (g !== qGen || !state.inf.tokens) return;
  const i = token ? state.inf.tokens.findIndex(t => t.includes(token)) : -1;
  const pos = i >= 0 ? +document.querySelectorAll("#inf-toks button")[i].dataset.i : -1;
  state.inf.pos = pos;
  document.querySelectorAll("#inf-toks button").forEach(x => x.classList.toggle("on", +x.dataset.i === pos));
  inferView(); showTokenPaths(pos);
}
// a tool call as one short inline line: neuron L15 N6254 · search_lens "Sure"
function toolLabel(name, a = {}) {
  const ref = a.layer != null && a.n != null ? `L${a.layer} N${a.n}` : a.layer != null ? `L${a.layer}` : "";
  const short = s => (s.length > 90 ? s.slice(0, 90) + "…" : s);
  const extra = a.command ? short(a.command) : a.file_path ? a.file_path.replace(/^\/Users\/[^/]+/, "~")
    : a.pattern ? `"${a.pattern}"` : a.query ? `"${a.query}"` : a.neurons ? t("agent.neurons", { n: a.neurons.length }) : "";
  return [name.replace(/_/g, " "), ref, extra].filter(Boolean).join(" ");
}
function prettyOut(t) {
  try { return JSON.stringify(JSON.parse(t), null, 1); } catch { return t; }
}
function addMsg(cls, html) {
  const th = $("#a-thread"); th.querySelector(".a-empty")?.remove();
  const d = document.createElement("div"); d.className = `a-msg ${cls}`; d.innerHTML = html; th.append(d);
  th.scrollTop = th.scrollHeight; return d;
}
// one turn: the reply streams into a new assistant message, text and tool calls interleaved in the order they happen
// ---- the chat: one claude process per chat on the server and one event stream here. Messages go in any time:
// mid-reply they reach Claude between tool calls. Stop ends the reply only; background work keeps running and
// reports back in a turn of its own, which renders like any other. ----
function ensureChatId() { if (!agent.chat) agent.chat = crypto.randomUUID(); return agent.chat; }
let evCtl = null, evChat = null;
async function openEvents() {
  const chat = agent.chat; if (!chat || (evCtl && evChat === chat)) return;
  if (evCtl) evCtl.abort();
  const ctl = evCtl = new AbortController(); evChat = chat;
  while (!ctl.signal.aborted) {
    try {
      const r = await fetch(`/api/agent/events?chat=${encodeURIComponent(chat)}`, { signal: ctl.signal });
      const rd = r.body.getReader(), dec = new TextDecoder(); let buf = "";
      for (;;) {
        const { value, done } = await rd.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, i); buf = buf.slice(i + 1);
          if (line && evCtl === ctl) onAgentEvent(JSON.parse(line));
        }
      }
    } catch { if (ctl.signal.aborted) return; }
    await new Promise(r => setTimeout(r, 1500));  // reconnect (server restarts); its state event settles a turn that ended meanwhile
  }
}
function closeEvents() { if (evCtl) evCtl.abort(); evCtl = evChat = null; }
// the reply box under the latest message; its "thinking…" line moves with it
function replyBox() {
  if (agent.box) return agent.box;
  $("#a-thread").querySelectorAll(".a-busy").forEach(x => x.remove());
  $("#a-thread").querySelectorAll(".a-msg.assistant:empty").forEach(x => x.remove());  // a reply that moved under a newer message
  const box = agent.box = addMsg("assistant", "");
  agent.seg = null;
  if (agent.busy) { const busy = document.createElement("div"); busy.className = "a-busy"; box.append(busy); }
  return box;
}
const putInBox = el => { const box = replyBox(), busy = box.querySelector(".a-busy"); busy ? box.insertBefore(el, busy) : box.append(el); };
function setBusy(b) { agent.busy = b; setSendMode(b); }
function onAgentEvent(e) {
  const th = $("#a-thread"), stick = th.scrollHeight - th.scrollTop - th.clientHeight < 80;
  if (e.t === "state") {
    // (re)connected: the turn in progress replays from its start, so what was drawn of it is drawn again
    if (agent.box && e.busy) { agent.box.innerHTML = ""; agent.seg = null; agent.reply = ""; }
    if (e.busy) { setBusy(true); replyBox(); if (!agent.box.querySelector(".a-busy")) { const d = document.createElement("div"); d.className = "a-busy"; agent.box.append(d); } }
    else if (agent.busy && !agent.sending) finishTurn();  // a message still on its way makes the server busy in a moment
    return;
  }
  if (e.t === "ping" || e.t === "sent") return;
  if (e.t === "ui") return agentUi(e);
  if (e.t === "done") return finishTurn();
  if (!agent.busy) { setBusy(true); clearTurnControls(); }  // a turn Claude started itself (background work finished)
  if (e.t === "text") {
    if (!agent.seg) { agent.seg = document.createElement("div"); agent.seg.className = "a-md"; agent.segText = ""; putInBox(agent.seg); }
    agent.segText += e.text; agent.reply += e.text; agent.seg.innerHTML = md(agent.segText);
  } else if (e.t === "tool") { agent.seg = null; putInBox(toolDetails(e)); }
  else if (e.t === "tool_result") { if (agent.box) fillTool(agent.box, e); }
  else if (e.t === "error") { const d = document.createElement("div"); d.className = "a-err"; d.textContent = e.error; putInBox(d); }
  if (stick) th.scrollTop = th.scrollHeight;
}
function finishTurn() {
  const box = agent.box, reply = agent.reply;
  $("#a-thread").querySelectorAll(".a-busy").forEach(x => x.remove());
  agent.box = agent.seg = null; agent.reply = "";
  setBusy(false); addTurnControls(); markFresh("claude"); saveAgent();
  for (const w of agent.waiters.splice(0)) w({ reply, box });
}
async function agentTurn(message, shown, shownCls = "user", atts = []) {
  const f = state.focus, th = $("#a-thread");
  if (pastStash !== null) { th.innerHTML = pastStash; pastStash = null; $("#a-past").classList.remove("on"); }
  clearAgentMarks(); clearTurnControls();
  const ub = addMsg(shownCls, esc(shown) + (atts.length ? `<span class="a-att">📎 ${atts.map(a => esc(a.name)).join(", ")}</span>` : ""));
  ub.dataset.msg = message; ub.dataset.shown = shown; ub.dataset.cls = shownCls; ub.dataset.att = JSON.stringify(atts);
  agent.box = agent.seg = null;  // the reply, or the rest of the one in progress, goes under this message
  ensureChatId(); openEvents();
  setBusy(true); replyBox(); th.scrollTop = th.scrollHeight; saveAgent();
  const out = new Promise(res => agent.waiters.push(res));
  agent.sending++;
  try {
    const r = await api("/api/agent/send", { chat: agent.chat, agent: agent.kind, message, attachments: atts.map(a => a.id),
      ...(f && f.kind === "neuron" ? { layer: f.layer, n: f.n } : f && f.kind === "head" ? { layer: f.layer, head: f.n } : {}) });
    if (r.error) throw new Error(agentError(r));
  } catch (err) {
    const d = document.createElement("div"); d.className = "a-err"; d.textContent = String(err.message || err); putInBox(d); finishTurn();
  } finally { agent.sending--; }
  return out;
}
async function stopAgent() {
  if (!agent.busy) return;
  const d = document.createElement("div"); d.className = "a-stopped"; d.textContent = t("agent.stopped"); putInBox(d);
  const r = agent.chat ? await api("/api/agent/interrupt", { chat: agent.chat }).catch(() => ({})) : {};
  if (!r.running && agent.busy) finishTurn();  // nothing running server side: settle here
}
function toolDetails(e) {
  const d = document.createElement("details"); d.className = "a-tool"; if (e.id) d.dataset.id = e.id;
  d.innerHTML = `<summary>${esc(toolLabel(e.name, e.input || {}))}</summary><pre class="a-io"><b>${esc(e.name)}</b> ${esc(JSON.stringify(e.input || {}, null, 1))}</pre><pre class="a-io out running">${t("agent.running")}</pre>`;
  return d;
}
function fillTool(box, e) {
  const d = [...box.querySelectorAll("details.a-tool")].find(x => x.dataset.id === e.id);
  if (d) { const o = d.querySelector(".out"); o.textContent = prettyOut(e.output); o.classList.remove("running"); o.classList.toggle("err", e.error); }
}
// ---- past chats: "past" lists the sessions on disk; picking one rebuilds its thread and later turns resume it ----
let pastStash = null;  // the thread shown before the list opened
const ago = ts => {
  const s = Date.now() / 1000 - ts;
  return s < 3600 ? t("agent.minutesAgo", { n: Math.max(1, Math.round(s / 60)) })
    : s < 86400 ? t("agent.hoursAgo", { n: Math.round(s / 3600) })
    : new Date(ts * 1000).toLocaleDateString(getLang());
};
async function renderPastList() {
  const th = $("#a-thread");
  th.innerHTML = `<div class="dim">…</div>`;
  const list = await api("/api/agent/chats");
  if (pastStash === null) return;
  const tag = a => a === "codex" ? "codex · " : a === "dsh" ? "DSH · " : "";
  th.innerHTML = list.length ? list.map(c => `<button class="a-pastrow${c.chat === agent.chat ? " on" : ""}" data-chat="${c.chat}" data-agent="${c.agent || "claude"}"><span class="t">${esc(c.title || "—")}</span><span class="m">${tag(c.agent || "claude")}${ago(c.t)} · ${tp("agent.messages", c.turns)}</span></button>`).join("")
    : `<div class="dim">—</div>`;
}
async function togglePast() {
  const th = $("#a-thread");
  if (pastStash !== null) { th.innerHTML = pastStash; pastStash = null; $("#a-past").classList.remove("on"); addTurnControls(); return; }
  pastStash = th.innerHTML; $("#a-past").classList.add("on");
  await renderPastList();
}
async function openPast(chat, kind) {
  const r = await api(`/api/agent/history?chat=${encodeURIComponent(chat)}`);
  pastStash = null; $("#a-past").classList.remove("on");
  const th = $("#a-thread"); th.innerHTML = EMPTY();
  let box = null, seg = null, segText = "";
  for (const e of r.events || []) {
    if (e.t === "user") {
      const ub = addMsg("user", esc(e.text));
      Object.assign(ub.dataset, { msg: e.text, shown: e.text, cls: "user", att: "[]" });
      box = seg = null; continue;
    }
    if (!box) { box = addMsg("assistant", ""); seg = null; }
    if (e.t === "text") {
      if (!seg) { seg = document.createElement("div"); seg.className = "a-md"; box.append(seg); segText = ""; }
      segText += (segText ? "\n\n" : "") + e.text; seg.innerHTML = md(segText);
    } else if (e.t === "tool") { seg = null; box.append(toolDetails(e)); }
    else if (e.t === "tool_result") fillTool(box, e);
    else if (e.t === "error") { seg = null; const d = document.createElement("div"); d.className = "a-err"; d.textContent = e.error; box.append(d); }
  }
  closeEvents(); if (agent.busy) { agent.box = null; finishTurn(); }
  agent.chat = chat; setAgentKind(kind); clearAgentMarks(); addTurnControls(); saveAgent(); openEvents();
  th.scrollTop = th.scrollHeight;
}
$("#a-thread").addEventListener("click", e => { const b = e.target.closest(".a-pastrow"); if (b) openPast(b.dataset.chat, b.dataset.agent); });
async function explainFocus() {
  const f = state.focus; if (!f) return;
  if (f.kind === "head") return agentTurn(`Explain attention head L${f.layer} h${f.n}: write the head report.`, t("agent.explain", { ref: `L${f.layer} h${f.n}` }));
  if (f.kind !== "neuron") return;
  const { reply, box } = await agentTurn(`Explain L${f.layer} N${f.n}: write the autointerp report.`, t("agent.explain", { ref: `L${f.layer} N${f.n}` }));
  const name = reply.match(/\*\*Name\*\*:?\s*(.+)/)?.[1].replace(/[*`]/g, "").trim();
  if (name) {
    const b = document.createElement("button"); b.id = "a-usename"; b.dataset.name = name; b.textContent = t("agent.useName", { name });
    b.onclick = () => { if (isFocus(f.layer, f.n, "neuron")) setNameField(name); saveName({ layer: f.layer, n: f.n, name }); b.remove(); };
    box.append(b);
  }
}
function sendChat() {
  const msg = $("#a-input").value.trim();
  if (agent.pending.some(a => !a.id) || (!msg && !agent.pending.length)) return false;
  const atts = agent.pending; agent.pending = []; renderPending();
  $("#a-input").value = ""; growInput();
  agentTurn(msg || "(see the attached files)", msg || t("agent.attached"), "user", atts);
  return true;
}
function setSendMode(busy) {
  const b = $("#a-send"); b.disabled = false; b.classList.toggle("stop", busy);
  b.textContent = busy ? "■" : "↑"; const label = busy ? t("agent.stop") : t("agent.send"); b.title = label; b.setAttribute("aria-label", label);
}

// ---- attachments: picked, dropped or pasted files upload right away and go with the next message ----
agent.pending = [];
function renderPending() {
  $("#a-files").innerHTML = agent.pending.map((a, i) => `<span class="a-file${a.id ? "" : " up"}">${esc(a.name)}<button data-i="${i}" aria-label="${esc(t("agent.remove"))}">×</button></span>`).join("");
}
$("#a-files").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { agent.pending.splice(+b.dataset.i, 1); renderPending(); } });
async function attachFiles(files) {
  for (const file of files) {
    const a = { name: file.name || t("agent.pasted"), id: null }; agent.pending.push(a); renderPending();
    const data = await new Promise((ok, bad) => { const r = new FileReader(); r.onload = () => ok(String(r.result).split(",")[1] || ""); r.onerror = bad; r.readAsDataURL(file); });
    try {
      const res = await api("/api/agent/upload", { name: a.name, type: file.type, data });
      if (res.error) throw new Error(res.error);
      a.id = res.id;
    } catch (err) { agent.pending.splice(agent.pending.indexOf(a), 1); addMsg("assistant", `<div class="a-err">${esc(a.name)}: ${esc(String(err.message || err))}</div>`); }
    renderPending();
  }
}
$("#a-attach").onclick = () => $("#a-file").click();
$("#a-file").addEventListener("change", e => { attachFiles([...e.target.files]); e.target.value = ""; });
agentEl.addEventListener("dragover", e => { if (e.dataTransfer?.types?.includes("Files")) { e.preventDefault(); agentEl.classList.add("drop"); } });
agentEl.addEventListener("dragleave", e => { if (!agentEl.contains(e.relatedTarget)) agentEl.classList.remove("drop"); });
agentEl.addEventListener("drop", e => { if (!e.dataTransfer?.files?.length) return; e.preventDefault(); agentEl.classList.remove("drop"); attachFiles([...e.dataTransfer.files]); });
$("#a-input").addEventListener("paste", e => { const fs = [...(e.clipboardData?.files || [])]; if (fs.length) { e.preventDefault(); attachFiles(fs); } });

// ---- edit / reroll the latest turn: cut the claude session back to before it, then resend or hand it back ----
function clearTurnControls() { document.querySelectorAll("#a-thread .a-ctl").forEach(x => x.remove()); }
function lastTurn() {
  const users = [...document.querySelectorAll("#a-thread .a-msg.user")], u = users[users.length - 1];
  if (!u) return null;
  const after = []; for (let x = u.nextElementSibling; x; x = x.nextElementSibling) after.push(x);
  return { u, after };
}
function addTurnControls() {
  clearTurnControls();
  const turn = lastTurn(); if (!turn) return;
  const uc = document.createElement("div"); uc.className = "a-ctl"; uc.innerHTML = `<button data-act="edit" title="${esc(t("agent.editTitle"))}">${t("agent.edit")}</button>`;
  turn.u.append(uc);
  const a = turn.after.find(x => x.classList.contains("assistant"));
  if (a) { const ac = document.createElement("div"); ac.className = "a-ctl"; ac.innerHTML = `<button data-act="reroll" title="${esc(t("agent.rerollTitle"))}">${t("agent.reroll")}</button>`; a.append(ac); }
}
async function rewindLast(act) {
  const turn = lastTurn(); if (!turn) return;
  if (agent.busy) { agent.box = null; finishTurn(); }
  const u = turn.u, msg = u.dataset.msg ?? u.childNodes[0]?.textContent ?? "", shown = u.dataset.shown ?? msg;
  const cls = u.dataset.cls || "user", atts = JSON.parse(u.dataset.att || "[]");
  if (agent.chat) { try { await api("/api/agent/rewind", { chat: agent.chat, text: msg }); } catch {} }
  for (const x of [u, ...turn.after]) x.remove();
  if (!$("#a-thread").children.length) $("#a-thread").innerHTML = EMPTY();
  if (act === "reroll") agentTurn(msg, shown, cls, atts);
  else { $("#a-input").value = cls.includes("interp") || shown !== msg ? shown : msg; growInput(); agent.pending = atts; renderPending(); $("#a-input").focus(); addTurnControls(); saveAgent(); }
}
$("#a-thread").addEventListener("click", e => { const b = e.target.closest(".a-ctl button"); if (b) rewindLast(b.dataset.act); });
function growInput() { const t = $("#a-input"); t.style.height = "auto"; t.style.height = `${Math.min(200, t.scrollHeight)}px`; }
$("#a-input").addEventListener("input", growInput);
// Enter sends, also while Claude works: the message reaches it between tool calls
$("#a-input").addEventListener("keydown", e => {
  if (e.key !== "Enter" || e.shiftKey) return;
  e.preventDefault();
  sendChat();
});
// the button sends, or stops the reply while Claude works (Enter always sends: mid-reply the message goes straight in)
$("#a-form").addEventListener("submit", e => { e.preventDefault(); if (agent.busy) stopAgent(); else sendChat(); });
$("#a-new").onclick = () => { pastStash = null; $("#a-past").classList.remove("on"); resetAgent(); clearAgentMarks(); };
$("#a-past").onclick = togglePast;
{ let pref = "claude"; try { pref = localStorage.getItem("explorer.agentKind") || "claude"; } catch {} setAgentKind(pref); }
loadAgent(); renderAgentActions(); openEvents();
syncEdge(); updateInsets();

// ---------- language ----------
// a switch redraws everything derived from a translated string; open neurons, searches and conversations stay put
function relabelAgentThread() {
  const th = $("#a-thread");
  th.querySelectorAll(".a-empty").forEach(el => { el.innerHTML = t("agent.empty"); });
  th.querySelectorAll(".a-stopped").forEach(el => { el.textContent = t("agent.stopped"); });
  th.querySelectorAll("details.a-tool .a-io.out.running").forEach(el => { el.textContent = t("agent.running"); });
  th.querySelectorAll(".a-ctl button[data-act=edit]").forEach(b => { b.textContent = t("agent.edit"); b.title = t("agent.editTitle"); });
  th.querySelectorAll(".a-ctl button[data-act=reroll]").forEach(b => { b.textContent = t("agent.reroll"); b.title = t("agent.rerollTitle"); });
  const un = $("#a-usename");
  if (un) un.textContent = t("agent.useName", { name: un.dataset.name || un.textContent.replace(/^use name:\s*/, "").replace(/^使用名称：/, "") });
}
function relabel() {
  hideTip();
  applyTranslations();
  syncPlaceholders();
  syncEdge();
  setAgentKind(agent.kind);
  setSendMode(agent.busy);
  $("#c-go").textContent = running ? t("chat.stop") : t("chat.run");
  if (map.layerMode !== null) $("#layer-back").textContent = t("map.allLayers", { l: map.layerMode });
  if (state.webTitle) $("#web-z").title = t("web.title", state.webTitle);
  if (state.statusText) $("#status").textContent = localizeStatus(state.statusText);
  const f = state.focus;
  if (f?.kind === "random") $("#f-id").textContent = t("focus.idRandom", { layer: f.layer, n: f.n });
  if ((f?.kind === "neuron" || f?.kind === "random") && state.unit) {
    if (state.unit.stats) $("#f-rank").textContent = t("focus.rank", { rank: state.unit.rank ?? "—", n: state.unit.n_ranked, mass: (state.unit.stats.mass ?? state.unit.stats.z).toFixed(1), nullPct: (state.unit.stats.null_pct * 100).toFixed(0) });
    renderExamples(state.unit);
    renderDetails(state.unit);
    renderLensDetails();
    renderConnections();
  }
  renderList();
  if (fireResults) renderResults(fireResults, $("#run-rows"));
  renderAgentActions();
  if (!running) renderThread();
  relabelAgentThread();
  if (agent.pending.length) renderPending();
  if (pastStash !== null) renderPastList();
}
document.addEventListener("i18nchange", relabel);
syncPlaceholders();
