"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const REASONS = {
  filler: "заминка / паразит", soft: "слово-паразит", repeat: "повтор", falsestart: "оговорка",
  sound: "звук", manual: "вручную", range: "вырезанный участок", retake: "дубль (остался поздний)",
};
const PAUSE_FIELDS = [
  ["max_pause", "Макс. пауза во фразе, с", 0.05],
  ["max_pause_sentence", "Между предложениями, с", 0.05],
  ["join_pause", "На месте вырезки, с", 0.01],
  ["lead", "Тишина в начале, с", 0.05],
  ["tail", "Тишина в конце, с", 0.05],
];
const CLEAN_FIELDS = [
  ["denoise", "Шумоподавление, дБ", 1],
  ["dereverb", "Подавление эха, 0–2", 0.1],
  ["gate", "Гейт пауз, дБ ниже речи", 1],
  ["dehum", "Гул сети", null],
  ["highpass", "Срез низов, Гц", 10],
  ["deess", "Де-эссер, дБ", 1],
  ["leveler", "Выравнивание громкости, 0–1", 0.1],
  ["compress", "Компрессор", 0.5],
  ["lufs", "Громкость, LUFS", 1],
  ["limit", "Лимитер, дБ", 0.5],
];
const OUTPUT_FIELDS = [
  ["crossfade_ms", "Кроссфейд на склейках, мс", 1],
];
const SENT_END = /[.!?…]+["'»”)]*$/;
const RULER = 20;

let P = null;          // текущий проект
let peaks = [];        // огибающая исходника
let outPeaks = null;   // {name, duration, peaks} результата
let lastMtime = 0;
let tokEls = [];
let timed = [];        // токены со временем
let mapped = [];       // [new_s, new_e, i] — где слова в результате
let sel = null;        // выделенные токены [a, b]
let anchor = null;
let range = null;      // выделенный участок времени исходника [t1, t2]
let active = "orig";   // какую дорожку играем
let playingEl = null;
let scaleDb = store("voicer.scale") === "db";
const view = { t0: 0, pps: 50 };
let tlStatic = null;   // заранее нарисованная статичная часть таймлайна

// ---------- утилиты ----------
function store(k, v) {
  try {
    if (v === undefined) return localStorage.getItem(k);
    localStorage.setItem(k, v);
  } catch { return null; }
}
async function api(url, opts = {}) {
  const r = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let m = r.statusText;
    try { m = (await r.json()).detail || m; } catch {}
    throw new Error(m);
  }
  return r.json();
}
function fmtT(t, d = 1) {
  if (t == null || isNaN(t)) return "--:--";
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(d).padStart(d ? 3 + d : 2, "0")}`;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
let toastTimer;
function toast(msg, err = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.className = err ? "err" : "";
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), err ? 6000 : 3000);
}
let statusTimer;
function status(msg, cls = "", ms = 6000) {
  const el = $("#status");
  el.textContent = msg;
  el.className = "save-status " + cls;
  el.title = msg;
  clearTimeout(statusTimer);
  if (ms) statusTimer = setTimeout(() => { el.textContent = ""; el.className = "save-status"; }, ms);
}
const textOf = (t) => (P.edit.fixes[t.i] ?? t.t);
function effT(t) {  // границы токена с учётом заданных вручную краёв вырезки
  const b = P.edit.bounds && P.edit.bounds[t.i];
  return b ? { s: b.s ?? t.s, e: b.e ?? t.e } : { s: t.s, e: t.e };
}
const isAudio = () => !!(P && P.kind === "audio" && P.status === "ready");
const stale = () => !!(P && P.last_render && P.rev > P.last_render.rev);
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const byName = (by) => ({ ui: "вы", cli: "Claude", mcp: "Claude", claude: "Claude", auto: "авто" }[by] || by);

// ---------- проекты и загрузка ----------
async function loadList() {
  const list = await api("/api/projects");
  $("#plist").innerHTML = list.map((p) => {
    const info = p.kind === "audio" ? fmtT(p.duration) : `${p.words} слов`;
    const st = p.status && p.status !== "ready" ? ` · ${esc(p.status.slice(0, 30))}` : "";
    return `<a href="#${esc(p.id)}" data-id="${esc(p.id)}" class="${P && P.id === p.id ? "active" : ""}" title="Правый клик — удалить">
      <span class="pn">${esc(p.name)}</span><small>${p.kind === "audio" ? "аудио" : "текст"} · ${info}${st}</small></a>`;
  }).join("") || `<div class="none">Проектов пока нет</div>`;
  return list;
}
$("#plist").addEventListener("click", (e) => {
  const a = e.target.closest("a[data-id]");
  if (!a) return;
  e.preventDefault();
  openProject(a.dataset.id);
});
$("#plist").addEventListener("contextmenu", (e) => {
  const a = e.target.closest("a[data-id]");
  if (!a) return;
  e.preventDefault();
  showPop(e.clientX, e.clientY, `<div class="mi" data-del="1">Удалить проект…</div>`, (pop) => {
    pop.querySelector("[data-del]").onclick = () => { hidePop(); deleteProject(a.dataset.id); };
  });
});
async function deleteProject(id) {
  if (!confirm("Удалить проект вместе с файлами?")) return;
  await api(`/api/projects/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (P && P.id === id) {
    P = null;
    $("#aOrig").removeAttribute("src");
    $("#aOut").removeAttribute("src");
    history.replaceState(null, "", "#");
    renderAll();
  }
  loadList();
}

let jobTimer = null;
const packJobs = new Map();  // задача пака → проект: открыть, когда будет готов
async function pollJobs() {
  const jobs = await api("/api/jobs");
  for (const j of jobs) if (packJobs.has(j.id) && j.project) packJobs.set(j.id, j.project);
  for (const [id, pid] of packJobs) {
    if (jobs.some((j) => j.id === id)) continue;
    packJobs.delete(id);
    if (pid) { await loadList(); openProject(pid); }
  }
  $("#jobs").innerHTML = jobs.map((j) => `
    <div class="job ${j.error ? "err" : ""}">
      <b>${esc(j.file)}</b><span class="msg">${esc(j.error ? "Ошибка: " + j.error : j.msg)}</span>
      ${j.error ? ` <button class="mini-btn" data-drop="${j.id}">скрыть</button>` :
        `<div class="bar"><i style="width:${Math.round(j.progress * 100)}%"></i></div>`}
    </div>`).join("");
  if (jobs.some((j) => !j.done)) jobTimer = setTimeout(pollJobs, 800);
  else { jobTimer = null; await loadList(); }
}
$("#jobs").addEventListener("click", async (e) => {
  const id = e.target.dataset.drop;
  if (id) { await api(`/api/jobs/${id}`, { method: "DELETE" }); pollJobs(); }
});
async function uploadFiles(files) {
  for (const f of files) {
    const fd = new FormData();
    fd.append("file", f);
    fd.append("lang", $("#lang").value);
    try {
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.statusText);
      if (j.id) { await loadList(); openProject(j.id); }
      else status(`«${f.name}»: распознаю речь…`, "flash");
    } catch (e) { toast(`${f.name}: ${e.message}`, true); }
  }
  if (!jobTimer) pollJobs();
}
async function uploadPack(files) {
  if (!files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  fd.append("lang", $("#lang").value);
  try {
    const r = await fetch("/api/upload_pack", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    packJobs.set(j.job, null);
    status(`Пак из ${files.length} файлов: склеиваю, распознаю, ищу дубли…`, "flash");
  } catch (e) { toast(`Пак: ${e.message}`, true); }
  if (!jobTimer) pollJobs();
}
for (const b of ["#btnAdd", "#btnAdd2"]) $(b).onclick = () => $("#file").click();
$("#file").onchange = (e) => { uploadFiles([...e.target.files]); e.target.value = ""; };
$("#btnPack").onclick = () => $("#filePack").click();
$("#filePack").onchange = (e) => { uploadPack([...e.target.files]); e.target.value = ""; };
for (const b of ["#btnText", "#btnText2"]) $(b).onclick = () => $("#dlgText").showModal();
$("#textForm").addEventListener("submit", async (e) => {
  if (e.submitter?.value !== "ok") return;
  const text = $("#textBody").value.trim();
  if (!text) return;
  try {
    const r = await api("/api/text", { method: "POST", body: JSON.stringify({ text, name: $("#textName").value.trim() }) });
    $("#textBody").value = $("#textName").value = "";
    await loadList();
    openProject(r.id);
  } catch (err) { toast(err.message, true); }
});
let dragDepth = 0;
document.addEventListener("dragenter", (e) => {
  if (e.dataTransfer?.types?.includes("Files")) { dragDepth++; $("#dropOverlay").hidden = false; }
});
document.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("#dropOverlay").hidden = true; } });
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  $("#dropOverlay").hidden = true;
  if (e.dataTransfer?.files?.length) uploadFiles([...e.dataTransfer.files]);
});

// ---------- открытие проекта ----------
async function openProject(id) {
  try {
    P = await api(`/api/projects/${encodeURIComponent(id)}`);
  } catch (e) { toast(e.message, true); return; }
  lastMtime = P.mtime;
  if (location.hash.slice(1) !== P.id) history.replaceState(null, "", "#" + P.id);
  peaks = isAudio() ? await api(`/api/projects/${P.id}/peaks`) : [];
  const a = $("#aOrig");
  a.pause(); $("#aOut").pause();
  if (isAudio()) a.src = `/api/projects/${P.id}/source`; else a.removeAttribute("src");
  setOutSrc();
  await loadOutPeaks();
  sel = range = null; anchor = null;
  setActive("orig");
  renderAll();
  fitView();
  loadList();
}
function setOutSrc() {
  const b = $("#aOut");
  const last = P && P.last_render;
  if (last && P.renders.includes(last.name)) {
    const url = `/api/projects/${P.id}/out/${encodeURIComponent(last.name)}?v=${last.rev}`;
    if (!b.src.endsWith(url)) b.src = url;
  } else b.removeAttribute("src");
}
async function loadOutPeaks() {
  outPeaks = null;
  if (isAudio() && P.last_render && P.renders.includes(P.last_render.name)) {
    try {
      const r = await api(`/api/projects/${P.id}/peaks_out`);
      if (r && r.peaks && r.name === P.last_render.name) outPeaks = r;
    } catch {}
  }
}

function renderAll() {
  const has = !!P;
  $("#empty").hidden = has;
  $("#transcript").hidden = !has;
  $("#rightBody").hidden = !has;
  const audio = isAudio();
  $("#timeline").hidden = $("#split").hidden = !audio;
  for (const el of ["#cleanBox", "#fmt", "#btnProcess"]) $(el).hidden = has && !audio;
  $("#btnProcess").disabled = !audio;
  $("#btnDl").disabled = $("#btnUndo").disabled = !has;
  for (const s of ["pauses", "clean", "output"]) $(`[data-sec="${s}"]`).hidden = !audio;
  renderTop();
  if (has) { renderTranscript(); renderRight(); }
  buildTl();
  drawTl();
}

function renderTop() {
  $("#pname").textContent = P ? P.name : "";
  const vc = !!(P && P.settings.voice_clean);
  $("#voiceClean").checked = vc;
  $("#cleanBox").classList.toggle("on", vc);
  if (P) $("#fmt").value = P.settings.format || "mp3";
  $("#stale").hidden = !(outPeaks && stale());
  if (!P) { $("#tstats").textContent = ""; return; }
  const cutN = Object.keys(P.cuts).length;
  const words = P.tokens.filter((t) => t.kind !== "sound").length;
  $("#tstats").textContent = isAudio()
    ? `${words} слов · вырезано ${cutN} · ${fmtT(P.duration)} → ${fmtT(P.edited_duration)}`
    : P.kind === "audio" ? (P.status === "processing" ? "распознавание…" : P.status) : `${words} слов · вырезано ${cutN}`;
}

// ---------- транскрипт ----------
function renderTranscript() {
  const box = $("#transcript");
  tokEls = [];
  timed = P.tokens.filter((t) => t.s !== undefined);
  mapped = Object.entries(P.map || {}).map(([i, [s, e]]) => [s, e, +i]).sort((a, b) => a[0] - b[0]);
  const soft = new Set(P.flags.soft);
  const pauses = P.edit.pauses;
  const frag = document.createDocumentFragment();
  let para = document.createElement("p");
  P.tokens.forEach((t, k) => {
    const el = document.createElement("span");
    el.className = "tok";
    el.dataset.i = t.i;
    let title = t.s !== undefined ? fmtT(t.s, 2) : "";
    if (t.kind === "sound") {
      el.classList.add("sound");
      if (t.hes) el.classList.add("hes");
      el.textContent = `${t.hes ? "заминка" : "звук"} ${(t.e - t.s).toFixed(1)}с`;
      title += t.hes ? " · заминка, которой нет в тексте" : " · звук без слова (вдох, шум)";
    } else {
      el.textContent = textOf(t);
      if (t.p != null && t.p < 0.5) el.classList.add("low");
      if (soft.has(t.i)) el.classList.add("soft");
    }
    if (P.cuts[t.i] !== undefined) {
      el.classList.add("cut");
      el.dataset.why = P.cuts[t.i];
      title += ` · вырезано: ${REASONS[P.cuts[t.i]] || P.cuts[t.i]}`;
    }
    if (sel && t.i >= sel[0] && t.i <= sel[1]) el.classList.add("sel");
    el.title = title;
    tokEls[t.i] = el;
    para.append(el, " ");
    const nx = P.tokens[k + 1];
    if (nx && t.s !== undefined) {
      const gap = nx.s - t.e;
      const ov = pauses[t.i];
      const limit = SENT_END.test(textOf(t)) ? P.settings.max_pause_sentence : P.settings.max_pause;
      if (gap >= 0.3 || ov !== undefined) {
        const c = document.createElement("span");
        c.className = "pause";
        c.dataset.after = t.i;
        if (ov !== undefined) { c.classList.add("set"); c.textContent = `${gap.toFixed(1)}→${(+ov).toFixed(2)}`; }
        else if (gap > limit) {
          c.classList.add("long");
          const d = gap.toFixed(1) === (+limit).toFixed(1) ? 2 : 1;
          c.textContent = `${gap.toFixed(d)}→${(+limit).toFixed(d)}`;
        } else c.textContent = gap.toFixed(1);
        c.title = "Пауза, с. Клик — задать длину";
        para.append(c, " ");
      }
      if (gap >= 1.5) { frag.append(para); para = document.createElement("p"); }
    } else if (t.s === undefined && SENT_END.test(textOf(t)) && para.childNodes.length > 120) {
      frag.append(para); para = document.createElement("p");
    }
  });
  frag.append(para);
  box.replaceChildren(frag);
  box.classList.toggle("hide-cut", !$("#tglCut").classList.contains("on"));
  playingEl = null;
  updateSelInfo();
}
$("#tglCut").onclick = () => {
  const b = $("#tglCut");
  b.classList.toggle("on");
  $("#transcript").classList.toggle("hide-cut", !b.classList.contains("on"));
};

$("#transcript").addEventListener("click", async (e) => {
  const pz = e.target.closest(".pause");
  if (pz) { editPause(+pz.dataset.after, pz); return; }
  const el = e.target.closest(".tok");
  if (!el) return;
  const i = +el.dataset.i;
  if (e.ctrlKey || e.metaKey) {
    await ops([P.cuts[i] !== undefined ? `keep ${i}` : `cut ${i}`], true);
    return;
  }
  range = null;
  if (e.shiftKey && anchor !== null) sel = [Math.min(anchor, i), Math.max(anchor, i)];
  else {
    anchor = i;
    sel = [i, i];
    const t = P.tokens[i];
    if (t.s !== undefined) {
      setActive("orig");
      const a = $("#aOrig");
      a.currentTime = Math.max(0, t.s - 0.12);
      a.play();
      revealTime(t.s, t.e);
    }
  }
  paintSel();
});
function paintSel() {
  $$(".tok.sel").forEach((x) => x.classList.remove("sel"));
  if (sel) for (let k = sel[0]; k <= sel[1]; k++) tokEls[k]?.classList.add("sel");
  updateSelInfo();
  buildTl();
  drawTl();
}
function updateSelInfo() {
  const box = $("#selInfo");
  if (sel && P) {
    const n = sel[1] - sel[0] + 1;
    $("#selText").textContent = n === 1 ? `«${textOf(P.tokens[sel[0]])}»` : `${n} слов`;
    box.hidden = false;
  } else if (range) {
    $("#selText").textContent = `${range[0].toFixed(2)}–${range[1].toFixed(2)} с`;
    box.hidden = false;
  } else box.hidden = true;
}
function rangeTokens() {
  return P.tokens.filter((t) => t.s !== undefined && Math.min(t.e, range[1]) - Math.max(t.s, range[0]) > 0.5 * (t.e - t.s));
}
function idsLabel(ids) {
  ids = [...new Set(ids)].sort((a, b) => a - b);
  const parts = [];
  for (let k = 0; k < ids.length;) {
    let j = k;
    while (j + 1 < ids.length && ids[j + 1] === ids[j] + 1) j++;
    parts.push(j > k ? `${ids[k]}-${ids[j]}` : `${ids[k]}`);
    k = j + 1;
  }
  return parts.join(",");
}
async function cutSelection(keep = false) {
  if (!P) return;
  if (sel) return ops([`${keep ? "keep" : "cut"} ${sel[0]}-${sel[1]}`]);
  if (!range) return;
  const ids = rangeTokens().map((t) => t.i);
  const span = `${range[0].toFixed(3)}-${range[1].toFixed(3)}`;
  if (keep) return ops([...(ids.length ? [`keep ${idsLabel(ids)}`] : []), `uncuttime ${span}`]);
  return ops([ids.length ? `cut ${idsLabel(ids)}` : `cuttime ${span}`]);
}
$("#selCut").onclick = () => cutSelection(false);
$("#selKeep").onclick = () => cutSelection(true);

// ---------- всплывающее окошко ----------
function showPop(x, y, html, init) {
  const pop = $("#pop");
  pop.innerHTML = html;
  pop.hidden = false;
  const r = pop.getBoundingClientRect();
  pop.style.left = Math.max(8, Math.min(x, innerWidth - r.width - 8)) + "px";
  pop.style.top = Math.max(8, Math.min(y, innerHeight - r.height - 8)) + "px";
  init?.(pop);
}
function hidePop() { $("#pop").hidden = true; }
document.addEventListener("mousedown", (e) => {
  if (!e.target.closest("#pop")) hidePop();
  if (!e.target.closest(".menu")) $("#dlmenu").hidden = true;
});
function editPause(i, chip) {
  const cur = P.edit.pauses[i];
  const r = chip.getBoundingClientRect();
  showPop(r.left, r.bottom + 6, `
    <div class="pop-title">Пауза после «${esc(textOf(P.tokens[i]))}», с</div>
    <div class="pop-row"><input class="num" id="popVal" type="number" step="0.05" min="0" value="${cur ?? ""}" placeholder="авто">
    <button class="tb accent" id="popOk">OK</button><button class="tb" id="popAuto">Авто</button></div>`, (pop) => {
    const inp = pop.querySelector("#popVal");
    inp.focus();
    const apply = (v) => { hidePop(); ops([v === "" ? `pause ${i} auto` : `pause ${i} ${v}`]); };
    pop.querySelector("#popOk").onclick = () => apply(inp.value.trim().replace(",", "."));
    pop.querySelector("#popAuto").onclick = () => apply("");
    inp.onkeydown = (e) => {
      if (e.key === "Enter") apply(inp.value.trim().replace(",", "."));
      if (e.key === "Escape") hidePop();
    };
  });
}

// ---------- правая панель ----------
function fillForm(form, fields) {
  if (form.contains(document.activeElement)) return;
  form.innerHTML = fields.map(([k, label, step]) => {
    const v = P.settings[k];
    const custom = P.settings_custom.includes(k);
    const def = P.settings_defaults[k];
    const tip = custom ? `задано в проекте; по умолчанию ${def ?? "выкл."}` : "значение по умолчанию";
    const cls = `num${custom ? " custom" : ""}`;
    const input = k === "dehum"
      ? `<select class="${cls}" data-key="${k}" title="${esc(tip)}">${[["0", "выкл."], ["auto", "авто"], ["50", "50 Гц"], ["60", "60 Гц"]]
          .map(([o, t]) => `<option value="${o}" ${String(v) === o ? "selected" : ""}>${t}</option>`).join("")}</select>`
      : `<input class="${cls}" data-key="${k}" type="number" step="${step}" value="${v ?? ""}" placeholder="выкл." title="${esc(tip)}">`;
    return `<label class="w-row"><span class="w-label">${label}</span>${input}</label>`;
  }).join("");
}
function renderRight() {
  if (document.activeElement !== $("#notes")) $("#notes").value = P.notes || "";
  if (isAudio()) {
    fillForm($("#fPauses"), PAUSE_FIELDS);
    fillForm($("#fClean"), CLEAN_FIELDS);
    fillForm($("#fOutput"), OUTPUT_FIELDS);
    const vc = !!P.settings.voice_clean;
    $("#fClean").classList.toggle("off", !vc);
    $("#cleanState").textContent = vc ? "включена" : "выключена";
    $("#cleanState").className = "sec-state" + (vc ? " on" : "");
  }
  $("#log").innerHTML = P.log.slice().reverse().map((x) =>
    `<li><time>${esc(x.ts.slice(11, 16))}</time><span class="by ${esc(x.by)}">${esc(byName(x.by))}</span>${esc(x.msg)}</li>`).join("");
}
for (const f of ["#fPauses", "#fClean", "#fOutput"]) {
  $(f).addEventListener("change", (e) => {
    const k = e.target.dataset.key;
    if (!k) return;
    const v = e.target.value.trim().replace(",", ".");
    const empty = ["lufs", "limit"].includes(k) ? "off" : f === "#fClean" ? "0" : "default";
    ops([`set ${k} ${v === "" ? empty : v}`], true);
  });
}
$$(".insp-section").forEach((sec) => {
  const key = "voicer.sec." + sec.dataset.sec;
  if (store(key) === "closed") sec.classList.add("closed");
  sec.querySelector(".insp-sec-head").onclick = () => {
    sec.classList.toggle("closed");
    store(key, sec.classList.contains("closed") ? "closed" : "open");
  };
});
$("#btnAuto").onclick = () => {
  const opts = $$("[data-auto]").filter((x) => x.checked)
    .map((x) => (x.dataset.auto === "soft" ? $("#softMode").value : x.dataset.auto));
  if (!opts.length) return toast("Отметьте хотя бы одно правило", true);
  ops([`auto ${opts.join(" ")}`]);
};
$("#btnReset").onclick = () => confirm("Вернуть все вырезанные слова и паузы?") && ops(["reset"]);
$("#btnPauseDefaults").onclick = () => ops(PAUSE_FIELDS.map(([k]) => `set ${k} default`));
$("#btnPreset").onclick = () => ops(["studio"]);
$("#btnReanalyze").onclick = async () => {
  const b = $("#btnReanalyze");
  b.disabled = true;
  status("Пересчитываю паузы и заминки…", "flash", 0);
  try {
    const r = await api(`/api/projects/${P.id}/reanalyze`, { method: "POST" });
    P = r.project;
    lastMtime = P.mtime;
    renderAll();
    status(r.msg, "flash");
  } catch (e) { toast(e.message, true); status(""); }
  b.disabled = false;
};
$("#btnNotes").onclick = async () => {
  await api(`/api/projects/${P.id}/notes`, { method: "POST", body: JSON.stringify({ notes: $("#notes").value }) });
  $("#notesState").textContent = "сохранено";
  setTimeout(() => ($("#notesState").textContent = ""), 2000);
};
$("#cmd").addEventListener("keydown", async (e) => {
  if (e.key !== "Enter") return;
  const v = e.target.value.trim();
  if (!v) return;
  try {
    const r = await ops([v], true);
    $("#cmdout").className = "dim small";
    $("#cmdout").textContent = r.report.join("\n");
    e.target.value = "";
  } catch (err) {
    $("#cmdout").className = "small err";
    $("#cmdout").textContent = err.message;
  }
});

// ---------- правки ----------
async function ops(cmds, quiet = false) {
  if (!P) return;
  try {
    const r = await api(`/api/projects/${P.id}/ops`, { method: "POST", body: JSON.stringify({ commands: cmds }) });
    P = r.project;
    lastMtime = P.mtime;
    renderTop(); renderTranscript(); renderRight(); buildTl(); drawTl();
    status(r.report.join(" · "), "flash", quiet ? 4000 : 7000);
    return r;
  } catch (e) {
    toast(e.message, true);
    throw e;
  }
}
async function undo() {
  if (!P) return;
  try {
    P = await api(`/api/projects/${P.id}/undo`, { method: "POST" });
    lastMtime = P.mtime;
    renderAll();
    status("Отменено", "flash");
  } catch (e) { toast(e.message, true); }
}
$("#btnUndo").onclick = undo;
$("#voiceClean").onchange = (e) => ops([`set voice_clean ${e.target.checked ? "on" : "off"}`], true);
$("#fmt").onchange = () => ops([`set format ${$("#fmt").value}`], true);

$("#btnProcess").onclick = async () => {
  const b = $("#btnProcess");
  b.disabled = true;
  b.textContent = "Обработка…";
  status(P.settings.voice_clean ? "Монтаж и очистка голоса…" : "Монтаж…", "flash", 0);
  try {
    const r = await api(`/api/projects/${P.id}/render`, { method: "POST", body: JSON.stringify({ format: $("#fmt").value }) });
    P = await api(`/api/projects/${P.id}`);
    lastMtime = P.mtime;
    setOutSrc();
    await loadOutPeaks();
    setActive("out");
    renderAll();
    $("#aOut").currentTime = 0;
    $("#aOut").play();
    status(`Готово: ${fmtT(r.original)} → ${fmtT(r.duration)}`, "flash");
  } catch (e) { toast(e.message, true); status(""); }
  b.disabled = !isAudio();
  b.textContent = "Обработать";
};

$("#btnDl").onclick = () => {
  const m = $("#dlmenu");
  if (!m.hidden) { m.hidden = true; return; }
  const base = `/api/projects/${P.id}`;
  const audio = P.renders.filter((n) => !/\.(txt|srt|vtt)$/.test(n));
  const items = [];
  if (P.kind === "audio") {
    items.push(`<div class="mh">Аудио</div>`);
    items.push(...(audio.length ? audio.map((r) => `<a href="${base}/out/${encodeURIComponent(r)}?download=1">${esc(r)}</a>`)
      : [`<div class="none">Появится после «Обработать»</div>`]));
    items.push(`<div class="sep"></div>`);
  }
  items.push(`<div class="mh">Текст</div>`, `<a href="${base}/export/txt">Текст после правок (.txt)</a>`,
    `<a href="${base}/export/txt?orig=1">Исходный текст (.txt)</a>`);
  if (P.kind === "audio") items.push(`<div class="sep"></div>`, `<div class="mh">Субтитры</div>`,
    `<a href="${base}/export/srt">К результату (.srt)</a>`, `<a href="${base}/export/vtt">К результату (.vtt)</a>`,
    `<a href="${base}/export/srt?orig=1">К исходнику (.srt)</a>`);
  m.innerHTML = items.join("");
  m.hidden = false;
};

// ---------- воспроизведение ----------
function setActive(src) {
  active = src;
  $$("#srcSel .seg").forEach((b) => b.classList.toggle("on", b.dataset.src === src));
  $("#labOrig").classList.toggle("active", src === "orig");
  $("#labOut").classList.toggle("active", src === "out");
  (src === "orig" ? $("#aOut") : $("#aOrig")).pause();
  drawTl();
}
$$("#srcSel .seg").forEach((b) => (b.onclick = () => setActive(b.dataset.src)));
function curAudio() { return active === "orig" ? $("#aOrig") : $("#aOut"); }
function togglePlay() {
  const a = curAudio();
  if (!a.getAttribute("src")) { if (active === "out") toast("Сначала нажмите «Обработать»"); return; }
  a.paused ? a.play() : a.pause();
}
$("#btnPlay").onclick = togglePlay;
for (const a of [$("#aOrig"), $("#aOut")]) {
  a.addEventListener("play", () => { $("#btnPlay").textContent = "❚❚"; });
  a.addEventListener("pause", () => { if ($("#aOrig").paused && $("#aOut").paused) $("#btnPlay").textContent = "▶"; });
  a.addEventListener("seeked", () => drawTl());
}

function findIdx(arr, t, key) {
  let lo = 0, hi = arr.length - 1, ans = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (key(arr[m]) <= t) { ans = m; lo = m + 1; } else hi = m - 1;
  }
  return ans;
}
function tick() {
  if (isAudio()) {
    const a = curAudio();
    const dur = active === "orig" ? P.duration : (outPeaks ? outPeaks.duration : a.duration || 0);
    $("#tlTime").textContent = `${fmtT(a.currentTime)} / ${fmtT(dur)}`;
    let i = null;
    if (!a.paused) {
      if (active === "orig") {
        const k = findIdx(timed, a.currentTime, (t) => t.s);
        if (k >= 0 && a.currentTime <= timed[k].e + 0.05) i = timed[k].i;
      } else if (!stale()) {
        const k = findIdx(mapped, a.currentTime, (m) => m[0]);
        if (k >= 0 && a.currentTime <= mapped[k][1] + 0.05) i = mapped[k][2];
      }
      follow(a.currentTime);
      drawTl();
    }
    const el = i !== null ? tokEls[i] : null;
    if (el !== playingEl) {
      playingEl?.classList.remove("playing");
      el?.classList.add("playing");
      if (el) {
        const box = $("#transBody").getBoundingClientRect(), r = el.getBoundingClientRect();
        if (r.top < box.top + 20 || r.bottom > box.bottom - 20) el.scrollIntoView({ block: "center", behavior: "smooth" });
      }
      playingEl = el;
    }
  }
  requestAnimationFrame(tick);
}

// ---------- таймлайн ----------
const tl = $("#tl");
function tlSize() { return { w: tl.clientWidth, h: tl.clientHeight }; }
function tracks(h) {
  const th = (h - RULER) / 2;
  return { src: { y: RULER, h: th }, out: { y: RULER + th, h: th } };
}
function totalDur() { return Math.max(P?.duration || 0, outPeaks ? outPeaks.duration : 0, 1); }
const X = (t) => (t - view.t0) * view.pps;
const T = (x) => view.t0 + x / view.pps;
function clampView() {
  const { w } = tlSize();
  if (!w) return;
  view.pps = Math.min(Math.max(view.pps, ((w - 2) / totalDur()) * 0.5), 3000);
  view.t0 = Math.min(Math.max(0, view.t0), Math.max(0, totalDur() - (w - 2) / view.pps));
}
function fitView() {
  const { w } = tlSize();
  if (!w) return;
  view.pps = (w - 2) / totalDur();
  view.t0 = 0;
  buildTl();
  drawTl();
}
function zoomAt(x, factor) {
  const t = T(x);
  view.pps *= factor;
  clampView();
  view.t0 = t - x / view.pps;
  clampView();
  buildTl();
  drawTl();
}
function revealTime(s, e) {
  const { w } = tlSize();
  if (X(s) < 0 || X(e) > w) {
    view.t0 = s - (w / view.pps) * 0.2;
    clampView();
    buildTl();
    drawTl();
  }
}
function follow(t) {
  const { w } = tlSize();
  const x = X(t);
  if (x > w * 0.92 || x < 0) {
    view.t0 = t - (w / view.pps) * 0.08;
    clampView();
    buildTl();
  }
}
function amp(v) {
  if (!scaleDb) return v;
  return Math.max(0, (20 * Math.log10(Math.max(v, 1e-6)) + 60) / 60);
}
function keptMask(w) {
  const keep = new Uint8Array(w);  // какие пиксели исходника войдут в результат
  for (const [s, e] of P.segments || []) {
    if (s === null) continue;
    const x0 = Math.max(0, Math.floor(X(s))), x1 = Math.min(w, Math.ceil(X(e)));
    for (let x = x0; x < x1; x++) keep[x] = 1;
  }
  return keep;
}
function drawWave(g, data, dur, y, h, w, colorAt) {
  if (!data.length || !dur) return;
  const n = data.length, mid = y + h / 2, half = h / 2 - 2;
  for (let x = 0; x < w; x++) {
    const t0 = T(x), t1 = T(x + 1);
    if (t0 > dur) break;
    if (t1 < 0) continue;
    const i0 = Math.max(0, Math.floor((t0 / dur) * n));
    const i1 = Math.min(n, Math.max(i0 + 1, Math.ceil((t1 / dur) * n)));
    let v = 0;
    for (let i = i0; i < i1; i++) if (data[i] > v) v = data[i];
    const a = Math.max(1, amp(v) * half);
    g.fillStyle = colorAt(x);
    g.fillRect(x, mid - a, 1, a * 2);
  }
}
function niceStep(pps) {
  const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  return steps.find((s) => s * pps >= 70) || 600;
}
function buildTl() {
  const { w, h } = tlSize();
  if (!w || !h || !isAudio()) { tlStatic = null; return; }
  const dpr = devicePixelRatio || 1;
  const off = document.createElement("canvas");
  off.width = Math.round(w * dpr); off.height = Math.round(h * dpr);
  const g = off.getContext("2d");
  g.scale(dpr, dpr);
  const C = {
    bg0: cssVar("--bg0"), bg1: cssVar("--bg1"), line: cssVar("--line"), text: cssVar("--text"),
    faint: cssVar("--text-faint"), gold: cssVar("--gold"), orange: cssVar("--orange"),
  };
  const R = tracks(h);
  g.fillStyle = C.bg0; g.fillRect(0, 0, w, h);
  // линейка
  g.fillStyle = C.bg1; g.fillRect(0, 0, w, RULER);
  g.fillStyle = C.line; g.fillRect(0, RULER - 1, w, 1);
  const step = niceStep(view.pps);
  g.font = "10px Consolas, monospace";
  g.textBaseline = "alphabetic";
  for (let t = Math.floor(view.t0 / step) * step; X(t) < w; t += step) {
    const x = Math.round(X(t)) + 0.5;
    if (x < 0) continue;
    g.fillStyle = C.line; g.fillRect(x, RULER - 6, 1, 6);
    g.fillStyle = C.faint;
    g.fillText(step < 1 ? t.toFixed(1) : fmtT(t, 0), x + 3, 12);
  }
  // исходник: то, что уйдёт, затемнено
  const keep = keptMask(w);
  g.fillStyle = "rgba(0,0,0,0.38)";
  for (let x = 0; x < w; x++) if (!keep[x] && T(x) <= P.duration) g.fillRect(x, R.src.y, 1, R.src.h);
  const band = 17;
  drawWave(g, peaks, P.duration, R.src.y + band + 2, R.src.h - band - 4, w,
    (x) => (keep[x] ? "rgba(89,227,226,0.85)" : "rgba(107,100,79,0.75)"));
  // слова поверх волны
  g.font = "11px Segoe UI, sans-serif";
  g.textBaseline = "middle";
  for (const t of P.tokens) {
    if (t.s === undefined) continue;
    const et = effT(t);
    const x0 = X(et.s), x1 = X(et.e);
    if (x1 < 0 || x0 > w) continue;
    const cut = P.cuts[t.i] !== undefined;
    const snd = t.kind === "sound";
    const selected = sel && t.i >= sel[0] && t.i <= sel[1];
    const bw = Math.max(2, x1 - x0);
    g.fillStyle = cut ? "rgba(255,68,68,0.16)" : snd ? "rgba(230,131,82,0.16)" : "rgba(89,227,226,0.10)";
    g.fillRect(x0, R.src.y + 2, bw, band);
    g.strokeStyle = selected ? C.gold : cut ? "rgba(255,68,68,0.55)" : snd ? "rgba(230,131,82,0.5)" : "rgba(89,227,226,0.35)";
    g.lineWidth = selected ? 1.5 : 1;
    g.strokeRect(x0 + 0.5, R.src.y + 2.5, bw - 1, band - 1);
    if (bw > 14) {
      g.save();
      g.beginPath(); g.rect(x0 + 2, R.src.y + 2, bw - 4, band); g.clip();
      g.fillStyle = cut ? C.faint : snd ? C.orange : C.text;
      g.fillText(snd ? (t.hes ? "заминка" : "звук") : textOf(t), x0 + 3, R.src.y + 2 + band / 2);
      g.restore();
    }
  }
  // вырезанные участки времени и края вырезок, за которые можно тянуть
  for (const [rs, re] of P.edit.ranges) {
    const x0 = X(rs), x1 = X(re);
    if (x1 < 0 || x0 > w) continue;
    g.fillStyle = "rgba(255,68,68,0.16)";
    g.fillRect(x0, R.src.y + 2, x1 - x0, band);
    if (x1 - x0 > 40) { g.fillStyle = C.faint; g.fillText("участок", x0 + 3, R.src.y + 2 + band / 2); }
  }
  for (const h of cutHandles()) {
    const x = Math.round(X(h.t));
    if (x < -2 || x > w + 2) continue;
    g.fillStyle = h.set ? "rgba(253,207,130,0.75)" : "rgba(255,68,68,0.6)";
    g.fillRect(x - (h.side === "s" ? 0 : 1), R.src.y + 2, 2, R.src.h - 4);
    g.fillRect(h.side === "s" ? x : x - 5, R.src.y + 2, 6, 3);
    g.fillRect(h.side === "s" ? x : x - 5, R.src.y + R.src.h - 5, 6, 3);
  }
  // результат
  g.fillStyle = C.line; g.fillRect(0, R.out.y, w, 1);
  g.font = "12px Segoe UI, sans-serif";
  if (outPeaks) {
    const st = stale();
    g.globalAlpha = st ? 0.4 : 1;
    drawWave(g, outPeaks.peaks, outPeaks.duration, R.out.y + 3, R.out.h - 6, w, () => "rgba(137,229,220,0.9)");
    if (!st && P.segments && Math.abs(outPeaks.duration - P.edited_duration) < 0.1) {
      g.fillStyle = "rgba(230,131,82,0.55)";  // места склеек
      let pos = 0;
      P.segments.forEach(([s, e], k) => {
        pos += s === null ? e : e - s;
        const x = Math.round(X(pos));
        if (k < P.segments.length - 1 && x >= 0 && x <= w) g.fillRect(x, R.out.y + 2, 1, R.out.h - 4);
      });
    }
    g.globalAlpha = 1;
    if (st) { g.fillStyle = C.orange; g.fillText("результат устарел — нажмите «Обработать»", 10, R.out.y + 14); }
  } else {
    g.fillStyle = C.faint;
    g.fillText("Результата пока нет — нажмите «Обработать»", 10, R.out.y + R.out.h / 2);
  }
  tlStatic = off;
}
function cutHandles() {
  // края вырезок: подряд идущие вырезанные слова — одна область; плюс вырезанные участки времени
  const hs = [];
  const tim = P.tokens.filter((t) => t.s !== undefined);
  const bnd = P.edit.bounds || {};
  for (let k = 0; k < tim.length; k++) {
    if (P.cuts[tim[k].i] === undefined) continue;
    let j = k;
    while (j + 1 < tim.length && P.cuts[tim[j + 1].i] !== undefined) j++;
    const a = tim[k], b = tim[j], ea = effT(a), eb = effT(b);
    hs.push({ kind: "tok", side: "s", i: a.i, t: ea.s, lo: 0, hi: eb.e - 0.02, set: bnd[a.i]?.s != null });
    hs.push({ kind: "tok", side: "e", i: b.i, t: eb.e, lo: ea.s + 0.02, hi: P.duration, set: bnd[b.i]?.e != null });
    k = j;
  }
  P.edit.ranges.forEach(([rs, re, why]) => {
    hs.push({ kind: "range", side: "s", t: rs, lo: 0, hi: re - 0.02, span: [rs, re], why, set: true });
    hs.push({ kind: "range", side: "e", t: re, lo: rs + 0.02, hi: P.duration, span: [rs, re], why, set: true });
  });
  return hs;
}
function hitHandle(x, y) {
  const R = tracks(tlSize().h);
  if (!P || y < R.src.y || y > R.src.y + R.src.h) return null;
  let best = null, bd = 6;
  for (const h of cutHandles()) {
    const d = Math.abs(X(h.t) - x);
    if (d < bd) { bd = d; best = h; }
  }
  return best;
}
let hover = null;
function drawTl() {
  const { w, h } = tlSize();
  const dpr = devicePixelRatio || 1;
  if (tl.width !== Math.round(w * dpr) || tl.height !== Math.round(h * dpr)) {
    tl.width = Math.round(w * dpr); tl.height = Math.round(h * dpr);
    buildTl();
  }
  const g = tl.getContext("2d");
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, tl.width, tl.height);
  if (!tlStatic) return;
  g.drawImage(tlStatic, 0, 0);
  g.scale(dpr, dpr);
  const R = tracks(h);
  if (range) {
    const x0 = X(range[0]), x1 = X(range[1]);
    g.fillStyle = "rgba(253,207,130,0.14)";
    g.fillRect(x0, R.src.y, x1 - x0, R.src.h);
    g.fillStyle = "rgba(253,207,130,0.8)";
    g.fillRect(x0, R.src.y, 1, R.src.h);
    g.fillRect(x1, R.src.y, 1, R.src.h);
  }
  const gold = cssVar("--gold");
  const hh = drag && drag.handle ? drag.handle : hover;
  if (hh) {
    const cur = drag && drag.handle && drag.cur != null ? drag.cur : hh.t;
    const xo = X(hh.t), xc = X(cur);
    if (xc !== xo) {
      // к вырезке добавится (красный) или вернётся в запись (бирюзовый)
      const grow = hh.side === "s" ? cur < hh.t : cur > hh.t;
      g.fillStyle = grow ? "rgba(255,68,68,0.28)" : "rgba(89,227,226,0.22)";
      g.fillRect(Math.min(xo, xc), R.src.y, Math.abs(xc - xo), R.src.h);
    }
    g.fillStyle = gold;
    g.fillRect(Math.round(xc) - 1, R.src.y, 2, R.src.h);
    if (drag && drag.handle) {
      g.font = "11px Consolas, monospace";
      g.textBaseline = "top";
      const label = `${cur.toFixed(2)} с`;
      const tx = hh.side === "s" ? xc - g.measureText(label).width - 6 : xc + 6;
      g.fillText(label, tx, R.src.y + 22);
    }
  }
  const head = (t, tr, on) => {
    const x = X(t);
    if (x < 0 || x > w) return;
    g.globalAlpha = on ? 1 : 0.35;
    g.fillStyle = gold;
    g.fillRect(Math.round(x) - 1, tr.y, 2, tr.h);
    g.beginPath(); g.moveTo(x - 5, tr.y); g.lineTo(x + 5, tr.y); g.lineTo(x, tr.y + 6); g.fill();
    g.globalAlpha = 1;
  };
  head($("#aOrig").currentTime, R.src, active === "orig");
  if (outPeaks) head($("#aOut").currentTime, R.out, active === "out");
}

// мышь: клик — перейти, перетаскивание — выделить участок, колесо — прокрутка, Ctrl+колесо — зум
let drag = null;
tl.addEventListener("mousedown", (e) => {
  if (!isAudio() || e.button !== 0) return;
  const r = tl.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const handle = hitHandle(x, y);
  drag = { x0: x, onOut: y >= tracks(r.height).out.y, moved: false, ctrl: e.ctrlKey || e.metaKey, shift: e.shiftKey, handle };
});
tl.addEventListener("mousemove", (e) => {
  if (drag) return;
  const r = tl.getBoundingClientRect();
  const h = isAudio() ? hitHandle(e.clientX - r.left, e.clientY - r.top) : null;
  tl.style.cursor = h ? "col-resize" : "crosshair";
  if (h !== hover && !(h && hover && h.kind === hover.kind && h.side === hover.side && h.t === hover.t)) { hover = h; drawTl(); }
});
tl.addEventListener("mouseleave", () => { if (hover) { hover = null; drawTl(); } });
tl.addEventListener("dblclick", (e) => {
  const r = tl.getBoundingClientRect();
  const h = hitHandle(e.clientX - r.left, e.clientY - r.top);
  if (h && h.kind === "tok" && h.set) ops([`edge ${h.i} auto`]);
});
window.addEventListener("mousemove", (e) => {
  if (!drag || drag.onOut) return;
  const r = tl.getBoundingClientRect();
  const x = Math.min(Math.max(0, e.clientX - r.left), r.width);
  if (drag.handle) {
    if (Math.abs(x - drag.x0) > 2) drag.moved = true;
    if (drag.moved) {
      drag.cur = Math.min(Math.max(T(x), drag.handle.lo), drag.handle.hi);
      drawTl();
    }
    return;
  }
  if (Math.abs(x - drag.x0) > 4) {
    drag.moved = true;
    range = [Math.max(0, T(Math.min(drag.x0, x))), Math.min(P.duration, T(Math.max(drag.x0, x)))];
    if (sel) { sel = null; $$(".tok.sel").forEach((el) => el.classList.remove("sel")); buildTl(); }
    updateSelInfo();
    drawTl();
  }
});
window.addEventListener("mouseup", async () => {
  if (!drag) return;
  const d = drag;
  drag = null;
  if (d.handle && d.moved && d.cur != null) {
    const h = d.handle, v = d.cur.toFixed(3);
    if (h.kind === "tok") await ops([`edge ${h.i} ${h.side === "s" ? "start" : "end"} ${v}`], true);
    else {
      const [a, b] = h.side === "s" ? [d.cur, h.span[1]] : [h.span[0], d.cur];
      await ops([`uncuttime ${h.span[0].toFixed(3)}-${h.span[1].toFixed(3)}`,
                 `cuttime ${a.toFixed(3)}-${b.toFixed(3)} ${h.why || "manual"}`], true);
    }
    return;
  }
  if (d.moved) { updateSelInfo(); return; }
  const t = T(d.x0);
  if (d.onOut) {
    if (!outPeaks) return;
    setActive("out");
    $("#aOut").currentTime = Math.min(Math.max(0, t), outPeaks.duration);
    drawTl();
    return;
  }
  const tok = P.tokens.find((k) => k.s !== undefined && effT(k).s <= t && t <= effT(k).e);
  if (tok && d.ctrl) { await ops([P.cuts[tok.i] !== undefined ? `keep ${tok.i}` : `cut ${tok.i}`], true); return; }
  range = null;
  if (tok && d.shift && anchor !== null) sel = [Math.min(anchor, tok.i), Math.max(anchor, tok.i)];
  else if (tok) { sel = [tok.i, tok.i]; anchor = tok.i; }
  else sel = null;
  setActive("orig");
  $("#aOrig").currentTime = Math.min(Math.max(0, t), P.duration);
  if (tok) tokEls[tok.i]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  paintSel();
});
tl.addEventListener("wheel", (e) => {
  if (!isAudio()) return;
  e.preventDefault();
  const r = tl.getBoundingClientRect();
  if (e.ctrlKey || e.metaKey) zoomAt(e.clientX - r.left, Math.exp(-e.deltaY * 0.0015));
  else {
    view.t0 += (Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY) / view.pps;
    clampView();
    buildTl();
    drawTl();
  }
}, { passive: false });
$("#zoomIn").onclick = () => zoomAt(tlSize().w / 2, 1.6);
$("#zoomOut").onclick = () => zoomAt(tlSize().w / 2, 1 / 1.6);
$("#zoomFit").onclick = fitView;
function paintScale() { $$("#scaleSel .seg").forEach((b) => b.classList.toggle("on", (b.dataset.scale === "db") === scaleDb)); }
$$("#scaleSel .seg").forEach((b) => (b.onclick = () => {
  scaleDb = b.dataset.scale === "db";
  store("voicer.scale", scaleDb ? "db" : "lin");
  paintScale();
  buildTl();
  drawTl();
}));
paintScale();

// разделитель «транскрипт / таймлайн»
(() => {
  const saved = +store("voicer.tlh");
  if (saved) document.documentElement.style.setProperty("--tlh", saved + "px");
  $("#split").addEventListener("mousedown", (e) => {
    const startY = e.clientY, startH = $("#timeline").getBoundingClientRect().height;
    $("#split").classList.add("active");
    const move = (ev) => {
      const hgt = Math.min(Math.max(150, startH - (ev.clientY - startY)), $("#center").clientHeight - 140);
      document.documentElement.style.setProperty("--tlh", hgt + "px");
      drawTl();
    };
    const up = () => {
      $("#split").classList.remove("active");
      store("voicer.tlh", Math.round($("#timeline").getBoundingClientRect().height));
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  });
})();
window.addEventListener("resize", () => { clampView(); drawTl(); });

// ---------- клавиатура ----------
document.addEventListener("keydown", (e) => {
  if (!P || e.target.closest?.("input, textarea, select, dialog")) return;
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if ((e.key === "Delete" || e.key === "Backspace") && (sel || range)) { e.preventDefault(); cutSelection(false); }
  else if (e.key === "Escape") {
    if (drag && drag.handle) { drag = null; drawTl(); return; }  // отмена перетаскивания края
    sel = range = null; anchor = null; paintSel();
  }
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
});

// ---------- синхронизация: правки Claude видны сразу ----------
async function poll() {
  if (P && !document.hidden) {
    try {
      const r = await api(`/api/projects/${P.id}/rev`);
      if (r.mtime !== lastMtime) {
        const wasProcessing = P.status !== "ready";
        const prevRender = P.last_render ? P.last_render.rev : null;
        P = await api(`/api/projects/${P.id}`);
        lastMtime = P.mtime;
        if (wasProcessing && P.status === "ready") await openProject(P.id);
        else {
          if (P.last_render && P.last_render.rev !== prevRender) { setOutSrc(); await loadOutPeaks(); }
          renderAll();
        }
        if (r.by !== "ui") {
          const b = $("#banner");
          b.textContent = `Изменено (${byName(r.by)}): ${r.msg || ""}`;
          b.hidden = false;
          clearTimeout(b._t);
          b._t = setTimeout(() => (b.hidden = true), 8000);
        }
      }
    } catch {}
  }
  setTimeout(poll, 1500);
}
window.addEventListener("hashchange", () => {
  const id = decodeURIComponent(location.hash.slice(1));
  if (id && (!P || P.id !== id)) openProject(id);
});

// ---------- старт ----------
(async () => {
  renderAll();
  const list = await loadList();
  pollJobs();
  const id = decodeURIComponent(location.hash.slice(1));
  if (id) await openProject(id);
  else if (list.length) await openProject(list[0].id);
  requestAnimationFrame(tick);
  poll();
  setInterval(loadList, 10000);
})();
