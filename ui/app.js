// Tetris x Laya front end: draws frames from the Python server and fills the metrics panel.
const TURNS = ["spawn", "right", "flip", "left"];
let COLORS = {};   // filled from CSS so the pieces follow the theme
function readColors() {
  COLORS = Object.fromEntries("IOTSZJLG".split("").map((p) => [p, css("--p-" + p)]));
}
const $ = (id) => document.getElementById(id);

const canvas = $("board");
const ctx = canvas.getContext("2d");
const nextCanvas = $("next");
const nctx = nextCanvas.getContext("2d");

let frame = null;      // last server frame
let running = false;
let busy = false;      // a request is in flight: never overlap steps
let timer = null;
let pieceTimes = [];   // client timestamps of recent pieces, for pieces/sec
let history = [];      // {height, holes, cleared} per piece, for the sparkline

// ---------------------------------------------------------------- server
async function post(path, body = {}) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

async function loadPolicies() {
  const list = await (await fetch("/api/policies")).json();
  const sel = $("policy");
  sel.innerHTML = "";
  for (const p of list) {
    const o = new Option(p.available ? p.name : `${p.name} (not trained)`, p.name);
    o.disabled = !p.available;
    sel.add(o);
  }
  const first = list.find((p) => p.available);
  if (first) sel.value = first.name;
}

// ---------------------------------------------------------------- control loop
function tickMs() { return 1000 / Number($("speed").value); }

async function newGame() {
  setStatus(`loading ${$("policy").value}…`);
  frame = await post("/api/reset", { policy: $("policy").value });
  pieceTimes = [];
  history = [];
  $("overlay").hidden = true;
  render();
  setStatus(running ? "running" : "ready", running ? "running" : "");
}

async function stepOnce() {
  if (busy) return;
  busy = true;
  try {
    if (!frame || frame.game.done) await newGame();
    const before = frame.game.board;
    frame = await post("/api/step", { mask: $("mask").checked });
    pieceTimes.push(performance.now());
    if (pieceTimes.length > 20) pieceTimes.shift();
    const holes = frame.game.holes.reduce((a, b) => a + b, 0);
    history.push({ height: Math.max(...frame.game.heights), holes,
                   cleared: (frame.game.last.cleared || []).length,
                   dug: history.length ? holes > history[history.length - 1].holes : false });
    if (history.length > 120) history.shift();
    renderPanel();
    await animateDrop(before);   // the piece falls; the board only updates once it lands
    render();
    if (frame.game.done) await gameOver();
  } finally {
    busy = false;
  }
}

async function gameOver() {
  const g = frame.game;
  showOverlay(`${g.lines} lines`, `${g.end_cause} after ${g.pieces} pieces`);
  if (running && $("auto").checked) {
    await new Promise((r) => setTimeout(r, 1100));
    if (running) await newGame();
  } else {
    setRunning(false);
  }
}

async function loop() {
  if (!running) return;
  const t0 = performance.now();
  try {
    await stepOnce();
  } catch (e) {
    return fail(e);
  }
  timer = setTimeout(loop, Math.max(0, tickMs() - (performance.now() - t0)));
}

function setRunning(on) {
  running = on;
  $("play").textContent = on ? "Pause" : "Play";
  clearTimeout(timer);
  setStatus(on ? "running" : "paused", on ? "running" : "");
  if (on) loop();
}

function setStatus(text, cls = "") {
  const s = $("status");
  s.textContent = text;
  s.className = "status " + cls;
}

function fail(e) {
  setRunning(false);
  setStatus(e.message, "error");
}

// ---------------------------------------------------------------- board drawing
function fitBoard() {
  // the board is 10 x 20 cells: pick a cell size that fits the window height, then size the canvas
  const avail = Math.min(window.innerHeight - 250, 620);
  const cell = Math.max(12, Math.min(30, Math.floor(avail / 20)));
  document.documentElement.style.setProperty("--cell", cell + "px");
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = cell * 10 + "px";
  canvas.style.height = cell * 20 + "px";
  canvas.width = cell * 10 * dpr;
  canvas.height = cell * 20 * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawBoard();
}

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function block(c, x, y, cell, color, alpha = 1) {
  const p = Math.max(1, cell * 0.06);
  const x0 = x * cell + p, y0 = y * cell + p, w = cell - 2 * p, r = cell * 0.22;
  c.save();
  c.globalAlpha = alpha;
  const g = c.createLinearGradient(x0, y0, x0, y0 + w);
  g.addColorStop(0, shade(color, 0.16));
  g.addColorStop(0.55, color);
  g.addColorStop(1, shade(color, -0.16));
  c.fillStyle = g;
  c.beginPath();
  c.roundRect(x0, y0, w, w, r);
  c.fill();
  c.strokeStyle = shade(color, -0.3);
  c.lineWidth = Math.max(0.6, cell * 0.035);
  c.stroke();
  if (cell >= 16) {   // a highlight on the top edge: too busy to be worth it on a tiny board
    c.globalAlpha = alpha * 0.5;
    c.strokeStyle = shade(color, 0.42);
    c.lineWidth = Math.max(0.8, cell * 0.06);
    c.beginPath();
    c.moveTo(x0 + r, y0 + c.lineWidth);
    c.lineTo(x0 + w - r, y0 + c.lineWidth);
    c.stroke();
  }
  c.restore();
}

/* Lighten (k > 0) or darken (k < 0) a #rrggbb colour. */
function shade(hex, k) {
  const m = /^#?([0-9a-f]{6})$/i.exec((hex || "").trim());
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  const ch = [n >> 16 & 255, n >> 8 & 255, n & 255].map((v) =>
    Math.max(0, Math.min(255, Math.round(k > 0 ? v + (255 - v) * k : v * (1 + k)))));
  return "rgb(" + ch.join(",") + ")";
}

function drawWell(cell, w, h) {
  ctx.fillStyle = css("--well");
  ctx.fillRect(0, 0, cell * w, cell * h);
  ctx.strokeStyle = css("--grid");
  ctx.lineWidth = 1;
  for (let x = 1; x < w; x++) { ctx.beginPath(); ctx.moveTo(x * cell, 0); ctx.lineTo(x * cell, cell * h); ctx.stroke(); }
  for (let y = 1; y < h; y++) { ctx.beginPath(); ctx.moveTo(0, y * cell); ctx.lineTo(cell * w, y * cell); ctx.stroke(); }
}

function drawBoard() {
  const cols = 10, rows = 20;
  const cell = parseFloat(canvas.style.width) / cols;
  if (!cell) return;
  if (!frame) return drawWell(cell, cols, rows);   // before the first frame arrives
  const g = frame.game;
  drawWell(cell, g.w, g.h);

  const d = frame.decision;
  // the teacher's placement, as an outline, when it differs from what Laya played
  if (d && d.teacher && !(d.teacher.turn === d.turn && d.teacher.col === d.col)) {
    ctx.strokeStyle = css("--teacher");
    ctx.lineWidth = 2;
    for (const [x, y] of d.teacher.cells) {
      if (y < 0) continue;
      ctx.strokeRect(x * cell + 2, y * cell + 2, cell - 4, cell - 4);
    }
  }
  drawStack(g.board, cell);
  // highlight the piece that was just placed
  if (g.last && !g.done) {
    for (const [x, y] of g.last.cells) {
      if (y < 0) continue;
      ctx.strokeStyle = "rgba(255,255,255,.55)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x * cell + 2, y * cell + 2, cell - 4, cell - 4);
    }
  }
}

// A piece is placed with one hard drop, so without this it would simply appear in the stack.
// The fall is cosmetic: the server already decided where the piece lands.
function animateDrop(beforeBoard) {
  const g = frame.game, d = frame.decision;
  const cells = g.last && g.last.cells;
  if (!cells || !d || !$("anim").checked || cells.some(([, y]) => y < 0)) return Promise.resolve();
  const cell = parseFloat(canvas.style.width) / 10;
  if (!cell) return Promise.resolve();
  const top = Math.min(...cells.map(([, y]) => y));
  const height = Math.max(...cells.map(([, y]) => y)) - top + 1;
  const from = -height;                     // just above the ceiling
  const rows = top - from;
  if (rows <= 0) return Promise.resolve();
  // ~55 ms per row of fall, but never more than most of one tick: the animation must not set the pace
  const ms = Math.min(rows * 55, Math.max(70, tickMs() * 0.8));
  const color = COLORS[d.piece] || css("--accent");
  const t0 = performance.now();
  return new Promise((done) => {
    const step = (now) => {
      const k = Math.min(1, (now - t0) / ms);
      const dy = Math.round(from + rows * (0.25 * k + 0.75 * k * k)) - top;   // starts moving, then accelerates
      drawWell(cell, 10, 20);
      drawStack(beforeBoard, cell);
      ctx.save();
      ctx.globalAlpha = 0.28;
      for (const [x, y] of cells) block(ctx, x, y, cell, color);   // ghost: where it will land
      ctx.restore();
      for (const [x, y] of cells) {
        const yy = y + dy;
        if (yy >= 0) block(ctx, x, yy, cell, color);
      }
      if (k < 1) requestAnimationFrame(step); else flashClears(cell, beforeBoard, cells, color).then(done);
    };
    requestAnimationFrame(step);
  });
}

function flashClears(cell, beforeBoard, cells, color) {
  const cleared = frame.game.last.cleared;
  if (!cleared || !cleared.length) return Promise.resolve();
  const t0 = performance.now(), ms = 160;
  return new Promise((done) => {
    const step = (now) => {
      const k = Math.min(1, (now - t0) / ms);
      drawWell(cell, 10, 20);
      drawStack(beforeBoard, cell);
      for (const [x, y] of cells) block(ctx, x, y, cell, color);
      ctx.fillStyle = "rgba(255,255,255," + (0.85 * (1 - Math.abs(2 * k - 1))).toFixed(3) + ")";
      for (const y of cleared) ctx.fillRect(0, y * cell, cell * 10, cell);
      if (k < 1) requestAnimationFrame(step); else done();
    };
    requestAnimationFrame(step);
  });
}

function drawStack(board, cell) {
  for (let y = 0; y < board.length; y++) {
    for (let x = 0; x < board[y].length; x++) {
      if (board[y][x] !== ".") block(ctx, x, y, cell, COLORS[board[y][x]] || css("--accent"));
    }
  }
}

// Stack height over the recent past. The interesting thing this model does is dig itself out of
// trouble, and that only shows up as a shape over time.
function drawSpark() {
  const c = $("spark"), n = history.length;
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth || 260, H = 75;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);

  const pad = 3, top = pad, bot = H - pad;
  // Auto-scale: a teacher that holds the stack at 4 rows would be a flat line against a 20-row axis.
  const peak = Math.max(6, ...history.map((p) => p.height));
  const scale = Math.min(20, peak + 2);
  const y = (h) => bot - (bot - top) * Math.min(1, h / scale);
  const gridEvery = scale <= 8 ? 2 : scale <= 14 ? 4 : 5;
  g.strokeStyle = css("--line-soft");
  g.lineWidth = 1;
  g.font = "9px ui-monospace, Menlo, monospace";
  g.fillStyle = css("--muted");
  for (let h = gridEvery; h < scale; h += gridEvery) {
    g.beginPath(); g.moveTo(0, y(h)); g.lineTo(W, y(h)); g.stroke();
    g.fillText(String(h), 2, y(h) - 2);
  }
  if (n < 2) return;
  const x = (i) => (W * i) / Math.max(1, history.length - 1);

  g.beginPath();                         // filled area under the height line
  g.moveTo(x(0), bot);
  history.forEach((p, i) => g.lineTo(x(i), y(p.height)));
  g.lineTo(x(n - 1), bot);
  g.closePath();
  g.fillStyle = "color-mix(in srgb, " + css("--accent") + " 16%, transparent)";
  g.fill();

  g.beginPath();
  history.forEach((p, i) => (i ? g.lineTo(x(i), y(p.height)) : g.moveTo(x(i), y(p.height))));
  g.strokeStyle = css("--accent");
  g.lineWidth = 1.6;
  g.stroke();

  history.forEach((p, i) => {            // a tick per line clear, amber when a hole appeared
    if (p.cleared) {
      g.fillStyle = css("--accent");
      g.fillRect(x(i) - 0.9, y(p.height) - 4, 1.8, 4);
    }
    if (p.dug) {
      g.fillStyle = css("--teacher");
      g.beginPath(); g.arc(x(i), y(p.height), 1.7, 0, Math.PI * 2); g.fill();
    }
  });
}

function drawNext() {
  const g = frame.game;
  const cells = g.shapes[g.next][0];
  const w = Math.max(...cells.map((c) => c[0])) + 1;
  const h = Math.max(...cells.map((c) => c[1])) + 1;
  const dpr = window.devicePixelRatio || 1;
  const W = 88, H = 58;
  nextCanvas.width = W * dpr; nextCanvas.height = H * dpr;
  nctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  nctx.clearRect(0, 0, W, H);
  const cell = Math.min(W / (w + 1), H / (h + 1), 18);
  const ox = (W - w * cell) / 2, oy = (H - h * cell) / 2;
  nctx.save();
  nctx.translate(ox, oy);
  for (const [x, y] of cells) block(nctx, x, y, cell, COLORS[g.next]);
  nctx.restore();
}

function showOverlay(title, sub) {
  const o = $("overlay");
  o.innerHTML = `<div>${title}<small>${sub}</small></div>`;
  o.hidden = false;
}

// ---------------------------------------------------------------- metrics
function buildBars() {
  $("turnBars").innerHTML = TURNS.map((t, i) => `
    <div class="bar" id="turn-${i}">
      <canvas width="44" height="32"></canvas>
      <span class="label">${t}</span>
      <div class="track"><div class="fill"></div></div>
      <span class="val">—</span>
    </div>`).join("");
  $("colstrip").innerHTML = Array.from({ length: 10 }, (_, i) => `
    <div class="col" id="col-${i}">
      <div class="track"><div class="fill"></div></div>
      <span class="lab">${i + 1}</span>
    </div>`).join("");
}

// tiny previews of the current piece in each of the 4 turns, next to the turn bars
function drawTurnIcons() {
  const g = frame.game;
  const shapes = g.shapes[g.piece];
  for (let i = 0; i < 4; i++) {
    const c = $("turn-" + i).querySelector("canvas");
    const cells = shapes[i % shapes.length];
    const dpr = window.devicePixelRatio || 1;
    c.width = 44 * dpr; c.height = 32 * dpr;
    const cc = c.getContext("2d");
    cc.setTransform(dpr, 0, 0, dpr, 0, 0);
    cc.clearRect(0, 0, 44, 32);
    const w = Math.max(...cells.map((p) => p[0])) + 1;
    const h = Math.max(...cells.map((p) => p[1])) + 1;
    const cell = Math.min(44 / w, 32 / h, 8);
    cc.save();
    cc.translate((44 - w * cell) / 2, (32 - h * cell) / 2);
    for (const [x, y] of cells) block(cc, x, y, cell, COLORS[g.piece]);
    cc.restore();
  }
}

const fmt = (v, d = 1) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
const pct = (v) => (v === null || v === undefined ? "—" : (v * 100).toFixed(0) + "%");

function render() {
  drawBoard();
  renderPanel();
}

function renderPanel() {
  drawNext();
  drawTurnIcons();
  drawSpark();
  const { game: g, stats: s, decision: d } = frame;

  if ($("sLines").textContent !== String(g.lines)) {
    const el = $("sLines");
    el.textContent = g.lines;
    el.classList.remove("bump");
    void el.offsetWidth;        // restart the animation
    el.classList.add("bump");
  }
  $("sScore").textContent = g.score;
  $("sLevel").textContent = g.level;
  $("mPieces").textContent = g.pieces;
  $("mTetris").textContent = g.clears[4];
  $("mHigh").textContent = s.high;
  $("mGames").textContent = s.games;
  $("mAvg").textContent = fmt(s.avg);
  $("mHeight").textContent = Math.max(...g.heights);

  $("mLat").textContent = fmt(s.lat_last);
  $("mLatAvg").textContent = fmt(s.lat_avg);
  $("mLatP95").textContent = fmt(s.lat_p95);
  const span = pieceTimes.length > 1 ? (pieceTimes.at(-1) - pieceTimes[0]) / 1000 : 0;
  $("mPps").textContent = span ? fmt((pieceTimes.length - 1) / span) : "—";
  $("mPasses").textContent = d && d.passes_ms.length ? d.passes_ms.map((x) => fmt(x) + " ms").join("  ·  ") : "—";
  $("mModel").textContent = `${s.policy} · ${s.device}`;
  $("mAgree").textContent = pct(s.agree);
  $("mOver").textContent = s.overrides;

  const nTurns = g.shapes[g.piece].length;
  for (let i = 0; i < 4; i++) {
    const row = $("turn-" + i);
    const p = d ? d.turn_probs[i] : 0;
    row.querySelector(".fill").style.width = (p * 100).toFixed(1) + "%";
    row.querySelector(".val").textContent = d ? (p * 100).toFixed(0) + "%" : "—";
    row.classList.toggle("chosen", !!d && d.turn === i);
    row.classList.toggle("teacher", !!d && d.teacher && d.teacher.turn === i);
    row.classList.toggle("na", i >= nTurns);  // a flipped I is the same I: those turns are unused
  }
  for (let i = 0; i < 10; i++) {
    const col = $("col-" + i);
    const p = d ? d.col_probs[i] : 0;
    col.querySelector(".fill").style.height = (p * 100).toFixed(1) + "%";
    col.classList.toggle("chosen", !!d && d.col === i);
    col.classList.toggle("teacher", !!d && d.teacher && d.teacher.col === i);
  }
  $("confBar").style.width = d ? (d.confidence * 100).toFixed(0) + "%" : "0";
  $("confVal").textContent = d ? fmt(d.confidence, 2) : "—";
  $("maskBadge").hidden = !(d && d.masked);
  $("stateTurn").textContent = d ? d.states.turn : "—";
  $("stateCol").textContent = d ? d.states.column : "—";
}

// ---------------------------------------------------------------- wiring
$("play").onclick = () => setRunning(!running);
$("step").onclick = () => { setRunning(false); stepOnce().catch(fail); };
$("reset").onclick = () => newGame().catch(fail);
$("policy").onchange = () => newGame().catch(fail);
$("speed").oninput = () => { $("speedOut").textContent = $("speed").value + "/s"; };
document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT"].includes(e.target.tagName) && e.target.type !== "checkbox" && e.target.type !== "range") return;
  if (e.code === "Space") { e.preventDefault(); setRunning(!running); }
  else if (e.key === "s") $("step").click();
  else if (e.key === "n") $("reset").click();
});
$("theme").onclick = () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) { /* private window: theme just won't persist */ }
  readColors();
  if (frame) render();
};
window.addEventListener("resize", fitBoard);

(async () => {
  readColors();
  buildBars();
  fitBoard();                 // size and paint the empty well before the model loads
  try {
    await loadPolicies();
    await newGame();
  } catch (e) {
    fail(e);
  }
})();
