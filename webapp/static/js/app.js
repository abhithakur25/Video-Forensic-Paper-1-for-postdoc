const $ = (id) => document.getElementById(id);
const drop = $("drop"), file = $("file"), go = $("go"), chosen = $("chosen");
const spin = $("spin"), err = $("err"), results = $("results");

drop.addEventListener("click", () => file.click());
["dragenter", "dragover"].forEach(e =>
  drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("hot"); }));
["dragleave", "drop"].forEach(e =>
  drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("hot"); }));
drop.addEventListener("drop", ev => {
  if (ev.dataTransfer.files.length) { file.files = ev.dataTransfer.files; pick(); }
});
file.addEventListener("change", pick);

function pick() {
  if (!file.files.length) return;
  const f = file.files[0];
  chosen.textContent = `${f.name} · ${(f.size / 1048576).toFixed(1)} MB`;
  go.disabled = false;
}

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!file.files.length) return;
  go.disabled = true; go.textContent = "Analyzing…";
  spin.classList.add("show");
  err.classList.remove("show"); results.classList.remove("show");

  const fd = new FormData();
  fd.append("video", file.files[0]);
  try {
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    render(data);
  } catch (ex) {
    err.textContent = "Analysis failed — " + ex.message;
    err.classList.add("show");
  } finally {
    spin.classList.remove("show");
    go.disabled = false; go.textContent = "Analyze for Forgery";
  }
});

function render(d) {
  // score
  $("sVal").textContent = d.score.value;
  const band = $("sBand");
  band.textContent = d.score.band + " evidence";
  band.className = "band band-" + d.score.band;
  $("sBar").style.width = d.score.value + "%";

  $("terms").innerHTML = d.score.terms.map(t => `
    <div class="term">
      <span class="n">${t.name}</span>
      <span class="w">${(t.normalised * 100).toFixed(1)}% × ${t.weight}</span>
      <span class="d">${t.detail}</span>
    </div>`).join("");

  // notice
  const n = $("notice");
  if (d.notice) { n.textContent = d.notice; n.style.display = "block"; }
  else n.style.display = "none";

  // metrics
  const m = d.meta, g = d.glcm;
  $("kv").innerHTML = [
    ["Frames sampled", m.sampled], ["Total frames", m.total_frames],
    ["FPS", m.fps || "—"], ["Face ROI", m.face_roi ? "detected" : "not found"],
    ["GLCM contrast", g.contrast.toFixed(3)],
    ["GLCM homogeneity", g.homogeneity.toFixed(3)],
    ["GLCM energy", g.energy.toFixed(3)],
    ["GLCM correlation", g.correlation.toFixed(3)],
    ["Elapsed", m.elapsed_s + " s"],
  ].map(([k, v]) => `<div><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

  // stages
  $("stages").innerHTML = d.stages.filter(s => s.image).map(s => `
    <div class="card">
      <img src="${s.image}" alt="${s.title}" loading="lazy">
      <div class="cap"><h4>${s.title}</h4><p>${s.note}</p></div>
    </div>`).join("");

  chart(d.series);
  results.classList.add("show");
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function chart(s) {
  const W = 800, H = 170, P = 16;
  const norm = (a) => {
    if (!a || !a.length) return [];
    const mx = Math.max(...a), mn = Math.min(...a);
    return mx - mn < 1e-9 ? a.map(() => 0.5) : a.map(v => (v - mn) / (mx - mn));
  };
  const path = (a, col) => {
    if (a.length < 2) return "";
    const d = a.map((v, i) =>
      `${i ? "L" : "M"}${P + i * (W - 2 * P) / (a.length - 1)},${H - P - v * (H - 2 * P)}`).join(" ");
    return `<path d="${d}" fill="none" stroke="${col}" stroke-width="2.2"
             stroke-linejoin="round" stroke-linecap="round"/>`;
  };
  const marks = (idx, a, col) => (idx || []).map(i => {
    if (i >= a.length) return "";
    const x = P + i * (W - 2 * P) / Math.max(a.length - 1, 1);
    return `<circle cx="${x}" cy="${H - P - a[i] * (H - 2 * P)}" r="4" fill="${col}"/>`;
  }).join("");

  const kf = norm(s.keyframe), fl = norm(s.flow);
  const grid = [0.25, 0.5, 0.75].map(f =>
    `<line x1="${P}" x2="${W - P}" y1="${P + f * (H - 2 * P)}" y2="${P + f * (H - 2 * P)}"
      stroke="#1c2740" stroke-width="1"/>`).join("");

  $("chart").innerHTML = grid + path(kf, "#60a5fa") + path(fl, "#f59e0b")
    + marks(s.keyframe_peaks, kf, "#ef4444") + marks(s.flow_peaks, fl, "#ef4444");
}
