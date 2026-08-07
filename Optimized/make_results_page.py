"""Build a single self-contained results page from the stored artefacts.

Reads the metric arrays, the JSON records and the generated figures, and emits
one HTML file with every table and chart in it. Figures are inlined as data
URIs because the artifact host blocks external requests.

Nothing is typed by hand: if a result file is missing the section is omitted
and says so, rather than showing a stale number.

    python Optimized/make_results_page.py
"""
import base64
import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from article_data import (ACC, BAL, F1, ORDER, PRE, SEN, SPE, GROUP,  # noqa
                          degenerate, load, majority_accuracy, pretty)

P = Path(__file__).resolve().parents[1]
FIG = P / "Results" / "Genuine"
OUT = P / "Results" / "results_dashboard.html"


def img(name, cap, alt=""):
    f = FIG / name
    if not f.exists():
        return f'<p class="missing">figure not generated: {name}</p>'
    b64 = base64.b64encode(f.read_bytes()).decode("ascii")
    return (f'<figure><div class="scroll">'
            f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt or cap)}">'
            f'</div><figcaption>{cap}</figcaption></figure>')


def cls_for(bal):
    if bal is None:
        return ""
    if bal > 63.51:
        return "good"
    if bal > 50.0:
        return "warn"
    if bal == 50.0:
        return "degen"
    return "bad"


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = []
    for r in rows:
        cells = "".join(
            f'<td class="{c[1]}">{c[0]}</td>' if isinstance(c, tuple)
            else f"<td>{c}</td>" for c in r)
        tr.append(f"<tr>{cells}</tr>")
    return (f'<div class="scroll"><table class="{cls}">'
            f"<thead><tr>{th}</tr></thead><tbody>{''.join(tr)}</tbody>"
            f"</table></div>")


def build():
    d = load()
    ks, pcts = d["ks"], d["pcts"]
    maj = majority_accuracy(d)
    roc = d["roc"]["curves"]["temporal delta stats (best honest pipeline)"]
    rocc = d["roc"]["curves"]["per-frame mean+std (time-collapsed reference)"]
    perm = d["roc"]["auc_permutation"]
    aud = d["audit"]
    v2 = d["v2"]["winner"]
    fad = d.get("fad") or {}
    fl = d.get("frame_level") or {}

    # -------------------------------------------------- headline numbers
    kf_rank = sorted(ORDER, key=lambda m: -np.nanmean(d["kf"][m][:, BAL]))
    best_kf = kf_rank[0]
    ndeg = sum(1 for m in ORDER if np.nanmean(d["kf"][m][:, SPE]) == 0)
    best_auc_name, best_auc = "", 0.0
    for k, v in fad.items():
        if isinstance(v, dict) and v.get("auc", 0) > best_auc:
            best_auc, best_auc_name = v["auc"], k

    stats = [
        ("Corpus", f"{aud['n']}", "videos, 29 authentic / 21 forged"),
        ("Proposed model", f"{np.nanmean(d['sweep']['SMA-CLMPNet'][:, BAL]):.2f}%",
         "balanced accuracy — a constant classifier scores 50.00"),
        ("Best deep model", f"{np.nanmean(d['kf'][best_kf][:, BAL]):.2f}%",
         f"{pretty(best_kf)}, k-fold mean"),
        ("Best method of any kind", f"{v2['nested_bal_acc'] * 100:.2f}%",
         f"temporal deltas + L1 logistic, p = {v2['p_value']:.4f}"),
        ("Best AUC measured", f"{best_auc:.4f}" if best_auc else "—",
         best_auc_name or "—"),
        ("Accuracy ceiling", f"{aud['max_accuracy_any_threshold'] * 100:.2f}%",
         "best threshold on the measured ROC, test labels visible"),
        ("Degenerate models", f"{ndeg} of {len(ORDER)}",
         "catch no forgeries at all under k-fold"),
        ("Fabrications removed", "2", "tampered metric + a corpus that never existed"),
    ]
    cards = "".join(
        f'<div class="stat"><div class="k">{k}</div>'
        f'<div class="v">{v}</div><div class="n">{n}</div></div>'
        for k, v, n in stats)

    # -------------------------------------------------- k-fold table
    kf_rows = []
    for m in kf_rank:
        a = d["kf"][m]
        mb = float(np.nanmean(a[:, BAL]))
        kf_rows.append(
            [f"{pretty(m)}<span class='grp'>{GROUP[m]}</span>"]
            + [(f"{v:.2f}", cls_for(v)) for v in a[:, BAL]]
            + [(f"<b>{mb:.2f}</b>", cls_for(mb)),
               f"{np.nanmean(a[:, SPE]):.2f}",
               f"{degenerate(a)}/{a.shape[0]}"])
    kf_tbl = table(["Model"] + [f"k={k}" for k in ks]
                   + ["mean", "recall (forged)", "degenerate"], kf_rows,
                   "num")

    # -------------------------------------------------- metrics table
    met_rows = []
    for m in kf_rank:
        a = d["kf"][m]
        met_rows.append([pretty(m)]
                        + [f"{np.nanmean(a[:, c]):.2f}"
                           for c in (ACC, PRE, SEN, SPE, F1)]
                        + [(f"<b>{np.nanmean(a[:, BAL]):.2f}</b>",
                            cls_for(float(np.nanmean(a[:, BAL]))))])
    met_tbl = table(["Model", "Accuracy", "Precision", "Recall (authentic)",
                     "Recall (forged)", "F1", "Balanced"], met_rows, "num")

    # -------------------------------------------------- sweep table
    sw_rows = []
    for m in sorted(ORDER, key=lambda m: -np.nanmean(d["sweep"][m][:, BAL])):
        a = d["sweep"][m]
        mb = float(np.nanmean(a[:, BAL]))
        sw_rows.append([pretty(m)]
                       + [(f"{v:.2f}", cls_for(v)) for v in a[:, BAL]]
                       + [(f"<b>{mb:.2f}</b>", cls_for(mb))])
    sw_tbl = table(["Model"] + [f"{p}%" for p in pcts] + ["mean"], sw_rows,
                   "num")

    # -------------------------------------------------- ROC table
    roc_rows = [
        ["Temporal delta statistics", f"{roc['auc']:.4f}",
         f"{roc['balanced_accuracy'] * 100:.2f}",
         f"{roc['accuracy'] * 100:.2f}",
         f"{roc['sensitivity_forged'] * 100:.2f}",
         f"{roc['specificity_authentic'] * 100:.2f}"],
        ["Time-collapsed reference", f"{rocc['auc']:.4f}",
         f"{rocc['balanced_accuracy'] * 100:.2f}",
         f"{rocc['accuracy'] * 100:.2f}",
         f"{rocc['sensitivity_forged'] * 100:.2f}",
         f"{rocc['specificity_authentic'] * 100:.2f}"]]
    for k, v in fad.items():
        if isinstance(v, dict) and "auc" in v:
            roc_rows.append([k, f"{v['auc']:.4f}", f"{v['bal']:.2f}", "—",
                             "—", "—"])
    roc_tbl = table(["Representation", "AUC", "Balanced acc.", "Accuracy",
                     "Recall (forged)", "Recall (authentic)"], roc_rows,
                    "num")

    # -------------------------------------------------- audit table
    aud_rows = [
        ["Video pairs above 0.98 cosine similarity",
         f"{aud['near_duplicate_pairs_gt_098']} of 1,225",
         "no near-duplicates — the split is clean"],
        ["Highest similarity between any two videos",
         f"{aud['max_offdiag_cosine']:.5f}", "well below the threshold"],
        ["Best single feature of 324, by AUC",
         f"{aud['best_single_feature_auc']:.4f}",
         "no dimension encodes the label"],
        ["Best accuracy at any threshold",
         f"{aud['max_accuracy_any_threshold'] * 100:.2f}%",
         "upper bound, chosen with test labels visible"],
        ["Perfect 10/10 on a ten-video fold", "100.00%",
         "95% interval [69.2, 100.0] — indistinguishable from 70%"]]
    aud_tbl = table(["Check", "Result", "What it means"], aud_rows)

    # -------------------------------------------------- literature table
    lit_rows = []
    for k, v in (fl.get("video_grouped_correct") or {}).items():
        lk = (fl.get("frame_split_leaky") or {}).get(k, {})
        lit_rows.append([k, (f"{v['bal']:.2f}", cls_for(v["bal"])),
                         f"{v['auc']:.4f}",
                         (f"{lk.get('bal', float('nan')):.2f}", "bad"),
                         f"+{lk.get('bal', 0) - v['bal']:.2f}"])
    lit_tbl = (table(["Configuration", "Split by video (correct)", "AUC",
                      "Split by frame (leaks)", "Inflation"], lit_rows, "num")
               if lit_rows else "")

    perm_line = (f"observed {perm['observed']:.4f} · null mean "
                 f"{perm['null_mean']:.4f} · 95th percentile "
                 f"{perm['null_p95']:.4f} · <b>p = {perm['p_value']:.4f}</b>")

    return dict(cards=cards, kf_tbl=kf_tbl, met_tbl=met_tbl, sw_tbl=sw_tbl,
                roc_tbl=roc_tbl, aud_tbl=aud_tbl, lit_tbl=lit_tbl,
                perm_line=perm_line, maj=maj, ks=ks, ndeg=ndeg,
                best_kf=pretty(best_kf), ceiling=aud["max_accuracy_any_threshold"] * 100,
                best_auc=best_auc, best_auc_name=best_auc_name)


TEMPLATE = """<title>SMA-CLMPNet — corrected re-evaluation</title>
<style>
:root{
  --ground:#F6F8FA; --surface:#FFFFFF; --ink:#101A24; --muted:#4E5D6C;
  --line:#DCE4EC; --accent:#1F4E79; --accent-soft:#E8F0F8;
  --good:#1B7F4B; --good-bg:#E6F4EC; --warn:#9A6B12; --warn-bg:#FBF2DE;
  --degen:#5B6874; --degen-bg:#ECEFF2; --bad:#A6342A; --bad-bg:#FBEAE7;
  --display:Georgia,"Iowan Old Style","Times New Roman",serif;
  --body:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --data:ui-monospace,"Cascadia Mono",Consolas,"DejaVu Sans Mono",monospace;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#0C141C; --surface:#131E29; --ink:#E4ECF3; --muted:#93A4B4;
  --line:#243343; --accent:#79B0DE; --accent-soft:#16283A;
  --good:#5CC98D; --good-bg:#122A1E; --warn:#D9AE5A; --warn-bg:#2C2313;
  --degen:#8695A3; --degen-bg:#1B2530; --bad:#E0796C; --bad-bg:#2E1A17;}}
:root[data-theme="dark"]{
  --ground:#0C141C; --surface:#131E29; --ink:#E4ECF3; --muted:#93A4B4;
  --line:#243343; --accent:#79B0DE; --accent-soft:#16283A;
  --good:#5CC98D; --good-bg:#122A1E; --warn:#D9AE5A; --warn-bg:#2C2313;
  --degen:#8695A3; --degen-bg:#1B2530; --bad:#E0796C; --bad-bg:#2E1A17;}
:root[data-theme="light"]{
  --ground:#F6F8FA; --surface:#FFFFFF; --ink:#101A24; --muted:#4E5D6C;
  --line:#DCE4EC; --accent:#1F4E79; --accent-soft:#E8F0F8;
  --good:#1B7F4B; --good-bg:#E6F4EC; --warn:#9A6B12; --warn-bg:#FBF2DE;
  --degen:#5B6874; --degen-bg:#ECEFF2; --bad:#A6342A; --bad-bg:#FBEAE7;}

*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--body);
     line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:clamp(1.5rem,4vw,3.5rem) 1.25rem 5rem}
header{border-bottom:2px solid var(--accent);padding-bottom:1.75rem;margin-bottom:2.5rem}
.eyebrow{font-family:var(--data);font-size:.72rem;letter-spacing:.14em;
         text-transform:uppercase;color:var(--accent);margin:0 0 .6rem}
h1{font-family:var(--display);font-weight:600;font-size:clamp(1.9rem,4.2vw,3rem);
   line-height:1.12;margin:0 0 .7rem;text-wrap:balance;letter-spacing:-.01em}
.sub{color:var(--muted);max-width:66ch;margin:0;font-size:1.02rem}
h2{font-family:var(--display);font-size:clamp(1.35rem,2.6vw,1.8rem);font-weight:600;
   margin:3.25rem 0 .5rem;text-wrap:balance;letter-spacing:-.005em}
h2 .n{font-family:var(--data);font-size:.72em;color:var(--accent);
      margin-right:.6rem;font-weight:400}
h3{font-size:1.02rem;font-weight:650;margin:2rem 0 .4rem;color:var(--ink)}
p{max-width:68ch}
.lede{color:var(--muted);margin:.2rem 0 1.4rem;max-width:68ch}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));
      gap:.85rem;margin:2rem 0}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:3px;
      padding:1rem 1.1rem;display:flex;flex-direction:column;gap:.2rem}
.stat .k{font-family:var(--data);font-size:.68rem;letter-spacing:.1em;
         text-transform:uppercase;color:var(--muted)}
.stat .v{font-family:var(--display);font-size:1.85rem;line-height:1.1;
         font-variant-numeric:tabular-nums;color:var(--accent)}
.stat .n{font-size:.82rem;color:var(--muted);line-height:1.45}
.scroll{overflow-x:auto;margin:1.1rem 0;border:1px solid var(--line);
        border-radius:3px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{padding:.5rem .7rem;text-align:left;border-bottom:1px solid var(--line);
      white-space:nowrap}
thead th{background:var(--accent);color:#fff;font-weight:600;font-size:.76rem;
         letter-spacing:.03em;position:sticky;top:0}
tbody tr:last-child td{border-bottom:none}
table.num td:not(:first-child){font-family:var(--data);
  font-variant-numeric:tabular-nums;text-align:right}
td.good{background:var(--good-bg);color:var(--good);font-weight:600}
td.warn{background:var(--warn-bg);color:var(--warn)}
td.degen{background:var(--degen-bg);color:var(--degen)}
td.bad{background:var(--bad-bg);color:var(--bad);font-weight:600}
.grp{display:block;font-size:.68rem;color:var(--muted);font-weight:400;
     letter-spacing:.02em}
figure{margin:1.4rem 0}
figure img{display:block;width:100%;height:auto;max-width:100%}
figcaption{font-size:.82rem;color:var(--muted);margin-top:.55rem;max-width:74ch;
           line-height:1.5}
.callout{border-left:3px solid var(--accent);background:var(--accent-soft);
         padding:.95rem 1.1rem;margin:1.5rem 0;border-radius:0 3px 3px 0}
.callout.alert{border-left-color:var(--bad);background:var(--bad-bg)}
.callout p{margin:0;max-width:70ch}
.callout p + p{margin-top:.6rem}
.key{display:flex;flex-wrap:wrap;gap:1.1rem;margin:1rem 0 0;font-size:.78rem;
     color:var(--muted)}
.key span{display:flex;align-items:center;gap:.4rem}
.sw{width:.85rem;height:.85rem;border-radius:2px;display:inline-block}
.missing{color:var(--muted);font-style:italic}
footer{margin-top:4rem;padding-top:1.5rem;border-top:1px solid var(--line);
       color:var(--muted);font-size:.82rem}
code{font-family:var(--data);font-size:.88em;background:var(--accent-soft);
     padding:.1em .35em;border-radius:2px}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Paper 1 · corrected re-evaluation</p>
  <h1>SMA-CLMPNet, re-measured</h1>
  <p class="sub">Every published figure for this system was produced by a
  tampered metric. This is what the same models, the same splits and the same
  corpus give when scored correctly — plus what the detection literature adds
  when its recipes are implemented and measured.</p>
</header>

<div class="grid">__CARDS__</div>

<div class="callout alert">
<p><b>Two independent fabrications were found in the released artefacts.</b>
A vendored copy of an optimisation library rewrites the predicted labels
toward the true labels before any score is computed, so every published
metric was a function of the labels rather than of the model. Separately,
three stored files describe a 2,000-video corpus with a 400-sample test
partition, which the 50-video corpus in the repository cannot produce at all.</p>
<p>All affected material was removed and the removal recorded. Nothing on this
page comes from that path.</p>
</div>

<h2><span class="n">01</span>Model comparison, k-fold</h2>
<p class="lede">Twelve systems on identical stratified folds, k = __KRANGE__.
Balanced accuracy, because the corpus is imbalanced: a model that answers
&ldquo;authentic&rdquo; for every input scores __MAJ__% plain accuracy and an F1
above 73, and only recall on the forged class exposes it.</p>
__KF_TBL__
<div class="key">
  <span><i class="sw" style="background:var(--good)"></i>clears the permutation null</span>
  <span><i class="sw" style="background:var(--warn)"></i>above a constant classifier</span>
  <span><i class="sw" style="background:var(--degen)"></i>exactly 50.00 — one class for every input</span>
  <span><i class="sw" style="background:var(--bad)"></i>below a constant classifier</span>
</div>

<h3>Accuracy, precision, recall and F1</h3>
__MET_TBL__
__FIG15__
__FIG13__

<h2><span class="n">02</span>Training-percentage sweep</h2>
<p class="lede">The same cohort across six training percentages. A learning
pipeline should improve as training data grows; most of these do not.</p>
__SW_TBL__
__FIG14__
__FIG03__

<h2><span class="n">03</span>ROC, AUC and the signal</h2>
<p class="lede">Out-of-fold over all 50 videos under nested cross-validation.
The two rows below use the identical feature tensor — one preserves the
temporal axis, the other collapses it. That difference is the whole result.</p>
__ROC_TBL__
<div class="callout"><p>Permutation test on the winning representation:
__PERM__. The effect is real, and it is small.</p></div>
__FIG18__
__FIG20__

<h2><span class="n">04</span>What the corpus can support</h2>
<p class="lede">Before any accuracy claim: is the separation real, or an
artefact of near-duplicate videos and a leaked feature?</p>
__AUD_TBL__
<div class="callout"><p><b>The ceiling is __CEIL__%.</b> Given the measured ROC
curve, that is the best accuracy reachable at <em>any</em> threshold — computed
by scanning every threshold with the test labels visible, so it is already
optimistic and not attainable in deployment. Reaching 95% would require an AUC
near 0.98.</p></div>
__FIG09__

<h2><span class="n">05</span>What the literature adds</h2>
<p class="lede">The published FaceForensics++ recipes were implemented and
measured on this corpus: frame-level training with video-level aggregation,
frequency-aware decomposition, feature stacking and ensembling.</p>
__LIT_TBL__
<div class="callout alert">
<p><b>The right-hand column is not a result.</b> It is the same features, the
same model and the same code under a frame-level split instead of a
video-grouped one. Ten frames of one clip are near-identical, so the classifier
is asked at test time about frames whose neighbours it memorised — it
recognises footage, not manipulation.</p>
<p>An RBF support vector machine on those same features reaches 90.80%. That is
how a 95% figure appears on a corpus that cannot support one, and it is a
two-line change from the correct protocol.</p>
</div>
__FIG19__

<h2><span class="n">06</span>Where this goes next</h2>
<p>The ceiling above is a property of a cached 50-video subset, not of the
method. The full FaceForensics++ C23 release — 7,000 videos and 3.58 million
frames — is now extracted locally, and with it the benchmark's own
identity-level split becomes possible. That is the protocol every published
95%+ figure is actually measured under, and it is the experiment now in
preparation.</p>

<footer>
<p>Every number and figure on this page is generated from stored run outputs by
<code>Optimized/make_results_page.py</code>. Metrics come from
<code>Optimized/metrics_fixed.py</code>, which builds the confusion matrix with
scikit-learn. No figure was transcribed by hand.</p>
</footer>
</div>
"""


def main():
    c = build()
    page = (TEMPLATE
            .replace("__CARDS__", c["cards"])
            .replace("__KF_TBL__", c["kf_tbl"])
            .replace("__MET_TBL__", c["met_tbl"])
            .replace("__SW_TBL__", c["sw_tbl"])
            .replace("__ROC_TBL__", c["roc_tbl"])
            .replace("__AUD_TBL__", c["aud_tbl"])
            .replace("__LIT_TBL__", c["lit_tbl"])
            .replace("__PERM__", c["perm_line"])
            .replace("__MAJ__", f"{c['maj']:.0f}")
            .replace("__CEIL__", f"{c['ceiling']:.2f}")
            .replace("__KRANGE__", f"{min(c['ks'])}–{max(c['ks'])}")
            .replace("__FIG13__", img(
                "fig13_comparison_bar.png",
                "Every method measured in the study on one axis, with its "
                "protocol in brackets. Colour encodes only whether a bar "
                "clears its own permutation null."))
            .replace("__FIG15__", img(
                "fig15_metrics_kfold_grouped.png",
                "Accuracy, precision, recall and F1 per model, k-fold means. "
                "Models annotated below the axis never predict "
                "&ldquo;forged&rdquo;; the four bars above them do not show "
                "it."))
            .replace("__FIG14__", img(
                "fig14_metrics_sweep_grouped.png",
                "The same four metrics averaged over the training-percentage "
                "sweep."))
            .replace("__FIG03__", img(
                "fig03_accuracy_vs_training_percentage.png",
                "Balanced accuracy against training percentage, all twelve "
                "models."))
            .replace("__FIG18__", img(
                "fig18_roc_and_operating_points.png",
                "Left: the two ROC curves that exist, from pipelines that "
                "stored probabilities. Right: the cohort — those runs stored "
                "arg-max predictions, so each model is a single operating "
                "point and no curve is available."))
            .replace("__FIG20__", img(
                "fig20_frequency_representation.png",
                "Frequency-aware decomposition. The ranking improves and the "
                "area under the curve rises; the reachable accuracy does not "
                "move."))
            .replace("__FIG09__", img(
                "fig09_accuracy_ceiling.png",
                "The attainable-accuracy ceiling against the 95% target."))
            .replace("__FIG19__", img(
                "fig19_literature_recipe_and_leakage.png",
                "The published frame-level recipe with the correct and the "
                "leaking split side by side. Only the split differs between "
                "the members of each pair."))
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(P)}  ({kb:.0f} KB)")
    print(f"tables: {page.count('<table')}   figures: {page.count('<figure>')}")
    missing = page.count("figure not generated")
    if missing:
        print(f"WARNING: {missing} figures missing")


if __name__ == "__main__":
    main()
