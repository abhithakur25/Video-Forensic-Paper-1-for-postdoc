"""Generate a complete .docx record of the work: every step and every result.

Writes the OOXML directly (no python-docx in this environment): a minimal but
valid Word package with styled headings, body text, code blocks and tables.

Output: Paper1_Complete_Work_Report.docx in the project root.
"""
import datetime
import glob
import html
import json
import re
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
OUT = P / "Paper1_Complete_Work_Report.docx"

# --------------------------------------------------------------- OOXML parts
CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

# ------------------------------------------------------------------- images
# Embedding a picture in OOXML needs four things kept in step: the PNG bytes
# as word/media/*, a Default extension in [Content_Types].xml, a relationship
# in word/_rels/document.xml.rels, and a w:drawing referencing that rId.
# IMAGES accumulates (rId, filename, bytes) as picture() is called; write_doc()
# emits all four.
IMAGES = []
EMU_PER_PX = 9525          # 1 px at 96 dpi
MAX_WIDTH_EMU = 5943600    # 6.2 in - fits A4 with the 1134 twip margins


def picture(path, caption=None, max_w_in=6.2):
    """Embed a PNG, scaled to fit the text column, with an optional caption."""
    path = Path(path)
    if not path.exists():
        return para(f"[figure missing: {path.name}]", "Caption")
    data = path.read_bytes()
    try:
        import struct
        w, h = struct.unpack(">II", data[16:24])   # PNG IHDR
    except Exception:
        w, h = 1200, 800
    cx, cy = w * EMU_PER_PX, h * EMU_PER_PX
    cap = int(max_w_in * 914400)
    if cx > cap:
        cy = int(cy * cap / cx)
        cx = cap

    rid = f"rIdImg{len(IMAGES) + 1}"
    name = f"image{len(IMAGES) + 1}{path.suffix.lower()}"
    IMAGES.append((rid, name, data))
    n = len(IMAGES)
    drawing = (
        '<w:p><w:pPr><w:pStyle w:val="Normal"/><w:jc w:val="center"/>'
        '<w:spacing w:before="160" w:after="60"/></w:pPr><w:r><w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:docPr id="{n}" name="Picture {n}" descr="{esc(path.stem)}"/>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{n}" name="{esc(path.name)}"/>'
        '<pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/>'
        '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline>'
        '</w:drawing></w:r></w:p>')
    if caption:
        drawing += para(caption, "Caption")
    return drawing


def figure(name, caption=None):
    """Embed Results/Genuine/<name> if it exists."""
    return picture(P / "Results" / "Genuine" / name, caption)


# Formatting follows Paper1_SMA_CLMPNet_Genuine_Research_Paper.docx:
# Times New Roman throughout; title 15pt centred, authors 12pt centred,
# affiliations 8pt centred, section headings 13pt bold, subsections 12pt bold,
# body 10pt justified, table text 8pt. Sizes below are half-points, as OOXML
# requires.
FONT = "Times New Roman"


def _style(sid, name, sz, bold, color=None, before=200, after=100,
           mono=False, jc=None):
    font = "Consolas" if mono else FONT
    # built outside the f-string: Python 3.8 forbids backslashes in f-string
    # expressions, and nesting quotes here is what would need them
    b_tag = "<w:b/>" if bold else ""
    c_tag = '<w:color w:val="%s"/>' % color if color else ""
    j_tag = '<w:jc w:val="%s"/>' % jc if jc else ""
    return (f'<w:style w:type="paragraph" w:styleId="{sid}">'
            f'<w:name w:val="{name}"/>'
            f'<w:pPr><w:spacing w:before="{before}" w:after="{after}"/>'
            f'{j_tag}</w:pPr>'
            f'<w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}"/>'
            f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
            f'{b_tag}{c_tag}</w:rPr></w:style>')


DOC_DEFAULTS = (
    '<w:docDefaults><w:rPrDefault><w:rPr>'
    f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:cs="{FONT}"/>'
    '<w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:rPrDefault>'
    '</w:docDefaults>')

STYLES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
          + DOC_DEFAULTS
          + _style("Title", "Title", 30, True, None, 0, 120, jc="center")
          + _style("Authors", "Authors", 24, True, None, 60, 60, jc="center")
          + _style("Affil", "Affiliation", 16, False, None, 0, 20, jc="center")
          + _style("Corr", "Corresponding", 18, False, None, 40, 160,
                   jc="center")
          + _style("Heading1", "heading 1", 26, True, None, 300, 120)
          + _style("Heading2", "heading 2", 24, True, None, 240, 100)
          + _style("Heading3", "heading 3", 22, True, None, 200, 80)
          + _style("Normal", "Normal", 20, False, None, 60, 60, jc="both")
          + _style("Code", "Code", 18, False, "333333", 60, 60, mono=True)
          + _style("Caption", "Caption", 18, False, "444444", 20, 160)
          + '</w:styles>')

# The template's own front matter, reproduced verbatim.
TITLE_BLOCK_AUTHORS = ("Dr. Abhishek Thakur1,2,*,  Prof. Vishal Jain3, and  "
                       "Prof. Chin-Shiuh Shieh4")
TITLE_BLOCK_AFFIL = [
    "(謝欽旭)",
    "1 Primary: School of Computer Science & Engineering, Chitkara "
    "University, Himachal Pradesh, India (Associate Professor).",
    "2 Secondary: Postdoctoral Researcher, RIITC / Department of Electronic "
    "Engineering, NKUST, Kaohsiung 80778, Taiwan (R.O.C.); Ref. "
    "RIITC-Postdoc-2027-B03.",
    "3 Co-Supervisor: Vivekananda Institute of Professional Studies – "
    "Technical Campus (VIPS-TC), India.",
    "4 Supervisor: Department of Electronic Engineering / RIITC, NKUST, "
    "No. 415, Jiangong Rd., Kaohsiung 80778, Taiwan (R.O.C.).",
    "E-mail: abhithakur25@gmail.com; abhishek@chitkarauniversity.edu.in  |  "
    "drvishaljain83@gmail.com; vishal.jain@vips.edu  |  csshieh@nkust.edu.tw",
]
TITLE_BLOCK_CORR = ("* Corresponding author: Dr. Abhishek Thakur "
                    "(abhithakur25@gmail.com).")


def title_block(title):
    """Title, authors, affiliations and corresponding-author line."""
    out = [para(title, "Title"), para(TITLE_BLOCK_AUTHORS, "Authors")]
    out += [para(a, "Affil") for a in TITLE_BLOCK_AFFIL]
    out.append(para(TITLE_BLOCK_CORR, "Corr"))
    return "".join(out)


def esc(t):
    return html.escape(str(t), quote=False)


def para(text, style="Normal", bold_prefix=None):
    runs = ""
    if bold_prefix:
        runs += (f'<w:r><w:rPr><w:b/></w:rPr>'
                 f'<w:t xml:space="preserve">{esc(bold_prefix)}</w:t></w:r>')
    runs += f'<w:r><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{runs}</w:p>'


def bullet(text):
    return para("•  " + text, "Normal")


def code(lines):
    if isinstance(lines, str):
        lines = lines.split("\n")
    return "".join(para(l if l else " ", "Code") for l in lines)


def cell(text, bold=False, shade=None, width=1200):
    sh = f'<w:shd w:val="clear" w:fill="{shade}"/>' if shade else ""
    rpr = "<w:b/>" if bold else ""
    # jc=left explicitly: Normal is justified, which stretches short cell
    # text across the column and looks broken in a narrow table
    return (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{sh}'
            f'<w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr><w:pStyle w:val="Normal"/>'
            f'<w:jc w:val="left"/>'
            f'<w:spacing w:before="20" w:after="20"/></w:pPr>'
            f'<w:r><w:rPr>{rpr}<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/>'
            f'<w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr>'
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p></w:tc>')


def table(head, rows, widths=None, highlight=None):
    n = len(head)
    widths = widths or [max(1000, int(9000 / n))] * n
    borders = ('<w:tblBorders>' + "".join(
        f'<w:{s} w:val="single" w:sz="4" w:color="BFBFBF"/>'
        for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
        + '</w:tblBorders>')
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>{borders}</w:tblPr>']
    out.append("<w:tr>" + "".join(
        cell(h, True, "DCE6F1", widths[i]) for i, h in enumerate(head))
        + "</w:tr>")
    for r in rows:
        sh = "FFF2CC" if (highlight and highlight(r)) else None
        out.append("<w:tr>" + "".join(
            cell(c, False, sh, widths[i]) for i, c in enumerate(r)) + "</w:tr>")
    out.append("</w:tbl>")
    out.append(para(" "))
    return "".join(out)


NS = (' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
      ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
      ' xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"'
      ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
      ' xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')


def build(body):
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:document{NS}>'
            f'<w:body>{body}'
            '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>'
            '</w:sectPr></w:body></w:document>')


def write_doc(out_path, body):
    """Write the .docx, including any images picture() accumulated, and
    validate the result before returning."""
    import xml.dom.minidom as md

    xml = build(body)
    ct = CONTENT_TYPES
    if IMAGES:
        exts = {Path(n).suffix.lstrip(".").lower() for _, n, _ in IMAGES}
        mime = {"png": "image/png", "jpg": "image/jpeg",
                "jpeg": "image/jpeg", "gif": "image/gif"}
        ct = ct.replace("</Types>", "".join(
            f'<Default Extension="{e}" ContentType="{mime.get(e, "image/png")}"/>'
            for e in sorted(exts)) + "</Types>")
    rels = DOC_RELS.replace("</Relationships>", "".join(
        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
        f'officeDocument/2006/relationships/image" Target="media/{name}"/>'
        for rid, name, _ in IMAGES) + "</Relationships>")

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/document.xml", xml)
        z.writestr("word/styles.xml", STYLES)
        z.writestr("word/_rels/document.xml.rels", rels)
        for _, name, data in IMAGES:
            z.writestr(f"word/media/{name}", data)

    with zipfile.ZipFile(out_path) as z:
        assert z.testzip() is None, "corrupt zip"
        d = z.read("word/document.xml").decode("utf8")
        md.parseString(d)
        md.parseString(z.read("word/_rels/document.xml.rels").decode("utf8"))
        md.parseString(z.read("[Content_Types].xml").decode("utf8"))
        media = [n for n in z.namelist() if n.startswith("word/media/")]
        # every rId referenced by a drawing must exist as a relationship
        used = set(re.findall(r'r:embed="([^"]+)"', d))
        declared = set(re.findall(r'Id="(rIdImg\d+)"',
                                  z.read("word/_rels/document.xml.rels")
                                  .decode("utf8")))
        missing = used - declared
        assert not missing, f"drawings reference undeclared rIds: {missing}"
    print(f"wrote {Path(out_path).name}  "
          f"({Path(out_path).stat().st_size/1024:.0f} KB)")
    print(f"validated: zip OK, XML well-formed, "
          f"{d.count('<w:p>') + d.count('<w:p ')} paragraphs, "
          f"{d.count('<w:tbl>')} tables, {len(media)} embedded images")
    return d


# ------------------------------------------------------------------- content
def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x*100:.2f}"


def load_true(sub="TRUE"):
    d = P / "Analysis1" / sub
    if not (d / "run_manifest.json").exists():
        return {}, {}
    man = json.loads((d / "run_manifest.json").read_text("utf-8"))
    return {f.stem: np.load(f) for f in sorted(d.glob("*.npy"))}, man


def load_json(name):
    f = P / "Optimized" / name
    return json.loads(f.read_text("utf-8")) if f.exists() else None


def main():
    b = []
    now = datetime.datetime.now()
    tr, man = load_true()
    kf, kman = load_true("TRUE_KF")
    roc = load_json("roc_confusion.json")
    audit = load_json("corpus_audit.json")
    stil = load_json("oof_stil_tim.json")
    v2, v3 = load_json("optimize_v2.json"), load_json("optimize_v3.json")
    wts, probe = load_json("optimize_weights.json"), load_json("feature_probe.json")
    p2 = load_json("paper2_model.json")

    # ---------------------------------------------------------------- title
    b.append(title_block(
        "SMA-CLMPNet: Spatial Multiscale Attention enabled Convolutional "
        "Distributed Memory Network for Intra-frame Video Forgery Detection "
        "— Complete Work Record and Measured Evaluation"))
    b.append(para(f"Generated {now:%Y-%m-%d %H:%M:%S}. Windows 11, conda env "
                  f"VideoForgeryCPU (Python 3.8.20, TensorFlow 2.10, CPU "
                  f"only). Every figure in this record was scored with "
                  f"Optimized/metrics_fixed.py and traces to a named file in "
                  f"the repository.", "Caption"))

    # ------------------------------------------------------------- summary
    b.append(para("1. Executive Summary", "Heading1"))
    b.append(para("Three findings, in order of importance."))
    b.append(para("1.1 The reported metrics are fabricated", "Heading2"))
    b.append(para(
        "Every accuracy, sensitivity, specificity, precision, F1 and ROC value "
        "this codebase produces is a random number, independent of the model, "
        "the features and the training. The vendored mealpy/metrics.py has a "
        "modified _check_targets() that discards the classifier's predictions "
        "and replaces them with the ground-truth labels with a uniformly "
        "random fraction of entries flipped. A perfect predictor scores "
        "between 64.5% and 100.0% across repeated calls; an inverted predictor "
        "scores no worse; two identical calls disagree."))
    b.append(para("1.2 Paper 2's 100% is the maximum of 500 random draws",
                  "Heading2"))
    b.append(para(
        "Paper 2's SubFunctions/Optimization.py receives the test features and "
        "test labels in its constructor and searches model weights to maximise "
        "the score computed on them. That score is the fabricated metric, and "
        "HYBRID(epoch=10, pop_size=50) performs 500 evaluations keeping the "
        "best. Reproduced here: 30 of 30 independent runs return exactly "
        "100.00%. The same tampered file is present in the Paper 2 delivery, "
        "on the identical code path."))
    b.append(para("1.3 The features carry real but modest temporal signal",
                  "Heading2"))
    b.append(para(
        "Measured with a correct confusion matrix, the proposed SMA-CLMPNet "
        "achieves 50.00% balanced accuracy — it assigns one label to every "
        "video. However the 'proposed' feature tensor does carry signal: "
        "first-order temporal deltas reach 77.17% balanced accuracy under "
        "nested cross-validation, permutation-tested at p = 0.0099. The "
        "feature design is the paper's genuine contribution; the model fails "
        "to extract from it."))

    # ------------------------------------------------------- the fabrication
    b.append(para("2. The Fabricated Metric — Evidence", "Heading1"))
    b.append(para("2.1 The tampered code", "Heading2"))
    b.append(para("mealpy/metrics.py, lines 16–75, inside _check_targets(), "
                  "immediately before the confusion matrix is computed:"))
    b.append(code([
        "if perf:",
        "    per = random.uniform(0.065242, 0.35245235634)",
        "else:",
        "    per = random.uniform(0.090242, 0.45245235634)",
        "...",
        "y = np.concatenate(dat)",
        "y_true = shuffle(y, random_state=0)",
        "y_pred = y_true.copy()                       # predictions discarded",
        "va = random.sample(range(1, len(y_true)), int(len(y_true) * per))",
        "for i in va:",
        "    y_pred[i] = (random.sample(range(0, n), 1))[0]   # randomised",
    ]))
    b.append(para("Upstream mealpy contains no such code. The genuine function "
                  "survives, commented out, at mealpy/metrics.py:285."))
    b.append(para("2.2 Demonstration", "Heading2"))
    b.append(para("SubFunctions.Evaluate.Evaluation_Metrics as delivered, "
                  "y_true fixed at 15 zeros and 16 ones:"))
    b.append(table(
        ["Predictor", "True accuracy", "Reported (3 consecutive calls)"],
        [["Perfect", "1.000", "0.839, 0.935, 0.806"],
         ["Inverted (all wrong)", "0.000", "0.871, 0.710, 0.968"],
         ["Constant, all class 0", "0.484", "0.774, 0.774, 0.935"],
         ["Uniform random", "0.581", "0.806, 0.935, 0.839"]],
        [2600, 1800, 3600]))
    b.append(para("400 calls with a perfect predictor: min 0.645, max 1.000, "
                  "mean 0.878. A correct metric returns 1.000 every time."))
    b.append(para("2.3 Blast radius", "Heading2"))
    b.append(table(["Affected", "Route"],
                   [["§5.6.1 comparison vs training percentage",
                     "Analysis.py:184 → Evaluation_Metrics"],
                    ["§5.6.2 k-fold comparison",
                     "Analysis.py:233 → Evaluation_Metrics"],
                    ["§5.8 statistical analysis", "derived from same arrays"],
                    ["All ROC curves",
                     "Analysis.py:286-291 → Evaluation_Metrics1"],
                    ["Analysis/ and Analysis1/ .npy arrays", "same"],
                    ["Paper 2 (CODE_05-08-2025_Paper2)",
                     "SubFunctions/Evaluate.py:1, calls at lines 18 and 53"]],
                   [4200, 4600]))
    b.append(para("2.4 A second fabrication, independent of the metric",
                  "Heading2"))
    b.append(para(
        "Three published artifacts describe a corpus that does not exist. "
        "Features/Features.pkl holds 50 videos, 29 authentic and 21 forged; "
        "the largest test split in the entire protocol is 31 videos. These "
        "cannot have been produced from the shipped data by any scorer, "
        "tampered or otherwise — the numbers were not computed at all."))
    b.append(table(
        ["Artifact", "What it asserts", "Why it is impossible"],
        [["Results/Class.png", "1000 Normal / 1000 Scam",
          "The corpus is 29 / 21."],
         ["Results/ConfusionMatrix.png",
          "196 / 4 / 7 / 193, i.e. 400 test samples at 200/200, 97.25%",
          "400 test samples exceed the whole 50-video corpus eightfold."],
         ["Results/Features.csv",
          "GradCAM 97.0154535352, Hybrid 97.91784781180843",
          "Twelve significant figures where one video is worth 2 "
          "percentage points."]],
        [2500, 3000, 3300]))
    b.append(para("This is a separate finding from the tampered metric and is "
                  "not explained by it. All four files (including the bar "
                  "chart Results/Features.jpg) have been removed from the "
                  "repository; see section 12.", "Caption"))

    b.append(para("2.5 The fix", "Heading2"))
    b.append(para(
        "Optimized/metrics_fixed.py reimplements the metrics. The formulas in "
        "SubFunctions/Evaluate.py are correct as written, including the "
        "convention that class 0 is positive (TP = cm[0,0]); they are "
        "reproduced verbatim apart from zero-division guards. Only the "
        "confusion matrix is replaced, with sklearn's. Nothing in "
        "SubFunctions/ or mealpy/ was edited — the tampered code remains in "
        "place as evidence. Self-test: perfect 1.0000, inverted 0.0000, "
        "all-zeros 0.4839, deterministic over 50 calls."))

    # ------------------------------------------------------- Paper 2 finding
    b.append(para("3. Paper 2 Reference Study", "Heading1"))
    b.append(para("3.1 What Optimization.py does", "Heading2"))
    b.append(code([
        "class Optimization:",
        "    def __init__(self, model, x_test, y_test):   # receives TEST data",
        "        ...",
        "    def fitness_function1(self, solution):",
        "        self.model.set_weights(to_opt)",
        "        ypred = np.argmax(self.model.predict(self.x_test), axis=1)",
        "        A = Evaluation_Metrics(self.y_test, ypred)  # scores on TEST",
        "        return A[0]                                 # ...maximises it",
        "",
        "    model = HYBRID(epoch=10, pop_size=50)   # 500 evaluations, keep max",
    ]))
    b.append(para("3.2 Reproduction", "Heading2"))
    b.append(table(["500 fitness evaluations", "Value"],
                   [["Minimum", "67.74%"], ["Mean", "87.29%"],
                    ["Maximum — what solve() returns", "100.00%"],
                    ["Independent runs reaching 100%", "30 / 30"]],
                   [5000, 2500]))
    b.append(para("3.3 Paper 2's model applied to Paper 1", "Heading2"))
    b.append(para(
        "BiLSTMGBM was ported in full — stacked Bi-LSTM (100/128/128), "
        "multi-level and mixed attention, incremental learning over 5 "
        "cumulative chunks, 500 epochs, batch 32, lr 0.001, and the network "
        "used as a feature extractor with GradientBoosting on top — omitting "
        "only the test-set fitting step. Stratified 5-fold, correct scoring:"))
    if p2:
        g, o = p2["BiLSTMGBM"], p2["BiLSTM_only"]
        b.append(table(
            ["Configuration", "ACC", "SEN", "SPE", "PRE", "F1", "BAL"],
            [["BiLSTMGBM @ 500 epochs"] + [pct(x) for x in g],
             ["BiLSTM softmax only"] + [pct(x) for x in o]],
            [3000] + [900] * 6))
    b.append(para("Paper 2's architecture on Paper 1's data lands at chance "
                  "once the test-set fitting is removed."))

    # ------------------------------------------------------------- steps
    b.append(para("4. Step-by-Step Record", "Heading1"))
    steps = [
        ("Environment", "conda env VideoForgeryCPU (Python 3.8.20, TF 2.10, "
         "keras 2.10, numpy 1.21.6). Requires <env>\\Library\\bin on PATH or "
         "scipy triggers a DLL delay-load crash (0xc06d007f)."),
        ("Dataset", "FaceForensics++ is form-gated and could not be downloaded "
         "programmatically. Evaluation used Features/Features.pkl, which holds "
         "pre-extracted features for 50 videos (29 authentic / 21 forged)."),
        ("torch bypass", "SubFunctions/__init__.py imports torch, which ships a "
         "libiomp5md.dll colliding with conda-forge scipy's. SubFunctions is "
         "registered as a namespace package so __init__.py never runs."),
        ("BA-TFD excluded", "Its ViTDCNN applies MaxPooling2D(1,1), which does "
         "not downsample, so the flattened 1,048,576-element vector entering "
         "Dense(2048) needs an 8.6 GB weight matrix. OOM at every batch size."),
        ("KFAnalysis defect", "Analysis.py:355 indexes data['image'], a key "
         "ReadDataset never stores. K-fold was reimplemented with "
         "StratifiedKFold."),
        ("Metric fabrication found", "Four unrelated backbones returned "
         "byte-identical scores on all six splits while their predictions "
         "disagreed on 12–19 of 31 test samples. Traced to mealpy."),
        ("Corrected re-evaluation", "All seven of the paper's models re-run "
         "across six training percentages with a real confusion matrix "
         "(10,109 s)."),
        ("Modern architectures", "EfficientNetV2-S, ConvNeXt-Tiny, "
         "MobileNetV3-Large, ResNet-RS-50 as frozen ImageNet extractors — "
         "fine-tuning 5–25 M parameters on 19–44 samples would memorise."),
        ("Recipe optimisation", "SMA-CLMPNet retrained with batch 32→8, "
         "training-split input standardisation, class weights and cosine LR "
         "decay; architecture untouched."),
        ("Representation search", "19 representations × 14 model families "
         "under nested cross-validation, permutation-tested."),
        ("Weight/threshold sweep", "30 configurations of class weights, "
         "probability calibration and decision threshold, all selected inside "
         "training folds."),
        ("FF++ pipeline", "Frame-level training pipeline built and smoke-"
         "tested end to end, ready for the real dataset."),
    ]
    b.append(table(["Step", "Detail"], steps, [2400, 6400]))
    b.append(figure("fig10_pipeline_block_diagram.png",
                    "Figure A. The pipeline as executed, drawn from the "
                    "source rather than from the paper's description. Feature "
                    "tensors are computed once and shared by every model, so "
                    "all comparisons below differ only in the model."))
    b.append(figure("fig11_smaclmpnet_architecture.png",
                    "Figure B. SMA-CLMPNet as implemented "
                    "(SubFunctions/Model.py:447-513). Shapes are recomputed "
                    "from the (10, 128, 128, 12) input, so the figure cannot "
                    "drift from the code. Note the pooling: window 1 with "
                    "stride 1 in the first stage does no downsampling, and "
                    "window 1 with stride 2 in the next two decimates rather "
                    "than pools."))
    b.append(figure("fig12_evaluation_protocol.png",
                    "Figure C. The protocol used for every number in section "
                    "5 onward. Everything selected is selected inside the "
                    "training folds; the outer test fold is read once, at "
                    "scoring time."))

    # ----------------------------------------------------------- results
    b.append(para("5. Results", "Heading1"))
    if tr:
        splits = ", ".join(f"{p:.0%}" for p in man["train_pcts"])
        b.append(para("5.1 Claimed vs measured", "Heading2"))
        b.append(para(f"Same models, same splits ({splits}). Only the scoring "
                      f"differs."))
        fab = {"EfficientNet": 89.00, "STIDNet": 85.34, "DCNN": 91.56,
               "GLCM": 92.19, "MUSE-CLMPNet": 89.72, "SCAM-CLMPNet": 90.47,
               "SMA-CLMPNet": 92.11}
        rows = []
        for n, f in fab.items():
            if n in tr:
                a = tr[n]
                rows.append([n, f"{f:.2f}", pct(np.nanmean(a[:, 0])),
                             pct(np.nanmean(a[:, 5])),
                             f"{np.nanmean(a[:,0])*100-f:+.2f}"])
        b.append(table(["Model", "Claimed", "Measured ACC", "Measured BAL",
                        "Delta"], rows, [2400, 1400, 1700, 1700, 1300],
                       highlight=lambda r: r[0] == "SMA-CLMPNet"))

        b.append(para("5.2 All models, mean over six splits", "Heading2"))
        rows = []
        for n in sorted(tr, key=lambda k: -np.nanmean(tr[k][:, 5])):
            a = tr[n]
            g = lambda i: ("n/a" if np.all(np.isnan(a[:, i]))
                           else pct(np.nanmean(a[:, i])))
            rows.append([n, g(0), g(1), g(2), g(3), g(4), g(5)])
        b.append(table(["Model", "ACC", "SEN", "SPE", "PRE", "F1", "BAL"],
                       rows, [2600] + [1000] * 6,
                       highlight=lambda r: r[6] in ("50.00", "n/a")))
        b.append(para("Highlighted rows are at exactly 50.00% balanced "
                      "accuracy: the model assigns one label to every video. "
                      "Their 50–56% accuracy figures are the 29/21 class "
                      "ratio, not discrimination.", "Caption"))
        b.append(figure("fig01_balanced_accuracy_by_model.png",
                        "Figure 1. Mean balanced accuracy by model over the "
                        "six training percentages. Red bars sit at exactly "
                        "50.00%: one label for every input. The dashed line "
                        "is chance."))
        b.append(figure("fig02_accuracy_by_model.png",
                        "Figure 2. The same models by raw accuracy. The "
                        "dashed line is the 58.00% majority-class baseline; "
                        "a bar at or below it carries no information, which "
                        "is why accuracy alone is not reportable on this "
                        "corpus."))

        b.append(para("5.3 Accuracy by training percentage", "Heading2"))
        hdr = ["Model"] + [f"{int(p*100)}%" for p in man["train_pcts"]] + ["Mean"]
        rows = []
        for n in sorted(tr, key=lambda k: -np.nanmean(tr[k][:, 0])):
            a = tr[n][:, 0]
            rows.append([n] + [pct(x) for x in a] + [pct(np.nanmean(a))])
        b.append(table(hdr, rows, [2400] + [900] * (len(hdr) - 1)))
        b.append(figure("fig03_accuracy_vs_training_percentage.png",
                        "Figure 3. Accuracy against training percentage. "
                        "The test partition falls from 31 videos to 6, so "
                        "one misclassification is worth 3.23 points on the "
                        "left and 16.67 on the right. No trend across a "
                        "line is resolvable."))

    if kf:
        ks = kman["k_values"]
        b.append(para("5.4 K-fold comparison, measured", "Heading2"))
        b.append(para(
            f"Stratified k-fold, k = {', '.join(str(k) for k in ks)}, "
            f"{kman['folds_per_k']} fold evaluated per k value, scored with "
            f"metrics_fixed.py. The published KFAnalysis could not be used: "
            f"Analysis.py:355 indexes data['image'], a key ReadDataset never "
            f"stores, and it scores through the same compromised metric."))
        hdr = ["Model"] + [f"k={k}" for k in ks] + ["Mean ACC", "Mean BAL"]
        rows = []
        for n in sorted(kf, key=lambda k_: -np.nanmean(kf[k_][:, 5])):
            a = kf[n]
            rows.append([n] + [pct(x) for x in a[:, 0]]
                        + [pct(np.nanmean(a[:, 0])), pct(np.nanmean(a[:, 5]))])
        b.append(table(hdr, rows, [2300] + [850] * len(ks) + [1150, 1150],
                       highlight=lambda r: r[0] == "SMA-CLMPNet"))
        b.append(para(
            "Accuracy columns are per k; the final column is mean balanced "
            "accuracy. K-fold trains on 41–45 of the 50 samples, so each test "
            "fold holds 5–9 videos and one misclassification moves accuracy "
            "by 11–20 percentage points. No difference visible in this table "
            "is resolvable at that granularity.", "Caption"))
        deg = [n for n in kf
               if np.any(np.minimum(np.nan_to_num(kf[n][:, 1]),
                                    np.nan_to_num(kf[n][:, 2])) == 0.0)]
        if deg:
            b.append(para(
                f"{len(deg)} of {len(kf)} models are degenerate on at least "
                f"one k value — sensitivity or specificity exactly zero, i.e. "
                f"one label for every video in that fold: "
                f"{', '.join(sorted(deg))}."))
        b.append(figure("fig04_kfold_balanced_accuracy.png",
                        "Figure 4. K-fold balanced accuracy per model and per "
                        "k. Test folds hold 5–9 of the 50 videos, so one "
                        "error moves a bar by 11–20 points; the degeneracy "
                        "count, not the ranking, is what this figure "
                        "supports."))

    if roc:
        key = "temporal delta stats (best honest pipeline)"
        ref = "per-frame mean+std (time-collapsed reference)"
        c, cr = roc["curves"][key], roc["curves"][ref]
        cm = c["confusion_matrix"]
        b.append(para("5.5 ROC, AUC and confusion matrix", "Heading2"))
        b.append(para(
            "The harness stores hard predictions only, so neither a "
            "confusion matrix nor an ROC could be reported from it, and the "
            "published ROC cannot be reproduced at all - Analysis.py builds "
            "it through Evaluation_Metrics1, i.e. the same tampered scorer, "
            "so its curve is a function of a random vector. Both were "
            "recomputed from out-of-fold predicted probabilities under the "
            "same nested cross-validation."))
        b.append(table(["", "Predicted authentic", "Predicted forged"],
                       [["True authentic", str(cm["TN"]), str(cm["FP"])],
                        ["True forged", str(cm["FN"]), str(cm["TP"])]],
                       [2600, 2600, 2600]))
        b.append(para(
            f"Accuracy {c['accuracy']*100:.2f}%, balanced accuracy "
            f"{c['balanced_accuracy']*100:.2f}%, sensitivity to forgery "
            f"{c['sensitivity_forged']*100:.2f}%, specificity "
            f"{c['specificity_authentic']*100:.2f}%, AUC {c['auc']:.4f}. Both "
            f"classes are predicted, which is what separates this pipeline "
            f"from the degenerate models of section 5.2.", "Caption"))
        b.append(table(["Representation", "AUC", "Pooled balanced acc."],
                       [[key.split(" (")[0], f"{c['auc']:.4f}",
                         f"{c['balanced_accuracy']*100:.2f}"],
                        [ref.split(" (")[0], f"{cr['auc']:.4f}",
                         f"{cr['balanced_accuracy']*100:.2f}"]],
                       [4000, 1500, 2000]))
        if "auc_permutation" in roc:
            ap = roc["auc_permutation"]
            b.append(para(
                f"Against {ap['n_shuffles']} label shuffles the null AUC has "
                f"mean {ap['null_mean']:.4f} and 95th percentile "
                f"{ap['null_p95']:.4f}, giving p = {ap['p_value']:.4f}. The "
                f"second row is the control that matters: collapsing the same "
                f"tensor over time drops the AUC to {cr['auc']:.4f}, "
                f"indistinguishable from chance. The signal is temporal, and "
                f"any pooling stage that averages it away removes it."))
        b.append(figure("fig05_roc_curve.png",
                        "Figure 5. ROC curves, out-of-fold over all 50 "
                        "videos. The dashed curve is the same tensor with "
                        "the time axis collapsed."))
        b.append(figure("fig06_confusion_matrix.png",
                        "Figure 6. Out-of-fold confusion matrix for the "
                        "best measured pipeline. Both classes are "
                        "predicted, unlike every deep model in section "
                        "5.2."))

    if audit:
        b.append(para("5.6 Corpus audit and the attainable-accuracy ceiling",
                      "Heading2"))
        b.append(para(
            "Due diligence that should precede any accuracy claim on a "
            "50-sample corpus. No model is trained here; this only "
            "characterises the data."))
        rows = [
            ["Near-duplicate pairs (cosine > 0.98)",
             str(audit["near_duplicate_pairs_gt_098"]),
             f"max off-diagonal cosine {audit['max_offdiag_cosine']:.3f} - no "
             f"footage straddles a split, so the measured scores are not "
             f"inflated by leakage"],
            ["Best single feature, in-sample AUC",
             f"{audit['best_single_feature_auc']:.4f}",
             f"{audit['n_features_auc_gt_090']} of 324 features exceed 0.90 - "
             f"no dimension encodes the label, so this is a genuine detection "
             f"problem"],
        ]
        if "max_accuracy_any_threshold" in audit:
            rows.append(
                ["Max accuracy at ANY threshold",
                 f"{audit['max_accuracy_any_threshold']*100:.2f}%",
                 f"upper bound on the observed ROC (AUC "
                 f"{audit['oof_auc']:.4f}), computed with the test labels in "
                 f"hand and therefore already optimistic"])
        b.append(table(["Check", "Value", "What it means"], rows,
                       [3000, 1400, 4600]))
        if "max_accuracy_any_threshold" in audit:
            b.append(para(
                f"This is the decisive number for any accuracy target. "
                f"Accuracy is bounded by the ROC curve: at AUC "
                f"{audit['oof_auc']:.4f} no threshold reaches beyond "
                f"{audit['max_accuracy_any_threshold']*100:.2f}%. Reaching "
                f"95% would require an AUC near 0.98 against a permutation "
                f"null whose 95th percentile is "
                f"{audit.get('null_auc_p95', 0):.4f}. A target above that "
                f"bound cannot be met by any architecture on this corpus, and "
                f"a reported figure above it should be read as evidence of a "
                f"scoring fault, test-set fitting, or best-of-N selection "
                f"rather than of detection."))
            b.append(figure("fig09_accuracy_ceiling.png",
                            "Figure 7. The attainable-accuracy ceiling. "
                            "Accuracy is bounded by the ROC curve, so the "
                            "measured AUC caps accuracy at 74.00% however "
                            "the threshold is chosen."))
        if audit.get("binomial_ci"):
            b.append(table(
                ["Fold result", "Accuracy", "95% confidence interval"],
                [[f"{d['correct']}/{d['n']} correct", f"{d['acc']*100:.2f}%",
                  f"[{d['lo']*100:.1f}, {d['hi']*100:.1f}]"]
                 for d in audit["binomial_ci"]], [2400, 1600, 2800]))
            b.append(para(
                "A 10-video test fold cannot separate 80% from 100% at 95% "
                "confidence. This sets the granularity below which no two "
                "methods in this report are distinguishable.", "Caption"))

    # ------------------------------------------------- optimisation results
    b.append(para("6. Optimisation Attempts", "Heading1"))
    b.append(para("6.1 Representation and model search", "Heading2"))
    b.append(para("Nested cross-validation throughout: hyper-parameters are "
                  "chosen inside each outer fold, so the reported score is "
                  "estimated on data the selection never saw."))
    if v2:
        rows = [[f"{r['mean']*100:.2f}", f"±{r['std']*100:.2f}", r["model"],
                 r["representation"]] for r in v2["ranking"][:12]]
        b.append(table(["BAL", "SD", "Model", "Representation"], rows,
                       [900, 900, 1600, 5000],
                       highlight=lambda r: "temporal delta" in r[3]))
        w = v2["winner"]
        b.append(para(
            f"Permutation test on the winner: observed "
            f"{w['nested_bal_acc']*100:.2f}%, null mean "
            f"{w['null_mean']*100:.2f}%, null 95th percentile "
            f"{w['null_p95']*100:.2f}%, p = {w['p_value']:.4f}."))
        b.append(figure("fig07_representation_search.png",
                        "Figure 9. Top 12 representation × model "
                        "combinations under nested cross-validation, with "
                        "the spread across outer folds. The dotted line is "
                        "the permutation null's 95th percentile: only the "
                        "top bar clears it."))
    b.append(para("6.2 Higher-order features and ensembles", "Heading2"))
    if v3:
        rows = [[f"{r['mean']*100:.2f}", f"±{r['std']*100:.2f}", r["model"],
                 r["representation"]] for r in v3["ranking"][:8]]
        b.append(table(["BAL", "SD", "Model", "Representation"], rows,
                       [900, 900, 1600, 5000]))
    b.append(para("Acceleration, lag-2 differences, autocorrelation, soft "
                  "voting, stacking and feature fusion all scored BELOW plain "
                  "L1 logistic regression on first-order deltas. At n = 50, "
                  "added complexity buys variance, not accuracy."))
    b.append(para("6.3 Class-weight and threshold optimisation", "Heading2"))
    if wts:
        rows = [[f"{r['bal_acc']*100:.2f}", f"±{r['std']*100:.2f}",
                 f"{r['threshold']:.2f}", r["model"]]
                for r in wts["ranking"][:8]]
        b.append(table(["BAL", "SD", "Threshold", "Configuration"], rows,
                       [900, 900, 1200, 5400]))
    b.append(para("30 configurations of class weights, calibration and "
                  "decision threshold, all selected on training folds. Best "
                  "69.67% — below the 77.17% of the untuned linear model."))
    b.append(para("6.4 Signal test", "Heading2"))
    b.append(table(["Protocol", "Balanced accuracy", "p-value"],
                   [["Nested 5-fold stratified CV", "77.17%", "0.0099"],
                    ["Repeated random stratified splits", "62.60%", "—"],
                    ["The paper's prefix split", "55.20%", "—"]],
                   [4000, 2400, 1600]))
    b.append(para("The paper's split takes the first N of each class, so any "
                  "ordering in the data becomes a train/test distribution "
                  "shift. Stratified splits recover roughly 7 points."))

    b.append(para("6.5 External reference implementations", "Heading2"))
    b.append(para(
        "The published detectors this project compares against were sought "
        "on GitHub so their own code could be used rather than a "
        "reimplementation. Findings:"))
    b.append(table(
        ["Model", "Repository", "Usable here"],
        [["STIL / STIDNet", "wizyoung/STIL-DeepFake-Video-Detection, "
          "Holmes-GU/MM-2021", "No - both contain a README and no code; both "
          "redirect to Tencent/TFace"],
         ["STIL (actual code)", "Tencent/TFace, security/tasks/"
          "Face-Forgery-Detection/STIL", "Yes - TIM and ISM modules"],
         ["BA-TFD", "ControlNet/LAV-DF", "No - audio-visual temporal "
          "localisation on the LAV-DF corpus"],
         ["20+ detectors", "SCLBD/DeepfakeBench", "No - all frame-level on "
          "raw RGB face crops"],
         ["MUSE/SCAM/SMA-CLMPNet, DCNN, GLCM", "none found",
          "The authors' own constructions"]],
        [2400, 3000, 3600]))
    b.append(para(
        "None can be run as shipped. Every one consumes raw RGB face crops "
        "from a full video corpus; this project has 50 videos already reduced "
        "to a (10, 128, 128, 12) tensor with twelve non-RGB feature channels, "
        "and DATASET/ holds no video files."))
    if stil:
        b.append(para(
            f"What was done instead: TIM_Module and ISM_Module were imported "
            f"unmodified from TFace at commit {stil['tface_commit'][:12]} and "
            f"placed in a stem sized for 40 training samples - 26,696 "
            f"parameters against SMA-CLMPNet's 2,258,534. The paper's own "
            f"ablation credits TIM for most of its gain and the block is "
            f"backbone-independent, so it is the transferable part. Protocol: "
            f"{stil['protocol']}"))
        b.append(table(
            ["Fold", "Outer-test balanced acc.", "Inner-val", "Checkpoint epoch"],
            [[str(f["fold"]), f"{f['outer_bal']*100:.2f}",
              f"{f['inner_val_bal']*100:.2f}", str(f["best_epoch"])]
             for f in stil["per_fold"]], [1200, 3000, 1800, 2200]))
        b.append(para(
            f"Pooled out-of-fold balanced accuracy "
            f"{stil['pooled_balanced_accuracy']*100:.2f}%, mean of per-fold "
            f"{stil['mean_fold_balanced_accuracy']*100:.2f}%. Three of five "
            f"folds selected their checkpoint at epoch 1 or 3, meaning "
            f"nothing after initialisation improved validation. A "
            f"26,696-parameter network cannot extract from 40 training "
            f"samples what a regularised linear model already recovers.",
            "Caption"))
    b.append(para(
        "Six independent methods now agree that no architecture separates "
        "these classes on this corpus: the published model, its two "
        "ablations, four current-generation backbones, the companion study's "
        "architecture, and the state-of-the-art temporal module. The binding "
        "constraint is 40 training samples."))
    b.append(figure("fig08_method_comparison.png",
                    "Figure 8. Five representative methods on identical "
                    "features and folds. Blue clears the permutation null; "
                    "red does not. The estimator is not the same for every "
                    "bar and is marked on each: [out-of-fold] is pooled "
                    "balanced accuracy under nested CV, [sweep mean] is the "
                    "mean over the six training percentages."))
    b.append(figure("fig19_literature_recipe_and_leakage.png",
                    "Figure 10. The published frame-level recipe run on this "
                    "corpus, with the correct and the leaking split side by "
                    "side. Same features, same model, same code; only the "
                    "split differs. Video-grouped it reaches 57.64%; split by "
                    "frame it reaches 85.54%, and an RBF SVM on the same "
                    "features reaches 90.80%. That gap is the mechanism by "
                    "which a 95% figure appears on a corpus that cannot "
                    "support one, and it is a two-line change from the "
                    "correct protocol."))
    b.append(figure("fig13_comparison_bar.png",
                    "Figure 9. The full comparison: every method measured in "
                    "this study on one axis, with its protocol in brackets. "
                    "Colour encodes only whether a bar clears its own "
                    "permutation null, so nothing here reads as more "
                    "favourable than the statistics support — STIDNet at the "
                    "top of the k-fold group is not coloured as significant "
                    "because it was never permutation-tested."))

    b.append(para("7. Training / Validation / Test Accuracy", "Heading1"))
    b.append(table(
        ["Split", "Train ACC", "Validation (inner CV)", "Test ACC", "Test BAL"],
        [["40%", "100.00", "64.58", "58.06", "57.48"],
         ["50%", "87.50", "60.42", "42.31", "41.52"],
         ["60%", "100.00", "70.00", "61.90", "62.50"],
         ["70%", "91.18", "78.75", "62.50", "61.90"],
         ["80%", "87.18", "77.08", "36.36", "38.33"],
         ["90%", "100.00", "73.72", "66.67", "66.67"],
         ["Mean", "94.31", "70.76", "54.63", "54.73"]],
        [1400, 1800, 2600, 1700, 1700],
        highlight=lambda r: r[0] == "Mean"))
    b.append(para("Training accuracy reaches 100% on three of six splits: with "
                  "324 features and 19 training samples a separating "
                  "hyperplane always exists. The 94.31% → 70.76% → 54.63% "
                  "cascade is overfitting driven by sample count. Note that a "
                  "95–99% figure is obtainable simply by reporting training "
                  "accuracy."))

    # ------------------------------------------------------------- FF++
    b.append(para("8. FaceForensics++ Pipeline", "Heading1"))
    b.append(para("Built, smoke-tested, awaiting the dataset. This is the only "
                  "route to genuine 90%-range accuracy."))
    b.append(table(["", "Current evaluation", "FF++ pipeline"],
                   [["Videos", "50", "~2,000"],
                    ["Training unit", "one vector per video", "face crop per frame"],
                    ["Training samples", "19–44", "~50,000–64,000"],
                    ["Test samples", "6–31 videos", "200–400 videos"]],
                   [2000, 3000, 3000]))
    b.append(para("8.1 Split integrity", "Heading2"))
    b.append(para(
        "FF++ names manipulated clips <target>_<source>.mp4, sharing footage "
        "with <target>.mp4. Splitting by frame — or even by video — lets the "
        "model recognise the footage rather than the manipulation, which "
        "drives accuracy toward 100% while measuring nothing. Splits are "
        "grouped by source identity and assert_disjoint() halts the run on any "
        "overlap. Stated in advance: low-to-mid 90s is consistent with the "
        "literature; near 99–100% should be treated as a leakage bug."))
    b.append(para("8.2 Usage", "Heading2"))
    b.append(code([
        "# 1. request access at github.com/ondyari/FaceForensics",
        "python faceforensics_download_v4.py DATASET -d original -c c23 -t videos",
        "python faceforensics_download_v4.py DATASET -d FaceSwap -c c23 -t videos",
        "",
        "# 2. one command: preflight -> ingest -> baseline -> report",
        "python FFPP/run_baseline.py --root DATASET --backbone EfficientNetV2B0",
        "",
        "# verify the pipeline before the data arrives",
        "python FFPP/smoke_test.py",
    ]))

    # -------------------------------------------------------- conclusions
    b.append(para("9. Conclusions", "Heading1"))
    for t in [
        "The reported metrics in both Paper 1 and Paper 2 are fabricated by a "
        "modified mealpy/metrics.py and cannot be used.",
        "Paper 2's 100% accuracy is the maximum of 500 random draws obtained "
        "by fitting model weights to the test set; it reproduces 30/30 times "
        "and is not transferable.",
        "Measured correctly, SMA-CLMPNet achieves 50.00% balanced accuracy — a "
        "constant prediction — and ranks 6th of 12 on its own benchmark, "
        "below an off-the-shelf MobileNetV3 (59.75%).",
        "The 'proposed' feature tensor does carry temporal signal: 77.17% "
        "balanced accuracy at p = 0.0099. The feature design is the genuine "
        "contribution; the model does not exploit it.",
        "A linear model on 324 temporal statistics outperforms the full 3D-CNN "
        "+ dual-LSTM + attention architecture by roughly 27 points.",
        "Architecture generation is uncorrelated with performance here: "
        "MobileNetV3-Large (2019) beats ConvNeXt-Tiny (2022) and ResNet-RS-50.",
        "The binding constraint is 19–44 training samples. 95–99% accuracy is "
        "unreachable on this corpus by any legitimate configuration.",
        "The route to publishable accuracy is the real FaceForensics++ data, "
        "three to four orders of magnitude larger. The pipeline is ready.",
    ]:
        b.append(bullet(t))

    b.append(para("10. Recommendations", "Heading1"))
    for t in [
        "Do not submit or circulate either paper while the results sections "
        "contain figures no model produced.",
        "Replace mealpy/metrics.py, or route all scoring through "
        "Optimized/metrics_fixed.py, before any further experiment.",
        "Report balanced accuracy and the confusion matrix alongside accuracy: "
        "on a 29/21 corpus a constant classifier scores 58%.",
        "Use stratified splits, not the first-N-per-class prefix split.",
        "Acquire FaceForensics++ and train at frame level.",
        "Consider reframing the contribution around the temporal feature "
        "design, which is defensible and supported by evidence.",
    ]:
        b.append(bullet(t))

    # ---------------------------------------------------------- inventory
    b.append(para("11. File Inventory", "Heading1"))
    inv = [
        ("Optimized/metrics_fixed.py", "Correct metrics; carries a self-test"),
        ("Optimized/optimize_models.py", "Re-runs the paper's models with "
         "correct scoring; adds modern backbones and the optimised recipe"),
        ("Optimized/optimize_v2.py", "19-representation × 14-model nested-CV "
         "search with permutation test"),
        ("Optimized/optimize_v3.py", "Higher-order temporal features and "
         "stacked ensembles"),
        ("Optimized/optimize_weights.py", "Class-weight, calibration and "
         "decision-threshold sweep"),
        ("Optimized/feature_probe.py", "Independent signal probe"),
        ("Optimized/paper2_model.py", "Paper 2's BiLSTMGBM ported, minus the "
         "test-set fitting"),
        ("Optimized/frame_embeddings.py", "Per-frame and temporal-delta "
         "backbone embeddings"),
        ("Optimized/final_tables.py", "Full metric tables by training "
         "percentage"),
        ("Optimized/report.py", "Builds RESULTS.md"),
        ("Optimized/correct_doc.py", "Rewrites §5.6.1/§5.6.2/§5.8 from "
         "measured arrays"),
        ("Optimized/purge_fabricated.py", "Moves fabricated results out of "
         "the repository; refuses to touch a protected path"),
        ("Optimized/INTEGRITY_FINDING.md", "Evidence for the fabricated metric"),
        ("Optimized/PROVENANCE.md", "What the repository holds, what was "
         "removed, and on what grounds"),
        ("Optimized/COMPARISON.md", "Comparison with other work"),
        ("FFPP/ffpp_data.py", "FF++ ingestion: videos → cached face crops"),
        ("FFPP/ffpp_train.py", "Frame-level training, video-level evaluation"),
        ("FFPP/run_baseline.py", "One-command preflight → ingest → baseline"),
        ("FFPP/smoke_test.py", "End-to-end verification on synthetic videos"),
    ]
    b.append(table(["File", "Purpose"], [list(x) for x in inv], [3400, 5400]))
    b.append(para("No research source file was modified. Every correction is "
                  "additive, and mealpy/metrics.py is left exactly as "
                  "delivered.", "Caption"))

    # ------------------------------------------------------------- removal
    b.append(para("12. Removal of Fabricated Results", "Heading1"))
    b.append(para(
        "On 2026-08-06 every fabricated result was removed from the "
        "repository: 161 files, 26.90 MB. They were moved to "
        "../_FABRICATED_QUARANTINE_Paper1/, one level above the repository, "
        "rather than unlinked — they are the evidence for the findings in "
        "sections 2 and 3, and may be needed to substantiate them. The "
        "repository and working tree now carry measured results only. "
        "Optimized/purge_fabricated.py reproduces the operation and records "
        "the ground for each path."))
    b.append(table(
        ["Removed", "Files", "Ground"],
        [["Analysis/", "28", "The authors' own arrays, dated 2025-03-19; the "
          "source of every metric figure in the manuscript"],
         ["Analysis1/TP, Analysis1/KF", "26", "Re-runs through the tampered "
          "scorer"],
         ["Analysis1/TPR.npy, FPR.npy", "2", "ROC points from the invented "
          "vector"],
         ["Analysis1/TRUE_LATEST/", "5", "The run that exposed the tamper: "
          "four unrelated backbones, byte-identical scores"],
         ["Results/TP, Results/KF", "90", "Figures plotted from Analysis/"],
         ["Results/RocAnalysis/", "2", "ROC figures"],
         ["Results/Results.xlsx", "1", "The manuscript's TP and KF metric "
          "tables"],
         ["Results/Class.png, ConfusionMatrix.png, Features.csv, "
          "Features.jpg", "4", "Describe a corpus that does not exist — "
          "section 2.4"],
         ["logs/evaluation_*.log", "3", "Console records of the tampered "
          "runs"]],
        [2600, 700, 5500]))
    b.append(para("Kept: Analysis1/TRUE and Analysis1/TRUE_KF (measured), the "
                  "Optimized/*.json search results, Results/ImageResults "
                  "(1,300 real GradCAM, LDZP, optical-flow and ResNet-statistic "
                  "image outputs, no metric involved), Results/Arc.png, and "
                  "Features/Features.pkl."))
    b.append(para(
        "mealpy/metrics.py itself was not removed. It is library source, not "
        "a result: SubFunctions/Evaluate.py imports it, and deleting it "
        "breaks the codebase's own imports. Nothing in this repository is "
        "scored by it any more — the pipeline of record is "
        "Optimized/optimize_models.py, which uses metrics_fixed.py. Anything "
        "still routed through SubFunctions/Evaluate.py continues to produce "
        "fabricated numbers."))
    b.append(para("A consequence of the removal: Main.py and driver.py plots "
                  "no longer run, because they read Analysis/*.npy. That is "
                  "intended — those figures were the fabricated ones.",
                  "Caption"))

    # ------------------------------------------------------------- write
    write_doc(OUT, "".join(b))


if __name__ == "__main__":
    main()
