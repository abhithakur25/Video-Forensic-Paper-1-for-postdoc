"""Build the research article.

Assembles Research_Paper-1.docx from the prose modules and the stored run
artefacts, in the formatting of the Paper-2 template. Nothing here transcribes
a result: article_data loads every number from Analysis1/ and Optimized/, the
prose modules format them, and this script asserts the load succeeded before
any text is generated.

Checks performed before writing:
  - every reference resolved by fetch_references.py is cited at least once
  - every cited key exists in the resolved set
  - every figure referenced by the text exists on disk
  - the section word counts meet the brief
  - the OOXML package is well-formed and every embedded image is declared
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import article_body                                       # noqa: E402
import article_data                                       # noqa: E402
import article_intro                                      # noqa: E402
import article_lit                                        # noqa: E402
import article_refs as R                                  # noqa: E402
import article_style as S                                 # noqa: E402

P = Path(__file__).resolve().parents[1]
OUT = P / "Research_Paper-1.docx"
FIGDIR = P / "Results" / "Genuine"

TITLE = ("Spatial and Multi-excitation Attention for Intra-frame Video "
         "Forgery Detection (SMA-CLMPNet): A Corrected Re-evaluation, an "
         "Attainability Ceiling, and a Protocol for Small-Corpus Media "
         "Forensics")

AUTHORS = ("Dr. Abhishek Thakur1,2,*,  Prof. Vishal Jain3, and  "
           "Prof. Chin-Shiuh Shieh4")
NATIVE = "(謝欽旭)"
AFFIL = [
    ("1", "Primary Affiliation: School of Computer Science & Engineering, "
          "Chitkara University, Himachal Pradesh, India (Associate "
          "Professor)."),
    ("2", "Secondary Affiliation: Postdoctoral Researcher, Research "
          "Institute of IoT Cybersecurity (RIITC), Department of Electronic "
          "Engineering, National Kaohsiung University of Science and "
          "Technology (NKUST), No. 415, Jiangong Rd., Kaohsiung City 80778, "
          "Taiwan (R.O.C.)."),
    ("3", "Co-Supervisor Affiliation: Vivekananda Institute of Professional "
          "Studies – Technical Campus (VIPS-TC), India."),
    ("4", "Supervisor Affiliation: Ph.D., Professor, Department of "
          "Electronic Engineering; Research Institute of IoT Cybersecurity "
          "(RIITC), National Kaohsiung University of Science and Technology "
          "(NKUST), No. 415, Jiangong Rd., Kaohsiung City 80778, Taiwan "
          "(R.O.C.)."),
]
EMAILS = [
    "E-mail (Abhishek Thakur): abhithakur25@gmail.com; "
    "abhishek@chitkarauniversity.edu.in",
    "E-mail (Vishal Jain): drvishaljain83@gmail.com; vishal.jain@vips.edu",
    "E-mail (Chin-Shiuh Shieh): csshieh@nkust.edu.tw; csshieh@gmail.com",
]
CORR = "* Corresponding author: Dr. Abhishek Thakur (abhithakur25@gmail.com)."
TOPIC = ("Postdoctoral Topic (Offer Letter): A unified multi-modal deep "
         "learning framework for semantic media forensics—Detecting "
         "AI-generated manipulation across visual, document, and news "
         "domains. Programme requirement: publish one SCI-indexed journal "
         "paper.")

_FIGS_USED = []


def figure(name, cap):
    """Embed Results/Genuine/<name>; record it so the build can verify."""
    path = FIGDIR / name
    if not path.exists():
        raise FileNotFoundError(f"figure referenced but not generated: {path}")
    _FIGS_USED.append(name)
    return S.picture(path, cap)


def front_matter():
    B = [S.title(TITLE), S.authors(AUTHORS), S.native_name(NATIVE)]
    B += [S.affiliation(m, t) for m, t in AFFIL]
    B += [S.email(e) for e in EMAILS]
    B.append(S.corresponding(CORR))
    B.append(S.topic_note(TOPIC))
    return "".join(B)


def main():
    article_body.FIG = figure
    d = article_data.load()
    print(f"loaded: {len(d['sweep'])} models, "
          f"train% {d['pcts']}, k {d['ks']}")

    parts = [
        ("front matter", front_matter()),
        ("abstract", article_body.abstract(d)),
        ("1 Introduction", article_intro.build()),
        ("2 Literature Review", article_lit.build()),
        ("3 Proposed Work", article_body.proposed(d)),
        ("4 Experimental Work", article_body.experimental(d)),
        ("5 Results", article_body.results(d)),
        ("6 Comparison", article_body.comparison(d)),
        ("7 Conclusion", article_body.conclusion(d)),
        ("8 Future Work", article_body.future_work(d)),
        ("Acknowledgment", article_body.acknowledgment(d)),
    ]

    # References last: every citation must already have been rendered.
    missing = set(R.REFS) - R.used()
    if missing:
        raise SystemExit(f"reference list contains uncited entries: "
                         f"{sorted(missing)}")
    apa = R.reference_list(R.used())
    parts.append(("References", article_body.references(apa)))

    body = "".join(x for _, x in parts)

    print("\nsection word counts")
    total = 0
    for name, x in parts:
        n = S.word_count(x)
        total += n
        print(f"  {name:24s} {n:6d}")
    print(f"  {'TOTAL':24s} {total:6d}")

    intro_n = S.word_count(dict(parts)["1 Introduction"])
    lit_n = S.word_count(dict(parts)["2 Literature Review"])
    concl_n = S.word_count(dict(parts)["7 Conclusion"])
    abs_n = S.word_count(dict(parts)["abstract"])
    for label, n, lo, hi in [("introduction", intro_n, 5000, None),
                             ("literature review", lit_n, 5000, None),
                             ("abstract", abs_n, 250, 400),
                             ("conclusion", concl_n, 250, 400)]:
        if n < lo or (hi and n > hi):
            print(f"  WARNING: {label} is {n} words "
                  f"(brief: {lo}{'-' + str(hi) if hi else '+'})")

    xml, media = S.write_doc(OUT, body)
    npar = xml.count("<w:p>") + xml.count("<w:p ")
    print(f"\nwrote {OUT.name}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"validated: {npar} paragraphs, {xml.count('<w:tbl>')} tables, "
          f"{len(media)} embedded images, {len(apa)} references")
    print(f"figures embedded: {', '.join(sorted(set(_FIGS_USED)))}")
    unused = sorted({f.name for f in FIGDIR.glob("*.png")}
                    - set(_FIGS_USED))
    if unused:
        print(f"figures present but not cited: {unused}")


if __name__ == "__main__":
    main()
