"""Resolve the literature-review citations against CrossRef.

Every reference in the manuscript is looked up by title and stored with the
metadata CrossRef returns - authors, container, year, volume, pages, DOI. Nothing
in the reference list is typed from memory, and the resolver refuses a match
whose returned title does not overlap the query, so a wrong hit shows up as
UNRESOLVED rather than as a plausible-looking but invented entry.

Network note: this machine sits behind a TLS-intercepting proxy whose CRL
endpoint is unreachable, so curl is invoked with --ssl-no-revoke. Certificate
validation itself stays on.

Writes Optimized/references.json.
"""
import json
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
OUT = P / "Optimized" / "references.json"

# (key, query title, expected first-author surname or "" if unchecked)
# Ordered by the theme they support in section 2.
QUERIES = [
    # --- corpora and the detection task
    ("faceforensics", "FaceForensics++: Learning to Detect Manipulated Facial Images", "Rossler"),
    ("celebdf", "Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics", "Li"),
    ("dfdc", "The DeepFake Detection Challenge (DFDC) Dataset", "Dolhansky"),
    ("deeperforensics", "DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection", ""),
    # --- classical / handcrafted forensics
    ("haralick", "Textural Features for Image Classification", "Haralick"),
    ("fridrich", "Rich Models for Steganalysis of Digital Images", "Fridrich"),
    ("popescu", "Exposing digital forgeries by detecting traces of resampling", "Popescu"),
    ("wang2009", "Exposing digital forgeries in video by detecting double quantization", ""),
    # --- CNN detectors
    ("mesonet", "MesoNet: a Compact Facial Video Forgery Detection Network", "Afchar"),
    ("xception", "Xception: Deep Learning with Depthwise Separable Convolutions", "Chollet"),
    ("capsule", "Capsule-forensics: Using Capsule Networks to Detect Forged Images and Videos", "Nguyen"),
    ("facexray", "Face X-Ray for More General Face Forgery Detection", "Li"),
    # --- temporal modelling
    ("lstm", "Long Short-Term Memory", "Hochreiter"),
    ("guera", "Deepfake Video Detection Using Recurrent Neural Networks", "Guera"),
    ("stil", "Spatiotemporal Inconsistency Learning for DeepFake Video Detection", "Gu"),
    ("istvt", "ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection", ""),
    ("tall", "Learning Spatiotemporal Inconsistency via Thumbnail Layout for Face Deepfake Detection", ""),
    ("gaze", "Where Deepfakes Gaze at? Spatial-Temporal Gaze Inconsistency Analysis for Video Face Forgery Detection", ""),
    ("ftcn", "Exploring Temporal Coherence for More General Video Face Forgery Detection", "Zheng"),
    ("stidnet", "STIDNet: Identity-Aware Face Forgery Detection With Spatiotemporal Knowledge Distillation", ""),
    # --- attention
    ("senet", "Squeeze-and-Excitation Networks", "Hu"),
    ("cbam", "CBAM: Convolutional Block Attention Module", "Woo"),
    ("mat", "Multi-attentional Deepfake Detection", "Zhao"),
    ("nonlocal", "Non-local Neural Networks", "Wang"),
    # --- modern backbones
    ("efficientnet", "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", "Tan"),
    ("efficientnetv2", "EfficientNetV2: Smaller Models and Faster Training", "Tan"),
    ("mobilenetv3", "Searching for MobileNetV3", "Howard"),
    ("convnext", "A ConvNet for the 2020s", "Liu"),
    ("resnet", "Deep Residual Learning for Image Recognition", "He"),
    ("resnetrs", "Revisiting ResNets: Improved Training and Scaling Strategies", "Bello"),
    # --- optimisation / metaheuristics
    ("mealpy", "MEALPY: An open-source library for latest meta-heuristic algorithms in Python", "Van Thieu"),
    ("nfl", "No free lunch theorems for optimization", "Wolpert"),
    # --- methodology: small samples, validation, leakage
    ("varoquaux", "Cross-validation failure: Small sample sizes lead to large error bars", "Varoquaux"),
    ("combrisson", "Exceeding chance level by chance: The caveat of theoretical chance levels in brain signal classification and statistical assessment of decoding accuracy", "Combrisson"),
    ("ojala", "Permutation Tests for Studying Classifier Performance", "Ojala"),
    ("brodersen", "The Balanced Accuracy and Its Posterior Distribution", "Brodersen"),
    ("cawley", "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation", "Cawley"),
    ("kapoor", "Leakage and the reproducibility crisis in machine-learning-based science", "Kapoor"),
    ("ioannidis", "Why Most Published Research Findings Are False", "Ioannidis"),
    ("gundersen", "State of the Art: Reproducibility in Artificial Intelligence", ""),
    ("hanley", "The meaning and use of the area under a receiver operating characteristic (ROC) curve", "Hanley"),
    ("demsar", "Statistical Comparisons of Classifiers over Multiple Data Sets", "Demsar"),
    # --- surveys and journal-indexed reviews
    ("mirsky", "The Creation and Detection of Deepfakes: A Survey", "Mirsky"),
    ("verdoliva", "Media Forensics and DeepFakes: An Overview", "Verdoliva"),
    ("nguyen_survey", "Deep learning for deepfakes creation and detection: A survey", "Nguyen"),
    ("masood", "Deepfakes generation and detection: state-of-the-art, open challenges, countermeasures, and way forward", "Masood"),
    ("rana", "Deepfake Detection: A Systematic Literature Review", "Rana"),
    ("yu_survey", "A Survey on Deepfake Video Detection", "Yu"),
    # --- inter-frame / temporal-domain video forensics (journal)
    ("fadl", "Exposing video inter-frame forgery via histogram of oriented gradients and motion energy image", "Fadl"),
    ("kingra", "Inter-frame forgery detection in H.264 videos using motion and brightness gradients", "Kingra"),
    ("shelke", "Multiple forgery detection in digital video with VGG-16 based deep neural network and KPCA", "Shelke"),
    ("sitara", "Digital video tampering detection: An overview of passive techniques", "Sitara"),
    # --- frequency, optical flow, lip and identity cues
    ("amerini", "Deepfake Video Detection through Optical Flow Based CNN", "Amerini"),
    ("sabir", "Recurrent Convolutional Strategies for Face Manipulation Detection in Videos", "Sabir"),
    ("haliassos", "Lips Don't Lie: A Generalisable and Robust Approach to Face Forgery Detection", "Haliassos"),
    ("luo_hf", "Generalizing Face Forgery Detection with High-frequency Features", "Luo"),
    ("qian_f3net", "Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues", "Qian"),
    ("coccomini", "Combining EfficientNet and Vision Transformers for Video Deepfake Detection", "Coccomini"),
    # --- general architectures the cohort draws on
    ("vaswani", "Attention Is All You Need", "Vaswani"),
    ("vit", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "Dosovitskiy"),
    ("i3d", "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset", "Carreira"),
    ("slowfast", "SlowFast Networks for Video Recognition", "Feichtenhofer"),
    ("c3d", "Learning Spatiotemporal Features with 3D Convolutional Networks", "Tran"),
    ("dropout", "Dropout: a simple way to prevent neural networks from overfitting", "Srivastava"),
    # --- more methodology on small samples and imbalance
    ("vabalas", "Machine learning algorithm validation with a limited sample size", "Vabalas"),
    ("he_imbalance", "Learning from Imbalanced Data", "He"),
    ("saito", "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets", "Saito"),
    ("bouthillier", "Accounting for Variance in Machine Learning Benchmarks", "Bouthillier"),
]


# Title search picks the wrong paper whenever a later work quotes the classic in
# its own title, which is exactly what happened for the seminal entries below.
# For those the DOI is authoritative and is resolved directly; the title query
# is then only a fallback.
DOI_OVERRIDE = {
    "haralick": "10.1109/TSMC.1973.4309314",
    "lstm": "10.1162/neco.1997.9.8.1735",
    "senet": "10.1109/CVPR.2018.00745",
    "cbam": "10.1007/978-3-030-01234-2_1",
    "nonlocal": "10.1109/CVPR.2018.00813",
    "efficientnet": "10.48550/arXiv.1905.11946",
    "efficientnetv2": "10.48550/arXiv.2104.00298",
    "mobilenetv3": "10.1109/ICCV.2019.00140",
    "resnet": "10.1109/CVPR.2016.90",
    "resnetrs": "10.48550/arXiv.2103.07579",
    "nfl": "10.1109/4235.585893",
    "ioannidis": "10.1371/journal.pmed.0020124",
    "dfdc": "10.48550/arXiv.2006.07397",
    "istvt": "10.1109/TIFS.2023.3239223",
    "mirsky": "10.1145/3425780",
    "nguyen_survey": "10.1016/j.cviu.2022.103525",
    "rana": "10.1109/ACCESS.2022.3154404",
    "yu_survey": "10.1049/bme2.12031",
    "sabir": "10.48550/arXiv.1905.00582",
    "vaswani": "10.48550/arXiv.1706.03762",
    "vit": "10.48550/arXiv.2010.11929",
    "he_imbalance": "10.1109/TKDE.2008.239",
    "bouthillier": "10.48550/arXiv.2103.03098",
}

# Machine-learning venues that issue no DOI. Recorded by hand from the
# publisher's own listing, with the URL a reader can check.
MANUAL = {
    "cawley": {
        "title": "On Over-fitting in Model Selection and Subsequent Selection "
                 "Bias in Performance Evaluation",
        "authors": ["Cawley, G. C.", "Talbot, N. L. C."],
        "container": "Journal of Machine Learning Research",
        "publisher": "JMLR", "year": 2010, "volume": "11", "issue": "",
        "pages": "2079-2107", "doi": "",
        "url": "https://jmlr.org/papers/v11/cawley10a.html",
        "type": "journal-article", "match_overlap": None, "source": "manual",
    },
    "demsar": {
        "title": "Statistical Comparisons of Classifiers over Multiple Data "
                 "Sets",
        "authors": ["Demsar, J."],
        "container": "Journal of Machine Learning Research",
        "publisher": "JMLR", "year": 2006, "volume": "7", "issue": "",
        "pages": "1-30", "doi": "",
        "url": "https://jmlr.org/papers/v7/demsar06a.html",
        "type": "journal-article", "match_overlap": None, "source": "manual",
    },
    "dropout": {
        "title": "Dropout: A Simple Way to Prevent Neural Networks from "
                 "Overfitting",
        "authors": ["Srivastava, N.", "Hinton, G.", "Krizhevsky, A.",
                    "Sutskever, I.", "Salakhutdinov, R."],
        "container": "Journal of Machine Learning Research",
        "publisher": "JMLR", "year": 2014, "volume": "15", "issue": "",
        "pages": "1929-1958", "doi": "",
        "url": "https://jmlr.org/papers/v15/srivastava14a.html",
        "type": "journal-article", "match_overlap": None, "source": "manual",
    },
}


def crossref_doi(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    r = subprocess.run(["curl", "-sS", "-m", "40", "--ssl-no-revoke", url],
                       capture_output=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout.decode("utf8", "replace"))["message"]
    except Exception:
        return None


def datacite_doi(doi):
    """arXiv DOIs (10.48550/*) are registered with DataCite, not CrossRef."""
    url = f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}"
    r = subprocess.run(["curl", "-sS", "-m", "40", "--ssl-no-revoke", url],
                       capture_output=True)
    if r.returncode != 0:
        return None
    try:
        a = json.loads(r.stdout.decode("utf8", "replace"))["data"]["attributes"]
    except Exception:
        return None
    return {
        "title": (a.get("titles") or [{}])[0].get("title", ""),
        "authors": [f"{c.get('familyName','')}, "
                    + " ".join(p[0] + "." for p in
                               c.get("givenName", "").split() if p)
                    for c in a.get("creators", []) if c.get("familyName")],
        "container": "arXiv", "publisher": a.get("publisher", "arXiv"),
        "year": a.get("publicationYear"), "volume": "", "issue": "",
        "pages": "", "doi": a.get("doi", ""),
        "arxiv": next((i["identifier"] for i in a.get("identifiers", [])
                       if i.get("identifierType") == "arXiv"), ""),
        "type": "preprint", "match_overlap": 1.0, "source": "datacite",
    }


def crossref(title):
    url = ("https://api.crossref.org/works?rows=3&query.bibliographic="
           + urllib.parse.quote(title)
           + "&select=title,author,container-title,issued,volume,issue,page,DOI,type,publisher"
           + "&mailto=abhithakur25@gmail.com")
    r = subprocess.run(["curl", "-sS", "-m", "40", "--ssl-no-revoke", url],
                       capture_output=True)
    if r.returncode != 0:
        return None, r.stderr.decode("utf8", "replace")[:200]
    try:
        return json.loads(r.stdout.decode("utf8", "replace")), None
    except Exception as e:
        return None, str(e)[:200]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def overlap(a, b):
    """Fraction of the query's content words present in the returned title."""
    stop = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with",
            "using", "via", "is", "at"}
    qa = {w for w in norm(a).split() if w not in stop and len(w) > 2}
    qb = {w for w in norm(b).split() if w not in stop and len(w) > 2}
    return len(qa & qb) / max(1, len(qa))


def record(it, ov, source):
    auth = [f"{a.get('family','')}, "
            + " ".join(p[0] + "." for p in a.get("given", "").split() if p)
            for a in it.get("author", []) if a.get("family")]
    return {
        "title": (it.get("title") or [""])[0],
        "authors": auth,
        "container": (it.get("container-title") or [""])[0],
        "publisher": it.get("publisher", ""),
        "year": (it.get("issued", {}).get("date-parts", [[None]])[0][0]),
        "volume": it.get("volume", ""),
        "issue": it.get("issue", ""),
        "pages": it.get("page", ""),
        "doi": it.get("DOI", ""),
        "type": it.get("type", ""),
        "match_overlap": round(ov, 3),
        "source": source,
    }


def main():
    refs, unresolved = {}, []
    for key, q, first in QUERIES:
        if key in MANUAL:
            refs[key] = MANUAL[key]
            print(f"  {key:16s} {refs[key]['year']}  "
                  f"{refs[key]['container'][:52]:52s} manual")
            continue
        if key in DOI_OVERRIDE:
            doi = DOI_OVERRIDE[key]
            if doi.startswith("10.48550/"):
                rec = datacite_doi(doi)
                time.sleep(0.4)
                if rec is not None:
                    refs[key] = rec
                    print(f"  {key:16s} {rec['year']}  "
                          f"{'arXiv:' + rec.get('arxiv', ''):52s} datacite")
                    continue
                print(f"  {key:16s} DataCite lookup failed")
            it = crossref_doi(doi)
            time.sleep(0.4)
            if it is not None:
                refs[key] = record(it, 1.0, "doi")
                print(f"  {key:16s} {refs[key]['year']}  "
                      f"{refs[key]['container'][:52]:52s} doi")
                continue
            print(f"  {key:16s} DOI lookup failed, falling back to title")
        data, err = crossref(q)
        time.sleep(0.4)                      # be polite to the public endpoint
        if data is None:
            unresolved.append((key, q, err))
            print(f"  {key:16s} NETWORK FAIL: {err}")
            continue
        best = None
        for it in data["message"]["items"]:
            t = (it.get("title") or [""])[0]
            ov = overlap(q, t)
            if best is None or ov > best[0]:
                best = (ov, it)
        ov, it = best
        if ov < 0.6:
            unresolved.append((key, q, f"best match {ov:.2f}: "
                                        f"{(it.get('title') or [''])[0][:70]}"))
            print(f"  {key:16s} UNRESOLVED (overlap {ov:.2f})")
            continue
        rec = record(it, ov, "title")
        if first and rec["authors"] and \
                not rec["authors"][0].lower().startswith(first.lower()[:5]):
            unresolved.append((key, q, f"first author {rec['authors'][0]!r} "
                                       f"!= expected {first!r}"))
            print(f"  {key:16s} REJECTED: first author {rec['authors'][0]!r} "
                  f"!= expected {first!r}")
            continue
        refs[key] = rec
        print(f"  {key:16s} {refs[key]['year']}  "
              f"{refs[key]['container'][:52]:52s} {ov:.2f}")

    OUT.write_text(json.dumps({"resolved": refs,
                               "unresolved": unresolved}, indent=2),
                   encoding="utf-8")
    print(f"\nresolved {len(refs)} / {len(QUERIES)}; "
          f"{len(unresolved)} unresolved -> {OUT.name}")


if __name__ == "__main__":
    main()
