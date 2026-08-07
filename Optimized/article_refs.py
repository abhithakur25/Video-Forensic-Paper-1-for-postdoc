"""APA-7 reference list and in-text citations, built from references.json.

Every entry comes from Optimized/references.json, which fetch_references.py
resolved against CrossRef (or DataCite for arXiv DOIs). Nothing here invents
bibliographic data: the formatter only rearranges what the registrar returned.

cite("stil") renders "(Gu et al., 2021)" and records the key as used.
citet("stil") renders "Gu et al. (2021)". At build time, used() lets the caller
assert that every key in the reference list is actually cited in the text and
that every cited key exists - an uncited entry or a dangling citation is a bug,
not a formatting preference.
"""
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1]
_DATA = json.loads((P / "Optimized" / "references.json")
                   .read_text(encoding="utf-8"))
REFS = _DATA["resolved"]
_USED = set()

# Container names CrossRef returns for proceedings are long and inconsistent;
# these are the short forms used in the reference list.
SHORT = {
    "2019 IEEE/CVF International Conference on Computer Vision (ICCV)":
        "Proceedings of the IEEE/CVF International Conference on Computer "
        "Vision (ICCV)",
    "2021 IEEE/CVF International Conference on Computer Vision (ICCV)":
        "Proceedings of the IEEE/CVF International Conference on Computer "
        "Vision (ICCV)",
    "2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)":
        "Proceedings of the IEEE/CVF International Conference on Computer "
        "Vision Workshops (ICCVW)",
    "2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)":
        "Proceedings of the IEEE/CVF Conference on Computer Vision and "
        "Pattern Recognition (CVPR)",
    "2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)":
        "Proceedings of the IEEE/CVF Conference on Computer Vision and "
        "Pattern Recognition (CVPR)",
    "2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)":
        "Proceedings of the IEEE/CVF Conference on Computer Vision and "
        "Pattern Recognition (CVPR)",
    "2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition":
        "Proceedings of the IEEE/CVF Conference on Computer Vision and "
        "Pattern Recognition (CVPR)",
    "2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)":
        "Proceedings of the IEEE Conference on Computer Vision and Pattern "
        "Recognition (CVPR)",
    "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)":
        "Proceedings of the IEEE Conference on Computer Vision and Pattern "
        "Recognition (CVPR)",
    "2015 IEEE International Conference on Computer Vision (ICCV)":
        "Proceedings of the IEEE International Conference on Computer Vision "
        "(ICCV)",
    "Proceedings of the 29th ACM International Conference on Multimedia":
        "Proceedings of the 29th ACM International Conference on Multimedia "
        "(MM '21)",
    "2018 IEEE International Workshop on Information Forensics and Security (WIFS)":
        "Proceedings of the IEEE International Workshop on Information "
        "Forensics and Security (WIFS)",
    "ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)":
        "Proceedings of the IEEE International Conference on Acoustics, "
        "Speech and Signal Processing (ICASSP)",
    "2018 15th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)":
        "Proceedings of the IEEE International Conference on Advanced Video "
        "and Signal Based Surveillance (AVSS)",
    "2009 Ninth IEEE International Conference on Data Mining":
        "Proceedings of the IEEE International Conference on Data Mining "
        "(ICDM)",
    "2010 20th International Conference on Pattern Recognition":
        "Proceedings of the International Conference on Pattern Recognition "
        "(ICPR)",
    "Proceedings of the 11th ACM workshop on Multimedia and security":
        "Proceedings of the 11th ACM Workshop on Multimedia and Security "
        "(MM&Sec '09)",
    "Lecture Notes in Computer Science": "Lecture Notes in Computer Science",
}


def _year(k):
    return REFS[k].get("year") or "n.d."


def _surname(a):
    return a.split(",")[0].strip()


def label(k):
    """Author-date label: 'Gu et al.' / 'Kingra & Aggarwal' / 'Haralick et al.'"""
    a = REFS[k]["authors"]
    if not a:
        return REFS[k]["title"].split(":")[0][:40]
    if len(a) == 1:
        return _surname(a[0])
    if len(a) == 2:
        return f"{_surname(a[0])} & {_surname(a[1])}"
    return f"{_surname(a[0])} et al."


def cite(*keys):
    """Parenthetical citation, semicolon-separated, sorted by year."""
    for k in keys:
        if k not in REFS:
            raise KeyError(f"no reference for {k!r}")
        _USED.add(k)
    ks = sorted(keys, key=lambda k: (str(_year(k)), label(k)))
    return "(" + "; ".join(f"{label(k)}, {_year(k)}" for k in ks) + ")"


def citet(k):
    """Narrative citation: 'Gu et al. (2021)'."""
    if k not in REFS:
        raise KeyError(f"no reference for {k!r}")
    _USED.add(k)
    return f"{label(k)} ({_year(k)})"


def used():
    return set(_USED)


def _authors_apa(a):
    if not a:
        return ""
    if len(a) == 1:
        return a[0]
    if len(a) <= 20:
        return ", ".join(a[:-1]) + ", & " + a[-1]
    return ", ".join(a[:19]) + ", ... " + a[-1]


def apa(k):
    """One APA-7 reference string."""
    r = REFS[k]
    parts = [_authors_apa(r["authors"])]
    parts.append(f"({r.get('year') or 'n.d.'}).")
    title = r["title"].rstrip(".")
    container = SHORT.get(r["container"], r["container"])
    is_journal = r["type"] in ("journal-article",) and container

    if is_journal:
        parts.append(title + ".")
        vol = r.get("volume") or ""
        iss = r.get("issue") or ""
        seg = container
        if vol:
            seg += f", {vol}"
            if iss:
                seg += f"({iss})"
        if r.get("pages"):
            seg += f", {r['pages']}"
        parts.append(seg + ".")
    elif r["type"] == "preprint" or container == "arXiv":
        arx = r.get("arxiv", "")
        parts.append(title + ".")
        parts.append(f"arXiv:{arx}." if arx else "arXiv preprint.")
    elif container:
        parts.append(title + ".")
        seg = "In " + container
        if r.get("pages"):
            seg += f" (pp. {r['pages']})"
        parts.append(seg + ".")
        if r.get("publisher"):
            parts.append(r["publisher"] + ".")
    else:
        parts.append(title + ".")

    if r.get("doi"):
        parts.append(f"https://doi.org/{r['doi']}")
    elif r.get("url"):
        parts.append(r["url"])
    return " ".join(x for x in parts if x).replace("  ", " ")


def reference_list(keys=None):
    """APA reference strings, alphabetical by first author surname."""
    keys = sorted(keys or _USED,
                  key=lambda k: (label(k).lower(), str(_year(k))))
    return [apa(k) for k in keys]
