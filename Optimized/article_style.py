"""OOXML emitters that reproduce the Paper-2 template's formatting exactly.

The template (CODE_05-08-2025_Paper2/Research_Paper-2.docx) uses direct run and
paragraph formatting rather than named styles, so this module does the same and
copies its measurements verbatim, read out of the template's own document.xml:

    element        spacing (before/after, twips)   run
    title          0 / 280,  centred               b, 16 pt
    authors        80 / 40,  centred               b, 12 pt
    affiliation    0  / 60,  centred               8 pt
    e-mail         0  / 40,  centred               b, 9 pt
    corresponding  0  / 80,  centred               i, 9 pt
    heading 1      280 / 120                       b, 14 pt, #1F4E79
    heading 2      200 / 120                       b, 12 pt, #1F4E79
    body           0  / 160, justified, 432 indent 12 pt
    abstract body  0  / 160, justified             11 pt
    keywords       0  / 240                        i, 10 pt
    caption        40 / 200, centred               i, 10 pt
    table header   shaded #1F4E79                  b, 9 pt, white
    line spacing   276 (1.15) everywhere

Sizes below are half-points and spacings are twips, as OOXML requires.
"""
import html
import re
import struct
import zipfile
from pathlib import Path

FONT = "Times New Roman"
ACCENT = "1F4E79"
LINE_ATTRS = 'w:line="276" w:lineRule="auto"'

IMAGES = []
EMU_PER_PX = 9525


def esc(t):
    return html.escape(str(t), quote=False)


def _rpr(sz, bold=False, italic=False, color=None, sup=False, mono=False):
    font = "Consolas" if mono else FONT
    out = [f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" '
           f'w:eastAsia="{font}"/>']
    out.append("<w:b/>" if bold else '<w:b w:val="0"/>')
    out.append("<w:i/>" if italic else '<w:i w:val="0"/>')
    if color:
        out.append(f'<w:color w:val="{color}"/>')
    out.append(f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>')
    if sup:
        out.append('<w:vertAlign w:val="superscript"/>')
    return "<w:rPr>" + "".join(out) + "</w:rPr>"


def _ppr(before, after, jc=None, indent=0, keep=False):
    out = [f'<w:spacing w:before="{before}" w:after="{after}" {LINE_ATTRS}/>']
    if indent:
        out.append(f'<w:ind w:firstLine="{indent}"/>')
    if jc:
        out.append(f'<w:jc w:val="{jc}"/>')
    if keep:
        out.append("<w:keepNext/>")
    return "<w:pPr>" + "".join(out) + "</w:pPr>"


def _p(text, ppr, rpr):
    return (f"<w:p>{ppr}<w:r>{rpr}"
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


# ------------------------------------------------------------- front matter
def title(t):
    return _p(t, _ppr(0, 280, "center"), _rpr(32, bold=True))


def authors(t):
    return _p(t, _ppr(80, 40, "center"), _rpr(24, bold=True))


def native_name(t):
    return _p(t, _ppr(0, 160, "center"), _rpr(20, italic=True))


def affiliation(marker, t):
    """Affiliation line with only the leading marker superscripted.

    The template superscripts the entire line, which renders the affiliations
    at roughly 5 pt and unreadable; the marker alone is plainly what was meant.
    """
    return (f'<w:p>{_ppr(0, 60, "center")}'
            f'<w:r>{_rpr(16, sup=True)}'
            f'<w:t xml:space="preserve">{esc(marker)}</w:t></w:r>'
            f'<w:r>{_rpr(16)}'
            f'<w:t xml:space="preserve">{esc(t)}</w:t></w:r></w:p>')


def email(t):
    return _p(t, _ppr(0, 40, "center"), _rpr(18, bold=True))


def corresponding(t):
    return _p(t, _ppr(0, 80, "center"), _rpr(18, italic=True))


def topic_note(t):
    return _p(t, _ppr(0, 240, "center"),
              _rpr(16, italic=True, color="333333"))


# ---------------------------------------------------------------- structure
def h1(t):
    return _p(t, _ppr(280, 120, keep=True), _rpr(28, bold=True, color=ACCENT))


def h2(t):
    return _p(t, _ppr(200, 120, keep=True), _rpr(24, bold=True, color=ACCENT))


def h3(t):
    return _p(t, _ppr(160, 100, keep=True), _rpr(22, bold=True, color=ACCENT))


def para(t, indent=True):
    return _p(t, _ppr(0, 160, "both", 432 if indent else 0), _rpr(24))


def abstract_para(t):
    return _p(t, _ppr(0, 160, "both"), _rpr(22))


def keywords(t):
    return (f'<w:p>{_ppr(0, 240)}'
            f'<w:r>{_rpr(20, bold=True, italic=True)}'
            f'<w:t xml:space="preserve">Keywords: </w:t></w:r>'
            f'<w:r>{_rpr(20, italic=True)}'
            f'<w:t xml:space="preserve">{esc(t)}</w:t></w:r></w:p>')


def caption(t):
    return _p(t, _ppr(40, 200, "center"), _rpr(20, italic=True))


def bullet(t):
    return _p("•  " + t, _ppr(0, 80, "both", 0), _rpr(24))


def numbered(n, t):
    return _p(f"({n})  " + t, _ppr(0, 80, "both", 0), _rpr(24))


def code(lines):
    if isinstance(lines, str):
        lines = lines.split("\n")
    return "".join(_p(l if l else " ", _ppr(0, 0, None, 0),
                      _rpr(17, mono=True, color="333333")) for l in lines)


def reference_entry(t):
    """APA hanging indent: 0.5 in hang, left aligned, no first-line indent."""
    ppr = ('<w:pPr><w:spacing w:before="0" w:after="100" '
           f'{LINE_ATTRS}/><w:ind w:left="720" w:hanging="720"/>'
           '<w:jc w:val="both"/></w:pPr>')
    return _p(t, ppr, _rpr(22))


# ------------------------------------------------------------------- tables
def cell(text, header=False, width=1200, shade=None, bold=False):
    if header:
        shade, bold = ACCENT, True
    sh = f'<w:shd w:val="clear" w:fill="{shade}"/>' if shade else ""
    rpr = _rpr(18, bold=bold, color="FFFFFF" if header else None)
    return (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{sh}'
            f'<w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr><w:jc w:val="center"/>'
            f'<w:spacing w:before="20" w:after="20" {LINE_ATTRS}/></w:pPr>'
            f'<w:r>{rpr}'
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p></w:tc>')


def table(head, rows, cap=None, widths=None, highlight=None):
    """Table with the template's shaded header. Caption goes above, IEEE-style."""
    n = len(head)
    widths = widths or [int(9360 / n)] * n
    borders = ("<w:tblBorders>" + "".join(
        f'<w:{s} w:val="single" w:sz="4" w:color="9CB7D4"/>'
        for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
        + "</w:tblBorders>")
    grid = "<w:tblGrid>" + "".join(
        f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"
    out = [caption(cap)] if cap else []
    out.append(f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>'
               f'<w:jc w:val="center"/>{borders}</w:tblPr>{grid}')
    out.append("<w:tr><w:trPr><w:tblHeader/></w:trPr>" + "".join(
        cell(h, header=True, width=widths[i]) for i, h in enumerate(head))
        + "</w:tr>")
    for r in rows:
        sh = "FFF2CC" if (highlight and highlight(r)) else None
        out.append("<w:tr>" + "".join(
            cell(c, width=widths[i], shade=sh) for i, c in enumerate(r))
            + "</w:tr>")
    out.append("</w:tbl>")
    out.append(_p(" ", _ppr(0, 120), _rpr(12)))
    return "".join(out)


# ------------------------------------------------------------------ figures
def picture(path, cap=None, max_w_in=6.1):
    path = Path(path)
    if not path.exists():
        return caption(f"[figure missing: {path.name}]")
    data = path.read_bytes()
    try:
        w, h = struct.unpack(">II", data[16:24])       # PNG IHDR
    except Exception:
        w, h = 1200, 800
    cx, cy = w * EMU_PER_PX, h * EMU_PER_PX
    lim = int(max_w_in * 914400)
    if cx > lim:
        cy, cx = int(cy * lim / cx), lim

    rid = f"rIdImg{len(IMAGES) + 1}"
    name = f"image{len(IMAGES) + 1}{path.suffix.lower()}"
    IMAGES.append((rid, name, data))
    n = len(IMAGES)
    out = (f'<w:p>{_ppr(160, 40, "center")}<w:r><w:drawing>'
           '<wp:inline distT="0" distB="0" distL="0" distR="0">'
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
           "</w:drawing></w:r></w:p>")
    return out + (caption(cap) if cap else "")


def page_break():
    return ('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')


# ---------------------------------------------------------------- packaging
NS = (' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
      ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
      ' xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"'
      ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
      ' xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')

CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
    "</Types>")

RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>")

DOC_RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            "</Relationships>")

STYLES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
          '<w:docDefaults><w:rPrDefault><w:rPr>'
          f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" '
          f'w:cs="{FONT}"/><w:sz w:val="24"/><w:szCs w:val="24"/>'
          "</w:rPr></w:rPrDefault><w:pPrDefault><w:pPr>"
          f'<w:spacing w:before="0" w:after="160" {LINE_ATTRS}/>'
          "</w:pPr></w:pPrDefault></w:docDefaults>"
          '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
          '<w:name w:val="Normal"/></w:style>'
          "</w:styles>")


def write_doc(out_path, body):
    """Zip the package and verify it before returning the document XML."""
    import xml.dom.minidom as md

    xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           f"<w:document{NS}><w:body>{body}"
           '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
           '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" '
           'w:left="1134" w:header="708" w:footer="708" w:gutter="0"/>'
           "</w:sectPr></w:body></w:document>")

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
        used = set(re.findall(r'r:embed="([^"]+)"', d))
        declared = set(re.findall(
            r'Id="(rIdImg\d+)"',
            z.read("word/_rels/document.xml.rels").decode("utf8")))
        assert not (used - declared), \
            f"drawings reference undeclared rIds: {used - declared}"
        media = [n for n in z.namelist() if n.startswith("word/media/")]
    return d, media


def word_count(body):
    """Words of visible text in a run of body XML."""
    return len(" ".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", body,
                                   re.S)).split())
