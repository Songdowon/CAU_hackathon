"""마크다운을 PDF로 뽑는다. 한글이 되는 도구가 컨테이너에 없어서 직접 만든 것.

reportlab 내장 CID 폰트(HYSMyeongJo/HYGothic)를 쓰므로 폰트 파일 설치가 필요 없다.
METHOD.md가 쓰는 문법만 다룬다: 제목, 문단, 파이프 표, 코드블록, 인용, 목록,
수평선, `**굵게**`, `` `코드` ``.

    python tools/md2pdf.py METHOD.md --out METHOD.pdf
"""
import argparse
import html
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (HRFlowable, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

SERIF, SANS = "HYSMyeongJo-Medium", "HYGothic-Medium"
INK, MUTED, RULE, ZEBRA = colors.HexColor("#1a1a1a"), colors.HexColor("#666666"), colors.HexColor("#d8d8d8"), colors.HexColor("#f4f4f4")


def register():
    for f in (SERIF, SANS):
        pdfmetrics.registerFont(UnicodeCIDFont(f))
    # CID 폰트는 굵은 변형이 없으므로 고딕을 bold 자리에 매핑한다.
    pdfmetrics.registerFontFamily(SERIF, normal=SERIF, bold=SANS, italic=SERIF,
                                  boldItalic=SANS)


def inline(text):
    """마크다운 인라인 → reportlab 마크업. 이스케이프를 먼저 한다."""
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", rf'<font face="{SANS}" color="#8a3b00">\1</font>', t)
    t = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", t)
    return t


def styles():
    base = dict(fontName=SERIF, textColor=INK, alignment=TA_LEFT)
    return {
        "h1": ParagraphStyle("h1", **base, fontSize=19, leading=25, spaceBefore=2, spaceAfter=9),
        "h2": ParagraphStyle("h2", **base, fontSize=14.5, leading=20, spaceBefore=17, spaceAfter=7),
        "h3": ParagraphStyle("h3", **base, fontSize=11.8, leading=16, spaceBefore=12, spaceAfter=5),
        "p": ParagraphStyle("p", **base, fontSize=9.4, leading=15.2, spaceAfter=7),
        "quote": ParagraphStyle("q", **{**base, "textColor": MUTED}, fontSize=9.4,
                                leading=15.2, leftIndent=9, spaceAfter=9),
        "li": ParagraphStyle("li", **base, fontSize=9.4, leading=14.6, leftIndent=11,
                             bulletIndent=2, spaceAfter=3),
        "code": ParagraphStyle("code", fontName=SANS, fontSize=8.1, leading=11.6,
                               textColor=colors.HexColor("#222222")),
        "cell": ParagraphStyle("cell", fontName=SERIF, fontSize=8.3, leading=11.4, textColor=INK),
        "cellh": ParagraphStyle("cellh", fontName=SANS, fontSize=8.3, leading=11.4, textColor=INK),
    }


def table_flowable(rows, st, width):
    """파이프 표 → Table. 첫 행은 헤더, 구분선 행은 이미 제거된 상태."""
    head, body = rows[0], rows[1:]
    data = [[Paragraph(inline(c), st["cellh"]) for c in head]]
    data += [[Paragraph(inline(c), st["cell"]) for c in r] for r in body]
    ncol = max(len(r) for r in data)
    data = [r + [Paragraph("", st["cell"])] * (ncol - len(r)) for r in data]
    # 첫 열은 설명이 길어 넓게, 나머지는 균등 분배.
    first = width * (0.34 if ncol > 2 else 0.5)
    rest = (width - first) / (ncol - 1) if ncol > 1 else 0
    t = Table(data, colWidths=[first] + [rest] * (ncol - 1), repeatRows=1, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
             ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK),
             ("LINEBELOW", (0, 1), (-1, -2), 0.3, RULE),
             ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK)]
    style += [("BACKGROUND", (0, i), (-1, i), ZEBRA) for i in range(2, len(data), 2)]
    t.setStyle(TableStyle(style))
    return t


def code_flowable(lines, st, width):
    body = "<br/>".join(html.escape(l).replace(" ", "&nbsp;") or "&nbsp;" for l in lines)
    t = Table([[Paragraph(body, st["code"])]], colWidths=[width], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f7f5")),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9)]))
    return t


def build(md, width):
    st, out, i, lines = styles(), [], 0, md.splitlines()
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):                      # 코드블록
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out += [code_flowable(lines[i + 1:j], st, width), Spacer(1, 8)]
            i = j + 1
        elif re.match(r"^\|.*\|", ln):                # 표
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            rows = [[c.strip() for c in r.strip().strip("|").split("|")]
                    for r in lines[i:j] if not re.match(r"^\|[\s:|-]+\|?$", r.strip())]
            out += [table_flowable(rows, st, width), Spacer(1, 10)]
            i = j
        elif ln.startswith("### "):
            out.append(Paragraph(inline(ln[4:]), st["h3"])); i += 1
        elif ln.startswith("## "):
            out.append(Paragraph(inline(ln[3:]), st["h2"])); i += 1
        elif ln.startswith("# "):
            out.append(Paragraph(inline(ln[2:]), st["h1"])); i += 1
        elif ln.startswith("> "):
            out.append(Paragraph(inline(ln[2:]), st["quote"])); i += 1
        elif re.match(r"^\s*[-*] ", ln):
            out.append(Paragraph(inline(re.sub(r"^\s*[-*] ", "", ln)), st["li"], bulletText="•")); i += 1
        elif re.match(r"^\s*\d+\. ", ln):
            out.append(Paragraph(inline(re.sub(r"^\s*(\d+)\. ", r"\1. ", ln)), st["li"])); i += 1
        elif ln.strip() in ("---", "***", "___"):
            out += [Spacer(1, 4), HRFlowable(width="100%", thickness=0.5, color=RULE), Spacer(1, 8)]
            i += 1
        elif not ln.strip():
            i += 1
        else:                                          # 문단 (연속 줄 합침)
            j = i
            while (j < len(lines) and lines[j].strip()
                   and not re.match(r"^(#|\||>|```|---|\s*[-*] |\s*\d+\. )", lines[j])):
                j += 1
            out.append(Paragraph(inline(" ".join(l.strip() for l in lines[i:j])), st["p"]))
            i = j
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    a = p.parse_args()
    register()
    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title=a.title or a.src, author="Team 노광탈")
    width = doc.width

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont(SERIF, 7.6)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(A4[0] - 19 * mm, 11 * mm, str(d.page))
        canvas.restoreState()

    doc.build(build(open(a.src).read(), width), onFirstPage=footer, onLaterPages=footer)
    print(f"{a.src} -> {a.out}")


if __name__ == "__main__":
    main()
