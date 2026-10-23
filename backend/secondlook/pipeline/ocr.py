"""Tesseract word boxes, used two ways.

1. Grounding: every value Gemma transcribed is looked up in Tesseract's text of the same image.
   A value that cannot be found printed anywhere is flagged for review, and a value that is found
   gets a source crop for the evidence view. Two independent readers agreeing is evidence;
   neither is ground truth.
2. Baseline: a regex parser over the OCR text, for Experiment A. The regexes were written
   against our three synthetic layouts, which favours the baseline.
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from decimal import Decimal

import pytesseract
from PIL import Image

from secondlook.models import Bill, ChargeLine, VatLine
from secondlook.pipeline.extract import parse_date, parse_num

_TESS = os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if os.path.exists(_TESS):
    pytesseract.pytesseract.tesseract_cmd = _TESS

Box = tuple[int, int, int, int]


@dataclass
class Line:
    words: list[tuple[str, Box]]

    @property
    def text(self) -> str:
        return " ".join(w for w, _ in self.words)


def ocr_lines(image: bytes) -> list[Line]:
    img = Image.open(io.BytesIO(image)).convert("RGB")
    d = pytesseract.image_to_data(img, config="--psm 3", output_type=pytesseract.Output.DICT)
    lines: dict[tuple, Line] = {}
    for i, w in enumerate(d["text"]):
        if not w.strip():
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        box = (d["left"][i], d["top"][i], d["left"][i] + d["width"][i], d["top"][i] + d["height"][i])
        lines.setdefault(key, Line([])).words.append((w, box))
    return sorted(lines.values(), key=lambda ln: (ln.words[0][1][1], ln.words[0][1][0]))


# ---------- grounding ----------

def _norm(s: str) -> str:
    return re.sub(r"[£,]", "", s.lower()).replace("–", "-").replace("—", "-")


def _forms(field: str, v) -> list[str]:
    if v is None:
        return []
    if hasattr(v, "strftime"):
        return [f"{v.day} {v:%b %Y}", v.strftime("%d/%m/%Y"), v.strftime("%d/%m/%y"), v.isoformat()]
    if isinstance(v, Decimal):
        out = {format(v, "f"), format(v.normalize(), "f")}
        if field.endswith("gbp"):
            out.add(f"{v:.2f}")
        return sorted(out, key=len, reverse=True)
    return [str(v)]


def _find(lines: list[Line], forms: list[str]) -> Box | None:
    for form in forms:
        pat = re.compile(r"(?<![\d.])" + re.escape(_norm(form)) + r"(?![\d])")
        for ln in lines:
            # character offsets of each word inside the joined, normalised line
            spans, pos = [], 0
            for w, box in ln.words:
                nw = _norm(w)
                spans.append((pos, pos + len(nw), box))
                pos += len(nw) + 1
            m = pat.search(" ".join(_norm(w) for w, _ in ln.words))
            if m:
                hit = [b for s, e, b in spans if s < m.end() and e > m.start()]
                return (min(b[0] for b in hit), min(b[1] for b in hit),
                        max(b[2] for b in hit), max(b[3] for b in hit))
    return None


def grounded_fields(bill: Bill) -> dict[str, object]:
    out = {k: getattr(bill, k) for k in ("account_ref", "tariff_name", "period_start", "period_end",
                                         "prev_reading", "curr_reading", "units_kwh", "subtotal_gbp", "total_gbp")}
    for i, ln in enumerate(bill.lines):
        for k in ("start", "end", "quantity", "rate_p", "amount_gbp"):
            out[f"lines[{i}].{k}"] = getattr(ln, k)
    for k, v in enumerate(bill.vat):
        for f in ("pct", "on_gbp", "amount_gbp"):
            out[f"vat[{k}].{f}"] = getattr(v, f)
    return {k: v for k, v in out.items() if v is not None}


def ground(bill: Bill, lines: list[Line]) -> dict[str, Box | None]:
    """Field path -> box where the transcribed value is printed, or None if it is not found."""
    return {k: _find(lines, _forms(k, v)) for k, v in grounded_fields(bill).items()}


# ---------- regex baseline ----------

DATE = r"\d{1,2} [A-Z][a-z]{2} \d{4}|\d{2}/\d{2}/\d{2,4}"
MONEY = r"£?(\d[\d,]*\.\d{2})"


def _safe(fn, s):
    try:
        return fn(s)
    except (ValueError, TypeError):
        return None


def baseline_extract(lines: list[Line]) -> Bill:
    text = "\n".join(ln.text for ln in lines)
    t1 = re.sub(r"\s+", " ", text)
    b = Bill()
    if m := re.search(r"\b(\d{2}-\d{6})\b", t1):
        b.account_ref = m.group(1)
    if m := re.search(r"(Standard Variable \(price cap\)|[A-Z][\w ]*(?:Fixed|Tracker|Economy 7|Agile)[\w ]*)", t1, re.I):
        b.tariff_name = m.group(1)
    if m := re.search(rf"PERIOD FROM ({DATE}).*?PERIOD TO ({DATE})", t1) or \
            re.search(rf"({DATE})\s*(?:to|-|–|—)\s*({DATE})", t1):
        b.period_start, b.period_end = _safe(parse_date, m.group(1)), _safe(parse_date, m.group(2))
    reads = re.findall(r"(\d{4,6})\s*\(([AE])\)", t1)
    if len(reads) >= 2:
        (p, pt), (c, ct) = reads[:2]
        b.prev_reading, b.prev_reading_type, b.curr_reading, b.curr_reading_type = int(p), pt, int(c), ct
    if m := re.search(r"CONSUMPTION KWH (\d[\d,]*)", t1) or re.search(r"\([AE]\)\s+(\d[\d,]*)\s*kWh", t1):
        b.units_kwh = _safe(parse_num, m.group(1))

    ranges = re.findall(rf"({DATE})\s*[-–—]\s*({DATE})", text)
    rows = []
    for m in re.finditer(rf"(\d[\d,]*)\s*(kWh|days?)\s*[x×]?\s*(\d+\.\d+)\s*p/(?:kWh|day)\s*{MONEY}", t1):
        rows.append(["energy" if m.group(2) == "kWh" else "standing", None, None, m.group(1), m.group(3), m.group(4)])
    tail = ranges[-len(rows):] if rows and len(ranges) >= len(rows) else []
    for r, (s, e) in zip(rows, tail):
        r[1], r[2] = s, e
    for m in re.finditer(rf"(ENERGY|STANDING)\s+({DATE})\s+({DATE})\s+(\d+)\s+(\d+\.\d+)\s+(\d[\d,]*\.\d{{2}})", t1):
        rows.append([m.group(1).lower(), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)])
    b.lines = [ChargeLine(kind=k, start=_safe(parse_date, s), end=_safe(parse_date, e), quantity=_safe(parse_num, q),
                          rate_p=_safe(parse_num, r), amount_gbp=_safe(parse_num, a)) for k, s, e, q, r, a in rows]

    if m := re.search(rf"(?:Subtotal|Before VAT|NET CHARGES)\s*{MONEY}", t1, re.I):
        b.subtotal_gbp = _safe(parse_num, m.group(1))
    b.vat = [VatLine(pct=_safe(parse_num, p), on_gbp=_safe(parse_num, o), amount_gbp=_safe(parse_num, a))
             for p, o, a in re.findall(rf"VAT\s*(?:at|@)?\s*(\d+(?:\.\d+)?)%\s*on\s*{MONEY}\s*{MONEY}", t1, re.I)]
    totals = re.findall(rf"\bTotal\b[^£\d]{{0,20}}{MONEY}", t1, re.I)
    if totals:
        b.total_gbp = _safe(parse_num, totals[-1])
    return b
