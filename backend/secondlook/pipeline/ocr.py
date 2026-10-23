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


def _row_window(lines: list[Line], amount) -> list[Line] | None:
    """The OCR row printing a charge line's amount, plus the row after it (where dates sit)."""
    if amount is None:
        return None
    for i, ln in enumerate(lines):
        if _find([ln], _forms("amount_gbp", amount)):
            return lines[i:i + 2]
    return None


def ground(bill: Bill, lines: list[Line]) -> dict[str, Box | None]:
    """Field path -> box where the transcribed value is printed, or None if it is not found."""
    # The subtotal and total often share digits; look for the subtotal only on its labelled line,
    # so a total copied into the subtotal is not "found".
    labelled = [ln for ln in lines if _SUBTOTAL_LABEL.search(ln.text)] or lines
    # A charge line's dates, quantity and rate must be printed on that charge's own row: the period
    # end date appears elsewhere on every bill, so a page-wide search would "find" a wrong one.
    rows = {i: _row_window(lines, ln.amount_gbp) or lines for i, ln in enumerate(bill.lines)}
    out = {}
    for k, v in grounded_fields(bill).items():
        where = lines
        if k == "subtotal_gbp":
            where = labelled
        elif k.startswith("lines[") and not k.endswith("amount_gbp"):
            where = rows[int(k[6:k.index("]")])]
        out[k] = _find(where, _forms(k, v))
    return out


# ---------- second-reader repair ----------

_SUBTOTAL_LABEL = re.compile(r"\b(sub ?total|before vat|net charges)\b", re.I)


def _numbers(lines: list[Line]) -> list[tuple[str, Box]]:
    out = []
    for ln in lines:
        for w, box in ln.words:
            for m in re.finditer(r"\d[\d,]*\.\d+", w):
                out.append((m.group().replace(",", ""), box))
    return out


def repair(bill: Bill, lines: list[Line], boxes: dict[str, Box | None]) -> tuple[Bill, list[dict]]:
    """Fill two known transcription slips from the OCR words, and say so.

    1. A rate the model shortened (54.48 for a printed 54.486): if exactly one printed number
       extends the model's digits, take the printed one.
    2. A subtotal copied from the total although VAT was charged: take the money on the line
       labelled Subtotal / Before VAT / Net charges, if there is exactly one.
    Only ungrounded values are touched, and every change is returned so the UI can show it.
    """
    b = bill.model_copy(deep=True)
    fixes: list[dict] = []
    nums = _numbers(lines)
    for i, ln in enumerate(b.lines):
        key = f"lines[{i}].rate_p"
        if ln.rate_p is None or boxes.get(key) is not None:
            continue
        got = format(ln.rate_p, "f")
        places = len(got.split(".")[1]) if "." in got else 0
        # printed digits the model shortened, by truncating or by rounding
        cands = {(t, box) for t, box in nums if "." in got and len(t) > len(got)
                 and (t.startswith(got) or Decimal(t).quantize(Decimal(1).scaleb(-places)) == ln.rate_p)}
        values = {t for t, _ in cands}
        if len(values) == 1:
            t, box = next(iter(cands))
            fixes.append({"field": key, "model": got, "printed": t, "box": box})
            ln.rate_p = Decimal(t)
    vat = sum((v.amount_gbp or Decimal(0)) for v in b.vat)
    if (b.subtotal_gbp is not None and b.subtotal_gbp == b.total_gbp and vat > 0):
        found = set()
        for ln in lines:
            if _SUBTOTAL_LABEL.search(ln.text):
                for w, box in ln.words:
                    m = re.fullmatch(r"£?(\d[\d,]*\.\d{2})", w)
                    if m:
                        found.add((m.group(1).replace(",", ""), box))
        if len({t for t, _ in found}) == 1:
            t, box = next(iter(found))
            fixes.append({"field": "subtotal_gbp", "model": format(b.subtotal_gbp, "f"), "printed": t, "box": box})
            b.subtotal_gbp = Decimal(t)
    return b, fixes


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
