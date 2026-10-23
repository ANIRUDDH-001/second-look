"""Render a Bill as a PNG in one of three fictional supplier layouts, recording field boxes.

The boxes are ground truth for where each value is printed, used to score extraction
and to show source crops. Suppliers, account numbers and addresses are fictional.
"""
from __future__ import annotations

import random
from datetime import date
from decimal import Decimal
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from secondlook.models import Bill

W, H = 1240, 1754  # A4 at 150 dpi
FONTS = Path("C:/Windows/Fonts")
STREETS = ["Elm Row", "Cedar Close", "Harbour Lane", "Mill Street", "Quarry Road", "Alder Way", "Kestrel Court"]


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


def d(x: date) -> str:
    return f"{x.day} {x:%b %Y}"


def gbp(x: Decimal) -> str:
    return f"£{x:,.2f}"


class Canvas:
    def __init__(self, bg="white"):
        self.img = Image.new("RGB", (W, H), bg)
        self.dr = ImageDraw.Draw(self.img)
        self.boxes: dict[str, list[int]] = {}

    def text(self, xy, s, f, fill="#1a1a1a", key=None, anchor="la"):
        self.dr.text(xy, s, font=f, fill=fill, anchor=anchor)
        if key:
            self.boxes[key] = [int(v) for v in self.dr.textbbox(xy, s, font=f, anchor=anchor)]

    def rect(self, box, fill=None, outline=None, width=1, radius=0):
        if radius:
            self.dr.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
        else:
            self.dr.rectangle(box, fill=fill, outline=outline, width=width)

    def line(self, xy, fill="#cccccc", width=1):
        self.dr.line(xy, fill=fill, width=width)


def _line_label(ln, i_kind: dict) -> str:
    return "Electricity used" if ln.kind == "energy" else "Standing charge"


def _qty(ln) -> str:
    return f"{ln.quantity} kWh" if ln.kind == "energy" else f"{ln.quantity} days"


def _rate(ln) -> str:
    return f"{ln.rate_p}p/kWh" if ln.kind == "energy" else f"{ln.rate_p}p/day"


def _reading(v: int, t: str | None) -> str:
    return f"{v:05d} ({'A' if t == 'A' else 'E'})"


def render_classic(bill: Bill, extras: dict, rng: random.Random, inject: str | None = None) -> Canvas:
    c = Canvas()
    navy = "#16325c"
    c.rect((0, 0, W, 150), fill=navy)
    c.text((70, 45), "Northwind", font("georgiab.ttf", 48), fill="white")
    c.text((70, 102), "ENERGY", font("arial.ttf", 22), fill="#9fb7d9")
    c.text((W - 70, 50), "Your electricity bill", font("arial.ttf", 30), fill="white", anchor="ra")
    c.text((W - 70, 95), f"Bill date {d(bill.bill_date)}", font("arial.ttf", 22), fill="#cfdcef", anchor="ra")
    f, fb, fs = font("arial.ttf", 24), font("arialbd.ttf", 24), font("arial.ttf", 20)
    c.text((70, 200), "Mx A. Resident", fb)
    c.text((70, 235), f"{rng.randint(2, 98)} {rng.choice(STREETS)}", f)
    c.text((70, 270), "London", f)
    c.rect((700, 190, W - 70, 320), outline="#b8c4d6", width=2)
    c.text((725, 205), "Account number", fs, fill="#555")
    c.text((725, 232), bill.account_ref, fb, key="account_ref")
    c.text((725, 272), "Tariff", fs, fill="#555")
    c.text((800, 272), bill.tariff_name, fs, key="tariff_name")

    c.text((70, 370), "Billing period", fb)
    c.text((70, 405), d(bill.period_start), f, key="period_start")
    c.text((c.boxes["period_start"][2] + 12, 405), "to", f)
    c.text((c.boxes["period_start"][2] + 45, 405), d(bill.period_end), f, key="period_end")

    c.text((70, 470), "Meter readings", font("arialbd.ttf", 28), fill=navy)
    y = 520
    c.rect((70, y, W - 70, y + 44), fill="#eef2f8")
    for x, h in ((90, "Meter"), (360, "Previous"), (620, "Current"), (900, "Units used")):
        c.text((x, y + 10), h, fb)
    y += 60
    c.text((90, y), "E10K" + str(rng.randint(10000, 99999)), f)
    c.text((360, y), _reading(bill.prev_reading, bill.prev_reading_type), f, key="prev_reading")
    c.text((620, y), _reading(bill.curr_reading, bill.curr_reading_type), f, key="curr_reading")
    c.text((900, y), f"{bill.units_kwh} kWh", fb, key="units_kwh")
    c.text((90, y + 45), "A = actual reading, E = estimated reading", fs, fill="#666")

    y = 720
    c.text((70, y), "Your charges", font("arialbd.ttf", 28), fill=navy)
    y += 50
    c.rect((70, y, W - 70, y + 44), fill="#eef2f8")
    for x, h in ((90, "Description"), (520, "Quantity"), (760, "Rate"), (W - 90, "Amount")):
        c.text((x, y + 10), h, fb, anchor="ra" if h == "Amount" else "la")
    y += 60
    for i, ln in enumerate(bill.lines):
        c.text((90, y), _line_label(ln, {}), f)
        c.text((90, y + 30), f"{d(ln.start)} – {d(ln.end)}", fs, fill="#666", key=f"lines[{i}].dates")
        c.text((520, y), _qty(ln), f, key=f"lines[{i}].quantity")
        c.text((760, y), _rate(ln), f, key=f"lines[{i}].rate_p")
        c.text((W - 90, y), gbp(ln.amount_gbp), f, key=f"lines[{i}].amount_gbp", anchor="ra")
        y += 72
        c.line((90, y - 10, W - 90, y - 10))
    y += 10
    c.text((760, y), "Subtotal", fb)
    c.text((W - 90, y), gbp(bill.subtotal_gbp), fb, key="subtotal_gbp", anchor="ra")
    for k, v in enumerate(bill.vat):
        y += 40
        c.text((760, y), f"VAT at {v.pct}% on {gbp(v.on_gbp)}", f, key=f"vat[{k}].label")
        c.text((W - 90, y), gbp(v.amount_gbp), f, key=f"vat[{k}].amount_gbp", anchor="ra")
    y += 60
    c.rect((700, y, W - 70, y + 70), fill=navy)
    c.text((725, y + 20), "Total charges", font("arialbd.ttf", 28), fill="white")
    c.text((W - 90, y + 18), gbp(bill.total_gbp), font("arialbd.ttf", 32), fill="white", key="total_gbp", anchor="ra")
    y += 110
    c.text((70, y), f"We'll collect this by Direct Debit on or after {d(extras['due_date'])}.", f)
    c.text((70, y + 40), f"Estimated annual cost on this tariff: {gbp(extras['annual_cost_gbp'])}", fs, fill="#555")
    c.text((70, y + 70), f"Same period last year you used {extras['last_year_kwh']} kWh.", fs, fill="#555")
    if inject:
        c.text((70, H - 120), inject, font("arial.ttf", 16), fill="#999")
    c.text((70, H - 70), "Northwind Energy is a fictional supplier. Synthetic bill generated for testing.",
           font("arial.ttf", 16), fill="#aaa")
    return c


def render_modern(bill: Bill, extras: dict, rng: random.Random, inject: str | None = None) -> Canvas:
    c = Canvas("#fbfaf7")
    green, ink = "#0f7b5f", "#1d2b28"
    f, fb, fs = font("segoeui.ttf", 24), font("seguisb.ttf", 24), font("segoeui.ttf", 20)
    c.text((70, 60), "brightside", font("segoeuib.ttf", 44), fill=green)
    c.text((300, 75), "power", font("segoeui.ttf", 28), fill="#5d7a72")
    c.text((W - 70, 70), f"Account {bill.account_ref}", fs, fill="#5d7a72", anchor="ra", key="account_ref")
    c.rect((70, 150, W - 70, 400), fill="white", outline="#e3e0d8", width=2, radius=24)
    c.text((110, 185), "Your bill for", fs, fill="#5d7a72")
    c.text((110, 220), d(bill.period_start), font("seguisb.ttf", 30), fill=ink, key="period_start")
    c.text((c.boxes["period_start"][2] + 14, 220), "–", font("seguisb.ttf", 30), fill=ink)
    c.text((c.boxes["period_start"][2] + 44, 220), d(bill.period_end), font("seguisb.ttf", 30), fill=ink, key="period_end")
    c.text((110, 290), "This period's charges", fs, fill="#5d7a72")
    c.text((110, 318), gbp(bill.total_gbp), font("segoeuib.ttf", 56), fill=green, key="total_gbp")
    c.text((W - 110, 185), "Issued", fs, fill="#5d7a72", anchor="ra")
    c.text((W - 110, 215), d(bill.bill_date), f, fill=ink, anchor="ra")
    c.text((W - 110, 290), "Plan", fs, fill="#5d7a72", anchor="ra")
    c.text((W - 110, 320), bill.tariff_name, f, fill=ink, anchor="ra", key="tariff_name")

    y = 450
    for k, (lab, val, key) in enumerate((
        ("Previous reading", _reading(bill.prev_reading, bill.prev_reading_type), "prev_reading"),
        ("Latest reading", _reading(bill.curr_reading, bill.curr_reading_type), "curr_reading"),
        ("You used", f"{bill.units_kwh} kWh", "units_kwh"),
    )):
        x0 = 70 + k * 375
        c.rect((x0, y, x0 + 350, y + 140), fill="white", outline="#e3e0d8", width=2, radius=18)
        c.text((x0 + 25, y + 25), lab, fs, fill="#5d7a72")
        c.text((x0 + 25, y + 65), val, font("seguisb.ttf", 34), fill=ink, key=key)
    c.text((70, y + 160), "(A) actual  (E) estimated", fs, fill="#8a9a95")

    y = 680
    c.text((70, y), "How we worked it out", font("seguisb.ttf", 30), fill=ink)
    y += 60
    for i, ln in enumerate(bill.lines):
        c.rect((70, y, W - 70, y + 92), fill="white", outline="#ece9e2", width=1, radius=12)
        c.text((100, y + 14), _line_label(ln, {}), fb, fill=ink)
        c.text((100, y + 50), f"{d(ln.start)} – {d(ln.end)}", fs, fill="#8a9a95", key=f"lines[{i}].dates")
        c.text((560, y + 30), f"{_qty(ln)}  ×  {_rate(ln)}", f, fill=ink, key=f"lines[{i}].qty_rate")
        c.text((W - 100, y + 30), gbp(ln.amount_gbp), fb, fill=ink, anchor="ra", key=f"lines[{i}].amount_gbp")
        y += 104
    y += 10
    c.text((560, y), "Before VAT", f, fill="#5d7a72")
    c.text((W - 100, y), gbp(bill.subtotal_gbp), fb, fill=ink, anchor="ra", key="subtotal_gbp")
    for k, v in enumerate(bill.vat):
        y += 42
        c.text((560, y), f"VAT {v.pct}% on {gbp(v.on_gbp)}", f, fill="#5d7a72", key=f"vat[{k}].label")
        c.text((W - 100, y), gbp(v.amount_gbp), fb, fill=ink, anchor="ra", key=f"vat[{k}].amount_gbp")
    y += 50
    c.line((560, y, W - 100, y), fill=green, width=3)
    y += 16
    c.text((560, y), "Total", font("segoeuib.ttf", 30), fill=ink)
    c.text((W - 100, y), gbp(bill.total_gbp), font("segoeuib.ttf", 30), fill=green, anchor="ra")
    y += 90
    c.rect((70, y, W - 70, y + 120), fill="#eaf4f0", radius=18)
    c.text((100, y + 22), f"Looking ahead: about {gbp(extras['annual_cost_gbp'])} a year at today's prices.", f, fill=ink)
    c.text((100, y + 62), f"Last year, same months: {extras['last_year_kwh']} kWh. Paid by Direct Debit on {d(extras['due_date'])}.", fs, fill="#5d7a72")
    if inject:
        c.text((70, H - 120), inject, font("segoeui.ttf", 16), fill="#b5b5b5")
    c.text((70, H - 70), "Brightside Power is fictional. Synthetic bill for software testing.", font("segoeui.ttf", 16), fill="#b5b5b5")
    return c


def render_statement(bill: Bill, extras: dict, rng: random.Random, inject: str | None = None) -> Canvas:
    c = Canvas()
    f, fb, fs = font("consola.ttf", 22), font("consolab.ttf", 22), font("consola.ttf", 18)
    c.text((70, 60), "MERIDIAN UTILITIES PLC", font("tahomabd.ttf", 34), fill="#7a1f2b")
    c.text((70, 105), "STATEMENT OF ACCOUNT - ELECTRICITY", font("tahoma.ttf", 22), fill="#444")
    c.line((70, 145, W - 70, 145), fill="#7a1f2b", width=3)
    rows = [
        ("ACCOUNT REF", bill.account_ref, "account_ref"),
        ("STATEMENT DATE", bill.bill_date.strftime("%d/%m/%Y"), None),
        ("TARIFF", bill.tariff_name, "tariff_name"),
        ("SUPPLY PERIOD FROM", bill.period_start.strftime("%d/%m/%Y"), "period_start"),
        ("SUPPLY PERIOD TO", bill.period_end.strftime("%d/%m/%Y"), "period_end"),
    ]
    y = 175
    for lab, val, key in rows:
        c.text((70, y), lab, fs, fill="#666")
        c.text((420, y), val, f, key=key)
        y += 36
    y += 20
    c.text((70, y), "METER  E10K" + str(rng.randint(10000, 99999)), fb)
    y += 40
    c.text((70, y), "PREVIOUS READ", fs, fill="#666")
    c.text((420, y), _reading(bill.prev_reading, bill.prev_reading_type), f, key="prev_reading")
    y += 34
    c.text((70, y), "PRESENT READ", fs, fill="#666")
    c.text((420, y), _reading(bill.curr_reading, bill.curr_reading_type), f, key="curr_reading")
    y += 34
    c.text((70, y), "CONSUMPTION KWH", fs, fill="#666")
    c.text((420, y), f"{bill.units_kwh}", fb, key="units_kwh")
    y += 34
    c.text((70, y), "READ TYPE: A=ACTUAL E=ESTIMATED C=CUSTOMER", fs, fill="#888")
    y += 60
    c.rect((70, y, W - 70, y + 36), fill="#f3ecec")
    for x, h, a in ((80, "ITEM", "la"), (360, "FROM", "la"), (540, "TO", "la"), (760, "QTY", "ra"),
                    (960, "RATE(p)", "ra"), (W - 80, "GBP", "ra")):
        c.text((x, y + 7), h, fb, anchor=a)
    y += 50
    for i, ln in enumerate(bill.lines):
        c.text((80, y), "ENERGY" if ln.kind == "energy" else "STANDING", f)
        c.text((360, y), ln.start.strftime("%d/%m/%y"), f, key=f"lines[{i}].start")
        c.text((540, y), ln.end.strftime("%d/%m/%y"), f, key=f"lines[{i}].end")
        c.text((760, y), f"{ln.quantity}", f, anchor="ra", key=f"lines[{i}].quantity")
        c.text((960, y), f"{ln.rate_p}", f, anchor="ra", key=f"lines[{i}].rate_p")
        c.text((W - 80, y), f"{ln.amount_gbp:,.2f}", f, anchor="ra", key=f"lines[{i}].amount_gbp")
        y += 38
    c.line((70, y, W - 70, y), fill="#999")
    y += 14
    c.text((760, y), "NET CHARGES", fb, anchor="ra")
    c.text((W - 80, y), f"{bill.subtotal_gbp:,.2f}", fb, anchor="ra", key="subtotal_gbp")
    for k, v in enumerate(bill.vat):
        y += 36
        c.text((760, y), f"VAT @ {v.pct}% ON {v.on_gbp:,.2f}", f, anchor="ra", key=f"vat[{k}].label")
        c.text((W - 80, y), f"{v.amount_gbp:,.2f}", f, anchor="ra", key=f"vat[{k}].amount_gbp")
    y += 44
    c.text((760, y), "TOTAL THIS PERIOD", fb, anchor="ra")
    c.text((W - 80, y), f"{bill.total_gbp:,.2f}", fb, anchor="ra", key="total_gbp")
    y += 36
    c.text((760, y), "BALANCE B/FWD", f, anchor="ra")
    c.text((W - 80, y), "0.00", f, anchor="ra")
    y += 70
    c.text((70, y), f"PROJECTED ANNUAL COST {extras['annual_cost_gbp']:,.2f}   PRIOR YEAR SAME PERIOD {extras['last_year_kwh']} KWH", fs, fill="#666")
    c.text((70, y + 32), f"DIRECT DEBIT DATE {extras['due_date'].strftime('%d/%m/%Y')}", fs, fill="#666")
    if inject:
        c.text((70, H - 120), inject.upper(), font("consola.ttf", 15), fill="#aaa")
    c.text((70, H - 70), "MERIDIAN UTILITIES IS FICTIONAL. SYNTHETIC STATEMENT FOR TESTING.", font("consola.ttf", 15), fill="#aaa")
    return c


LAYOUTS = {"classic": render_classic, "modern": render_modern, "statement": render_statement}


def render(layout: str, bill: Bill, extras: dict, seed: int, inject: str | None = None) -> Canvas:
    return LAYOUTS[layout](bill, extras, random.Random(seed), inject)
