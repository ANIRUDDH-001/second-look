"""SQLite persistence. Uploads are keyed by content hash, so the same image uploaded twice is one
record. Corrections are append-only; the current bill is always the extracted bill with every
correction replayed in order, so any past state can be rebuilt exactly.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from secondlook.models import Bill
from secondlook.pipeline.extract import parse_date, parse_num

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
  id TEXT PRIMARY KEY, household TEXT, filename TEXT, created_at TEXT NOT NULL,
  status TEXT NOT NULL, error TEXT, extracted TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS corrections (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, upload_id TEXT NOT NULL REFERENCES uploads(id),
  field TEXT NOT NULL, old_value TEXT, new_value TEXT, created_at TEXT NOT NULL);
"""

DATE_FIELDS = {"bill_date", "period_start", "period_end", "start", "end"}
INT_FIELDS = {"prev_reading", "curr_reading"}
TEXT_FIELDS = {"supplier", "account_ref", "tariff_name", "prev_reading_type", "curr_reading_type", "kind"}
_PATH = re.compile(r"^(?:(lines|vat)\[(\d+)\]\.)?([a-z_]+)$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def create(self, id: str, household: str | None, filename: str | None) -> bool:
        """True if this is a new upload; False if the same image was already stored."""
        with _lock:
            cur = self.db.execute("INSERT OR IGNORE INTO uploads(id, household, filename, created_at, status) "
                                  "VALUES (?,?,?,?, 'queued')", (id, household, filename, _now()))
            self.db.commit()
            return cur.rowcount == 1

    def update(self, id: str, **fields) -> None:
        with _lock:
            sets = ", ".join(f"{k}=?" for k in fields)
            vals = [json.dumps(v) if k in ("extracted", "result") and v is not None else v for k, v in fields.items()]
            self.db.execute(f"UPDATE uploads SET {sets} WHERE id=?", (*vals, id))
            self.db.commit()

    def get(self, id: str) -> dict | None:
        r = self.db.execute("SELECT * FROM uploads WHERE id=?", (id,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        for k in ("extracted", "result"):
            d[k] = json.loads(d[k]) if d[k] else None
        d["corrections"] = [dict(c) for c in self.db.execute(
            "SELECT seq, field, old_value, new_value, created_at FROM corrections WHERE upload_id=? ORDER BY seq", (id,))]
        return d

    def add_correction(self, id: str, field: str, old: str | None, new: str | None) -> None:
        with _lock:
            self.db.execute("INSERT INTO corrections(upload_id, field, old_value, new_value, created_at) VALUES (?,?,?,?,?)",
                            (id, field, old, new, _now()))
            self.db.commit()


def current_bill(extracted: dict, corrections: list[dict]) -> Bill:
    """Replay corrections, in order, onto the extracted bill."""
    d = json.loads(json.dumps(extracted))
    for c in corrections:
        set_field(d, c["field"], c["new_value"])
    return Bill.model_validate(d)


def parse_value(name: str, value: str | None):
    if value is None or str(value).strip() == "":
        return None
    if name in DATE_FIELDS:
        return parse_date(value).isoformat()
    if name in INT_FIELDS:
        return parse_num(value, integer=True)
    if name in TEXT_FIELDS:
        return str(value).strip()
    return str(parse_num(value))


def get_field(d: dict, path: str):
    m = _PATH.match(path)
    if not m:
        raise ValueError(f"unknown field {path!r}")
    group, idx, name = m.groups()
    if group:
        items = d.get(group) or []
        return items[int(idx)].get(name) if int(idx) < len(items) else None
    return d.get(name)


def set_field(d: dict, path: str, value: str | None) -> None:
    m = _PATH.match(path)
    if not m:
        raise ValueError(f"unknown field {path!r}")
    group, idx, name = m.groups()
    if name not in Bill.model_fields and name not in ("start", "end", "quantity", "rate_p", "amount_gbp",
                                                      "kind", "pct", "on_gbp"):
        raise ValueError(f"unknown field {path!r}")
    v = parse_value(name, value)
    if group:
        items = d.setdefault(group, [])
        i = int(idx)
        while len(items) <= i:
            items.append({"kind": "energy"} if group == "lines" else {})
        items[i][name] = v
    else:
        d[name] = v
