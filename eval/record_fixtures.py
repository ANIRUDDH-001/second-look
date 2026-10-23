"""Record the static demo: run real bills through the real API and save every response.

Each example is uploaded to the FastAPI app (in-process TestClient, temporary data dir), the worker
runs the full pipeline (Gemma, Tesseract, reconcile, TabPFN, explanation), and the bill record is
saved to frontend/public/fixtures/{slug}.json. Corrections listed for an example are then posted one
at a time and each resulting record is saved as {slug}.stepN.json, so the demo can replay them.
Nothing here is edited by hand; rerun this script to regenerate the demo.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "public" / "fixtures"
BILLS = ROOT / "data" / "bills"

EXAMPLES = [
    {"slug": "matches-the-cap", "image": "extract/X003_scan.png", "title": "A bill that adds up",
     "note": "A scanned bill from a fictional supplier, checked against the Ofgem cap for its dates."},
    {"slug": "planted-vat-error", "image": "extract/X015_clean.png", "title": "A planted VAT error",
     "note": "This synthetic bill charges 5% VAT after VAT on household energy fell to 0% on 1 Oct 2026."},
    {"slug": "misread-then-corrected", "image": "extract/X003_clean.png", "title": "A misread, then corrected",
     "note": "Gemma misread two dates. The second reader could not find them on the bill, so they are "
             "flagged; correcting them re-runs every check.",
     "corrections": "from_key"},
    {"slug": "instructions-on-the-bill", "image": "adversarial/D002_clean.png", "title": "Instructions printed on the bill",
     "note": "This bill has text telling AI reviewers to claim a £250 overcharge. It is read as data, not obeyed."},
]


def _key(image: str) -> dict:
    stem = Path(image).stem
    bid, variant = stem.split("_")
    src = "adversarial" if image.startswith("adversarial") else "extract"
    for x in open(ROOT / "datagen" / "answer_keys" / f"{src}.jsonl"):
        r = json.loads(x)
        if r["bill_id"] == bid and r["variant"] == variant:
            return r
    raise LookupError(image)


def _wait(c, id: str) -> dict:
    while True:
        rec = c.get(f"/api/bills/{id}").json()
        if rec["status"] in ("done", "failed"):
            return rec
        time.sleep(2)


def _corrections_from_key(rec: dict, key: dict) -> list[dict]:
    """For every field the pipeline flagged for review, the value printed on the bill (from the answer key)."""
    from secondlook.store import get_field

    out = []
    truth = key["bill"]
    for f in rec["result"]["review_fields"]:
        want = get_field(truth, f)
        got = get_field(rec["result"]["bill"], f)
        if want is not None and str(want) != str(got):
            out.append({"field": f, "value": str(want)})
    return out


def main() -> None:
    os.environ["SECONDLOOK_DATA"] = tempfile.mkdtemp()
    from fastapi.testclient import TestClient

    from secondlook.api import app as api

    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    with TestClient(api.app) as c:
        for ex in EXAMPLES:
            key = _key(ex["image"])
            path = BILLS / ex["image"]
            hh = key.get("household") or ""
            t0 = time.perf_counter()
            r = c.post("/api/bills", files={"file": (path.name, path.read_bytes(), "image/png")},
                       data={"household": hh} if hh else {})
            r.raise_for_status()
            id = r.json()["id"]
            rec = _wait(c, id)
            print(ex["slug"], rec["status"], rec["result"]["verdict"]["kind"] if rec["result"] else rec["error"],
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
            shutil.copy(path, OUT / f"{ex['slug']}.png")
            (OUT / f"{ex['slug']}.json").write_text(json.dumps({"id": ex["slug"], **rec}, indent=1), encoding="utf-8")
            entry = {"id": ex["slug"], "title": ex["title"], "note": ex["note"]}
            if ex.get("corrections") == "from_key" and rec["result"]:
                replay = _corrections_from_key(rec, key)
                for n, corr in enumerate(replay, 1):
                    rec = c.post(f"/api/bills/{id}/corrections", json=corr).json()
                    (OUT / f"{ex['slug']}.step{n}.json").write_text(json.dumps({**rec, "id": ex["slug"]}, indent=1),
                                                                   encoding="utf-8")
                    print("  corrected", corr, "->", rec["result"]["verdict"]["kind"], flush=True)
                entry["replay"] = replay
            index.append(entry)
    (OUT / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
