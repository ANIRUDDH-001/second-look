"""HTTP API for local use.

    uvicorn secondlook.api.app:app --port 8000

POST /api/bills                    upload an image (+ optional household); idempotent by content hash
GET  /api/bills/{id}               status and result (the UI polls this)
GET  /api/bills/{id}/image         the uploaded image
POST /api/bills/{id}/corrections   {field, value}: append a correction and re-run stages 2-5
GET  /api/households               smart-meter households available for usage context
GET  /healthz                      model server, OCR and tariff status
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from secondlook import llm
from secondlook.pipeline.context import ROOT
from secondlook.pipeline.ocr import _TESS
from secondlook.pipeline.run import analyse_bill, content_hash, households, run_image
from secondlook.store import Store, current_bill, get_field, set_field
from secondlook.tariff.engine import load_tariff

DATA = Path(os.environ.get("SECONDLOOK_DATA", ROOT / "data" / "app"))
UPLOADS = DATA / "uploads"
MAX_BYTES = 10 * 1024 * 1024


class JsonFormatter(logging.Formatter):
    def format(self, r: logging.LogRecord) -> str:
        d = {"t": self.formatTime(r, "%Y-%m-%dT%H:%M:%S"), "level": r.levelname, "msg": r.getMessage()}
        d.update(getattr(r, "extra", {}))
        return json.dumps(d)


log = logging.getLogger("secondlook")
_h = logging.StreamHandler()
_h.setFormatter(JsonFormatter())
log.addHandler(_h)
log.setLevel(logging.INFO)


def _log(msg: str, **extra) -> None:
    log.info(msg, extra={"extra": extra})


store = Store(DATA / "secondlook.sqlite3")
jobs: queue.Queue[str] = queue.Queue()


def _worker() -> None:
    while True:
        id = jobs.get()
        rec = store.get(id)
        if rec is None:
            continue
        store.update(id, status="running")
        try:
            result = run_image((UPLOADS / f"{id}.png").read_bytes(), rec["household"])
            store.update(id, status="done", extracted=result["bill"], result=result, error=None)
            _log("bill processed", id=id, verdict=result["verdict"]["kind"], timings=result["timings"])
        except Exception as e:  # noqa: BLE001 - a failed bill must not stop the worker
            store.update(id, status="failed", error=f"{type(e).__name__}: {e}")
            _log("bill failed", id=id, error=str(e))


app = FastAPI(title="Second Look")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _start() -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_worker, daemon=True).start()
    # Anything interrupted by a restart is simply run again: the pipeline is idempotent.
    for (id,) in store.db.execute("SELECT id FROM uploads WHERE status IN ('queued','running')"):
        jobs.put(id)


@app.get("/healthz")
def healthz() -> dict:
    t = load_tariff()
    return {"ok": True, "model": llm.MODEL, "model_available": llm.available(),
            "tesseract": bool(shutil.which("tesseract") or os.path.exists(_TESS)),
            "tariff": t.tariff_id, "tariff_version": t.version,
            "tariff_covers": [t.first_day.isoformat(), t.last_day.isoformat()], "queue": jobs.qsize()}


@app.get("/api/households")
def list_households() -> list[str]:
    return households()


@app.post("/api/bills")
async def upload(file: UploadFile = File(...), household: str | None = Form(None)) -> dict:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image over 10 MB")
    if household and household not in households():
        raise HTTPException(422, f"unknown household {household}")
    from secondlook.pipeline.extract import prepare_image  # validates that it is an image
    try:
        prepare_image(data)
    except Exception:
        raise HTTPException(415, "not a readable image") from None
    id = content_hash(data)
    if store.create(id, household, file.filename):
        (UPLOADS / f"{id}.png").write_bytes(data)
        jobs.put(id)
        _log("bill queued", id=id, household=household)
    return {"id": id, **_public(store.get(id))}


@app.get("/api/bills/{id}")
def get_bill(id: str) -> dict:
    rec = store.get(id)
    if rec is None:
        raise HTTPException(404, "no such bill")
    return {"id": id, **_public(rec)}


@app.get("/api/bills/{id}/image")
def get_image(id: str) -> FileResponse:
    p = UPLOADS / f"{id}.png"
    if not p.exists():
        raise HTTPException(404, "no such bill")
    return FileResponse(p)


class Correction(BaseModel):
    field: str
    value: str | None


@app.post("/api/bills/{id}/corrections")
def correct(id: str, c: Correction) -> dict:
    rec = store.get(id)
    if rec is None or rec["status"] != "done":
        raise HTTPException(409, "bill is not ready for corrections")
    bill_now = current_bill(rec["extracted"], rec["corrections"]).model_dump(mode="json")
    try:
        old = get_field(bill_now, c.field)
        set_field(json.loads(json.dumps(bill_now)), c.field, c.value)  # validate before storing
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    store.add_correction(id, c.field, None if old is None else str(old), c.value)
    rec = store.get(id)
    bill = current_bill(rec["extracted"], rec["corrections"])
    t0 = time.perf_counter()
    result = {**rec["result"], **analyse_bill(bill, rec["household"]), "bill": bill.model_dump(mode="json")}
    # A corrected field is now user-confirmed: it no longer needs review.
    corrected = {x["field"] for x in rec["corrections"]}
    result["review_fields"] = [f for f in rec["result"]["review_fields"] if f not in corrected]
    store.update(id, result=result)
    _log("correction applied", id=id, field=c.field, verdict=result["verdict"]["kind"],
         seconds=round(time.perf_counter() - t0, 3))
    return {"id": id, **_public(store.get(id))}


def _public(rec: dict) -> dict:
    return {k: rec[k] for k in ("status", "error", "household", "filename", "created_at", "result", "corrections")}
