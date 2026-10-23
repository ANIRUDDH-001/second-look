import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BillRecord, Box, DEMO, DemoEntry, Finding, Result, UsageContext, correct, demoIndex, getBill, imageUrl,
  listHouseholds, uploadBill,
} from "./api";

const VERDICT_WORD: Record<string, string> = {
  reconciled: "Reconciled",
  discrepancy: "Discrepancy",
  insufficient_evidence: "Not enough evidence",
  unsupported_tariff: "Tariff not supported",
};

const LABELS: Record<string, string> = {
  tariff_name: "Tariff", period_start: "Period from", period_end: "Period to",
  prev_reading: "Previous reading", curr_reading: "Current reading", units_kwh: "Units billed (kWh)",
  subtotal_gbp: "Before VAT (£)", total_gbp: "Total (£)",
  start: "from", end: "to", quantity: "quantity", rate_p: "rate (p)", amount_gbp: "amount (£)",
  pct: "rate (%)", on_gbp: "charged on (£)",
};

type Row = { path: string; label: string; value: string };
type Group = { title: string; rows: Row[] };

function str(v: unknown): string {
  return v === null || v === undefined ? "" : String(v);
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function day(v: unknown): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(str(v));
  return m ? `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]} ${m[1]}` : str(v) || "?";
}

function groups(bill: Result["bill"]): Group[] {
  const top = (keys: string[]) => keys.map((k) => ({ path: k, label: LABELS[k], value: str(bill[k]) }));
  const out: Group[] = [
    { title: "Period and tariff", rows: top(["tariff_name", "period_start", "period_end"]) },
    { title: "Meter", rows: top(["prev_reading", "curr_reading", "units_kwh"]) },
  ];
  bill.lines.forEach((ln, i) => {
    const what = ln.kind === "standing" ? "Standing charge" : "Electricity used";
    out.push({
      title: `${what}, ${day(ln.start)} to ${day(ln.end)}`,
      rows: ["quantity", "rate_p", "amount_gbp", "start", "end"].map((k) => ({
        path: `lines[${i}].${k}`, label: LABELS[k], value: str(ln[k]),
      })),
    });
  });
  bill.vat.forEach((v, k) => {
    out.push({
      title: `VAT at ${str(v.pct) || "?"}%`,
      rows: ["pct", "on_gbp", "amount_gbp"].map((f) => ({ path: `vat[${k}].${f}`, label: LABELS[f], value: str(v[f]) })),
    });
  });
  out.push({ title: "Totals", rows: top(["subtotal_gbp", "total_gbp"]) });
  return out;
}

// ---------------------------------------------------------------- bill image with evidence boxes

function BillImage({ id, boxes, active, review, onPick }: {
  id: string; boxes: Result["boxes"]; active: string | null; review: Set<string>; onPick: (f: string) => void;
}) {
  const [size, setSize] = useState<[number, number] | null>(null);
  const img = useRef<HTMLImageElement>(null);
  // A cached image can finish loading before React attaches onLoad, so also read it after mount.
  useEffect(() => {
    const el = img.current;
    if (el?.complete && el.naturalWidth) setSize([el.naturalWidth, el.naturalHeight]);
  }, [id]);
  const pct = (b: Box) => size && {
    left: `${(b[0] / size[0]) * 100}%`, top: `${(b[1] / size[1]) * 100}%`,
    width: `${((b[2] - b[0]) / size[0]) * 100}%`, height: `${((b[3] - b[1]) / size[1]) * 100}%`,
  };
  return (
    <figure className="sheet">
      <div className="sheet-frame">
        <img ref={img} src={imageUrl(id)} alt="The uploaded bill"
             onLoad={(e) => setSize([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])} />
        {size && Object.entries(boxes).map(([f, b]) => b && (
          <button key={f} type="button" aria-label={`Show ${f}`} onClick={() => onPick(f)}
                  className={`ev${active === f ? " on" : ""}${review.has(f) ? " review" : ""}`}
                  style={pct(b) ?? undefined} />
        ))}
      </div>
      <figcaption>
        Outlined values were found printed on the bill by a second reader (Tesseract OCR). Select one to find it in the list.
      </figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------- verdict, checks, usage

function VerdictBlock({ r }: { r: Result }) {
  const v = r.verdict;
  return (
    <section className={`verdict v-${v.kind}`} aria-live="polite">
      <h2>{VERDICT_WORD[v.kind]}</h2>
      <p className="headline">{v.headline}</p>
      {v.billed_total_gbp && (
        <dl className="totals">
          <div><dt>Billed</dt><dd>£{v.billed_total_gbp}</dd></div>
          {v.expected_total_gbp && <div><dt>Recomputed from the Ofgem cap</dt><dd>£{v.expected_total_gbp}</dd></div>}
          {v.difference_gbp && Number(v.difference_gbp) !== 0 && (
            <div><dt>Difference</dt><dd>{Number(v.difference_gbp) > 0 ? "+" : "−"}£{Math.abs(Number(v.difference_gbp)).toFixed(2)}</dd></div>
          )}
        </dl>
      )}
    </section>
  );
}

function CheckRow({ f, onHover }: { f: Finding; onHover: (p: string | null) => void }) {
  const target = f.fields.find((x) => x.includes("[")) ?? f.fields[0] ?? null;
  return (
    <li className={`check c-${f.status}`} onMouseEnter={() => onHover(target)} onMouseLeave={() => onHover(null)}>
      <span className="mark" aria-hidden>{f.status === "pass" ? "✓" : f.status === "fail" ? "✕" : f.status === "warn" ? "!" : "–"}</span>
      <div>
        <p className="rule">{f.rule}</p>
        {(f.expected || f.billed) && (
          <p className="nums">
            {f.expected && <>expected <b>{f.expected}</b></>}{f.expected && f.billed && ", "}
            {f.billed && <>billed <b>{f.billed}</b></>}
          </p>
        )}
        {f.detail && <p className="detail">{f.detail}</p>}
      </div>
    </li>
  );
}

function Checks({ findings, onHover }: { findings: Finding[]; onHover: (p: string | null) => void }) {
  const attention = findings.filter((f) => f.status === "fail" || f.status === "warn");
  const rest = findings.filter((f) => f.status !== "fail" && f.status !== "warn");
  if (!findings.length) return null;
  return (
    <section className="panel">
      <h3>Checks</h3>
      {attention.length > 0 && <ul className="checks">{attention.map((f) => <CheckRow key={f.check} f={f} onHover={onHover} />)}</ul>}
      <details open={attention.length === 0}>
        <summary>{rest.filter((f) => f.status === "pass").length} checks passed{rest.some((f) => f.status === "skip") ? ", some skipped for missing values" : ""}</summary>
        <ul className="checks">{rest.map((f) => <CheckRow key={f.check} f={f} onHover={onHover} />)}</ul>
      </details>
    </section>
  );
}

function Usage({ c }: { c: UsageContext | null }) {
  if (!c || c.label === "no_history") return null;
  const lo = Math.min(c.low90_kwh, c.billed_kwh) * 0.85;
  const hi = Math.max(c.high90_kwh, c.billed_kwh) * 1.1;
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  const said = { typical: "typical for this home", unusually_high: "higher than usual for this home",
                 unusually_low: "lower than usual for this home" }[c.label] ?? c.label;
  return (
    <section className="panel">
      <h3>Is this usage normal?</h3>
      <p>{c.billed_kwh.toLocaleString()} kWh over {c.days} days is <strong>{said}</strong>.</p>
      <div className="band" role="img"
           aria-label={`Typical range ${c.low90_kwh} to ${c.high90_kwh} kWh, billed ${c.billed_kwh} kWh`}>
        <div className="range" style={{ left: x(c.low90_kwh), width: `calc(${x(c.high90_kwh)} - ${x(c.low90_kwh)})` }} />
        <div className="mid" style={{ left: x(c.expected_kwh) }} />
        <div className={`you l-${c.label}`} style={{ left: x(c.billed_kwh) }}><span>billed</span></div>
      </div>
      <p className="band-scale"><span>{c.low90_kwh} kWh</span><span>{c.high90_kwh} kWh</span></p>
      <p className="note">
        The shaded range is where 90% of outcomes are expected, forecast by TabPFN from {c.history_days} days of this
        home's smart-meter history. It describes usage, not whether the bill is right.
      </p>
    </section>
  );
}

// ---------------------------------------------------------------- the field list with corrections

function Fields({ r, rec, active, setActive, onSave, saving }: {
  r: Result; rec: BillRecord; active: string | null; setActive: (f: string | null) => void;
  onSave: (field: string, value: string) => void; saving: boolean;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const corrected = new Set(rec.corrections.map((c) => c.field));
  const review = new Set(r.review_fields);
  const rowRefs = useRef<Record<string, HTMLLIElement | null>>({});
  useEffect(() => {
    if (active) rowRefs.current[active]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [active]);

  const repaired = new Map((r.repairs ?? []).map((x) => [x.field, x.model]));
  const status = (p: string, value: string) =>
    corrected.has(p) ? ["corrected", "You corrected this"]
      : repaired.has(p) ? ["repaired", `Gemma read ${repaired.get(p)}; taken as printed`]
      : review.has(p) ? ["review", value ? "Not found on the bill, check it" : "Not read, add it"]
      : r.boxes[p] ? ["found", "Found on the bill"] : ["plain", ""];

  return (
    <section className="panel">
      <h3>What was read from the bill</h3>
      <p className="note">Read by Gemma 4 running on this computer. It copies values and never calculates them.</p>
      {groups(r.bill).map((g) => (
        <div key={g.title} className="group">
          <h4>{g.title}</h4>
          <ul className="fields">
            {g.rows.map((row) => {
              const [cls, said] = status(row.path, row.value);
              return (
                <li key={row.path} ref={(el) => { rowRefs.current[row.path] = el; }}
                    className={`field f-${cls}${active === row.path ? " on" : ""}`}
                    onMouseEnter={() => setActive(row.path)} onMouseLeave={() => setActive(null)}>
                  <span className="label">{row.label}</span>
                  {editing === row.path ? (
                    <form className="edit" onSubmit={(e) => { e.preventDefault(); onSave(row.path, draft); setEditing(null); }}>
                      <input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)}
                             aria-label={`Correct ${row.label}`} />
                      <button type="submit" disabled={saving}>Save</button>
                      <button type="button" className="quiet" onClick={() => setEditing(null)}>Cancel</button>
                    </form>
                  ) : (
                    <>
                      <span className="value">{row.value || "—"}</span>
                      <span className="state">{said}</span>
                      <button type="button" className="quiet" onClick={() => { setEditing(row.path); setDraft(row.value); }}>
                        Edit
                      </button>
                    </>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </section>
  );
}

// ---------------------------------------------------------------- one bill

function BillView({ id, demo }: { id: string; demo?: DemoEntry }) {
  const qc = useQueryClient();
  const [step, setStep] = useState(0);
  const [active, setActive] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  useEffect(() => { setStep(0); setNotice(null); }, [id]);

  const q = useQuery({
    queryKey: ["bill", id, step],
    queryFn: () => getBill(id, step),
    refetchInterval: (query) => (!DEMO && ["queued", "running"].includes(query.state.data?.status ?? "") ? 2000 : false),
  });
  const save = useMutation({
    mutationFn: ({ field, value }: { field: string; value: string }) => correct(id, field, value),
    onSuccess: (rec) => { qc.setQueryData(["bill", id, 0], rec); setNotice(null); },
    onError: (e: Error) => setNotice(e.message),
  });

  function onSave(field: string, value: string) {
    if (!DEMO) return save.mutate({ field, value });
    const next = demo?.replay?.[step];
    if (next && next.field === field && next.value.trim() === value.trim()) setStep(step + 1);
    else setNotice(next
      ? `This recorded example can replay one correction: set ${next.field} to ${next.value}. Run Second Look locally to correct any field.`
      : "This recorded example has no further corrections. Run Second Look locally to correct any field.");
  }

  const rec = q.data;
  if (q.isError) return <p className="state-msg">Could not load this bill: {(q.error as Error).message}</p>;
  if (!rec) return <p className="state-msg">Loading the bill…</p>;
  if (rec.status === "failed") return <p className="state-msg">Checking failed: {rec.error}. Try uploading the image again.</p>;
  if (!rec.result) return (
    <p className="state-msg working">
      {rec.status === "queued" ? "Waiting to start…" : "Reading the bill. On a laptop CPU this takes about a minute."}
    </p>
  );
  const r = rec.result;
  return (
    <div className="bill-layout">
      <BillImage id={id} boxes={r.boxes} active={active} review={new Set(r.review_fields)} onPick={setActive} />
      <div className="findings">
        <VerdictBlock r={r} />
        {r.review_fields.length > 0 && (
          <p className="review-note">
            {r.review_fields.length} value{r.review_fields.length > 1 ? "s" : ""} could not be found printed on the bill.
            {r.review_fields.length > 1
              ? " They are highlighted below. Check them against the image and correct any that are wrong."
              : " It is highlighted below. Check it against the image and correct it if it is wrong."}
          </p>
        )}
        <section className="panel explain">
          <p>{r.explanation.text}</p>
          <p className="note">
            {r.explanation.source === "model"
              ? "Written by Gemma from the check results, then checked: every number in it appears above."
              : r.explanation.rejected
                ? `Standard wording used: the model's text was rejected (${r.explanation.reason}).`
                : "Standard wording."}
          </p>
        </section>
        {notice && <p className="review-note" role="status">{notice}</p>}
        <Checks findings={r.verdict.findings} onHover={setActive} />
        <Usage c={r.context} />
        <Fields r={r} rec={rec} active={active} setActive={setActive} onSave={onSave} saving={save.isPending} />
        {rec.corrections.length > 0 && (
          <section className="panel">
            <h3>Corrections</h3>
            <ol className="log">
              {rec.corrections.map((c) => (
                <li key={c.seq}>{c.field}: {c.old_value ?? "empty"} changed to {c.new_value ?? "empty"}</li>
              ))}
            </ol>
            <p className="note">Corrections are kept in order and the checks re-run on every change, so any earlier state can be rebuilt.</p>
          </section>
        )}
        <p className="provenance">
          Tariff {r.verdict.tariff_version ?? "?"}, sources:{" "}
          {r.verdict.tariff_sources.map((s, i) => <a key={s} href={s} rel="noreferrer" target="_blank">Ofgem {i + 1} </a>)}
          {" "}Timings: {Object.entries(r.timings).map(([k, v]) => `${k} ${v.toFixed(1)}s`).join(", ")}.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- shell

function Upload({ onDone }: { onDone: (id: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [hh, setHh] = useState("");
  const hhs = useQuery({ queryKey: ["households"], queryFn: listHouseholds });
  const up = useMutation({ mutationFn: () => uploadBill(file!, hh), onSuccess: (r) => onDone(r.id) });
  return (
    <form className="upload" onSubmit={(e) => { e.preventDefault(); if (file) up.mutate(); }}>
      <label>Bill image <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></label>
      <label>Smart-meter history
        <select value={hh} onChange={(e) => setHh(e.target.value)}>
          <option value="">None</option>
          {hhs.data?.map((h) => <option key={h} value={h}>{h}</option>)}
        </select>
      </label>
      <button type="submit" disabled={!file || up.isPending}>Check this bill</button>
      {up.isError && <span className="err">{(up.error as Error).message}</span>}
    </form>
  );
}

export default function App() {
  const [id, setId] = useState<string | null>(() => new URLSearchParams(location.search).get("bill"));
  const demos = useQuery({ queryKey: ["demo"], queryFn: demoIndex, enabled: DEMO });
  const entries = demos.data ?? [];
  const current = useMemo(() => entries.find((e) => e.id === id), [entries, id]);
  // an old ?bill= link may name an example that was re-recorded under another id
  useEffect(() => { if (DEMO && entries[0] && !current) setId(entries[0].id); }, [entries, current]);
  useEffect(() => {
    if (id) history.replaceState(null, "", `?bill=${id}`);
  }, [id]);

  return (
    <>
      <header className="top">
        <div className="brand">
          <h1>Second Look</h1>
          <p>Checks a UK electricity bill line by line against the Ofgem price cap.</p>
        </div>
        {DEMO ? (
          <nav className="examples" aria-label="Recorded examples">
            {entries.map((e) => (
              <button key={e.id} type="button" aria-pressed={e.id === id} onClick={() => setId(e.id)}>{e.title}</button>
            ))}
          </nav>
        ) : <Upload onDone={setId} />}
      </header>
      {DEMO && (
        <p className="recorded">
          <strong>Recorded example.</strong> These results were produced by the full pipeline on a laptop and saved;
          no model runs on this page. {current?.note}
        </p>
      )}
      <main>
        {id && (!DEMO || current) ? <BillView key={id} id={id} demo={current} /> : (
          <p className="state-msg">
            {DEMO ? (demos.isError ? "The recorded examples could not be loaded." : "Loading the recorded examples…")
              : "Choose a bill image to check. Everything runs on this computer."}
          </p>
        )}
      </main>
      <footer>
        All bills here are synthetic, from fictional suppliers. Usage comes from the Low Carbon London smart-meter
        trial (CC BY 4.0), moved forward in time to 2025–2026. Supports single-rate price-cap electricity only.
      </footer>
    </>
  );
}
