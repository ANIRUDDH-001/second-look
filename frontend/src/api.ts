// Live mode talks to the local FastAPI server. Demo mode (VITE_DEMO=1, the hosted build) reads
// recorded pipeline outputs from /fixtures and never runs a model.

export const DEMO = import.meta.env.VITE_DEMO === "1";

export type Box = [number, number, number, number];
export type Status = "pass" | "fail" | "skip" | "warn";
export type VerdictKind = "reconciled" | "discrepancy" | "insufficient_evidence" | "unsupported_tariff";

export interface Finding {
  check: string; component: string; status: Status; rule: string; fields: string[];
  expected: string | null; billed: string | null; detail: string | null;
}
export interface Verdict {
  kind: VerdictKind; headline: string; findings: Finding[];
  expected_total_gbp: string | null; billed_total_gbp: string | null; difference_gbp: string | null;
  tariff_version: string | null; tariff_sources: string[];
}
export interface UsageContext {
  method: string; days: number; billed_kwh: number; expected_kwh: number;
  low90_kwh: number; high90_kwh: number; label: string; history_days: number;
}
export interface Result {
  id: string; household: string | null;
  bill: Record<string, unknown> & { lines: Record<string, unknown>[]; vat: Record<string, unknown>[] };
  boxes: Record<string, Box | null>; review_fields: string[];
  verdict: Verdict; context: UsageContext | null;
  explanation: { text: string; source: "model" | "template"; rejected: string | null; reason: string | null };
  extraction: { attempts: number; problems: string[]; model_seconds: number };
  timings: Record<string, number>;
}
export interface Correction { seq: number; field: string; old_value: string | null; new_value: string | null; created_at: string }
export interface BillRecord {
  id: string; status: "queued" | "running" | "done" | "failed"; error: string | null;
  household: string | null; filename: string | null; created_at: string;
  result: Result | null; corrections: Correction[];
}
export interface DemoEntry { id: string; title: string; note: string; replay?: { field: string; value: string }[] }

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${r.status})`);
  }
  return r.json();
}

export const imageUrl = (id: string) => (DEMO ? `fixtures/${id}.png` : `/api/bills/${id}/image`);

export const demoIndex = () => fetch("fixtures/index.json").then((r) => json<DemoEntry[]>(r));

export function getBill(id: string, step = 0): Promise<BillRecord> {
  if (DEMO) return fetch(`fixtures/${id}${step ? `.step${step}` : ""}.json`).then((r) => json<BillRecord>(r));
  return fetch(`/api/bills/${id}`).then((r) => json<BillRecord>(r));
}

export function uploadBill(file: File, household: string): Promise<BillRecord> {
  const fd = new FormData();
  fd.append("file", file);
  if (household) fd.append("household", household);
  return fetch("/api/bills", { method: "POST", body: fd }).then((r) => json<BillRecord>(r));
}

export function correct(id: string, field: string, value: string): Promise<BillRecord> {
  return fetch(`/api/bills/${id}/corrections`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ field, value }),
  }).then((r) => json<BillRecord>(r));
}

export const listHouseholds = () => fetch("/api/households").then((r) => json<string[]>(r));
