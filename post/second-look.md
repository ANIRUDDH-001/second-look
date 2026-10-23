---
title: TITLE_TBD
published: false
tags: devchallenge, hf26challenge, opensource, ai
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 3: Slow and Steady](https://dev.to/challenges/hacktoberfest-week3-2026-10-19)*

## What I Built

**Second Look** checks a UK electricity bill line by line against the Ofgem price cap, and shows
where on the bill every number came from.

Checking a bill is the kind of job everybody skips. You need the cap rates for the right quarter,
the days on each side of a rate change, the standing charge per day, and the VAT rate, which falls
to 0% on 1 October 2026. Then you do the sums. Nobody does this, so nobody notices when a bill
charges the old quarter's rate, counts a day twice or adds VAT that no longer applies.

Second Look does the sums, and it is careful about what it claims. Every bill gets one of four
verdicts:

| Verdict | When |
|---|---|
| Reconciled | Every line recomputes from the cap for its dates, and the total matches. |
| Discrepancy | A check fails on values that are printed on the bill. You see the line, the expected value and the billed one. |
| Not enough evidence | A value is missing, estimated, or could not be found printed on the bill. It won't guess. |
| Tariff not supported | Anything that isn't single-rate price-cap electricity for a period it has rates for. |

It also says whether the period's usage looks normal for that home, as a TabPFN forecast range from
smart-meter history. That is context only, and it never changes the verdict.

NUMBERS_TBD

## Demo

DEMO_TBD

## Code

CODE_TBD

## How I Built It

The rule I held to all weekend: **the model transcribes, code reconciles, TabPFN puts the usage in
context, and the assistant explains.** No model ever produces a number the verdict depends on.

```
bill image
   │
   ├─► Gemma 4 E2B (Ollama, CPU) ── copies values into a strict JSON schema
   │                                 "you are a copier, not a calculator"
   ├─► Tesseract OCR ────────────── a second, independent reader
   │
   ▼
find each value on the bill ── where it belongs: a line's date on that line's row,
   │                              the subtotal on its labelled line; else flag it
   ▼
reconcile (Python, Decimal) ── Ofgem cap table, prorated across quarter changes,
   │                              VAT 5% → 0% on 1 Oct 2026  →  one of four verdicts
   ├─► TabPFN ─── 90% range for this period's kWh from the home's history (context only)
   ▼
explain (Gemma) ── 2–3 sentences from the computed payload; a guard rejects any
                   number not in the payload and any claim the verdict can't back
```

**1. Transcribe, don't compute.** Gemma 4 E2B gets the bill image and a JSON schema where every
value is a string. The prompt says to copy values exactly as printed and to use null when something
is missing. If the reply is malformed, it gets up to two retries, and the feedback is **only ever
about the format**. Telling a model "your total doesn't add up" is an invitation to make it add up,
and that would quietly erase the exact discrepancies the tool exists to find.

**2. Make every value prove it is on the page.** This is the part that changed most over the weekend.
Gemma is a good reader but not a perfect one. On my bills it had three habits:

- it shortened rates: `54.486p` became `54.48`;
- it copied the total into the subtotal;
- it put the billing period's end date on a charge line that actually ends mid-period.

So Tesseract reads the page independently, and each value Gemma copied has to be **found where it
belongs**. A charge line's dates and rate must be on that line's row, and the subtotal must be on the
line labelled *Subtotal* or *Before VAT*. Searching the whole page wasn't enough, because the
period end date is printed on every bill. A value that isn't found is shown with a red dashed
outline and the words "Not found on the bill, check it".

There is one narrow repair. When exactly one printed number extends the digits Gemma gave (Gemma
read `54.48`, the bill says `54.486`), the printed one is used and the UI says so.

**3. A discrepancy has to rest on evidence.** If a check fails on a value that wasn't found on the
bill, the failure might be a misread, not a billing error. The verdict then becomes "Not enough
evidence", and the field is highlighted. Second Look only says *Discrepancy* when every value behind
the failure is printed on the bill.

**4. Reconcile in plain code.** The Ofgem cap for 2025 Q1 to 2026 Q4 lives in a YAML file. Each row
has its effective dates and the Ofgem URL it came from. A billing period that crosses a quarter
boundary is split by day. Money is `Decimal`, and the net rate is the published rate divided by
1 + VAT, to three decimal places. The checks are:

- each line's rate against the cap for its dates;
- quantity × rate = amount;
- the standing charge covers each day exactly once;
- meter advance = units billed;
- the VAT rate for the date;
- subtotal + VAT = total.

**5. Context, not judgement.** TabPFN forecasts a 90% range for the period's kWh from the
home's own history. Two variants were measured against naive baselines (Experiment C):

- the TabPFN v2 regressor on lag features;
- TabPFN-TS on the daily series.

A bill can be correct and still unusually high, so the band is labelled "It describes usage, not
whether the bill is right."

**6. Explain, but only what was computed.** Gemma writes two or three sentences from a JSON
payload of the results. Then a guard checks them:

- every number in the text must appear in the payload;
- each verdict has words it may not use, for example "overcharge" on a reconciled bill.

If either check fails, the text is replaced with fixed template wording, and the UI shows which one
you got.

**Corrections are a log, not an edit.** Each correction is appended, the bill is rebuilt from the
extraction plus the log, and every check re-runs. Uploads are keyed by a hash of the image, so
uploading twice never runs twice.

The stack:

- **Backend:** FastAPI with one worker thread and SQLite.
- **Frontend:** React and TypeScript with TanStack Query. The bill sits beside the field list, and
  hovering a field lights up its box on the bill.
- **Tests:** 54, run with pytest. They cover tariff prorating, the VAT switch, the retry rule, the
  grounding rules, the evidence gate, the explanation guard and correction replay.

## Why Does Open Innovation Matter?

A bill shows your name, your address, your meter number, and a record of when you are home. That is
exactly the document you shouldn't upload to a cloud service to "check it for you". Open weights
are what make the private version possible: Gemma 4 E2B runs on a laptop CPU, and so does TabPFN.
The bill never leaves the machine. I checked that by watching the processes' network connections
while every experiment ran, and there were no connections beyond localhost.

Open data does the other half. The usage history is real smart-meter data from the Low Carbon London
trial, published by UK Power Networks under CC BY 4.0. I used 396 of its standard-tariff homes. The tariff is Ofgem's published cap.
Every rate in the table links to its Ofgem page, so anyone can audit the auditor.

## My Agent Session

AGENT_SESSION_TBD

## The Lazy Payoff

LAZY_PAYOFF_TBD

## Prize Categories

PRIZES_TBD

---

*Disclosure: I built this with an AI coding assistant (Claude Code), which wrote much of the code
under my direction. Every number in this post comes from scripts in the repository and can be
reproduced with `python -m eval.run_all`. All bills shown are synthetic, from fictional suppliers.*
