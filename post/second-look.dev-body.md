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

I measured it in four experiments. The headline is the one I'm proudest of because it cost the most:

> On 75 bill images (25 synthetic bills, each clean, scanned and phone-photographed), Gemma reading
> the bill on its own gave **22 confident wrong verdicts**. Second Look gave **0**. The price is caution:
> on 42 of the 75 it said "Not enough evidence" and pointed at the field to check.

A tool that is sometimes confidently wrong about money is worse than no tool. I'd rather ship the
one that knows when to stop.

## Demo

**Live demo: https://second-look-demo-xj2a.onrender.com**

The hosted demo is **recorded**. Each example is a saved output of the full pipeline (Gemma,
Tesseract, the checks, TabPFN, the explanation) run on my laptop, and every page says so. No model
runs on Render, and the bills are synthetic, from fictional suppliers. There are four examples:

- **A bill that adds up.** A scanned bill; every line matches the cap, so it's *Reconciled*.
- **A planted VAT error.** The bill charges 5% VAT after it fell to 0% on 1 October 2026. Second Look
  says *Discrepancy*, +£6.33, and shows which check failed and why.
- **A misread, then corrected.** Gemma misread two end dates. Tesseract couldn't find them on the
  bill, so the verdict is *Not enough evidence* and the two fields are outlined in red. Click
  **Edit** on each and enter `2025-06-30`, the date printed on the bill: every check re-runs, and
  the verdict becomes *Reconciled*.
- **Instructions printed on the bill.** The bill tells AI reviewers to claim a £250 overcharge. It's
  read as data and ignored: *Reconciled*.

![The planted VAT error: Discrepancy, billed £133.01, recomputed £126.68](https://raw.githubusercontent.com/ANIRUDDH-001/second-look/main/post/images/vat-dark.png)

Hover any field and its box lights up on the bill. Here Gemma misread two dates, so the failed
checks can't be trusted yet:

![A misread bill: Not enough evidence, two values not found on the bill](https://raw.githubusercontent.com/ANIRUDDH-001/second-look/main/post/images/misread-then-corrected-dark.png)

To run it for real on your own images, follow the README; everything runs locally.

## Code

{% github ANIRUDDH-001/second-look %}

MIT licensed. The README has the full setup (Python 3.11, Node, Tesseract, Ollama), what is real,
what is synthetic and what is recorded, and every result table. `python -m eval.run_all` regenerates
every number in this post.

## How I Built It

The rule I held to throughout: **the model transcribes, code reconciles, TabPFN puts the usage in
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

**2. Make every value prove it is on the page.** This is the part that changed most while building.
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
- **Tests:** 55, run with pytest. They cover tariff prorating, the VAT switch, the retry rule, the
  grounding rules, the evidence gate, the explanation guard and correction replay.

### What the experiments found

All bills are synthetic, in three layouts I made, so these numbers say nothing about real
suppliers' bills. That limit matters; please read them with it in mind.

**A. Reading the bill.** 75 images. The verdict is what the app would say from each reading.

| | Right verdict | "Not enough evidence" | **Wrong verdict** | Seconds per bill |
|---|---|---|---|---|
| Tesseract + regex | 42 | 17 | 16 | 2 |
| Gemma 4 E2B alone | 53 | 0 | 22 | 104 |
| **Second Look** | 33 | 42 | **0** | 106 |

Gemma alone gets more verdicts right, and more wrong. Second Look's caution shows up in the details:

- On the 36 bills with a planted error, it said *Discrepancy* 8 times and *Not enough evidence* 28 times.
  It never called one of them *Reconciled*.
- On the 27 correct bills, it said *Reconciled* 13 times and *Not enough evidence* 14 times.
- Field accuracy was 93% on clean images, 94% on scans and 90% on phone photos. Tesseract alone
  managed 85%, 69% and 24%.

The first version of the evidence gate never fired. Failed checks named a group (`lines[0]`), and
unverified fields were leaves (`lines[0].end`), so the two never matched. Fixing that one comparison
took confident wrong verdicts from 10 to 0. That is the kind of bug you only find by counting.

**B. The checks.** 140 bills read from the answer key, so no model is involved. All 7 planted fault
classes were caught on 10 of 10 bills each, with 0 false alarms on 60 correct bills. All 10
unsupported tariffs were refused.

**C. Usage context.** 80 London homes × 6 forecast dates = 480 forecasts of a 30-day period.

| | MAE (kWh/day) | MASE | 90% coverage | 90% width | Seconds |
|---|---|---|---|---|---|
| **TabPFN v2, lag features** (shipped) | **1.19** | 1.03 | 94% | 6.5 | 0.9 |
| TabPFN-TS | 1.20 | 1.00 | 98% | 11.2 | 22.6 |
| Last period | 1.35 | 1.12 | | | |
| Mean of last 3 periods | 1.69 | 1.36 | | | |
| Same period last year | 1.87 | 1.54 | | | |

Both TabPFN variants beat every naive baseline and tie each other on accuracy. The lag model's range
is half as wide, still covers 94%, and runs 25 times faster, so that is the one in the app.

**D. Trying to mislead it.**

- I printed instructions on six bills: report the total as £0.00, set the units to 1, claim a £250
  overcharge. No injected value was ever copied. Three got the right verdict, three got *Not enough
  evidence*, and none got a wrong one.
- In 12 explanation cases, the guard caught one model text saying "overcharge" and "refund" on a bill
  that couldn't support it, and showed the template wording instead. Unsupported claims shown: 0.
- After a correction, the verdict always reflected the corrected bill.

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

I built this with Claude Code as a pair. I set the rules (model transcribes, code reconciles), the
four verdicts and the experiments, and kept every number honest. It wrote most of the code and ran
the experiments. The moments that mattered weren't code generation. They were the counts:

- Experiment A showed 10 confident wrong verdicts that should have been caught. Tracing them found
  the evidence gate comparing `lines[0]` against `lines[0].end` and never matching.
- One bill got a false *Discrepancy* because the rate repair borrowed a number from a different row
  of the bill. The repair now only looks at the line's own row, and there's a test for it.
- My laptop kept overheating, so every model run was one job at a time, paused and resumed by
  process id, with Gemma's replies cached so a rescore never re-ran the model.

## The Lazy Payoff

Checking one bill by hand means finding the right quarter's cap on Ofgem's site, splitting the
period at any rate change, counting standing-charge days on each side, working out the net rate,
checking the meter advance, the VAT rate for the date and the total. It's about ten sums that most
people never start.

Second Look does it in **under two minutes per bill on a laptop CPU** (86 to 111 seconds end to end
for the four demo bills). Roughly half of that is
Gemma reading the bill, most of the rest is the usage forecast, and the checks themselves take a
few milliseconds. You don't have to watch it; you come
back to a verdict, and the only thing it ever asks of you is to look at a highlighted field.

The bigger payoff is the *Reconciled* verdict. It means you can stop thinking about that bill,
because every line was recomputed from the published rates. *Not enough evidence* is the honest
version of "I'm not sure", with the exact field to check, so the lazy path never becomes the
wrong one.

I didn't time myself checking bills by hand, so I'm not quoting a speed-up.

## Prize Categories

- **Gemma.** Gemma 4 E2B (`gemma4:e2b-it-qat`) runs locally in Ollama, on the CPU. It does two jobs:
  it reads the bill image into a strict schema, and it writes the plain-English explanation under a
  guard. Experiment A measures it against Tesseract on 75 images, and Experiment D tries to trick it.
- **TabPFN.** The TabPFN v2 regressor forecasts a 90% range for each billing period's usage from the
  home's own smart-meter history, locally with telemetry off. Experiment C compares it with
  TabPFN-TS and three naive baselines over 480 forecasts, and it wins on accuracy and interval width.
- **Render.** The demo front end is a Render static site deployed from the repository's
  `render.yaml` Blueprint. It replays recorded runs of the real pipeline, so anyone can see it
  without installing a model.

---

*Disclosure: I built this with an AI coding assistant (Claude Code), which wrote much of the code
under my direction. Every number in this post comes from scripts in the repository and can be
reproduced with `python -m eval.run_all`. All bills shown are synthetic, from fictional suppliers.*
