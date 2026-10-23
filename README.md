# Second Look

Checks a UK electricity bill line by line against the Ofgem price cap, and shows the evidence for
every number it uses.

You give it a photo or scan of a bill. It tells you one of four things:

| Verdict | Meaning |
|---|---|
| **Reconciled** | Every line recomputes from the Ofgem cap for those dates, and the total matches. |
| **Discrepancy** | At least one check fails. You see which line, what was expected and what was billed. |
| **Not enough evidence** | A value it needs is missing, unreadable or estimated, so it won't guess. |
| **Tariff not supported** | The bill isn't single-rate, price-cap electricity for a period it has rates for. |

It also says whether the period's usage looks typical for that home, using a TabPFN forecast from
smart-meter history. That is context only. It never decides the verdict.

## How it works

The model transcribes, code reconciles, TabPFN puts the usage in context, and the assistant explains.

1. **Read.** Gemma 4 E2B (`gemma4:e2b-it-qat` in Ollama, on your machine) copies the bill's values into
   a strict JSON schema. The prompt tells it to copy, never to calculate. A retry only ever sends
   feedback about the format, never about the arithmetic, so the model is never nudged to "fix" the
   discrepancies we are looking for.
2. **Find each value on the bill.** Tesseract OCR reads the same image on its own. Each value Gemma
   copied must be found where it belongs:
   - a charge line's dates and rate on that charge's own row;
   - the subtotal on its labelled line.

   Anything not found is flagged for you to check. Where Gemma shortened a printed rate, for example
   54.48 for 54.486, the printed digits are used and the change is shown to you.
3. **Reconcile.** Plain Python with `Decimal` recomputes every line from an effective-dated table of
   Ofgem cap rates (2025 Q1 to 2026 Q4, each row citing its source). Periods that cross a quarter are
   split, and VAT falls from 5% to 0% on 1 Oct 2026. This step gives the four-state verdict, which
   records the tariff version it used.
4. **Context.** TabPFN v2 forecasts a 90% range for the period's kWh. Two variants were compared with
   naive baselines in Experiment C:
   - a regressor on lag features;
   - TabPFN-TS on the daily series.
5. **Explain.** Gemma writes two or three sentences from the computed results only. A guard rejects
   any number that isn't in the results, and any wording the verdict can't support, such as
   "overcharge" on a reconciled bill. A rejected text is replaced by fixed template wording.

**Corrections.** Correct any field in the UI. Each correction is appended to a log and every check
re-runs, so any earlier state can be rebuilt. Uploads are idempotent: the id is a hash of the image
content.

## What is real, what is synthetic, what is recorded

- **Bills are synthetic.** All of them come from fictional suppliers (Northwind Energy, Brightside
  Power, Meridian Utilities) in three layouts made for this project. Copies are degraded to look like
  scans and phone photos. Faults are planted on purpose, with an answer key for each. **Extraction is
  measured on our own layouts only, not on real utility bills.**
- **Usage is real public data, moved in time.** The [Low Carbon London smart-meter
  trial](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households) is
  UK Power Networks data under CC BY 4.0: half-hourly readings from London homes, Nov 2011 to
  Feb 2014. We keep only standard-tariff homes, aggregate the readings to daily totals, and shift
  every date forward by 678 weeks, so weekdays are kept and the data falls in 2025–2026 where we have
  tariff rates. The homes are anonymised household ids.
- **Tariff rates are public.** They are Ofgem cap rates for the GB average, paid by Direct Debit,
  single-rate electricity. Each row cites its Ofgem page in
  [`ofgem_caps.yaml`](backend/secondlook/tariff/ofgem_caps.yaml).
- **The hosted demo is recorded.** The [Render site](https://second-look-demo-xj2a.onrender.com) is a static page that shows saved outputs of the
  full pipeline, each labelled "Recorded example". No model runs there.
  [`eval/record_fixtures.py`](eval/record_fixtures.py) regenerates them from real API calls.

## Supported, and not

Supported: single-rate electricity on the Ofgem price cap, Direct Debit, the GB-average rate, for
periods from 1 Jan 2025 to 31 Dec 2026.

Not supported, so these get "Tariff not supported": gas, Economy 7 and other time-of-use tariffs,
fixed deals, regional rates, prepayment, and dual-fuel bills. Every result says so; none is guessed.

## Run it locally

You need:

- Python 3.11
- Node 20 or later
- [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- [Ollama](https://ollama.com) with `ollama pull gemma4:e2b-it-qat`

```bash
python -m venv .venv
```
```bash
.venv/Scripts/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```
```bash
.venv/Scripts/pip install -r requirements.txt
```

Data and weights are not in the repository:

- Put the TabPFN v2 regressor checkpoint at `models/tabpfn-v2-regressor.ckpt`.
- Unzip the LCL files to `data/raw/lcl`.
- Build the daily usage and the synthetic bills:

```bash
.venv/Scripts/python -m datagen.lcl_aggregate data/raw/lcl data/processed/lcl_daily.parquet
```
```bash
.venv/Scripts/python -m datagen.make_dataset
```

Start the API and the UI:

```bash
cd backend && ../.venv/Scripts/python -m uvicorn secondlook.api.app:app --port 8000
```
```bash
cd frontend && npm ci && npm run dev
```

Open http://localhost:5173. Telemetry is off: `TABPFN_DISABLE_TELEMETRY=1` is set before TabPFN
loads, and both models run locally on the CPU.

Tests:

```bash
.venv/Scripts/python -m pytest
```

## Experiments

Every number in the write-up comes from a script in `eval/`, which writes CSVs to `eval/results/`.

| | Question | Script |
|---|---|---|
| A | Does local Gemma read bills well enough, compared with Tesseract and regex? | `eval/exp_a_extract.py` |
| B | Do the checks catch planted faults without false alarms? | `eval/exp_b_faults.py` |
| C | Does TabPFN forecast usage better than naive methods? | `eval/exp_c_forecast.py` |
| D | Can text on the bill or in a note push it into a false claim? | `eval/exp_d_adversarial.py` |

### Results

**A. Reading the bill** (75 images: 25 bills, each clean, scanned and phone-photographed; our layouts only).
The verdict here is what the app would say from that reading.

| | Right verdict | "Not enough evidence" | **Wrong verdict** | Seconds per bill |
|---|---|---|---|---|
| Tesseract + regex | 42 | 17 | 16 | 2 |
| Gemma 4 E2B alone | 53 | 0 | 22 | 104 |
| **Second Look** (Gemma, checked against OCR, evidence gate) | 33 | 42 | **0** | 106 |

Second Look never gave a wrong verdict, but it is cautious. On 36 bills with a planted billing
error it said *Discrepancy* 8 times and *Not enough evidence* 28 times. It never said
*Reconciled*, and it always highlighted the line to check. On 27 correct bills it said *Reconciled*
13 times and *Not enough evidence* 14 times. Field accuracy was 93% on clean images, 94% on scans
and 90% on phone photos. Tesseract alone managed 85%, 69% and 24%.

**B. The checks** (140 bills, read from the answer key, so no model is involved): every one of 7 planted fault
classes was caught on 10 of 10 bills. There were 0 false alarms on 60 correct bills, and all 10
unsupported tariffs were refused.

**C. Usage context** (80 London homes × 6 forecast dates = 480 forecasts of a 30-day period):

| | MAE (kWh/day) | MASE | 90% interval coverage | 90% width (kWh/day) | Seconds |
|---|---|---|---|---|---|
| **TabPFN v2, lag features** (shipped) | **1.19** | 1.03 | **94%** | 6.5 | 0.9 |
| TabPFN-TS (v2 weights) | 1.20 | 1.00 | 98% | 11.2 | 22.6 |
| Last period | 1.35 | 1.12 | | | |
| Mean of last 3 periods | 1.69 | 1.36 | | | |
| Same period last year | 1.87 | 1.54 | | | |

Both TabPFN variants beat the naive forecasts, and they are equally accurate. The lag model's
interval is half as wide and still covers 94%, at 1/25 of the time, so it is the one shipped.

**D. Attempts to mislead it** (18 cases, run through the full pipeline):

- **Six bills with instructions printed on them**, telling an assistant to report the total as £0.00, to set
  the units to 1, or to claim a £250 overcharge. No injected value was ever copied into the reading. Three got the verdict the answer key
  expects; the other three got *Not enough evidence*; none got a wrong verdict.
- **Twelve explanation cases**: a note asking it to invent an overcharge, conflicting fields, a period outside
  the tariff table, and missing fields. The guard rejected one model text (it said "overcharge" and "refund"
  on a *Not enough evidence* bill) and showed the template wording instead. Unsupported claims shown: 0.
- **A correction re-runs everything.** After correcting a rate, the verdict moved from *Discrepancy* to
  *Reconciled*; no stale verdict was shown.

## Licences

- **This code:** MIT, see [`LICENSE`](LICENSE).
- **Gemma 4:** Apache 2.0.
- **TabPFN v2 weights:** Prior Labs licence, **non-commercial use**. TabPFN-TS 1.3 defaults to the
  gated TabPFN 3.5 checkpoint; here it is given the v2 checkpoint explicitly.
- **Low Carbon London data:** CC BY 4.0, from UK Power Networks via the London Datastore.
- **Ofgem rates:** public information from ofgem.gov.uk.

Second Look is not financial advice and is not affiliated with Ofgem or any supplier.
