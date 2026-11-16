# Derivatives feasibility (T13) — conclusion: do not activate

**Prepared:** 14 November 2026. Facts below are from the cited official sources, retrieved on that date, or from this
project's own data and code, as stated. Nothing here is investment advice, and **no derivatives code is enabled**: the
bot trades Indian cash equities only (ADR-0020, ADR-0025). `strategies/nifty_options_seller.py` is intentionally not
wrapped by the strategy adapters.

## 1. Contract sizes against a ₹10,000 paper capital

| Underlying | Lot size | Source |
|---|---|---|
| NIFTY | **65** | NSE circular NSE/FAOP/70616 (3 Oct 2025, ref. 176/2025): <https://nsearchives.nseindia.com/content/circulars/FAOP70616.pdf>. Effective 28 Oct 2025 EOD; weekly/monthly contracts, and quarterly/half-yearly from 30 Dec 2025 EOD. |
| BANKNIFTY | **30** | Same circular. |
| FINNIFTY / MIDCPNIFTY | 60 / 120 | Same circular. |

The project's own contract master (`data/cache/instrument_master.json`, Angel One, retrieved 14 Nov 2026) shows the same
lot sizes (65, 30, 60, 120), which is a consistency check, not an independent exchange source.

**Notional.** One NIFTY lot controls `65 × index level`. As an *illustration only* (the index level is an assumption, not a
quote): at 25,000 one lot is ₹16.25 lakh of exposure, 162× a ₹10,000 account; one BANKNIFTY lot (30) is of the same order.
The exchange margin (SPAN plus exposure margin) for a futures position is a fraction of that notional and is published
by the clearing corporation and the broker's margin calculator; it must be read there on the day. It is far larger than
₹10,000 for futures and for short options.

**Defined-risk spreads do not rescue a ₹10,000 account.** A risk-defined vertical spread loses at most
`strike width × lot size` before costs. In the contract master the near-expiry strike spacing is 50 points for NIFTY (verified for the 1 Dec 2026 expiry) and 100 for BANKNIFTY (23 Nov 2026), so the
smallest NIFTY spread risks `50 × 65 = ₹3,250` (32.5% of ₹10,000) and the smallest BANKNIFTY spread `100 × 30 = ₹3,000`.
The system's planned-risk limit is **0.5% of equity = ₹50** per entry. A spread would need a width under one point, which
does not exist. At ₹100,000 the same 0.5% limit is ₹500, still far below ₹3,000+. **No listed index-option spread
fits the unchanged risk rules at either capital.** Buying a single option to "limit" risk still loses the whole premium
(a lot of ATM weekly premium is typically several thousand rupees), which again breaks the 0.5% limit.

## 2. Expiry calendar

Expiry weekdays are set by the exchange and shift for holidays, so they must be read from the contract master, never
assumed (ADR-0022). From the master on 14 Nov 2026:

- NIFTY: 17 Nov 2026 (Tue), **23 Nov 2026 (Mon, because 24 Nov is a holiday)**, 1/8/15 Dec (Tue), then 29 Dec (Tue) and
  later monthlies; the previous hard-coded Thursday rule was wrong.
- BANKNIFTY: monthly only (23 Nov, 29 Dec, 25 Jan 2027, …); no weekly series are listed.

`data.instrument_master.is_fno_expiry_day` implements the lookup and returns *unknown* (`None`) when the master is unavailable.

## 3. Stock options and physical settlement

SEBI moved stock derivatives to physical settlement by circular (SEBI/HO/MRD/DP/CIR/P/2018/67, 11 Apr 2018, extended by
SEBI/HO/MRD/DOPI/CIR/P/2018/161 of 31 Dec 2018 and SEBI/HO/MRD/DOP1/CIR/P/2019/28 of 8 Feb 2019,
<https://www.sebi.gov.in/sebi_data/attachdocs/feb-2019/1549625911612.pdf>, now subsumed in the Master Circular for Stock
Exchanges and Clearing Corporations). In-the-money stock options and stock futures that are held to expiry are settled by
**delivery of the shares**, so a small account can be assigned a position worth many times its capital unless it closes
before expiry. Index derivatives are cash-settled. The bot has no assignment or delivery workflow; it would have to flatten
every stock-derivative position before expiry, which is an additional failure mode to engineer and test.

## 4. Taxes and costs

- **STT** (Finance (No. 2) Act, 2024; NSE circular NSE/FATAX/63809, effective 1 Oct 2024, <https://www.kotakneo.com/uploads/FATAX_63809_06c32787d5.pdf>
  and BSE notice 20240916-61): sale of an option **0.10%** of premium (was 0.0625%), sale of a future **0.02%** (was 0.0125%),
  and 0.125% on an exercised option's intrinsic value, payable by the purchaser.
- Brokerage, exchange transaction charges, GST, SEBI fees and stamp duty add per-order costs; `backtester/costs.py` models
  the equity cash segment only and has **no derivatives cost model**. Costs matter more for a ₹10,000 account because
  per-order brokerage does not scale with size.

## 5. Evidence about outcomes

SEBI's study of individual traders in equity F&O reported that **93%** of over one crore individuals lost money between
FY22 and FY24, average loss about ₹2 lakh including costs, aggregate losses above ₹1.8 lakh crore
(PR No. 22/2024, 23 Sep 2024: <https://www.sebi.gov.in/media-and-notifications/press-releases/sep-2024/updated-sebi-study-reveals-93-of-individual-traders-incurred-losses-in-equity-fando-between-fy22-and-fy24-aggregate-losses-exceed-1-8-lakh-crores-over-three-years_86906.html>).
A follow-up study found nearly **91%** of individual traders lost money in FY25 and net losses widened to ₹1,05,603 crore
(PR No. 39/2025, 7 Jul 2025: <https://www.sebi.gov.in/sebi_data/attachdocs/jul-2025/1751900802566.pdf>).

## 6. Why options analytics are unavailable here

The system has **no open-interest provider**. `data/options_data.py` returns contract metadata only ("prices, OI and volume
are missing until a verified options data provider is integrated"); the options agent therefore abstains. There is no
option-chain, implied-volatility surface, Greeks feed, or OI history, and the NSE web endpoints are unauthenticated
scrapers that are not a dependable data source. Any options strategy would be flying blind, and the paper evidence that
T10 requires could not be gathered.

## 7. Conclusion

| Question | Answer |
|---|---|
| Can the unchanged 0.5% risk rule be met with an index future or option spread at ₹10,000 or ₹100,000? | No (sections 1) |
| Is there data to evaluate an options strategy? | No (section 6) |
| Is there a cost, assignment and expiry model for derivatives? | No (sections 2–4) |
| Does the evidence on retail outcomes support trying? | It argues for caution (section 5) |

**Decision: derivatives stay out of scope; no flag, adapter or strategy is activated.** Revisiting this requires, at a
minimum: an authenticated OI/option-chain provider, a derivatives cost and margin model, assignment/delivery handling for
stock options, a risk rule that is meaningful for lot-sized instruments, and T10 out-of-sample evidence.
