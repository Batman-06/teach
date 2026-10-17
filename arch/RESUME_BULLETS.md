# Resume Bullets — Forecasting → PCA/Gappy-POD → Optimization

Reordered and rewritten to lead with **forecasting**, then the new **PCA + Gappy-POD retrieval/estimation** work, then **optimization**, with drift-monitoring and infra as supporting points. Metrics marked `~` are **estimates** — confirm against your real numbers before using.

---

## Probabilistic Forecasting Framework *(lead)*

- **Built an end-to-end probabilistic forecasting pipeline from the ground up** — combining Monte Carlo simulation and XGBoost to produce electricity demand, solar, and wind generation profiles, replacing ad-hoc Excel workflows with a reproducible, automated system.
- **Improved forecast accuracy by ~20% (MAPE)** while delivering **calibrated uncertainty intervals** (validated for coverage, e.g. ~90% intervals holding ~90% empirically), verified through rigorous backtesting and hypothesis testing to enable risk-aware capacity decisions.
- **Generated full-year 8760-hour profiles** with per-hour uncertainty bands across 3 energy vectors (demand/solar/wind), supporting scenario analysis over `~N` sectors/regions.
- **Translated forecast gains into ~30% lower projected operating expenses (~$2–3M)** on simulated demand scenarios — directly tying model quality to business impact.

---

## Profile Retrieval & Partial-Data Estimation — PCA + Gappy POD *(new)*

- **Designed a retrieval-augmented estimation layer** that lets a user with only **partial** demand data (a few representative days instead of a full year) obtain a complete 8760-hour profile — cutting the onboarding data requirement by **~90%**.
- **Compressed each 8760-hour profile into a ~10-dimensional PCA shape-embedding** retaining **>95% of variance**, shrinking the similarity-search footprint by **~800×** and enabling sub-**~50 ms** k-NN retrieval of comparable historical profiles.
- **Implemented Gappy POD (masked least-squares projection)** to reconstruct missing hours directly from the PCA basis — recovering full-year shapes from as little as **~10–15% observed data** at **~5–8% reconstruction MAPE**, unifying retrieval and gap-filling into a single operation.
- **Separated shape from magnitude** (normalized profiles for matching, re-scaled by annual energy/PUE) so retrieval generalizes across customers of very different sizes within the same sector.
- **Hardened the stage with a validation + guardrail gate** — held-out reconstruction benchmarks across realistic missing-data patterns, per-request confidence scoring, and basis-versioning with drift-triggered refits — preventing silently-wrong estimates from reaching the optimizer.

---

## Energy-System Sizing Optimization *(focus)*

- **Replaced a legacy Differential Evolution + L-BFGS-B optimizer (>10 s solve) with Bayesian Optimization + L-BFGS-B**, cutting solve time to **2–3 s (~70–80% faster)** — fast enough for **live in-browser trial runs** showing instant ROI previews (~20% savings).
- **Architected a dual-solver pattern**: a fast heuristic/BO solver returns an estimated cost/time result in-session, while a **parallel LP model** runs asynchronously to compute the near-global optimum — giving users instant feedback without sacrificing eventual exactness.
- **Built an automated calibration loop** that compares heuristic vs LP results per (country·sector·location·year) group on a daily cadence, analyzes the deviation distribution (p75/p90 tails), and **auto-tunes the heuristic's hyper-parameters** to keep the optimality gap within an acceptable band — **holding the heuristic within ~X% of global optimum** while preserving browser-speed response.
- **Warm-started the LP from the heuristic solution**, reducing LP solve time by **~Y%** and yielding the optimality gap as a free by-product.

---

## Drift Monitoring & Benchmarking *(support)*

- **Engineered an event-driven, asynchronous drift-monitoring pipeline** (async I/O + concurrent workers) running distribution-shift checks across MCP, solar/wind, demand, and tariff/PPA inputs plus optimization outputs — **without blocking the core optimization workload**.
- **Formulated a parallel LP benchmark** to continuously validate the lightweight BO solution's near-global optimality, catching regressions before they reached users.

---

## Deployment & Infrastructure *(support)*

- **Containerized all frameworks with Docker** and automated deployments on **AWS via CI/CD** (GitHub Actions), ensuring reproducible builds and consistent dev/staging environments.
- **Designed the serving architecture** around DynamoDB (hot session results, TTL), S3 + Athena (analytics/comparison layer), and SQS + Batch/Fargate (async LP jobs) — a split-by-access-pattern store supporting both live sessions and the daily calibration batch.

---

### Fill-in checklist (replace the `~`/placeholders with real numbers)

- `~N` sectors/regions covered by the forecasting pipeline.
- Exact PCA dimension and variance retained on your real catalogue.
- Real reconstruction MAPE vs % observed (from your held-out tests).
- Retrieval latency measured end-to-end.
- Heuristic-vs-LP optimality gap `~X%` and LP warm-start speedup `~Y%`.
- Interval calibration coverage number.
