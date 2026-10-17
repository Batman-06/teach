# Testing, Validation & Monitoring for the PCA + Gappy POD Stage

Answering: do we need testing / validation / logging for the PCA + Gappy-POD embedding-and-reconstruction stage? **Yes — and arguably more than for the rest of the pipeline.** Here's why and exactly what to build.

---

## 0. Why it's non-negotiable here

- **Silent failure is the whole problem.** Gappy POD *always* returns numbers. If the masked least-squares is under-determined or ill-conditioned, you get a smooth, plausible-looking 8760 that is simply **wrong** — no exception, no crash.
- This stage sits **upstream of everything**: a bad reconstruction feeds the heuristic optimiser (wrong cost/time shown live) **and** the LP **and** the calibration loop (which then tunes hyper-params against corrupted data). One bad estimate contaminates three places.
- So treat validation as a **gate**: a reconstruction must prove itself before it's allowed downstream.

---

## 1. Offline validation (before deploy / on every basis refit)

- **Reconstruction accuracy via held-out masking:** take *complete* catalogue profiles, artificially mask hours, run Gappy POD, compare reconstructed vs truth. Report **MAE / RMSE / MAPE**, overall and per-hour / per-season.
- **Choose k honestly:** cross-validate reconstruction error against number of components k; combine with the **scree plot / explained-variance** curve. Pick the smallest k past the elbow — more components overfit the observed mask.
- **Robustness to the mask pattern:** this is the one people skip. Test many missing patterns, not just "random 20%":
  - random 10% / 50% / 80% missing,
  - only weekdays, only a few months, only daytime hours,
  - one contiguous block missing.
  Record how error degrades. This tells you your real-world floor, because users upload *structured* partial data, not random samples.
- **Retrieval quality (leave-one-out):** does the k-NN on PC scores actually return profiles that yield good estimates? Hold out a profile, retrieve neighbours, check the neighbour-based estimate error.
- **Conditioning check:** log the **condition number** of the reduced (masked) least-squares across test cases; establish the threshold above which results are untrustworthy.

---

## 2. Runtime guardrails (per request — the actual gate)

- **Input preconditions:** observed points **≥ k**, and **well spread** (not all in one day/hour). If not → drop k or fall back (per §6b ladder).
- **Output sanity checks:** non-negative, peak ≤ plausible capacity, annual energy consistent with PUE, no absurd ramps.
- **Residual check on observed hours:** after projection, how well does the reconstruction match the hours the user *did* give? A large residual on *known* points means the basis doesn't fit this profile → reject or fall back.
- **Emit a confidence score:** combine (residual on observed + number/spread of observations + condition number) into a 0–1 confidence. Surface it to the user in-session **and** attach it to the stored result so the calibration loop can **down-weight or exclude low-confidence estimates**.

---

## 3. What to log (per request)

Log enough to debug and to feed drift analysis later:

- `session_id`, group key (country·sector·location·year)
- **basis version** used (see §5)
- `k` used, **n observed**, mask pattern summary (e.g. % missing, coverage by month/hour)
- **reconstruction residual** on observed hours
- **condition number** of the solve
- which **fallback tier** fired (full Gappy POD / reduced-k / feature-based / metadata-only)
- output sanity pass/fail, **confidence score**
- latency

Ship these to **CloudWatch** (metrics/alarms) and persist alongside the session record in **DynamoDB → S3** so they're queryable in the same analytics layer as the LP/heuristic comparison.

---

## 4. Monitoring & drift over time

- **Data drift:** compare incoming uploads' PC-score distribution against the basis's training distribution. New sectors/countries or changing load patterns → the basis no longer represents reality.
- **Basis staleness → refit trigger:** watch **explained variance on recent data** and the **fallback / low-confidence rate**. When explained variance drops or fallback rate climbs past a threshold, **refit the PCA** on the enlarged catalogue.
- **Alerts:** rising mean residual, rising condition numbers, rising reject/fallback rate — all are early signals the stage is degrading before users complain.

---

## 5. Versioning (don't skip this)

- **Version the PCA basis** as an artifact: components, per-hour means, singular values, chosen k, training-data snapshot, and date. Store in **S3 + SSM Parameter Store** (same pattern as the Optuna hyper-params).
- **Tag every session result with the basis version** that produced it. When you refit, you can (a) reproduce old estimates, (b) A/B the new basis against the old, (c) explain a shift in the calibration loop as "basis changed," not "heuristic drifted."
- Refit is a **release**, not a hot-swap: validate the new basis offline (§1), compare to the current one, then promote.

---

## 6. Code-level tests (CI)

- **Unit:**
  - full-data projection equals exact PCA scores;
  - a synthetic profile built from known component weights is recovered within tolerance;
  - masked projection recovers a held-out curve within a pinned error threshold;
  - under-determined input (observed < k) correctly triggers the fallback, not a garbage result;
  - whitening divides by the right singular values; distance metric behaves.
- **Golden-dataset regression:** a fixed set of profiles + masks with **pinned error thresholds** — fails the build if a code change worsens reconstruction.
- **Property tests:** adding observations should not *increase* error beyond noise; shuffling complete-data hour order (with fixed convention) shouldn't change scores.
- Run these in **GitHub Actions** on every PR to the embedding code *and* on every basis refit.

---

## 7. One-line answer

- **Yes — build it, and make it a gate.** Offline reconstruction validation (incl. realistic mask patterns) + per-request guardrails with a confidence score + full logging + drift-triggered refit + a versioned basis + CI regression tests. The cost is modest; skipping it means silently wrong 8760s flowing into the optimiser and quietly corrupting the self-tuning loop.
