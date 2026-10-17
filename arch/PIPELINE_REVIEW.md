# Demand-Profile Optimisation Pipeline — Architecture Review

A point-wise walk-through of your proposed pipeline, what works, what to change, and a cleaner end-to-end flow. No code.

---

## 0. Quick verdict

- The overall shape is sound: **retrieve a shape → estimate the 8760 → optimise fast for the session → optimise exactly in the background → reconcile and self-tune offline.**
- The genuinely clever part is the **offline calibration loop** (using LP as ground truth to tune the heuristic). That is where most of the value is, and also where most of the risk is.
- The main things to fix: *where* data lives (DynamoDB alone is wrong for some of this), making LP truly async, not running LP on *every* session, and being careful about the statistics in the tuning loop.

---

## 1. Retrieval — is it necessary, what, and how

- **Yes, retrieval is necessary — but not "RAG" in the LLM sense.** It is a lookup + nearest-neighbour problem, not a semantic-search problem.
- **Why it's needed:** the user gives you partial data + machine specs + PUE. From that you cannot *invent* an 8760-hour curve out of nothing. You need a **reference load shape** to anchor the estimate. Retrieval supplies that shape.
- **What to retrieve:**
  - First choice: an **exact key match** on `(country, sector, year, location)` — your natural DynamoDB key. That's a direct `GetItem`, not "retrieval" really.
  - Fallback when no exact match: **nearest-neighbour** on the key dimensions (same sector, nearby location, adjacent year) plus **physical features** (PUE, machine mix, installed capacity, load factor).
  - What you actually use from the retrieved profile is the **normalised shape** (the 8760 curve scaled to sum/peak = 1), then you **re-scale it to the user's magnitude** using their known quantities (capacity, PUE, uptime).
- **How to retrieve:**
  - Keep it as a **tiered lookup**: (1) exact composite-key hit → (2) relax one dimension at a time (year → location → sector) → (3) feature-based k-NN over a small embedding/feature vector of each profile.
  - Only build a vector index if your profile catalogue is large and exact matches are often missing. If most sessions hit an exact or near-exact key, a vector DB is over-engineering.
- **Bottom line:** retrieval is a *prior/template provider*, not the answer. Treat it as "give me the best-matching shape to scale," and make the fallback behaviour explicit.

---

## 2. Estimating the 8760 values

- Make this an explicit, named stage — don't fold it silently into retrieval.
- It is essentially: **retrieved normalised shape × user-specific scaling + user's partial real data overriding where present.**
- Where the user gives real hours, **trust them and keep them**; only estimate the gaps. Blend: real where known, scaled-template where unknown, smoothed at the seams.
- **Record provenance per hour** (real vs estimated vs interpolated). This matters later: deviation between LP and heuristic on *estimated* hours means something different from deviation on *real* hours.
- Validate the output: exactly 8760 (or 8784 for leap years — decide now), no negatives, peak ≤ plausible capacity, annual energy consistent with PUE. Reject or flag bad uploads before they reach the optimiser.

---

## 3. In-session dual optimisation (heuristic + LP at once)

- **Fire both, use the heuristic (Optuna) for the live session** — good. The user gets a fast "estimated time & cost" answer.
- **But do not call LP as a blocking request.** LP for 8760-hour problems can take minutes to hours. The session request must return immediately with the heuristic result and a `session_id`; LP must be a **detached background job**.
- Recommended mechanics:
  - Heuristic → synchronous endpoint, returns in-session.
  - LP → enqueue a job (SQS / a queue), run on **worker compute that can take its time** (AWS Batch or Fargate, not a short-timeout Lambda). Lambda caps at 15 min — too short for global LP in many cases.
- **Reconsider running LP for every single session.** This is likely your biggest cost. Options:
  - Run LP only when (a) the group is under-sampled, (b) the heuristic's confidence is low, or (c) on a sampled fraction of sessions.
  - Or **deduplicate by group**: if `(country, sector, location, year)` already has a fresh LP result, reuse it instead of re-solving.
- **Warm-start the LP from the heuristic solution.** It converges faster and gives you the gap for free.
- Track LP job state explicitly: `PENDING → RUNNING → COMPLETE → FAILED/TIMEOUT`. Decide what a never-finishing LP does to the daily comparison (exclude it, don't let it stall the batch).

---

## 4. Where to store results

Split by access pattern — one store does not fit all of this.

- **Demand profiles (the 8760 catalogue):**
  - 8760 floats ≈ 35–70 KB. DynamoDB's 400 KB item limit survives it, but it's wasteful and makes scans expensive.
  - Better: **metadata + key in DynamoDB, the array in S3** (one object per profile, e.g. Parquet/compressed). Or keep in Dynamo only if you always fetch by exact key and never scan.
- **In-session heuristic result + `session_id`:**
  - **DynamoDB**, keyed by `session_id`. Fast, low-latency, exactly its sweet spot. Add a **TTL** so live-session junk expires.
- **LP result (arrives later, same `session_id`):**
  - Same DynamoDB table/item, **updated in place** with a status field — or a sibling item. One `session_id` → one record that gains the LP fields when ready. Avoid a second system you then have to join.
- **For the daily comparison you need to query by group, not by session:**
  - Add a **GSI** on `(country, sector, location, year)` so the batch job can pull all sessions in a group — *or*, cleaner, **stream completed results into S3 / a warehouse** (Athena over S3, or Redshift) and run the comparison as analytics there. DynamoDB is bad at the "scan everything from yesterday and group it" pattern; a warehouse is built for it.
- **Rule of thumb:** DynamoDB for the hot, keyed, live path; S3 (+ Athena) for the cold, analytical, batch path. Don't force the batch analytics onto DynamoDB.

---

## 5. The daily comparison + calibration loop

This is the heart of the design. Keep it, but tighten it.

- **Define the deviation metric precisely.** Deviation in *what*? Recommend comparing the **objective values (total cost, total time)**, not the raw decision schedules — two near-optimal schedules can differ a lot yet cost nearly the same. The relative optimality gap `(heuristic_obj − lp_obj) / lp_obj` is the clean, comparable number.
- **Don't assume the deviation is normally distributed.** It's usually skewed and heavy-tailed (heuristics are fine most of the time, then occasionally bad). Your instinct to look at **p75/p90 tails is correct** — lean on empirical quantiles and drop the normality assumption entirely.
- **Require a minimum sample size per group before tuning.** Many `(country, sector, location, year)` groups will be sparse. Tuning hyper-parameters off 3 sessions will chase noise. Options: set a threshold (e.g. ≥ N sessions), or **aggregate hierarchically** (tune at sector level when location-level data is thin).
- **Tune against a held-out slice, not the same data you measured on.** Otherwise you overfit the heuristic to recent sessions and regress on the next batch. Keep a validation set; only accept a hyper-parameter change if the tail deviation improves on data it wasn't tuned on.
- **Version and scope the hyper-parameters.** Store the Optuna config *per group* (or per cluster of groups), with a version and an effective date. You want to roll back if a tuning round makes things worse.
- **Close the loop safely:** after re-tuning, measure the next day's deviation for that group and confirm it actually dropped. Treat it as a control loop with regression monitoring, not a one-shot fix.
- **Watch for the moving-target problem:** changing hyper-parameters changes future heuristic outputs, which changes future deviations. Change slowly, one factor at a time, and keep the comparison apples-to-apples.
- **Timing:** "end of day" is fine as a default, but trigger by **data volume, not just the clock** — run per group once it has accumulated enough new LP results, so sparse groups wait and busy groups get tuned more often.

---

## 6. Risks & gaps to close

- **LP cost & latency** — running global LP per session is likely the dominant cost; sample or dedupe (see §3).
- **LP that never finishes** — needs a timeout + exclusion rule so it can't stall the batch or leave sessions half-stored.
- **Sparse groups** — most of the metadata cube will be thin; plan hierarchical fallback now (§5).
- **Garbage uploads** — validate the 8760 before optimising (§2); a bad profile poisons both the session and the calibration loop.
- **Estimated vs real hours** — deviation on estimated data partly measures *your estimate*, not the *heuristic*. Separate these or you'll tune the heuristic to compensate for estimation error.
- **Leap years / time zones / DST** — 8760 vs 8784, and hour alignment across locations. Pin this down early; it silently corrupts comparisons.
- **Feedback instability** — the tuning loop can oscillate; add damping and regression gates (§5).

---

## 7. Suggested additions

- **Group-level result cache / reuse:** serve a fresh LP result for a group instead of re-solving per session.
- **Warm-start LP from the heuristic** for faster convergence and a free gap measurement.
- **A learned surrogate (later):** once you have many (features → LP-optimal) pairs stored, you can eventually train a model to predict the LP outcome directly and shrink reliance on live LP. Your storage design should keep these pairs so this stays open.
- **A confidence score on the heuristic** shown in-session, so users know when the fast answer is trustworthy vs provisional.
- **Dashboards on the deviation distribution per group** — this is your product-health signal; make it visible, not just a batch log.

---

## 8. Cleaner end-to-end flow

1. **Upload** → validate partial data, PUE, machine specs; reject malformed input.
2. **Retrieve** → exact key match on `(country, sector, year, location)`; relax dimensions, then feature k-NN as fallback. Return a normalised load shape.
3. **Estimate 8760** → scale the retrieved shape to the user's magnitude; overlay the user's real hours; tag each hour real/estimated; validate.
4. **Optimise (live)** → heuristic endpoint returns fast; store result + `session_id` in **DynamoDB (with TTL)**.
5. **Optimise (background)** → enqueue LP job on long-running worker (Batch/Fargate); warm-start from heuristic; on completion update the same `session_id` record with status `COMPLETE`.
6. **Land for analytics** → stream completed session records into **S3 + Athena** (or a GSI) keyed/queryable by group.
7. **Reconcile (batch)** → per group with enough samples, compute the relative objective gap distribution; inspect p75/p90 tails.
8. **Calibrate** → where tail deviation is too large, re-tune Optuna hyper-parameters for that group (or its parent in the hierarchy), on a held-out slice; version the config.
9. **Verify** → next cycle, confirm the tail actually shrank; roll back if it regressed.

---

## 9. One-line answer to your three direct questions

- **Is retrieval necessary?** Yes — but as a *template/shape lookup* (keyed + k-NN fallback), not LLM-style RAG.
- **Where to store Optuna + session, then LP + session?** Hot path in **DynamoDB keyed by `session_id`** (LP updates the same record); batch path mirrored into **S3 + Athena** (or a GSI) for group-wise comparison.
- **Is the pipeline good?** Yes, the skeleton is good. Make LP async and sampled (not per-session-blocking), fix storage-by-access-pattern, and harden the calibration loop statistically (empirical tails, min samples, held-out tuning, versioned configs, regression gates).
