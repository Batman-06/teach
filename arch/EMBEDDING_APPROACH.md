# Embedding an 8760-hour Load Shape for k-NN

Answering `points.md`: how do we turn an 8760-value demand curve into a vector for nearest-neighbour search, which approach to use, and why.

---

## 0. The key reframing

- The 8760 values are *already* a vector — but a **bad** one for similarity search. Don't index it raw. Four reasons:
  1. **Too high-dimensional** — 8760 dims means the "curse of dimensionality": distances become meaningless and k-NN quality collapses.
  2. **Dominated by magnitude** — a big data centre and a small one with the *same shape* look far apart, even though you only care about shape (you scale by magnitude separately).
  3. **Phase-misaligned** — hour 0 of one profile vs another, DST, leap years, holidays → the same pattern shifted by a few hours reads as "different."
  4. **Noisy** — hour-to-hour jitter drowns the structure you actually want to match (diurnal / weekly / seasonal).
- So the real task is: **compress the 8760 into a low-dimensional, magnitude-invariant, noise-robust vector that captures the *shape*.**

---

## 1. Step zero — always normalise first

- Divide the curve by its **annual total** (or its **peak**) before doing anything else. Now you compare *shape*, not *size*.
- Magnitude (capacity, PUE, uptime) is handled in the separate scaling step of the pipeline — keep it out of the similarity vector.
- Decide 8760 vs 8784 (leap year) and a fixed hour-0 convention now, or phase misalignment will quietly wreck every distance.

---

## 2. Approach A — Hand-crafted feature vector  ⭐ recommended start

Instead of embedding 8760 points, compute a small set of **interpretable features** that describe the shape:

- **Load statistics:** load factor (mean/peak), base-load ratio (min/mean), coefficient of variation, peakiness.
- **Average daily profile:** collapse 365 days → one average 24-hour curve → **24 values**. This is the single most informative shape descriptor (the diurnal pattern).
- **Weekly pattern:** weekday-vs-weekend average, 7-day shape → captures business-cycle structure.
- **Seasonal pattern:** 12 monthly averages → captures summer/winter behaviour.
- **Dynamics:** ramp rates, variability, number/height of peaks.
- **Metadata (concatenate):** PUE, machine mix, installed capacity, plus categoricals (country, sector).

Result: ~**50–100 interpretable dimensions** instead of 8760.

**Why start here:**
- No training, no GPUs, works immediately even with a **small catalogue**.
- **Explainable** — you can say *why* two profiles matched (similar diurnal shape + same sector).
- Directly aligned to the physics you care about (daily/weekly/seasonal rhythm).
- Degrades gracefully with **partial uploads** (see §6).

---

## 3. Approach B — PCA / SVD on the shape  ⭐ recommended data-driven upgrade

- Reshape each normalised profile `8760 → 365 × 24` (days × hours).
- Stack all catalogue profiles and run **PCA (SVD)** across them; keep the top **k** components (typically k = 10–20 captures most variance).
- Each profile becomes its **k principal-component scores** → a compact, purely data-driven shape vector.

**Why:**
- No hand-engineering; the data tells you the dominant shape modes.
- Standard, well-proven technique for load profiles ("typical load shapes").
- Natural denoising — minor components (noise) are dropped.
- Cheap to compute and re-fit as the catalogue grows.
- Pairs well with Approach A: use PCA scores **for shape** + handcrafted/metadata features for the rest.

---

## 4. Approach C — Frequency domain (FFT / wavelets)

- Take the **FFT** of the normalised series; keep the dominant harmonics (daily ≈ 365 cycles/yr, weekly ≈ 52, yearly ≈ 1).
- A handful of coefficients (magnitude ± phase) compactly encode the periodic structure.
- **Why / when:** great when the signal is strongly periodic (most demand is). More robust to small phase shifts if you keep magnitude and drop phase. Use as an *alternative* shape vector, or concatenate a few FFT features into Approach A.

---

## 5. Approach D — Learned embedding (autoencoder)  — later, only if needed

- Train a **1-D convolutional / sequence autoencoder** (or a time-series transformer) to compress 8760 → a small latent vector; use the latent as the embedding.
- **Why wait:** needs a **large** catalogue to train well, costs GPU/time, and is **not interpretable**.
- **When to switch:** only after A/B are in production *and* you've measured that their retrieval quality is insufficient. Don't start here — it's over-engineering for day one.

> Note on **DTW (Dynamic Time Warping):** it's the classic time-series similarity measure and handles phase shifts well — but it produces a *pairwise distance*, not a fixed-length vector, so it doesn't drop into a vector k-NN index cleanly and is O(n²) per pair (expensive at 8760). Skip it for the index; the feature/PCA route is the right fit.

---

## 6. The subtlety you must design for — partial uploads

- The user uploads **partial** data, so you often **cannot compute the full-year features**. Your embedding must be computable from what's available.
- Practical pattern:
  1. **Filter hard on categoricals first** — country, sector (and roughly location/year). This is a DynamoDB/OpenSearch *filter*, not a distance.
  2. **k-NN on shape features within that filtered set**, using only the features you *can* derive from the partial upload (e.g. the observed average daily shape, observed load factor).
  3. If almost nothing is uploaded, fall back to **metadata-only** matching and lean on the retrieved shape as a pure template.
- OpenSearch supports **filtered k-NN** (categorical filter + vector distance together), which fits this exactly.

---

## 6b. If you use Approach B (PCA) — partial uploads via Gappy POD

Approach B needs the **complete** 365×24 vector to project onto the loading matrix, so partial uploads need special handling. The standard fix is **Gappy POD (masked least-squares projection)**.

- **The problem:** normally you'd project the full normalised 8760 onto the top-k loading vectors to get the scores. A partial upload has missing hours, so the full projection is undefined.
- **The fix — project using only the observed hours:**
  1. Keep the PCA basis (top-k loading vectors) fitted on your **complete** catalogue profiles.
  2. Take only the rows of the loading matrix for the hours the user **actually provided**.
  3. Solve a small **least-squares** problem for the k scores that best reconstruct the *observed* values from those rows.
  4. Those k scores **are** the embedding → run k-NN on them (inside the categorical filter, as in §6).
  - This is "Gappy POD" (Everson–Sirovich): PCA projection restricted to the observed mask.
- **Bonus — the 8760 estimate comes for free.** Multiply the recovered scores back through the **full** basis and you've reconstructed all 8760 hours. So retrieval and the "estimate 8760" stage become the *same* operation; you still overlay the user's real hours and keep provenance tags.
- **Conditioning requirement:** you need at least **k observed values**, ideally spread across the pattern (a few representative days, not one hour), or the least-squares is under-determined. If observed points < k or poorly spread, **drop k** (fewer components) or fall back.
- **Whiten before distance:** PC scores have very different variances (component 1 ≫ component k). Divide each score by its singular value, or the first component dominates the k-NN distance.
- **Fallback ladder:**
  1. Enough well-spread observations → Gappy POD scores → k-NN.
  2. Too few → reduce k, or match on handcrafted features you *can* compute (avg daily shape).
  3. Almost nothing → categorical/metadata-only match, use the retrieved profile as a pure template.
- **Robustness tip:** also fit PCA on **daily shapes** (each day = 24-dim, catalogue = many days), not just the full-year 365×24. A partial upload of a few days still yields solid daily-shape scores you can average — far more forgiving than needing a full year.

---

## 7. Making the distance behave

- **Standardise** every feature (z-score) before computing distance — otherwise a large-range feature (e.g. peak kW) dominates and small-range shape features are ignored.
- **Metric:** cosine similarity is natural for *shape* vectors; Euclidean on standardised features also works. Pick one and stay consistent.
- **Weight** feature groups if some matter more (e.g. weight diurnal shape above seasonal). You can hand-set weights or learn them from which retrievals actually produced good estimates.

---

## 8. Recommendation in one line

- **Start:** normalise → **handcrafted features + metadata** (Approach A), with categorical **filter → k-NN** for partial uploads.
- **Upgrade:** add **PCA scores** (Approach B) as the data-driven shape vector; optionally a few **FFT** features.
- **Only if proven necessary:** move the shape vector to a **learned autoencoder** (Approach D).
- **Why this order:** maximum retrieval quality per unit of effort, interpretable from day one, works with a small catalogue and with partial data, and leaves a clean upgrade path as data grows.
