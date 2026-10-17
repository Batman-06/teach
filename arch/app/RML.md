# Jagannath Mukkara

**M.Tech, Computer Science — Indian Statistical Institute, Kolkata**
+91 8074148733 · [jagan.mukkara12@gmail.com](mailto:jagan.mukkara12@gmail.com)

> Machine Learning / Data Scientist · Probabilistic Forecasting · Dimensionality Reduction & Retrieval · Optimization

---

## Education

| Course | College / University |
|---|---|
| M.Tech Computer Science | Indian Statistical Institute, Kolkata |
| B.Tech Computer Science | Sri Venkateswara University |

---

## Work Experience

### Lead Machine Learning Engineer — Sustainability Economics *(Dec '25 – Present)*

**Probabilistic Forecasting Framework**
- Built an end-to-end probabilistic forecasting pipeline **from the ground up**, combining Monte Carlo simulation with XGBoost to generate full-year 8760-hour electricity-demand, solar, and wind-generation profiles — replacing ad-hoc Excel workflows with a reproducible, automated system.
- Improved forecast accuracy by **~20% (MAPE)** while delivering **calibrated uncertainty intervals** (coverage-validated, e.g. ~90% intervals holding ~90% empirically), verified through rigorous backtesting and hypothesis testing to enable risk-aware capacity decisions.
- Translated forecast gains into a **~30% reduction in projected operating expense (est. $2–3M)** on simulated demand scenarios, tying model quality directly to business impact.

**Profile Retrieval & Partial-Data Estimation — PCA + Gappy POD**
- Designed a **retrieval-augmented estimation layer** that reconstructs a complete 8760-hour profile from a user's **partial** data (a few representative days), cutting the onboarding data requirement by **~90%**.
- Compressed each 8760-hour profile into a **~10-dimensional PCA shape-embedding retaining >95% variance**, shrinking the similarity-search footprint by **~800×** and enabling sub-**~50 ms** k-NN retrieval of comparable historical profiles.
- Implemented **Gappy POD (masked least-squares projection)** to reconstruct missing hours directly from the PCA basis — recovering full-year shapes from as little as **~10–15% observed data** at **~5–8% reconstruction MAPE** — unifying retrieval and gap-filling into a single operation.
- **Separated shape from magnitude** (normalized profiles for matching, re-scaled by annual energy / PUE) so retrieval generalizes across customers of very different sizes within the same sector.
- Hardened the stage with a **validation + guardrail gate**: held-out reconstruction benchmarks across realistic missing-data patterns, per-request confidence scoring, and basis-versioning with drift-triggered refits — preventing silently-wrong estimates from reaching the optimizer.

**Energy-System Sizing Optimization**
- Replaced a legacy Differential Evolution + L-BFGS-B optimizer (>10 s solve) with **Bayesian Optimization + L-BFGS-B**, cutting solve time to **2–3 s (~70–80% faster)** — fast enough for live in-browser trial runs with instant ROI previews (~20% savings).
- Architected a **dual-solver pattern**: a fast heuristic/BO solver returns an estimated cost/time result in-session while a **parallel LP model** runs asynchronously for the near-global optimum — instant feedback without sacrificing eventual exactness.
- Built an **automated calibration loop** comparing heuristic vs LP results per (country·sector·location·year) group on a daily cadence, analyzing the deviation distribution (p75/p90 tails) and **auto-tuning the heuristic's hyper-parameters** to hold the optimality gap within an acceptable band at browser speed.

**Drift Monitoring & Infrastructure** *(supporting)*
- Engineered an **event-driven, asynchronous drift-monitoring pipeline** (async I/O + concurrent workers) running distribution-shift checks across demand, solar/wind, MCP, and tariff/PPA inputs plus optimization outputs — without blocking the core workload.
- Containerized all frameworks with **Docker** and automated deployments on **AWS via CI/CD (GitHub Actions)**; designed the serving store around DynamoDB (hot session results, TTL), S3 + Athena (comparison/analytics), and SQS + Batch/Fargate (async LP jobs).

### AI Engineer — BluBridge *(Sept '25 – Oct '25)*
- Built an FX-style graph-capture system converting eager execution into static IR, enabling whole-graph optimization.
- Designed a tensor IR (SSA DAG) with shape inference and dependency tracking for kernel fusion and code generation.

### Data Scientist / Systems Engineer — Tata Consultancy Services *(April '21 – Nov '22)*
- Developed a Next-Best-Offer engine for a US bank using XGBoost across 150+ rolling continuous features to predict cross-sell conversion (**AUC-ROC 0.82**).
- Automated weekly batch inference for **2.5M+ retail accounts**, optimizing Precision@2 and lifting digital CTR by **24%**.

---

## Dissertation

**2.5D Dual-Encoder U-Net for Lesion Segmentation in Chest CT Scans** — Prof. Sarbani Palit
- Designed a CNN-based medical image segmentation architecture combining 2.5D learning with a dual-encoder U-Net for processing 3D chest CT scans.
- Achieved higher segmentation accuracy (**Dice 0.88**) than traditional 2D CNNs at comparable architectural complexity and resource use.

---

## Skills

- **Forecasting & Machine Learning:** XGBoost, Ensemble Learning, Monte Carlo Simulation, Probabilistic Forecasting, Time-Series Forecasting, SVM, Logistic/Linear Regression, Backtesting, Hypothesis Testing
- **Dimensionality Reduction & Retrieval:** PCA / POD, Gappy POD, Shape Embeddings, k-NN Retrieval, Masked Least-Squares Reconstruction, Variance-Retention Analysis
- **Optimization & Operations Research:** Bayesian Optimization, L-BFGS-B, Differential Evolution, MILP/LP, Constrained Optimization, Heuristic Optimization, Battery Dispatch Optimization
- **ML Evaluation & Metrics:** A/B Testing, Offline Evaluation, Cross-Validation, Backtesting, Calibration/Coverage, MAPE, RMSE, MAE, Model Benchmarking
- **Energy Systems & Simulation:** Electricity Demand / Solar / Wind Forecasting, 8760-Hour Dispatch Simulation, Energy System Sizing, Battery Modeling, ROI Modeling
- **Infrastructure & Deployment:** Docker, AWS (Lambda, S3, SQS), CI/CD, GitHub Actions, FastAPI, Git, Redis
- **Programming & Frameworks:** Python, SQL, PyTorch, NumPy, Pandas, Scikit-learn, SciPy, Matplotlib

---

*Metrics marked `~` are estimates — confirm against real numbers before sending.*
