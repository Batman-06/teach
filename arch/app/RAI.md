# Jagannath Mukkara

**M.Tech, Computer Science — Indian Statistical Institute, Kolkata**
+91 8074148733 · [jagan.mukkara12@gmail.com](mailto:jagan.mukkara12@gmail.com)

> AI Engineer · Agentic Systems & GraphRAG · LLM Inference & Serving · Optimization

---

## Education

| Course | College / University |
|---|---|
| M.Tech Computer Science | Indian Statistical Institute, Kolkata |
| B.Tech Computer Science | Sri Venkateswara University |

---

## Work Experience

### Lead Machine Learning Engineer — Sustainability Economics *(Dec '25 – Present)*

**Agentic RAG & Guardrails**
- Architected an **air-gapped GraphRAG workflow** using a LangGraph **Router → Reasoner** topology, combining dense vectors, sparse BM25, and Knowledge Graphs to resolve **multi-hop legal dependencies**.
- Engineered a **safety & compliance layer** enforcing grounded extraction and strict citation validation, with real-time hallucination checks to ensure verifiable outputs for enterprise compliance.

**LLM Inference & Serving**
- Deployed **self-hosted inference for Llama-3.1 (70B & 8B)** on a **6× RTX 6000 Ada** cluster via **vLLM, AWQ 4-bit, and PagedAttention**, reclaiming **100GB+ VRAM** while enabling **65K+ context windows**.
- Scaled the platform to **100+ concurrent enterprise users**, boosting end-to-end throughput by **~300%** via semantic caching, chunked prefill, and asynchronous task queues.

**Optimization & Benchmarking**
- Replaced a legacy Differential Evolution + L-BFGS-B model (>10 s solve) with **Bayesian Optimization + L-BFGS-B**, cutting solve time to **2–3 s (~30× speedup)** for live browser ROI previews (~20% savings).
- Formulated a **parallel MILP model** as a business sanity check, benchmarking and proving that the lightweight BO solver achieved near-global optimality.

**Deployment & Infrastructure** *(supporting)*
- Containerized all frameworks with **Docker** and automated deployments on **AWS via CI/CD (GitHub Actions)**, ensuring reproducible builds across dev and staging.

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

- **Agentic AI & Orchestration:** LangGraph, LangChain, Multi-Agent Systems, Supervisor–Worker Architecture, Asynchronous Task Queues
- **RAG & Retrieval:** Hybrid Retrieval, GraphRAG, Dense Vectors, BM25, Cross-Encoder Reranking, ChromaDB, Citation Grounding
- **LLM Inference & Optimization:** vLLM, AWQ Quantization, PagedAttention, FlashAttention, Semantic Caching, Chunked Prefill, Memory Optimization
- **Evaluation & ML Optimization:** A/B Testing, Offline Evaluation, LLM-as-a-Judge, Bayesian Optimization, L-BFGS-B, MILP, XGBoost
- **Infrastructure & Deployment:** Docker, AWS (Lambda, S3, SQS), CI/CD, GitHub Actions, FastAPI, Git, Redis
- **Programming & Frameworks:** Python, SQL, PyTorch

---

*Metrics (300% throughput, 30× speedup, VRAM/context figures) are carried from source bullets — confirm against real numbers before sending.*
