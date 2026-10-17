# Demand-Profile Optimisation Pipeline — Flowchart (with services/tooling)

Mermaid diagrams for the pipeline, with concrete AWS services and other tools named inside each cell. Four views: live session path, background LP path, offline calibration loop, and a combined view — plus a tooling summary and a stage→service table.

> Service choices are the sensible default stack, not the only option. Swap equivalents where your team already has expertise (e.g. GCP/Azure, or self-hosted).

---

## 1. Live session path (synchronous, fast)

```mermaid
flowchart TD
    U(["User uploads partial data + PUE + machine specs<br/><i>Browser SPA: React · Amplify<br/>Auth: Amazon Cognito</i>"]) --> API["API entry<br/><i>Amazon API Gateway<br/>large files via S3 presigned URL</i>"]
    API --> V{"Validate input<br/><i>AWS Lambda - Python · Pydantic</i>"}
    V -- invalid --> R1["Reject / flag upload<br/><i>CloudWatch log + user error</i>"]
    V -- valid --> RET["Retrieve reference shape<br/><i>AWS Lambda</i>"]

    RET --> K{"Exact key match?<br/>country · sector · year · location<br/><i>DynamoDB GetItem</i>"}
    K -- yes --> SHAPE["Use matched normalised shape"]
    K -- no --> RELAX["Relax dimensions year→location→sector<br/><i>DynamoDB Query + GSI</i>"]
    RELAX --> KNN{"Still no match?<br/><i>Amazon OpenSearch k-NN<br/>feature nearest-neighbour</i>"}
    KNN -- found --> SHAPE
    KNN -- fallback --> SHAPE

    SHAPE --> EST["Estimate 8760 values<br/>scale shape · overlay real hours · tag real/estimated<br/><i>AWS Lambda · NumPy/Pandas</i>"]
    EST --> V2{"Validate 8760<br/>count · non-negative · energy vs PUE<br/><i>AWS Lambda</i>"}
    V2 -- bad --> R1
    V2 -- ok --> OPT["Heuristic optimiser - estimated time & cost<br/><i>Optuna on ECS Fargate or SageMaker endpoint</i>"]

    OPT --> STORE[("Session store<br/><i>Amazon DynamoDB<br/>key=session_id · TTL</i>")]
    OPT --> RESP(["Return result to user in-session<br/><i>API Gateway · AppSync for live updates</i>"])
    OPT -. warm-start .-> ENQ["Enqueue LP job"]
```

---

## 2. Background LP path (asynchronous, exact)

```mermaid
flowchart TD
    ENQ["LP job enqueued<br/><i>Amazon SQS · EventBridge trigger</i>"] --> ORCH["Orchestrate run<br/><i>AWS Step Functions</i>"]
    ORCH --> WORK["Worker - global LP solve, warm-started<br/><i>AWS Batch or ECS Fargate<br/>Solver: Gurobi / HiGHS / OR-Tools / PuLP<br/>Image in Amazon ECR</i>"]
    WORK --> ST{"LP state<br/><i>Step Functions state</i>"}
    ST -- running --> WORK
    ST -- timeout/failed --> EXCL["Mark FAILED - exclude from batch<br/><i>DynamoDB status + CloudWatch alarm</i>"]
    ST -- complete --> UPD[("Update same session_id record<br/>status=COMPLETE + LP cost/time<br/><i>Amazon DynamoDB</i>")]
    UPD --> LAND["Stream to analytics lake<br/><i>DynamoDB Streams → Kinesis Firehose → S3</i>"]
    EXCL --> LAND
```

---

## 3. Offline calibration loop (batch, self-tuning)

```mermaid
flowchart TD
    TRIG(["Trigger: end-of-day OR enough new LP results<br/><i>Amazon EventBridge Scheduler</i>"]) --> PULL["Pull completed results per group<br/>country · sector · location · year<br/><i>Amazon Athena over S3 · AWS Glue catalog</i>"]
    PULL --> MIN{"Enough samples in group?"}
    MIN -- no --> AGG["Aggregate to parent level e.g. sector<br/><i>Athena / Pandas</i>"] --> GAP
    MIN -- yes --> GAP["Compute relative objective gap<br/>heuristic_obj − lp_obj / lp_obj<br/><i>SageMaker Processing or AWS Glue job</i>"]

    GAP --> DIST["Build empirical deviation distribution<br/>inspect p75 / p90 tails<br/><i>NumPy / Pandas</i>"]
    DIST --> CHK{"Tail deviation too large?"}
    CHK -- no --> OK["Keep current hyper-params"]
    CHK -- yes --> TUNE["Re-tune Optuna hyper-params on held-out slice<br/><i>Optuna on SageMaker / AWS Batch</i>"]
    TUNE --> VER[("Version & store config per group<br/><i>SSM Parameter Store or DynamoDB · S3</i>")]
    VER --> VERIFY{"Next cycle: tail shrank?"}
    VERIFY -- yes --> OK
    VERIFY -- no, regressed --> ROLL["Roll back to previous version<br/><i>SSM / DynamoDB versioned config</i>"] --> OK

    OK --> DASH["Health dashboards<br/><i>Amazon QuickSight / Grafana / CloudWatch</i>"]
```

---

## 4. Combined end-to-end view

```mermaid
flowchart LR
    subgraph LIVE["Live session - synchronous"]
        A(["Upload<br/><i>React · Cognito</i>"]) --> B["Validate<br/><i>API Gateway · Lambda</i>"]
        B --> C["Retrieve shape<br/><i>DynamoDB · OpenSearch k-NN</i>"]
        C --> D["Estimate 8760<br/><i>Lambda · NumPy</i>"]
        D --> E["Heuristic / Optuna<br/><i>ECS Fargate / SageMaker</i>"]
        E --> F(["Return to user<br/><i>API Gateway · AppSync</i>"])
    end

    subgraph STOREL["Storage - hot path"]
        G[("DynamoDB<br/>session_id · TTL")]
    end

    subgraph BG["Background - asynchronous"]
        H["Enqueue LP<br/><i>SQS · Step Functions</i>"] --> I["Worker LP solve<br/><i>AWS Batch/Fargate · Gurobi/HiGHS · ECR</i>"]
        I --> J[("Update session record<br/><i>DynamoDB</i>")]
    end

    subgraph ANALYTICS["Analytics - cold path"]
        K["Data lake<br/><i>DynamoDB Streams · Firehose · S3 · Glue · Athena</i>"]
    end

    subgraph CAL["Offline calibration loop"]
        L["Gap distribution p75/p90<br/><i>SageMaker Processing / Glue</i>"]
        L --> M{"Tail too large?"}
        M -- yes --> N["Re-tune Optuna<br/><i>Optuna · SageMaker</i>"]
        N --> O[("Versioned hyper-params<br/><i>SSM Parameter Store</i>")]
        M -- no --> P["Keep config"]
        O -.-> E
    end

    E --> G
    E -. warm-start .-> H
    G --> K
    J --> K
    K --> L
```

---

## 5. Cross-cutting tooling (not in the flow, but everywhere)

```mermaid
flowchart LR
    subgraph SRC["Source & CI/CD"]
        GH["GitHub<br/>code + PRs"] --> GHA["GitHub Actions<br/>build · test · lint"]
        GHA --> ECR["Amazon ECR<br/>container images"]
        GHA --> IAC["Terraform / AWS CDK<br/>infra as code"]
    end

    subgraph DEPLOY["Deploy targets"]
        ECR --> LAMBDA["Lambda / Fargate / Batch"]
        IAC --> AWSENV["All AWS resources"]
    end

    subgraph OBS["Observability & Security"]
        CW["CloudWatch - logs/metrics/alarms"]
        XR["AWS X-Ray - tracing"]
        IAM["AWS IAM - least privilege"]
        SEC["AWS Secrets Manager - solver license keys e.g. Gurobi"]
    end

    DEPLOY --> OBS
```

---

## 6. Stage → service quick reference

| Stage | Primary service(s) | Alternatives / notes |
|---|---|---|
| Frontend / upload | React + AWS Amplify, Amazon Cognito | Any SPA host; Auth0 instead of Cognito |
| API entry | Amazon API Gateway (+ AppSync for live updates) | ALB + ECS if containerised API |
| Input & 8760 validation | AWS Lambda (Python, Pydantic) | Fargate if long-running |
| Profile catalogue (8760 arrays) | Amazon S3 (Parquet) + metadata in DynamoDB | Dynamo-only if always keyed lookup |
| Exact / relaxed retrieval | DynamoDB GetItem + GSI Query | — |
| Similarity fallback | Amazon OpenSearch (k-NN) | pgvector on RDS; Pinecone |
| Heuristic optimiser | Optuna on ECS Fargate or SageMaker endpoint | Lambda if fast enough |
| Session results (hot) | Amazon DynamoDB (TTL) | — |
| LP queue & orchestration | Amazon SQS + AWS Step Functions + EventBridge | — |
| LP solver compute | AWS Batch / ECS Fargate | Gurobi / HiGHS / OR-Tools / CBC / PuLP |
| Solver container registry | Amazon ECR | Docker Hub (private) |
| Analytics landing | DynamoDB Streams → Kinesis Firehose → S3 | Direct S3 writes |
| Query / grouping (cold) | Amazon Athena + AWS Glue catalog | Redshift / EMR |
| Batch gap computation | SageMaker Processing / AWS Glue job | EMR Spark at scale |
| Re-tuning | Optuna on SageMaker / AWS Batch | — |
| Versioned hyper-params | AWS SSM Parameter Store / DynamoDB + S3 | — |
| Scheduling | Amazon EventBridge Scheduler | cron on a worker |
| Dashboards | Amazon QuickSight / Grafana / CloudWatch | — |
| Source & CI/CD | GitHub + GitHub Actions | GitLab CI; CodePipeline |
| Infra as code | Terraform / AWS CDK | CloudFormation / Pulumi |
| Secrets (e.g. Gurobi license) | AWS Secrets Manager | SSM SecureString |
| Observability | CloudWatch + AWS X-Ray | Datadog |
| Access control | AWS IAM (least privilege) | — |

---

### Legend

- **Rounded nodes** = start/end / user-facing events.
- **Rectangles** = processing steps (service named in italics).
- **Diamonds** = decisions / branches.
- **Cylinders** `[( )]` = data stores.
- **Dotted arrows** = asynchronous or feedback links (LP warm-start, hyper-param updates flowing back into the heuristic).
