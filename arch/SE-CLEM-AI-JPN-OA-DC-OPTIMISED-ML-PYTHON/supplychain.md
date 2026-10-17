# Supply Chain Optimization vs. Our Energy Optimization

This document explains the anatomy of a classic supply chain optimization
problem — its variables, constraints, penalties, objective, and outputs — and
maps each piece back to our energy system sizing/dispatch model
(`core/optimization.py`, `utils/zone_dispatch.py`, etc.).

---

## The mental model

A supply chain is a **network of nodes and arcs**:

```
Suppliers → Plants → Warehouses/DCs → Customers
   (nodes)         (arcs = flows between them)
```

You decide **what to build** (structure) and **what to move** (flow), over
**time periods**, to **meet demand at minimum cost**.

The most common archetype is a **multi-echelon network design + flow problem**.

---

## 1. Decision variables

These split into two natural tiers (like our sizing-vs-dispatch split):

**Strategic / structural (usually binary or integer)**
- `y[w] ∈ {0,1}` — do we open warehouse `w`? (facility location)
- `cap[w] ≥ 0` — how big do we build it? (sizing)
- `z[w,p] ∈ {0,1}` — does plant `p` produce a given product line, etc.

**Operational / flow (usually continuous)**
- `x[i,j,t] ≥ 0` — units shipped on arc `i→j` in period `t`
- `inv[w,t] ≥ 0` — inventory held at `w` at end of period `t`
- `prod[p,t] ≥ 0` — units produced at plant `p` in period `t`
- `back[c,t] ≥ 0` — unmet/backordered demand for customer `c` in period `t`

> **Our parallel:** `y`/`cap` = solar/wind/battery build sizes. `x` = zone
> dispatch. `inv` = battery state-of-charge. `back` = unmet load / spot-market
> purchase.

---

## 2. Constraints

The recurring types:

**a) Demand satisfaction** (the "must be met" rule)
```
Σ_i x[i,c,t]  +  back[c,t]  =  demand[c,t]      ∀ customer c, period t
```
Everything flowing into a customer plus backorders equals demand.

**b) Flow conservation / balance** (nothing appears or vanishes at a node)
```
inflow[w,t] + inv[w,t-1]  =  outflow[w,t] + inv[w,t]      ∀ warehouse w, t
```
> Identical to our battery balance: `SoC[t] = SoC[t-1] + charge - discharge`.
> Inventory balance = storage balance.

**c) Capacity limits**
```
Σ_j x[w,j,t]  ≤  cap[w]·y[w]              (can't ship from an unbuilt/oversized DC)
prod[p,t]     ≤  plant_capacity[p]
inv[w,t]      ≤  storage_capacity[w]
```
> Our parallel: battery can't discharge more than its size; generation ≤
> installed capacity.

**d) Linking constraints** (tie operational to structural)
```
x[w,j,t]  ≤  BigM · y[w]     — no flow through a warehouse unless it's open
```

**e) Service-level / business rules**
- Max lead time / distance served
- Min utilization ("if open, run it ≥ 60%")
- Single-sourcing (each customer served by exactly one DC)
- Safety-stock minimums

> Our parallel: RE utilization floor, CFE % target, spot-market reliance cap.

---

## 3. Objective function

Minimize **total cost = fixed + variable**:

```
min   Σ_w  fixed_cost[w]·y[w]                    ← facility/capex
    + Σ_w  unit_cap_cost[w]·cap[w]               ← sizing cost
    + Σ    transport_cost[i,j]·x[i,j,t]          ← moving goods
    + Σ    production_cost[p]·prod[p,t]           ← making goods
    + Σ    holding_cost[w]·inv[w,t]              ← carrying inventory
    + Σ    penalty·back[c,t]                      ← unmet demand
```

> Our parallel: capex (build) + grid/tariff cost + demand charges + spot-market
> cost. Same fixed-plus-variable skeleton.

---

## 4. Penalties — the key distinction

**In a classic MILP supply chain, most "constraints" are HARD** — the solver
won't return a solution that violates them. Infeasible is infeasible; there are
no penalties.

Penalties appear in two specific situations:

**a) Soft constraints deliberately modeled as costs.** The most common is a
**backorder/lost-sales penalty** — you *allow* demand to go unmet but charge a
stiff per-unit cost, so the optimizer meets demand only where it's economical:
```
+ Σ  stockout_penalty·back[c,t]
```
This is the direct analog of our `_market_penalty` and `_cfe_penalty` — turning
a "should" into a priced-in cost.

**b) When you can't use an exact solver** (nonlinear, black-box, or too large)
and switch to a **metaheuristic** (genetic algorithms, simulated annealing,
Bayesian optimization) — then you convert *every* hard constraint into a
penalty, because heuristics can't natively enforce constraints. **This is
exactly our situation:** the Optuna TPE model penalizes CFE shortfall, market
overshoot, etc., because TPE can't be handed hard constraints the way Gurobi
can.

So: **classic supply chain = mostly hard constraints + a few economic
penalties. Our energy model = everything soft/penalized, because the solver
demands it.**

---

## 5. Outputs

What the solver hands back:

- **The network design:** which facilities to open and at what size (`y*`, `cap*`)
- **The flow plan:** how much moves on every arc each period (`x*`)
- **The inventory plan:** stock levels over time (`inv*`)
- **Total cost** broken into components (capex vs transport vs holding vs penalty)
- **The optimality gap** (MILP only): "within 0.5% of the true optimum" — a
  guarantee the Bayesian approach *cannot* give
- **Shadow prices / duals** (LP only): "relaxing this warehouse's capacity by 1
  unit saves $X" — sensitivity info

> Our parallel: chosen build sizes, hourly dispatch schedule, battery SoC
> profile, cost breakdown. What we *lack* versus a MILP is the optimality gap
> and shadow prices — the price of using a black-box solver.

---

## Side-by-side summary

| Component | Classic supply chain (MILP) | Our energy model (Optuna) |
|---|---|---|
| **Structural vars** | open/size warehouses `y, cap` | solar/wind/battery build sizes |
| **Flow vars** | shipments `x[i,j,t]` | zone dispatch |
| **Storage vars** | inventory `inv[w,t]` | battery SoC |
| **Balance constraint** | flow conservation | charge/discharge balance |
| **Demand rule** | hard equality + backorder | hourly load meet |
| **Constraints** | mostly **hard** | all **soft/penalized** |
| **Penalties** | stockout, lost-sales | CFE, market, utilization |
| **Solver** | Gurobi/CPLEX/OR-Tools | TPE Bayesian search |
| **Output extras** | optimality gap, shadow prices | none (black-box) |

---

## Bottom line

The two problems share the same skeleton (structural + flow vars, balance +
demand + capacity constraints, fixed+variable cost objective). The divergence is
purely in the **solution method** — hard-constrained exact MILP vs penalty-based
black-box search — and that choice is driven by whether the cost function can be
written linearly. Ours can't (tariff layers, demand charges), so we're pushed
into penalties.
