"""
Sizing optimizer for Solar / Wind / Battery (Zone-based).

Strategy: Bayesian Optimization (Optuna TPE) for global search.
Constraints (RE utilization, battery utilization, peak/annual demand-meet)
are enforced via a structured penalty function.
"""

import logging
import optuna
import os

from config import config
from performance_utils import timeit

n_jobs = 1

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ===== Independent-zone + shared-battery optimiser =====
def _market_penalty(total_cost, market_meet_percent):
    """Cost-scaled, progressive soft floor on spot-market reliance."""
    cap = getattr(config, 'market_meet_percent', 5) or 5
    tol = 0.5  # percentage-point grace band before any penalty
    over = market_meet_percent - cap - tol
    if over <= 0:
        return 0.0
    scale = max(float(total_cost), 1.0) * 0.25  # dimensionally tied to annual cost
    return scale * over * over


def _cfe_penalty(total_cost, cfe_score, cfe_target):
    """Hard floor on requested CFE target; inactive when no target is set."""
    if cfe_target <= 0:
        return 0.0
    shortfall = float(cfe_target) - float(cfe_score) - 0.05
    if shortfall <= 0:
        return 0.0
    scale = max(float(total_cost), 1.0) * 100.0
    return scale * shortfall * shortfall


# How far above the requested floor a build may sit and still count as "meeting"
# it. Purely to absorb float noise around the boundary -- it is NOT an allowance
# to overshoot.
CFE_MEET_TOLERANCE = 0.05

# How far above the requested floor a build may DELIVER and still be accepted
# (percentage points). A user asking for 20% wants ~20%, not 70%. Cost alone does
# not police this: under the firmed-hybrid base the gas that backstops solar+wind
# carries no standby charge (zone_dispatch: firm_annual_cost is zeroed there), so
# a much greener build can score CHEAPER than a modest one and win on cost. Until
# that backstop is priced, cap the overshoot directly.
#
# This is a PREFERENCE, not a constraint: if no trial lands inside the band the
# selection falls back to every trial that met the floor, so a target is never
# made unreachable by this cap.
CFE_OVERSHOOT_BAND = 5.0


def _selection_pool(study):
    """[(cost, cfe, params)] for every completed trial, on DELIVERED values."""
    pool = []
    for t in study.trials:
        if t.value is None or t.state != optuna.trial.TrialState.COMPLETE:
            continue
        cfe = t.user_attrs.get('cfe')
        cost = t.user_attrs.get('cost')
        if cfe is None or cost is None:
            continue
        pool.append((float(cost), float(cfe), t.params))
    return pool


def _log_convergence(study, pool, label):
    """Make a non-converged answer visible instead of indistinguishable.

    Reports how late the winning trial arrived and how tightly the best ten
    cluster. A best found in the first half, with a wide top-ten spread, means
    the search stopped exploring long before it ran out of budget -- the answer
    is reproducible (fixed seed) but not converged, and the two look identical
    in the output unless this is said out loud.
    """
    if not pool:
        return
    ordered = sorted(pool, key=lambda r: r[0])
    best_cost = ordered[0][0]
    top = [r[0] for r in ordered[:10]]
    spread = ((max(top) - min(top)) / min(top) * 100.0) if min(top) > 0 else 0.0
    try:
        best_i = min(range(len(pool)), key=lambda i: pool[i][0])
        frac = (best_i + 1) / float(len(pool))
    except ValueError:
        frac = 1.0
    converged = frac <= 0.75 and spread <= 1.0
    logging.info(
        "%s convergence: %d trials, best cost %.6f found at %.0f%% through the "
        "budget, top-10 spread %.2f%% -> %s",
        label, len(pool), best_cost, frac * 100.0, spread,
        "converged" if converged else "NOT CONVERGED (raise OPT_MAXITER)")


def _cheapest_meeting_target(study, cfe_target):
    """Params of the best trial, decided on what it DELIVERED.

    study.best_params is the best PENALISED score (cost + market cap + CFE
    shortfall), which is not the cheapest build. With a steep penalty the
    sampler is pushed well clear of the boundary, so a build far above the
    target can win even though a cheaper build that also cleared it was
    evaluated -- more CFE than asked for AND more cost than necessary.

    Worse, the old code returned None for cfe_target <= 0, so the pure
    COST-minimisation path fell through to that same penalised best and never
    selected on cost at all. A cost-led run could therefore come out dearer
    than a CFE-constrained one, even though its feasible set is a superset.

    So the winner is always taken from the delivered (cost, cfe) pairs:
      * no floor      -> the cheapest trial, full stop;
      * floor met     -> cheapest among those meeting it, preferring builds
                         within CFE_OVERSHOOT_BAND rather than far above;
      * floor unmet   -> the highest CFE achieved, cheapest at that level --
                         the closest approach, chosen deterministically.
    """
    pool = _selection_pool(study)
    if not pool:
        return None

    if not cfe_target or cfe_target <= 0:
        _log_convergence(study, pool, "cost-led")
        cost, cfe, params = min(pool, key=lambda r: (r[0], -r[1]))
        logging.info("No CFE floor: taking the CHEAPEST of %d trials "
                     "(cost %.4f, cfe %.2f%%).", len(pool), cost, cfe)
        return params

    _log_convergence(study, pool, "cfe-led")
    met = [r for r in pool if r[1] >= cfe_target - CFE_MEET_TOLERANCE]
    if met:
        banded = [r for r in met if r[1] <= cfe_target + CFE_OVERSHOOT_BAND]
        sel = banded or met
        cost, cfe, params = min(sel, key=lambda r: (r[0], r[1]))
        logging.info(
            "CFE floor %.2f%%: %d/%d trials met it, %d within +%.1fpp; taking "
            "the CHEAPEST at %.2f%% (cost %.4f).",
            cfe_target, len(met), len(pool), len(banded), CFE_OVERSHOOT_BAND,
            cfe, cost)
        return params

    # Floor unreachable for this resource mix. The CFE penalty already trades
    # shortfall against cost, so its best score IS the right closest approach --
    # maximising CFE regardless of cost buys a fraction of a point for several
    # percent of spend. Fall back to it, exactly as before.
    logging.info(
        "CFE floor %.2f%% NOT met by any of %d trials; falling back to the best "
        "penalised score (the penalty trades shortfall against cost).",
        cfe_target, len(pool))
    return None
def _zone_penalty(metric_tuple, cfe_target=0.0):
    """Cost objective plus the zone soft floors: market-share and CFE target."""
    (RE_util, batt_util, peak_met, _annual, _s, _w, market_meet_percent,
     reliability_score, total_cost, cfe_score) = metric_tuple
    penalty = 0.0
    
    if market_meet_percent > 0:
        penalty += _market_penalty(total_cost, market_meet_percent)
    penalty += _cfe_penalty(total_cost, cfe_score, cfe_target)
    return total_cost + penalty


@timeit
def optimize_system_zone(df, metricsConfig, penalty_mode='adaptive'):
    """Independent-zone + shared-battery optimiser.

    Decision vector = one size per (zone, distributable source) the
    qualification matrix admits, PLUS a single shared battery. Solar lands
    in mid-merit, wind spans mid-base + mid-merit, and base competes as a
    firmed hybrid (its own solar+wind, sizeable down to zero = plain firm)
    with the firm source as the guaranteed backstop — peak is battery +
    market. Each zone is sized independently, the BESS shared. Returns the
    per-zone sizes and the shared battery size.
    """
    from utils.zone_dispatch import (
        build_dispatch_inputs, compute_zone_metrics, zone_sizing_plan,
        zone_owner_by_coincidence, resolve_edges, enforce_hybrid_pairing,
        resolve_cfe_target,
    )

    prebuilt = build_dispatch_inputs(df, metricsConfig)
    arrays, _bess, _firm, layer_zones, _uc, payload = prebuilt
    plan = zone_sizing_plan(payload, metricsConfig)  # {zone: [qualified sources]}
    max_demand = float(df['demand'].max())

    edges = resolve_edges(arrays['demand'], layer_zones)
    coincidence = zone_owner_by_coincidence(arrays, edges, plan)  # descriptive only
    knobs = [(zone, s) for zone, sources in plan.items() for s in sources]
    psp_enabled = bool(_uc.get('pspBase'))
    size_firm = bool((metricsConfig.get('customInputs') or {}).get('sizeFirmCapacity'))
    cfe_target = resolve_cfe_target(metricsConfig)

    base_clean_src = _uc.get('baseCleanSource')
    base_fossil_src = _uc.get('baseFossilSource')
    base_blend_enabled = bool(cfe_target > 0 and not _uc.get('firmedHybridBase')
                              and base_clean_src and base_fossil_src)
    upper_blend_enabled = bool(cfe_target > 0 and _uc.get('upperBlendAvailable'))

    if cfe_target > 0:
        logging.info(f"Zone optimisation CFE floor requested: {cfe_target:.2f}%")
    if base_blend_enabled:
        logging.info(f"CFE base blend enabled: clean='{base_clean_src}' + "
                     f"fossil='{base_fossil_src}'")
    if upper_blend_enabled:
        logging.info("CFE upper-zone firm blend enabled")

    # SAME floor as the constrained stage below. A cfeTarget=0 request runs
    # only this stage, so leaving it at OPT_MAXITER searched cost-led with a
    # quarter of the budget a CFE-led run gets (300 + this), and then the two
    # were compared. Cost-led has the LARGER feasible set; it must not also
    # get the smaller search.
    n_trials_uncon = max(40, config.OPT_MAXITER, 300)
    n_startup_uncon = max(10, n_trials_uncon // 5)
    n_trials_floor = max(n_trials_uncon, 300)
    n_startup_floor = max(n_startup_uncon, 60)
    if upper_blend_enabled:
        n_trials_floor = max(n_trials_floor, 500)
        n_startup_floor = max(n_startup_floor, 100)

    def decode(params, use_base=False, use_upper=False):
        sizes_by_zone = {}
        for zone, source in knobs:
            sizes_by_zone.setdefault(zone, {})[source] = params.get(f'{source}_{zone}', 0.0)
        enforce_hybrid_pairing(sizes_by_zone, payload)
        battery_size = params.get('battery', 0.0)
        psp_size = params.get('psp', 0.0) if psp_enabled else 0.0
        firm_cap = ({'midBase': params.get('firm_cap_midBase'),
                     'midMerit': params.get('firm_cap_midMerit')}
                    if size_firm else None)
        base_clean_frac = params.get('base_clean_frac') if use_base else None
        upper_clean_frac = params.get('upper_clean_frac') if use_upper else None
        return sizes_by_zone, battery_size, psp_size, firm_cap, base_clean_frac, upper_clean_frac

    def evaluate(decoded, eff_target):
        sizes_by_zone, battery_size, psp_size, firm_cap, base_clean_frac, upper_clean_frac = decoded
        tuple_ = compute_zone_metrics(
            sizes_by_zone, battery_size, df, metricsConfig,
            prebuilt=prebuilt, psp_size=psp_size, firm_cap=firm_cap,
            cfe_target=eff_target, base_clean_frac=base_clean_frac,
            upper_clean_frac=upper_clean_frac,
        )
        return tuple_, _zone_penalty(tuple_, cfe_target=eff_target)

    def _run(apply_floor, seed_params=None):
        use_base = base_blend_enabled and apply_floor
        use_upper = upper_blend_enabled and apply_floor
        eff_target = cfe_target if apply_floor else 0.0
        n_trials = n_trials_floor if apply_floor else n_trials_uncon
        n_startup = n_startup_floor if apply_floor else n_startup_uncon

        def optuna_objective(trial):
            params = {}
            for zone, source in knobs:
                params[f'{source}_{zone}'] = trial.suggest_float(f'{source}_{zone}', 0.0, max_demand)
            params['battery'] = trial.suggest_float('battery', 1.0, max_demand * 2)
            if size_firm:
                params['firm_cap_midBase'] = trial.suggest_float('firm_cap_midBase', 0.0, max_demand)
                params['firm_cap_midMerit'] = trial.suggest_float('firm_cap_midMerit', 0.0, max_demand)
            if psp_enabled:
                params['psp'] = trial.suggest_float('psp', 0.0, max_demand * 2)
            if use_base:
                params['base_clean_frac'] = trial.suggest_float('base_clean_frac', 0.0, 1.0)
            if use_upper:
                params['upper_clean_frac'] = trial.suggest_float('upper_clean_frac', 0.0, 1.0)
            try:
                tuple_, objective = evaluate(decode(params, use_base, use_upper), eff_target)
                trial.set_user_attr('cfe_score', tuple_[-1])
                trial.set_user_attr('total_cost', tuple_[-2])
                return objective
            except Exception as e:
                logging.warning(f"zone score error: {e}")
                return 1e10

        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=n_startup)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        if seed_params:
            try:
                study.enqueue_trial(seed_params)
            except Exception:
                pass
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)

        # Rank the floor-meeting trials on what they DELIVERED (cost, then CFE),
        # not on trial.value -- that is the penalised score, and ranking by it
        # re-admits the over-sized builds the floor penalty pushed clear of the
        # boundary. _cheapest_meeting_target also keeps the winner inside
        # CFE_OVERSHOOT_BAND of the target when any candidate is.
        params = (_cheapest_meeting_target(study, eff_target)
                  or study.best_params)
        return decode(params, use_base, use_upper), params

    if cfe_target > 0:
        decoded_uncon, params_uncon = _run(apply_floor=False)
        tuple_uncon, _objective_uncon = evaluate(decoded_uncon, 0.0)
        cfe_uncon = tuple_uncon[-1]
        if cfe_uncon >= cfe_target - 0.05:
            logging.info(f"CFE floor {cfe_target:.2f}% already met by cost-optimal "
                         f"build ({cfe_uncon:.2f}%); returning it")
            decoded = decoded_uncon
        else:
            seed = dict(params_uncon)
            if base_blend_enabled:
                seed['base_clean_frac'] = 0.0
            if upper_blend_enabled:
                seed['upper_clean_frac'] = 0.0
            decoded, _params = _run(apply_floor=True, seed_params=seed)
    else:
        decoded, _params = _run(apply_floor=False)

    (sizes_by_zone, battery_size, psp_size, firm_cap,
     base_clean_frac, upper_clean_frac) = decoded
    return {
        'sizes_by_zone': sizes_by_zone,
        'battery_size': battery_size,
        'psp_size': psp_size,
        'firm_cap': firm_cap,
        'zone_plan': plan,
        'zone_coincidence': coincidence,
        'owners_fixed': False,
        'penalty_mode': penalty_mode,
        'cfe_target': cfe_target,
        'green_available': False,
        'base_clean_frac': base_clean_frac,
        'base_clean_source': base_clean_src if base_clean_frac is not None else None,
        'base_fossil_source': base_fossil_src if base_clean_frac is not None else None,
        'upper_clean_frac': upper_clean_frac,
    }
