"""Per-layer blended tariff: allocate the T&D stack across the economic zones.

Each zone report already carries an ENERGY cost (own PPA generation + firm +
market/grid + storage fixed). The rest of the bill -- network volumetric charges,
demand charges, fixed/customer charges -- is computed once for the whole site and
never reaches the zones, so ``blended_cost_per_mwh`` is an energy cost, not a
tariff. This module closes that gap.

Allocation follows CAUSATION, not one blanket pro-rata:

  * DEMAND charges are caused by the MW drawn at the specific hour that sets the
    bill (a daily or monthly maximum). Because the zones are STACKED PERCENTILE
    SLABS of the same demand series, the billed MW decomposes across them at that
    hour with no residual -- so this is exact arithmetic, not an estimate.
  * VOLUMETRIC charges follow the MWh each band actually took, hour by hour, so a
    time-varying network rate is respected rather than averaged away.
  * FIXED charges are caused by no band at all. They are spread by energy share
    and reported on their own line so they can be read separately.

The invariant that keeps this honest is checked by the caller: the allocated
total must equal the site's annual T&D charge to the cent. If it does not, the
split is inventing money and must not be reported.

Countries opt in by giving their calculator ``demand_billing_events`` and
(optionally) ``volumetric_rate_series``. Without them a calculator is treated as
wholly volumetric, which is the safe default -- the totals still reconcile, only
the demand/volumetric attribution is coarser.
"""

import numpy as np

ZONES = ('base', 'midBase', 'midMerit', 'peak')


def _split_at_hour(bands, demand, i, amount):
    """Decompose ``amount`` across the slabs live at hour ``i``.

    The slabs sum to the hour's demand by construction, so the shares sum to 1
    and nothing is lost or created.
    """
    total = float(demand[i])
    if total <= 0:
        return {z: 0.0 for z in ZONES}
    return {z: amount * (float(bands[z][i]) / total) for z in ZONES}


def allocate_tnd_to_layers(df, bands, calc, fixed_total, total_charges=None):
    """Split the T&D stack across zones. Returns (alloc, parts).

    ``alloc`` is ``{zone: {'volumetric', 'demand', 'fixed'}}``; ``parts`` totals the
    same three classes site-wide so the caller can reconcile.
    """
    demand = np.asarray(df['demand'].values, dtype=float)
    alloc = {z: {'volumetric': 0.0, 'demand': 0.0, 'fixed': 0.0} for z in ZONES}
    band_mwh = {z: float(np.asarray(bands[z]).sum()) for z in ZONES}
    total_mwh = float(demand.sum())
    if total_charges is None:
        total_charges = float(calc.get_total_charges())

    # ---- demand-type charges, at the hours that actually set them -----------
    events = []
    fn = getattr(calc, 'demand_billing_events', None)
    if callable(fn):
        try:
            events = list(fn() or [])
        except Exception:
            events = []
    demand_total = 0.0
    for i, amount in events:
        amount = float(amount)
        demand_total += amount
        for z, v in _split_at_hour(bands, demand, int(i), amount).items():
            alloc[z]['demand'] += v

    # ---- everything else is volumetric; keep the residual exact ------------
    volumetric_total = float(total_charges) - demand_total
    rates = None
    fn = getattr(calc, 'volumetric_rate_series', None)
    if callable(fn):
        try:
            rates = fn()
        except Exception:
            rates = None
    if rates is not None:
        rates = np.asarray(rates, dtype=float)
        shaped = {z: float((np.asarray(bands[z], dtype=float) * rates).sum()) for z in ZONES}
        shaped_sum = sum(shaped.values())
        if shaped_sum > 0:
            # scale to the residual so rounding in the calculator cannot leak
            k = volumetric_total / shaped_sum
            for z in ZONES:
                alloc[z]['volumetric'] += shaped[z] * k
        else:
            rates = None
    if rates is None and total_mwh > 0:
        for z in ZONES:
            alloc[z]['volumetric'] += volumetric_total * (band_mwh[z] / total_mwh)

    # ---- fixed charges: caused by no band ---------------------------------
    if total_mwh > 0:
        for z in ZONES:
            alloc[z]['fixed'] += float(fixed_total) * (band_mwh[z] / total_mwh)

    parts = {'demand': demand_total, 'volumetric': volumetric_total,
             'fixed': float(fixed_total)}
    return alloc, parts


def layer_tariff_report(alloc, bands, zone_energy_cost):
    """Per-zone tariff lines: energy + T&D volumetric + demand + fixed.

    ``zone_energy_cost`` is the zone's existing energy-side cost in currency
    units. Bands with no energy report zero rather than dividing by zero.
    """
    out = {}
    for z in ZONES:
        mwh = float(np.asarray(bands[z]).sum())
        a = alloc.get(z, {})
        energy = float(zone_energy_cost.get(z, 0.0) or 0.0)
        total = energy + a.get('volumetric', 0.0) + a.get('demand', 0.0) + a.get('fixed', 0.0)
        per = (lambda v: round(v / mwh, 4)) if mwh > 0 else (lambda v: 0.0)
        out[z] = {
            'energy_mwh': round(mwh, 4),
            'energy_per_mwh': per(energy),
            'tnd_volumetric_per_mwh': per(a.get('volumetric', 0.0)),
            'demand_charge_per_mwh': per(a.get('demand', 0.0)),
            'fixed_per_mwh': per(a.get('fixed', 0.0)),
            'blended_tariff_per_mwh': per(total),
            'tnd_volumetric_total': round(a.get('volumetric', 0.0), 2),
            'demand_charge_total': round(a.get('demand', 0.0), 2),
            'fixed_total': round(a.get('fixed', 0.0), 2),
        }
    return out
