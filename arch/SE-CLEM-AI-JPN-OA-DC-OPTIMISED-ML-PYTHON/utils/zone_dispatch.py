"""Independent-zone dispatch with a single shared battery.

This realises the deck's "Right Source, Right Layer" stack as a real
optimisation degree of freedom rather than a post-hoc report:

  * Each economic zone (Base · Mid-Base · Mid-Merit · Peak) is an
    INDEPENDENT sub-problem — it is sized and dispatched against its own
    slice of demand only. Solar and wind share ONE generation profile each
    but are sized SEPARATELY per zone (``solar_midMerit`` differs from
    ``wind_midBase`` etc.), so zones do not spill generation into one
    another. Which sources a zone may size is driven by the qualification
    matrix in the payload (fit / conditional only), so solar lands in
    mid-merit, wind spans mid-base + mid-merit, base stays firm-only and
    peak is battery + market — exactly the deck's assignment.

  * The BESS is the ONE shared asset that couples the zones. It is sized
    once, holds a single SOC trajectory, charges from any zone's surplus
    renewables (plus cheap market) and discharges top-of-stack — peak
    first, then mid-merit, then mid-base (never base, where storage is a
    misfit). Its per-zone discharge is an accounting split of one asset and
    always sums back to the single battery (asserted by the caller).

The module returns both the per-zone economics (independent P&L) and the
aggregate metric tuple ``compute_metrics`` already emits, so it can drive
the optimiser objective and the reported result without changing either
contract.
"""

import logging

import numpy as np

from config import config as _cfg
from utils.get_cheap_source import cheap_source
from utils.grid_costs import grid_peak, market_peak
from utils.layer_qualification import (
    get_payload, select_base_load_source, select_base_load_blend,
)
from utils.source_feasibility import parse_feasible_sources, source_available

try:
    from numba import jit
except Exception:  # numba optional — fall back to a no-op decorator
    def jit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap if not args else args[0]

# Economic zones, top-of-stack first (battery-allocation priority order).
ECONOMIC_ZONES = ('peak', 'midMerit', 'midBase', 'base')
# Zones the shared battery is allowed to discharge into (peak/mid-merit are
# fit, mid-base conditional; base is a misfit for storage per the matrix).
BATTERY_SERVED_ZONES = ('peak', 'midMerit', 'midBase')
# Fallback ownership hint (used only when a band has no active hours to
# measure). Real ownership is EARNED by coincidence with the band's actual
# hours — see zone_owner_by_coincidence — not assumed: whether wind can lead
# mid-base depends entirely on the site (night-weighting + CUF), exactly the
# deck's "standalone wind only at exceptional sites (CUF>40%) + banking".
ZONE_OWNER = {'midMerit': 'solar', 'midBase': 'wind'}

# Source families the model can optimise. A source ACTIVATES the moment its
# cost is supplied in the request (see zone_firm_options / build_dispatch_inputs)
# — nothing else is hardcoded, so H2 / OCGT / diesel slot straight in.
#  - VARIABLE: profile x MW, sized per zone, dispatched use-or-lose first.
#  - FIRM: dispatchable on marginal $/MWh, cheapest-first, fossils last (only
#    fill what clean firm + RE cannot). Clean firm is carbon-free; fossils carry
#    emissions (factor read from the payload, gCO2/kWh == kg/MWh).
VARIABLE_SOURCES = ('solar', 'wind')
CLEAN_FIRM_SOURCES = ('hydro', 'biomass', 'geothermal', 'nuclear', 'h2Turbine')
FOSSIL_FIRM_SOURCES = ('naturalGas', 'ocgtNaturalGas', 'diesel', 'coal')
FIRM_SOURCES = CLEAN_FIRM_SOURCES + FOSSIL_FIRM_SOURCES
FIRM_COST_KEYS = {
    'hydro': 'hydroCost', 'biomass': 'biomassCost', 'geothermal': 'geothermalCost',
    'nuclear': 'nuclearCost', 'h2Turbine': 'h2TurbineCost',
    'naturalGas': 'naturalGasCost', 'ocgtNaturalGas': 'ocgtNaturalGasCost', 'diesel': 'dieselCost',
    'coal': 'coalCost',
}
# Firm CAPACITY cost ($/MW-year) — the fixed charge you pay to CONTRACT a
# dispatchable firm plant for the MW you rely on, independent of hours run.
# Without it a peaker that runs 3% of hours looks free to own; with it, low
# utilisation is correctly expensive (its capacity is amortised over few MWh),
# which is exactly the deck's "cheap to own but brutal $/MWh" peaker economics.
# Charged only on UPPER-zone merit dispatch (mid-base/mid-merit/peak); the base
# floor runs ~flat, so its per-MWh input already reflects amortised capacity.
FIRM_CAPACITY_COST_KEYS = {
    'hydro': 'hydroCapacityCost', 'biomass': 'biomassCapacityCost',
    'geothermal': 'geothermalCapacityCost', 'nuclear': 'nuclearCapacityCost',
    'h2Turbine': 'h2TurbineCapacityCost', 'naturalGas': 'naturalGasCapacityCost',
    'ocgtNaturalGas': 'ocgtNaturalGasCapacityCost', 'diesel': 'dieselCapacityCost', 'coal': 'coalCapacityCost',
}
# firmed-hybrid / RTC is a CLEAN firm base product (contractually shaped solar-
# wind+storage) — competes only for the base floor, not the per-zone merit.
DISTRIBUTABLE_SOURCES = ('solar', 'wind')
_QUALIFYING = ('fit', 'conditional')


def resolve_cfe_target(metricsConfig):
    """Read requested CFE target percentage from customInputs.cfeTarget."""
    custom_inputs = (metricsConfig or {}).get('customInputs') or {}
    value = custom_inputs.get('cfeTarget', custom_inputs.get('cfe_target'))
    if value is None:
        try:
            payload_inputs = get_payload(metricsConfig).get('customInputs') or {}
            value = payload_inputs.get('cfeTarget', payload_inputs.get('cfe_target'))
        except Exception:
            value = None
    if value is None:
        return 0.0
    if isinstance(value, str):
        value = value.strip().rstrip('%')
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        logging.warning(f"Invalid customInputs.cfeTarget value {value!r}; defaulting to 0")
        return 0.0


def _is_fossil(source):
    return source in FOSSIL_FIRM_SOURCES


def emission_factor(source, payload):
    """kg CO2 per MWh for a fossil firm source, from the payload (gCO2/kWh).

    naturalGas keeps the model's existing factor so current results are
    unchanged; others read carbonGPerKwh straight from the payload sources.
    """
    if source == 'naturalGas':
        return float(_cfg.NATURAL_GAS_EMISSION_FACTOR)
    c = (payload.get('sources', {}).get(source, {}) or {}).get('carbonGPerKwh')
    return float(c) if isinstance(c, (int, float)) else 0.0


_MARKET_PEAK = np.vectorize(market_peak)
_GRID_PEAK = np.vectorize(grid_peak)


def _india_grid_price(df):
    """India guaranteed grid-supply price (₹/MWh), by config.MARKET_TYPE:
    green grid -> grid_cost_val, fossil grid -> grid_cost_fossil_val.
    Returns None if neither column is present (no grid backstop available)."""
    green = str(getattr(_cfg, 'MARKET_TYPE', 'fossil')).lower() == 'green'
    col = 'grid_cost_val' if green else 'grid_cost_fossil_val'
    if col not in df.columns:
        col = 'grid_cost_val' if 'grid_cost_val' in df.columns else (
            'grid_cost_fossil_val' if 'grid_cost_fossil_val' in df.columns else None)
    return df[col].values.astype(float) if col else None





def get_grid_cfe_fraction(metricsConfig=None):
    """Carbon-free share of the GRID a site imports from, as a fraction in [0, 1].

    Imported grid energy is not all fossil -- a Norwegian grid is ~99% carbon
    free, an Indian one ~25% -- so counting every imported MWh as dirty
    understates the site's real 24/7 carbon-free score. This is the grid's OWN
    clean share, distinct from `green_grid`, which is clean energy the site
    explicitly PROCURES.

    Defaults to 0, which is exactly the previous behaviour (grid contributes
    nothing to CFE), so a payload that does not send it is unchanged.
    """
    if metricsConfig:
        custom = (metricsConfig.get('customInputs') or {})
        # 'gridCfePercent' is the name already wired through main.py /
        # main_local.py into config.GRID_CFE_PERCENT; the rest are aliases.
        for key in ('gridCfePercent', 'gridCfePercentage', 'gridCfePct', 'gridCfe'):
            val = metricsConfig.get(key)
            if val is None:
                val = custom.get(key)
            if val is not None:
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if v > 0:
                    return min(v / 100.0 if v > 1.0 else v, 1.0)
    val = getattr(_cfg, 'GRID_CFE_PERCENT', getattr(_cfg, 'GRID_CFE_PERCENTAGE', 0)) or 0
    try:
        v = float(val)
    except (TypeError, ValueError):
        return 0.0
    return min(v / 100.0 if v > 1.0 else v, 1.0) if v > 0 else 0.0


def get_firm_source_cuf(source, metricsConfig=None):
    """Capacity Utilization Factor (CUF) for a firm source as a float in range (0, 1].
    
    Checks metricsConfig (or customInputs) first, then falls back to config.
    E.g. 0.85 for 85%.
    """
    if not source or source in VARIABLE_SOURCES:
        return 1.0
        
    if metricsConfig:
        custom = (metricsConfig.get('customInputs') or {})
        firm_cuf_map = (
            metricsConfig.get('cuf') or custom.get('cuf') or
            metricsConfig.get('firmCuf') or custom.get('firmCuf') or {}
        )
        if isinstance(firm_cuf_map, dict):
            val = firm_cuf_map.get(source)
            if val is None and source == 'ocgtNaturalGas':
                val = firm_cuf_map.get('ocgt') or firm_cuf_map.get('ocgtNaturalGas')
            if val is not None and float(val) > 0:
                return float(val) / 100.0 if float(val) > 1.0 else float(val)
        elif firm_cuf_map:
            val = float(firm_cuf_map)
            if val > 0:
                return val / 100.0 if val > 1.0 else val
        
        for key in (f"{source}Cuf", f"{source}_cuf", f"{source.upper()}_CUF"):
            val = metricsConfig.get(key) or custom.get(key)
            if val and float(val) > 0:
                return float(val) / 100.0 if float(val) > 1.0 else float(val)

        if source == 'ocgtNaturalGas':
            for key in ("ocgtCuf", "ocgt_cuf", "OCGT_CUF", "ocgt"):
                val = metricsConfig.get(key) or custom.get(key)
                if val and float(val) > 0:
                    return float(val) / 100.0 if float(val) > 1.0 else float(val)

    cuf_dict = getattr(_cfg, 'FIRM_SOURCE_CUF', {})
    if isinstance(cuf_dict, dict):
        val = cuf_dict.get(source)
        if val is None and source == 'ocgtNaturalGas':
            val = cuf_dict.get('ocgt') or cuf_dict.get('ocgtNaturalGas')
        if val is not None and float(val) > 0:
            return float(val) / 100.0 if float(val) > 1.0 else float(val)

    attr_name = f"{source.upper()}_CUF"
    val = getattr(_cfg, attr_name, 0.85)
    if val and float(val) > 0:
        return float(val) / 100.0 if float(val) > 1.0 else float(val)

    return 0.85


def build_dispatch_inputs(df, metricsConfig):
    """Assemble the (arrays, bess-base, firm, layer_zones, unit_costs) inputs
    the zone dispatch needs from the dataframe + config, once per run.

    Everything here is independent of the battery *size* (Pmax is resolved
    per candidate size by :func:`compute_zone_metrics`).
    """
    bessType = metricsConfig['bessType']
    demand = df['demand'].values.astype(float)
    arrays = {
        'solar_p': df['Solar Power Per MW'].values.astype(float),
        'wind_p': df['Wind Power Per MW'].values.astype(float),
        'demand': demand,
        'market_price': df['MCP (Rs/MWh)'].values.astype(float),
        # India guaranteed grid-supply price (₹/MWh, time-of-day tariff) — the
        # always-available backstop when the open market is off (noOpenMarket).
        # green grid vs fossil grid selected by config.MARKET_TYPE.
        'grid_price': _india_grid_price(df),
        'q': df[f'MCP_{int(_cfg.MCP_PERCENTILE * 100)}th_Percentile'].values.astype(float),
        'is_market_peak': _MARKET_PEAK(df['hour'].values, df['demand'].values).astype(float),
        'is_grid_peak': _GRID_PEAK(df['hour'].values).astype(float),
        'hour': df['hour'].values,
        # calendar columns for the hourly simulation sheet (fall back gracefully)
        'day': df['day'].values if 'day' in df else np.arange(len(demand)),
        'month': df['month'].values if 'month' in df else np.zeros(len(demand)),
        # India banking peak-slot flag (1 = peak bucket hour for this state).
        'banking_peak': np.zeros(len(demand), dtype=np.int64),
    }
    bess_base = {
        'charge_eff': bessType['batteryChargingEfficiency'],
        'discharge_eff': bessType['batteryDischargingEfficiency'],
        'min_SOC': bessType['minSoc'],
        'max_SOC': bessType['maxSoc'],
        'initial_SOC': _cfg.INITIAL_SOC,
        'rte': bessType['roundTripEfficiency'],
        'dod': bessType['depthOfDischarge'],
        'backup_hours': bessType['batteryBackupHours'],
    }
    # firm base source: prefer the qualified base-load pick (deck discipline),
    # fall back to the plain cheapest source
    try:
        firm_name, firm_cost = select_base_load_source(metricsConfig)
    except Exception:
        firm_name, firm_cost = None, 0.0

    try:
        base_clean, base_fossil = select_base_load_blend(metricsConfig)
    except Exception:
        base_clean, base_fossil = None, None

    payload = get_payload(metricsConfig)
    custom_inputs = metricsConfig.get('customInputs') or {}
    layer_zones = (
        custom_inputs.get('layerZones')
        or payload.get('customInputs', {}).get('layerZones')
        or {"base": "p10", "midBase": "p40", "midMerit": "p90", "peak": "p100"}
    )
    # firm floor is sized to the base BAND edge (from layerZones), so the base
    # zone is always 100% firm with no market leak -- even if the base layer is
    # moved off p10. Falls back to base_load / config if the edge is degenerate.
    firm = float(resolve_edges(demand, layer_zones)[0]
                 or metricsConfig.get('base_load') or _cfg.BASE_LOAD)
    # per-source firm marginal costs + which zones each may dispatch into
    feasible_sources = parse_feasible_sources(metricsConfig)
    firm_costs = {s: metricsConfig.get(FIRM_COST_KEYS[s])
                  for s in FIRM_SOURCES
                  if metricsConfig.get(FIRM_COST_KEYS[s]) not in (None, 0)
                  and source_available(s, feasible_sources)}
    # firm CAPACITY costs ($/MW-year): request override first, else payload default
    payload_caps = payload.get('firmCapacityCosts') or {}
    firm_capacity_costs = {}
    for s in FIRM_SOURCES:
        v = metricsConfig.get(FIRM_CAPACITY_COST_KEYS[s])
        if v in (None, 0):
            v = payload_caps.get(s)
        if v not in (None, 0):
            firm_capacity_costs[s] = float(v)
    firm_options = zone_firm_options(payload, metricsConfig)

    def _zone_has_both(zone):
        srcs = [s for s in firm_options.get(zone, []) if s in firm_costs]
        return any(not _is_fossil(s) for s in srcs) and any(_is_fossil(s) for s in srcs)

    upper_blend_available = _zone_has_both('midBase') or _zone_has_both('midMerit')
    unit_costs = {
        'solarPPA': metricsConfig['solarPPA'],
        'windPPA': metricsConfig['windPPA'],
        'firmCost': firm_cost,
        'firmName': firm_name,
        'baseCleanSource': base_clean[0] if base_clean else None,
        'baseCleanCost': base_clean[1] if base_clean else None,
        'baseFossilSource': base_fossil[0] if base_fossil else None,
        'baseFossilCost': base_fossil[1] if base_fossil else None,
        'upperBlendAvailable': upper_blend_available,
        'firmCosts': firm_costs,
        'firmCapacityCosts': firm_capacity_costs,
        'firmOptions': firm_options,
        # which variable sources the matrix qualifies per zone — gates the
        # bottom-up carry-forward (excess only cascades where it qualifies)
        'zoneVarQualified': {z: set(srcs)
                             for z, srcs in zone_sizing_plan(payload, metricsConfig).items()},
        'batteryCost': metricsConfig['batteryCost'],
        'reliabilityScores': metricsConfig.get('reliabilityScores', {}),
        'payload': payload,   # for hybrid-pairing enforcement in dispatch
        'firmedHybridBase': firmed_hybrid_base_enabled(metricsConfig),
        # firm sources are take-or-pay PPAs by default (contract price, no spot
        # gate); set customInputs.firmTakeOrPay=false to model them as merchant tolls
        'firmTakeOrPay': ((metricsConfig.get('customInputs') or {}).get('firmTakeOrPay', True)),
        # NO-OPEN-MARKET scenario: the spot market does not exist at all -- firm
        # covers the whole residual (0% market for demand) AND the battery/PSP may
        # charge only from RE surplus, never from cheap spot. customInputs.noOpenMarket
        'noOpenMarket': bool((metricsConfig.get('customInputs') or {}).get('noOpenMarket')),
        # noExport: don't sell surplus RE (no export credit) -> the optimiser sizes
        # RE to serve demand, not to overbuild-and-export.
        'noExport': bool((metricsConfig.get('customInputs') or {}).get('noExport')),
        # India energy banking: mode (0=off / 1=generic) from the state; deposit
        # haircut and month-end lapse charge from config.
        'state': metricsConfig.get('state'),
        'bankingMode': 0,
        'bankingChargePct': 0.0,
        'recoveryBankCost': 0.0,
        # annual-mean green-grid price (₹/MWh) for automatic minimal-green procurement
        'greenGridPriceMean': (float(np.nanmean(df['grid_cost_val']))
                               if 'grid_cost_val' in df.columns else 0.0),
    }
    # optional second shared storage (pumped storage) — activates only when a
    # pspConfig is supplied; mirrors the battery (charge from RE, discharge into
    # top-of-stack deficit after the battery).
    # pumped storage activates only when a pspConfig is supplied AND pumped
    # storage is procurable at the location (e.g. CA lists it nowhere — it is
    # utility grid infrastructure, not a C&I product — so it stays off there)
    psp = metricsConfig.get('pspConfig') or {}
    if psp and source_available('pumpedStorage', parse_feasible_sources(metricsConfig)):
        unit_costs['pspBase'] = {
            'charge_eff': psp.get('batteryChargingEfficiency', 0.9),
            'discharge_eff': psp.get('batteryDischargingEfficiency', 0.9),
            'min_SOC': psp.get('minSoc', 0.1),
            'max_SOC': psp.get('maxSoc', 0.95),
            'initial_SOC': _cfg.INITIAL_SOC,
            'rte': psp.get('roundTripEfficiency', 0.8),
            'dod': psp.get('depthOfDischarge', 0.85),
            'backup_hours': psp.get('batteryBackupHours', 8),
            'cost': psp.get('pspCost', 0.0),
        }
    return arrays, bess_base, firm, layer_zones, unit_costs, payload


def compute_zone_metrics(sizes_by_zone, battery_size, df, metricsConfig,
                         prebuilt=None, return_full=False, psp_size=0.0, firm_cap=None,
                         cfe_target=0.0, green_available=False,
                         base_clean_frac=None, upper_clean_frac=None):
    """Score a per-zone sizing candidate through the shared-battery dispatch.

    Returns the 10-value metric tuple that ``compute_metrics`` emits (so it
    drops straight into the optimiser objective), or the full dispatch dict
    when ``return_full`` is set (for building the reported result).
    """
    if prebuilt is None:
        prebuilt = build_dispatch_inputs(df, metricsConfig)
    arrays, bess_base, firm, layer_zones, unit_costs, _payload = prebuilt

    backup_hours = bess_base['backup_hours']
    Pmax = (battery_size / backup_hours) if battery_size else 0.0
    bess = dict(bess_base)
    bess['Pmax'] = Pmax
    battery_fixed = unit_costs['batteryCost'] * Pmax
    uc = dict(unit_costs)
    uc['batteryFixed'] = battery_fixed

    result = dispatch_zone_system(
        sizes_by_zone, battery_size, arrays, bess, firm, layer_zones, uc,
        green_base_load=bool(metricsConfig.get('greenBaseLoad')),
        psp_size=psp_size, firm_cap=firm_cap,
        cfe_target=cfe_target, green_available=green_available,
        base_clean_frac=base_clean_frac, upper_clean_frac=upper_clean_frac,
    )
    if return_full:
        return result
    return result['metric_tuple']


def build_zone_result_entry(opt_result, df, metricsConfig):
    """Build the reported result for the independent-zone optimisation.

    Produces the standard aggregate keys downstream consumers read (sizes,
    percentages, costs, LCOE, CFE, emissions) computed from the independent
    per-zone dispatch, plus a ``zone_optimization`` block carrying the
    per-zone P&L and the single shared-battery bookkeeping.
    """
    from utils.demand_charge_cal import (
        compute_clem_demand_charges, basic_charge)

    sizes_by_zone = opt_result['sizes_by_zone']
    battery_size = opt_result['battery_size']
    full = compute_zone_metrics(sizes_by_zone, battery_size, df, metricsConfig,
                                return_full=True, psp_size=opt_result.get('psp_size', 0.0),
                                firm_cap=opt_result.get('firm_cap'),
                                cfe_target=opt_result.get('cfe_target', 0.0),
                                green_available=opt_result.get('green_available', False),
                                base_clean_frac=opt_result.get('base_clean_frac'),
                                upper_clean_frac=opt_result.get('upper_clean_frac'))
    agg = full['aggregate']
    (RE_util, batt_util, peak_met, annual_dc, solar_meet, wind_meet,
     market_meet, reliability_score, total_cost_per_mwh, cfe_score) = full['metric_tuple']

    state = metricsConfig['state']
    bessType = metricsConfig['bessType']
    total_demand = agg['total_demand']
    backup_hours = bessType['batteryBackupHours']

    solar_mw = sum((sz.get('solar') or 0.0) for sz in sizes_by_zone.values())
    wind_mw = sum((sz.get('wind') or 0.0) for sz in sizes_by_zone.values())

    firm_name, firm_cost = None, None
    try:
        firm_name, firm_cost = select_base_load_source(metricsConfig)
    except Exception:
        firm_name, firm_cost = cheap_source(metricsConfig)

    _trans = float(df['demand'].sum() * getattr(_cfg, 'TRANSMISSION_COST', 0))
    _basic = float(basic_charge(df))
    _demand_charge = float(compute_clem_demand_charges(df)[0])
    annual_tnd_charge = _trans + _basic + _demand_charge

    # ---- per-layer blended TARIFF ----------------------------------------
    # Each zone already reports an ENERGY cost; the T&D stack was site-wide only,
    # so blended_cost_per_mwh was never a tariff. Allocate that stack by causation
    # (demand charges at the hour whose MW set them, volumetric by the MWh each band
    # took, fixed by energy share) and publish a per-zone tariff alongside it. This
    # is reporting only -- no dispatch or total changes. If the allocation does not
    # reconcile to annual_tnd_charge it is dropped rather than reported wrong.
    layer_tariff = None
    try:
        from utils.layer_tariff import allocate_tnd_to_layers, layer_tariff_report
        class _JPNLayerCalc:
            def __init__(self, df, demand_side_total):
                self.df = df
                self._total = float(demand_side_total)

            def demand_billing_events(self):
                import numpy as _np
                d = self.df['demand'].values.astype(float)
                months = self.df['month'].values
                idx, wt = [], []
                for m in _np.unique(months):
                    sel = _np.where(months == m)[0]
                    j = int(sel[int(_np.argmax(d[sel]))])
                    idx.append(j)
                    wt.append(d[j])
                tw = sum(wt)
                if tw <= 0:
                    return []
                return [(i, self._total * w / tw) for i, w in zip(idx, wt)]

        _lt_calc = _JPNLayerCalc(df, float(_basic) + float(_demand_charge))
        _lt_charges = float(annual_tnd_charge)
        _lt_fixed = 0.0
        _lt_credit = 0.0
        _lt_bands = band_demand(df['demand'].values.astype(float), full['edges'])
        _alloc, _parts = allocate_tnd_to_layers(
            df, _lt_bands, _lt_calc, _lt_fixed, total_charges=_lt_charges)
        # The import T&D credit backs out network cost on imported MWh (their price
        # already bundles T&D). It is CAUSED by the bands that imported, so it is
        # returned to them in proportion to their imported energy rather than spread
        # over everyone -- otherwise self-generating bands would receive a credit for
        # imports they never made.
        if _lt_credit > 0:
            _imp = {z: sum(float((full['zones'].get(z, {}).get('energy_mwh') or {}).get(k, 0.0) or 0.0)
                           for k in ('market', 'grid', 'green_grid'))
                    for z in _alloc}
            _imp_tot = sum(_imp.values())
            _lt_credit = min(_lt_credit, _lt_charges + _lt_fixed)   # mirror the max(0, ...) clamp
            for z in _alloc:
                share = (_imp[z] / _imp_tot) if _imp_tot > 0 else (1.0 / len(_alloc))
                _alloc[z]['volumetric'] -= _lt_credit * share
            _parts['import_credit'] = -round(_lt_credit, 2)
            _parts['volumetric'] = round(_parts['volumetric'] - _lt_credit, 2)
        _allocated = sum(sum(v.values()) for v in _alloc.values())
        if abs(_allocated - annual_tnd_charge) <= max(0.01, abs(annual_tnd_charge) * 1e-9):
            # Energy cost per band, PLUS its share of charging the shared storage
            # from the grid. That charging sits in total_market_cost but in no band's
            # own cost, so without it the layer tariffs cannot add back to the site
            # bill. Attributed by discharge share -- the band that used the stored
            # energy pays for filling it. (zone cost_musd itself is left untouched.)
            _chg = (float(agg.get('battery_market_charge_cost', 0.0) or 0.0)
                    + float(agg.get('psp_market_charge_cost', 0.0) or 0.0))
            _disch = {z: float(((full['zones'].get(z, {}).get('energy_mwh') or {}).get('battery', 0.0) or 0.0)
                               + ((full['zones'].get(z, {}).get('energy_mwh') or {}).get('pumpedStorage', 0.0) or 0.0))
                      for z in _alloc}
            _dtot = sum(_disch.values())

            def _lt_e(z, key):
                return float(((full['zones'].get(z, {}).get('energy_mwh') or {})
                              .get(key, 0.0)) or 0.0)

            _band_mwh = {z: float(np.asarray(_lt_bands[z]).sum()) for z in _alloc}
            _mwh_tot = sum(_band_mwh.values())

            def _lt_share(z, weights, wtot):
                """Causal share, falling back to energy share when the causal
                basis is empty -- so a term is never silently dropped."""
                if wtot > 0:
                    return weights[z] / wtot
                return (_band_mwh[z] / _mwh_tot) if _mwh_tot > 0 else 0.0

            # Two further site-bill terms sit in NO band's cost_musd, so without
            # them the layer tariffs cannot add back to the bill:
            #   bank_recovery_cost -- the month-end lapse charge on banked own RE.
            #     Caused by the bands that DREW from the bank (the scheme runs to
            #     serve them), so it follows bank withdrawal.
            #   green_premium -- the extra paid to green the DIRTY residual.
            #     Caused by the bands that left one: market + grid + bank + the
            #     grid-charged slice of storage discharge, i.e. exactly the basis
            #     the aggregate `greenable` is built from.
            # Both are structurally zero in this engine today (bankingMode is
            # pinned to 0 and green_available is never enabled), but they are
            # attributed rather than assumed away so the guard stays honest if
            # either is ever switched on.
            _bank_c = float(agg.get('bank_recovery_cost', 0.0) or 0.0)
            _green_c = float(agg.get('green_premium', 0.0) or 0.0)
            _bank_e = {z: _lt_e(z, 'bank') for z in _alloc}
            _bank_tot = sum(_bank_e.values())
            _gridchg = float(agg.get('market_discharge', 0.0) or 0.0)
            _dirty = {z: (_lt_e(z, 'market') + _lt_e(z, 'grid') + _bank_e[z]
                          + (_gridchg * _disch[z] / _dtot if _dtot > 0 else 0.0))
                      for z in _alloc}
            _dirty_tot = sum(_dirty.values())
            # cost_musd is scaled by 1e7 in this repo (crore-style), NOT 1e6 --
            # multiplying back by 1e6 would report a tenth of every band's energy
            # cost and the reconciliation guard below would (rightly) drop the report.
            _lt_energy = {z: ((full['zones'].get(z, {}).get('cost_musd', 0.0) or 0.0) * 1e7
                             + _chg * _lt_share(z, _disch, _dtot)
                             + _bank_c * _lt_share(z, _bank_e, _bank_tot)
                             + _green_c * _lt_share(z, _dirty, _dirty_tot))
                          for z in _alloc}
            layer_tariff = layer_tariff_report(_alloc, _lt_bands, _lt_energy)
            layer_tariff['_totals'] = {k: round(v, 2) for k, v in _parts.items()}
        else:
            logging.warning(
                "Per-layer tariff not reported: allocation %.2f does not reconcile "
                "to the T&D total %.2f", _allocated, annual_tnd_charge)
    except Exception as e:
        logging.warning("Per-layer tariff skipped: %s", e)

    solar_cost = agg['solar_energy_cost']
    wind_cost = agg['wind_energy_cost']
    battery_cost = agg['battery_fixed']
    psp_cost = agg.get('psp_fixed', 0.0)
    market_cost = agg['total_market_cost']
    grid_cost = agg.get('total_grid_cost', 0.0)          # India grid backstop
    grid_energy = agg.get('grid_energy', 0.0)
    bank_recovery_cost = agg.get('bank_recovery_cost', 0.0)   # month-end lapse charge
    bank_discharge = agg.get('bank_discharge', 0.0)          # own RE served from bank
    firm_total_cost = agg['firm_total_cost']
    firm_source_energy = agg['firm_source_energy']
    firm_source_cost = agg['firm_source_cost']
    firm_source_capacity_cost = agg.get('firm_source_capacity_cost', {})
    firm_source_peak_mw = agg.get('firm_source_peak_mw', {})
    export_revenue = agg['export_revenue']

    green_premium = agg.get('green_premium', 0.0)   # cost of automatic minimal green
    total_cost_without_exports = (solar_cost + wind_cost + battery_cost + psp_cost
                                  + market_cost + grid_cost + firm_total_cost
                                  + bank_recovery_cost + annual_tnd_charge + green_premium)
    # ---- per-layer FINAL cost (tax applied the way the site bill applies it) --
    # Japan levies a 10% consumption tax on electricity, but this engine prices the
    # WHOLE site bill pre-tax: no rate is read from the payload, no config key holds
    # one, and no term of total_cost_without_exports carries one. So the tax is out
    # of scope here, not dropped from a formula that used to have it -- and the layer
    # tariffs must stay pre-tax too, or they would stop adding back to the site bill.
    # The rate is kept as an explicit knob (rather than the arithmetic being deleted)
    # so that if the site formula ever goes tax-inclusive, the split follows it: the
    # tax is charged for USING THE GRID, so it would land on everything delivered
    # over the network -- PPA energy, firm, market/grid and the network charges --
    # but NOT on the battery, which sits behind the meter.
    if layer_tariff:
        try:
            _lt_tax = 0.0
            _grand = 0.0
            for _z, _t in layer_tariff.items():
                if _z.startswith('_'):
                    continue
                _cbs = (full['zones'].get(_z) or {}).get('cost_by_source_musd') or {}
                # same 1e7 (crore-style) scale as cost_musd above
                _batt = float(_cbs.get('battery', 0.0) or 0.0) * 1e7
                _zenergy = float(_lt_energy.get(_z, 0.0) or 0.0)
                _ztnd = (_t['tnd_volumetric_total'] + _t['demand_charge_total']
                         + _t['fixed_total'])
                _taxable = (_zenergy - _batt) + _ztnd
                _final = _taxable * (1.0 + _lt_tax / 100.0) + _batt
                _mwh = _t['energy_mwh']
                _t['gst_rate_pct'] = _lt_tax
                _t['gst_exempt_battery_total'] = round(_batt, 2)
                _t['gst_total'] = round(_taxable * _lt_tax / 100.0, 2)
                _t['final_cost_total'] = round(_final, 2)
                _t['final_cost_per_mwh'] = round(_final / _mwh, 4) if _mwh else 0.0
                _grand += _final
            layer_tariff['_totals']['gst_rate_pct'] = _lt_tax
            layer_tariff['_totals']['final_cost_all_zones'] = round(_grand, 2)
        except Exception as e:
            logging.warning("Per-layer final cost skipped: %s", e)
            layer_tariff = None

    # Stronger guard than the T&D check above: the four FINAL layer costs must add
    # back to the site bill. The T&D check alone can pass while the layers are still
    # short -- tax, or the cost of charging storage from the grid -- so reconcile
    # against the number the customer actually pays.
    if layer_tariff and 'final_cost_all_zones' in (layer_tariff.get('_totals') or {}):
        _fin = layer_tariff['_totals']['final_cost_all_zones']
        if abs(_fin - total_cost_without_exports) > max(0.05, abs(total_cost_without_exports) * 1e-9):
            logging.warning(
                "Per-layer tariff not reported: final costs %.2f do not reconcile to "
                "the site bill %.2f (short by %.2f)", _fin, total_cost_without_exports,
                total_cost_without_exports - _fin)
            layer_tariff = None

    total_cost = total_cost_without_exports - export_revenue
    lcoe_without_selling = (total_cost_without_exports / (total_demand * 1000)) * 1e6 / 1e6
    lcoe_with_selling = (total_cost / (total_demand * 1000)) * 1e6 / 1e6

    # ---- emissions: dirty residual (market + grid + grid-charged storage) + fossil
    # firm. The green-procured portion (green_fill) is renewable, so it carries no
    # grid emission — only the remaining dirty residual and fossil firm do.
    grid_emission = total_demand * _cfg.GRID_EMISSION_FACTOR
    market_for_demand = agg['market_energy']
    market_discharge = agg['market_discharge']
    green_fill = agg.get('green_fill', 0.0)
    dirty_residual = max(0.0, market_for_demand + market_discharge + grid_energy - green_fill)
    clem_emission = dirty_residual * _cfg.GRID_EMISSION_FACTOR + agg.get('fossil_emissions', 0.0)
    avoided_emission = ((grid_emission - clem_emission) / grid_emission * 100) if grid_emission > 0 else 0.0

    battery_meet = (agg['battery_discharge'] / total_demand * 100) if total_demand else 0.0

    # CFE floor bookkeeping: report the requested target and whether the sized
    # build actually reaches it. A small tolerance absorbs rounding so a target
    # met to the last decimal still reads as met. With no target (0) the floor
    # is trivially satisfied.
    cfe_target = float(opt_result.get('cfe_target', 0) or 0)
    cfe_target_met = (cfe_target <= 0) or (cfe_score >= cfe_target - 0.05)
    if cfe_target > 0 and not cfe_target_met:
        logging.warning(
            f"CFE target {cfe_target:.2f}% not fully reached: achieved "
            f"{cfe_score:.2f}% (target may be infeasible for this demand/resource mix)")

    # per firm-source aggregate keys (% / annual cost). Base-floor sources carry
    # a nameplate MW (possibly split by a CFE clean/fossil blend); dispatch merit
    # sources use their peak delivered MW.
    _blend = full.get('base_blend') or {}
    if _blend.get('active'):
        base_floor_share = {c['source']: c['fraction'] for c in _blend.get('components', [])}
    elif firm_name:
        base_floor_share = {firm_name: 1.0}
    else:
        base_floor_share = {}
    base_nameplate = float(full.get('edges', (metricsConfig.get('base_load') or _cfg.BASE_LOAD,))[0]
                           or metricsConfig.get('base_load') or _cfg.BASE_LOAD)

    firm_sources_dict = {}
    firm_source_entry_fields = {}
    for s, energy in firm_source_energy.items():
        if energy <= 0 and s != firm_name:
            continue
        pct = (energy / total_demand * 100) if total_demand else 0.0
        contracted_mw = firm_source_peak_mw.get(s, 0.0)
        is_base_floor = s in base_floor_share
        floor_mw = base_nameplate * base_floor_share[s] if is_base_floor else 0.0
        s_mw = max(floor_mw, round(contracted_mw, 4))
        # report INSTALLED capacity: a firm plant at 85% CUF needs more
        # nameplate than the MW it delivers. Absent a payload CUF the
        # default is 0.85, matching the models that already do this.
        _cuf = get_firm_source_cuf(s, metricsConfig)
        s_mw = round(s_mw / _cuf, 4) if (_cuf > 0 and s_mw) else s_mw
        total_c = firm_source_cost.get(s, 0.0)
        cap_c = firm_source_capacity_cost.get(s, 0.0)
        annual_c = total_c / 1e7

        firm_source_entry_fields[f'{s}_mw'] = s_mw
        firm_source_entry_fields[f'{s}_percentage'] = pct
        firm_source_entry_fields[f'annual_{s}_cost'] = annual_c

        firm_sources_dict[s] = {
            'role': 'base_floor' if is_base_floor else 'dispatch_merit',
            'mw': (s_mw or None),
            'contracted_mw': s_mw,
            'peak_delivered_mw': round(contracted_mw, 4),
            'energy_mwh': energy,
            'percentage': pct,
            'annual_cost_musd': annual_c,
            'capacity_cost_musd': cap_c / 1e7,
            'variable_cost_musd': (total_c - cap_c) / 1e7,
            'effective_cost_per_mwh': (total_c / energy) if energy > 0 else 0.0,
            'is_clean': not _is_fossil(s),
        }

    battery_mw_val = (battery_size / backup_hours) if battery_size else 0.0
    grid_mw_val = float(np.max(full.get('hourly', {}).get('grid_mw', [0])))
    market_mw_val = float(np.max(full.get('hourly', {}).get('market_mw', [0])))

    entry = {
        'reliability_score': reliability_score * 100,

        # Solar
        'solar_mw': solar_mw,
        'solar_percentage': solar_meet,
        'annual_solar_cost': solar_cost / 1e7,

        # Wind
        'wind_mw': wind_mw,
        'wind_percentage': wind_meet,
        'annual_wind_cost': wind_cost / 1e7,

        # Battery
        'battery_mw': battery_mw_val,
        'battery_percentage': battery_meet,
        'annual_battery_cost': battery_cost / 1e7,
        'storage_capacity_mw': battery_mw_val,
        'storage_capacity_mwh': battery_size or 0,

        # Market
        'market_mw': market_mw_val,
        'market_percentage': market_meet,
        'annual_market_cost': market_cost / 1e7,

        # Grid
        'grid_mw': grid_mw_val,
        'grid_percentage': (grid_energy / total_demand * 100) if total_demand else 0.0,
        'annual_grid_cost': grid_cost / 1e7,
    }

    # Add active firm sources right after Grid
    entry.update(firm_source_entry_fields)

    # General metrics & optimization results
    entry.update({
        'bank_percentage': (bank_discharge / total_demand * 100) if total_demand else 0.0,
        're_utilization_percentage': RE_util,
        'bess_utilization_percentage': batt_util,
        'peak_demand_met_percentage': peak_met,
        'annual_demand_meet_percentage': annual_dc,
        'annual_tnd_charge': annual_tnd_charge / 1e7,
        'annual_operational_cost_cr': total_cost_without_exports / 1e7,
        'lcoe_without_selling': lcoe_without_selling,
        'cost_recovery_by_selling_re_energy_to_open_market': export_revenue / 1e7,
        'lcoe_with_selling_excess': lcoe_with_selling,
        'annual_hourly_cfe_score': cfe_score,
        'cfe_target': cfe_target,
        'cfe_target_met': cfe_target_met,
        'base_load_blend': full.get('base_blend'),
        'green_grid_percentage': (green_fill / total_demand * 100) if total_demand else 0.0,
        'green_grid_premium_cr': green_premium / 1e7,
        'avoided_emission_percentage': avoided_emission,
        'clem_ai_annual_emission': clem_emission / 1000,
        'grid_annual_emission': grid_emission / 1000,
        'yearlyDemand': float(total_demand),
        'openAccessCharges': 0,
        'baseLoadSource': firm_name,
        'availability': 100,
        'zone_optimization': {
            'mode': 'independent_zones_shared_battery',
            'note': 'Each zone is sized and dispatched independently against its own '
                    'demand band (generation does not spill between zones); the BESS is '
                    'a single shared asset that charges from any zone surplus and '
                    'discharges top-of-stack (peak -> mid-merit -> mid-base). No owners '
                    'are fixed: every qualification-matrix-admitted source is a free '
                    'variable and the optimiser picks the cheapest mix per zone.',
            'owners_fixed': opt_result.get('owners_fixed', False),
            'zone_plan': opt_result.get('zone_plan'),
            'zone_coincidence': opt_result.get('zone_coincidence'),
            'layer_zones': full.get('edges') and {
                'base': round(full['edges'][0], 3),
                'midBase': round(full['edges'][1], 3),
                'midMerit': round(full['edges'][2], 3),
            },
            'zones': full['zones'],
            'layer_tariff': layer_tariff,
            'layerWiseMW': full.get('layer_wise_mw'),
            'shared_battery': full['battery'],
            'pumped_storage': full.get('pumped_storage'),
            'firm_sources': firm_sources_dict,
        },

        # residual backstop: grid-only when the open market is off, else the cheaper
        # of open-market vs grid each hour
        'residual_backstop': ('grid' if bool((metricsConfig.get('customInputs') or {})
                                              .get('noOpenMarket')) else 'market+grid'),
    })
    return entry, full.get('hourly', {})


# A hybrid source qualifies a zone as a portfolio, not as its bare parts.
# The deck's mid-base owner is the solar-wind HYBRID + banking — bare wind is
# only a conditional fallback and bare solar is a misfit there. The payload's
# modelMapping represents that hybrid as solar + wind + BESS, so a distributable
# source is sizeable in a zone when the source itself qualifies OR a hybrid it
# belongs to qualifies. Banking is provided by the single shared battery.
DEFAULT_HYBRID_CONSTITUENTS = {'hybridSolarWind': ('solar', 'wind')}

# Zones where bare renewables may be sized directly. Base is excluded by
# default (firm-only). The deck admits RE into base only as a CONTRACTUALLY
# FIRMED hybrid; that is opt-in via customInputs.firmedHybridBase, which lets
# base size its own solar+wind (hybrid-paired) firmed by the shared battery,
# with the firm source as the residual backstop — exactly like mid-base.
_DISTRIBUTABLE_ZONES = ('midBase', 'midMerit', 'peak')


def firmed_hybrid_re_procurable(metricsConfig):
    """Whether a firmed-hybrid base can be built here at all.

    A firmed hybrid base is solar + wind + shared battery. BOTH variable sources
    must be procurable: the pairing is the whole point -- solar and wind are
    anti-correlated, and it is that complementarity (firmed by the shared battery)
    which lets a hybrid stand in for a firm base block. With only one of them the
    build is not a hybrid, it is a single intermittent source leaning on storage,
    so the mode must not engage -- it would strip the flat firm floor out of the
    base band and leave nothing behind to replace it.

    ``parse_feasible_sources`` returning None means the request declared no location
    restriction at all, which is permission by the same convention
    ``source_available`` uses everywhere else.
    """
    feasible = parse_feasible_sources(metricsConfig)
    if feasible is None:
        return True
    return ('solar' in feasible) and ('wind' in feasible)


def firmed_hybrid_base_enabled(metricsConfig):
    """Whether base may be a firmed hybrid (solar + wind + shared battery, with
    the firm source as the guaranteed backstop) rather than a flat firm-only floor.

    OFF BY DEFAULT. Set ``customInputs.firmedHybridBase`` to 1 / true and list
    the firmed hybrid in ``resources.feasibleEnergySources`` to opt into the RTC
    firmed-hybrid base.
    """
    ci = (metricsConfig or {}).get('customInputs') or {}
    feasible = parse_feasible_sources(metricsConfig)
    
    if bool(ci.get('firmedHybridBase', False) and source_available('firmedHybrid', feasible)):
        if not firmed_hybrid_re_procurable(metricsConfig):
            logging.warning(
                "firmedHybridBase requested but ignored: a firmed-hybrid base is "
                "solar+wind+battery and needs BOTH; resources.feasibleEnergySources "
                "does not carry solar and wind. Keeping the firm-only base floor.")
            return False
        return True
        
    cfe_target = float(ci.get('cfeTarget', 0) or 0.0)
    if cfe_target > 0:
        # In custom CFE, if there are no clean constant sources, use solar, wind, and battery in base zone.
        try:
            from utils.layer_qualification import select_base_load_blend
            bc, _ = select_base_load_blend(metricsConfig)
            clean_src = bc[0] if bc else None
        except Exception:
            clean_src = None
        if clean_src is None:
            return firmed_hybrid_re_procurable(metricsConfig)
            
    return False


def _re_zones(firmed_hybrid_base=False):
    """Zones that may size solar/wind.

    Base is included only when the input explicitly enables a feasible firmed
    hybrid base. Bottom-up order lets carry-forward flow base -> midBase ->
    midMerit -> peak in that case.
    """
    return (('base',) + _DISTRIBUTABLE_ZONES) if firmed_hybrid_base else _DISTRIBUTABLE_ZONES


def zone_owner_by_coincidence(arrays, edges, plan):
    """Decide each zone's owner from real coincidence with the band's hours.

    Ownership is earned, not assumed: the owner is the distributable source
    with the highest mean output during the hours the band is actually active.
    This is why solar owns a daytime mid-merit and wind owns a night-heavy
    mid-base HERE — but it flips automatically at a site where the band's
    hours or the resource shape differ. The measured coincidence (and each
    source's CUF) is returned so the report can show *why*, and flag when a
    lead leans on banking rather than a deck-qualifying standalone CUF.
    """
    demand = arrays['demand']
    profiles = {'solar': arrays['solar_p'], 'wind': arrays['wind_p']}
    bands = band_demand(demand, edges)
    owners = {}
    for zone, sources in plan.items():
        active = bands[zone] > 0
        n_active = int(active.sum())
        # coincidence is only meaningful for the profile-based (variable) sources
        var_sources = [s for s in sources if s in profiles]
        if n_active > 0 and var_sources:
            coincidence = {s: float(profiles[s][active].mean()) for s in var_sources}
        else:
            coincidence = {s: 0.0 for s in var_sources}
        owners[zone] = {
            'coincidence_pct': {s: round(v * 100, 2) for s, v in coincidence.items()},
            'cuf_pct': {s: round(float(profiles[s].mean()) * 100, 2) for s in var_sources},
        }
    return owners


_RE_PRICE_KEYS = {'solar': 'solarPPA', 'wind': 'windPPA'}


def _re_priced(metricsConfig):
    """Variable-RE sources that actually carry a PPA price in the request.

    A null / zero / unparseable price means the source cannot be bought, so it
    must not become a sizing knob -- and its price must never reach the per-zone
    cost arithmetic, where ``MWh * None`` raises TypeError.
    """
    mc = metricsConfig or {}
    out = set()
    for source, key in _RE_PRICE_KEYS.items():
        try:
            if float(mc.get(key) or 0) > 0:
                out.add(source)
        except (TypeError, ValueError):
            logging.warning("%s=%r is not a number; treating %s as unavailable.",
                            key, mc.get(key), source)
    return out


def zone_sizing_plan(payload, metricsConfig=None):
    """Which sources each zone may size, hybrid-aware and firm-inclusive.

    A source (solar / wind) is sizeable in a zone when EITHER the bare source
    is fit/conditional there OR a hybrid containing it is fit/conditional
    there. This is why mid-base is ``['solar', 'wind']`` (the hybrid owner),
    not bare ``['wind']`` — bare wind is only conditional in mid-base, but the
    solar-wind hybrid is the natural fit. Base is firm-only; peak is
    battery + market.

    Returns e.g. ``{'midBase': ['solar','wind','hydro','biomass','naturalGas'],
    'midMerit': ['solar','wind','hydro','naturalGas'], 'peak': ['hydro']}``.
    """
    matrix = payload.get('qualificationMatrix', {})
    hybrids = payload.get('hybridConstituents') or DEFAULT_HYBRID_CONSTITUENTS

    def qualifies(source, zone):
        if matrix.get(source, {}).get(zone) in _QUALIFYING:
            return True
        for hyb, parts in hybrids.items():
            if source in parts and matrix.get(hyb, {}).get(zone) in _QUALIFYING:
                return True
        return False

    # Only VARIABLE RE is a capacity (sizing) decision. Firm/dispatchable
    # sources are not sized -- they compete on marginal cost at dispatch time
    # (see zone_firm_options / the firm-merit step in dispatch_zone_system).
    # Base joins the sizeable zones only when the firmed-hybrid base is enabled.
    # Location whitelist: only RE procurable at the site is sizeable.
    feasible = parse_feasible_sources(metricsConfig)
    # A source with no PPA price is not procurable, whatever the whitelist says.
    # Sizing it would hand the optimiser free capacity, so drop it here -- the
    # same convention select_base_load_source uses for the firm candidates.
    priced = _re_priced(metricsConfig)
    plan = {}
    for zone in _re_zones(firmed_hybrid_base_enabled(metricsConfig)):
        allowed = [s for s in DISTRIBUTABLE_SOURCES
                   if qualifies(s, zone) and s in priced
                   and source_available(s, feasible)]
        if allowed:
            plan[zone] = allowed
    return plan


def hybrid_only_requirements(payload):
    """Sources that qualify a zone ONLY through a hybrid, and their partners.

    Per the deck, bare solar is a MISFIT in mid-base -- it only belongs there
    as the solar-wind hybrid. So solar's sole qualification path into mid-base
    is via ``hybridSolarWind``; it therefore requires its partner (wind) to be
    present. A source that qualifies on its own (solar in mid-merit -- the
    star; wind in mid-base -- conditional) carries no such requirement.

    Returns e.g. ``{'midBase': {'solar': ['wind']}}``.
    """
    matrix = payload.get('qualificationMatrix', {})
    hybrids = payload.get('hybridConstituents') or DEFAULT_HYBRID_CONSTITUENTS
    reqs = {}
    # base is included: if solar is ever sized there (firmed-hybrid base), it
    # only qualifies via the hybrid, so it needs wind — same rule as mid-base
    for zone in ('base',) + _DISTRIBUTABLE_ZONES:
        for s in VARIABLE_SOURCES:
            if matrix.get(s, {}).get(zone) in _QUALIFYING:
                continue                       # qualifies on its own -> no pairing
            partners = set()
            for hyb, parts in hybrids.items():
                if s in parts and matrix.get(hyb, {}).get(zone) in _QUALIFYING:
                    partners.update(p for p in parts if p != s and p in VARIABLE_SOURCES)
            if partners:
                reqs.setdefault(zone, {})[s] = sorted(partners)
    return reqs


def enforce_hybrid_pairing(sizes_by_zone, payload):
    """Zero any hybrid-only source that has no partner sized in its zone.

    Enforces the deck rule that bare solar cannot serve mid-base -- it must be
    part of a genuine hybrid. Mutates and returns ``sizes_by_zone``.
    """
    for zone, reqs in hybrid_only_requirements(payload).items():
        z = sizes_by_zone.get(zone)
        if not z:
            continue
        for src, partners in reqs.items():
            if (z.get(src, 0) or 0) > 0 and not any((z.get(p, 0) or 0) > 0 for p in partners):
                z[src] = 0.0
    return sizes_by_zone


def zone_firm_options(payload, metricsConfig):
    """Dispatchable firm sources each zone may draw on, per the matrix.

    These are NOT sized -- they are per-hour dispatch options with a marginal
    $/MWh (from ``metricsConfig``), competing with the market to fill a zone's
    residual. Only sources with a supplied cost are eligible. Returns e.g.
    ``{'midBase': ['hydro','biomass','naturalGas'], 'midMerit': [...],
    'peak': ['hydro']}``.
    """
    matrix = payload.get('qualificationMatrix', {})
    feasible = parse_feasible_sources(metricsConfig)
    available = [s for s in FIRM_SOURCES
                 if (metricsConfig or {}).get(FIRM_COST_KEYS[s]) not in (None, 0)
                 and source_available(s, feasible)]
    # under greenBaseLoad, carbon-disqualified fossils are dropped everywhere,
    # not just from the base source
    if (metricsConfig or {}).get('greenBaseLoad'):
        disq = set(payload.get('carbonDisqualifiedWhenGreenBaseLoad', []))
        available = [s for s in available if s not in disq]
    # base is a firm-merit backstop zone only under the firmed-hybrid base
    # (its residual after RE + battery is firmed by the qualified firm source)
    zones = _re_zones(firmed_hybrid_base_enabled(metricsConfig))
    options = {}
    for zone in zones:
        firms = [s for s in available if matrix.get(s, {}).get(zone) in _QUALIFYING]
        if firms:
            options[zone] = firms
            
    if 'base' in zones:
        try:
            from utils.layer_qualification import select_base_load_source
            fn, _ = select_base_load_source(metricsConfig)
        except Exception:
            fn, _ = cheap_source(metricsConfig)
        if fn and fn in available:
            if fn not in options.setdefault('base', []):
                options['base'].append(fn)
                
    return options


def _parse_zone(value, demand):
    if str(value).startswith('p'):
        return float(np.quantile(demand, float(str(value)[1:]) / 100.0))
    return float(value)


def resolve_edges(demand, layer_zones):
    """Upper MW edge of each band from the demand-quantile layer zones."""
    b_base = _parse_zone(layer_zones.get('base', 'p10'), demand)
    b_mid_base = _parse_zone(layer_zones.get('midBase', 'p40'), demand)
    b_mid_merit = _parse_zone(layer_zones.get('midMerit', 'p90'), demand)
    b_base = max(0.0, b_base)
    b_mid_base = max(b_mid_base, b_base)
    b_mid_merit = max(b_mid_merit, b_mid_base)
    return b_base, b_mid_base, b_mid_merit


def band_demand(demand, edges):
    """Split each hour's demand into the four band slices (they sum to demand)."""
    b_base, b_mb, b_mm = edges
    base = np.minimum(demand, b_base)
    mid_base = np.clip(demand - b_base, 0.0, b_mb - b_base)
    mid_merit = np.clip(demand - b_mb, 0.0, b_mm - b_mb)
    peak = np.clip(demand - b_mm, 0.0, None)
    return {'base': base, 'midBase': mid_base, 'midMerit': mid_merit, 'peak': peak}


# Hours in which the shared battery is allowed to DISCHARGE.
#
# grid_costs.grid_peak is a TARIFF clock (a ToD window). Where that window is a
# real tariff -- India -- it is also the right dispatch signal. Where it is not,
# gating discharge on it locks the battery out of the site's actual economic peak.
# Measured at AU-NSW: the peak band is a 15:00-20:00 ramp while the clock is
# [5-9, 17-22], so 21.6% of peak-band energy could never be served from storage
# and fell through to firm (hydro took 78% of the band). Meanwhile 05:00-09:00
# opened the gate when the peak band was empty, draining SOC before the real ramp.
#
# 'band'  -- clock hours UNION the hours the peak band is genuinely live (default)
# 'clock' -- tariff window only (previous behaviour; India ships this)
# Override per request with customInputs.batteryDischargeWindow.
BATTERY_DISCHARGE_WINDOW = 'band'


def battery_discharge_window(metricsConfig, is_grid_peak, peak_band):
    """Discharge-permitted mask. CHARGING still follows is_market_peak (the price
    signal) either way -- this only decides when discharge is allowed."""
    mode = str(((metricsConfig or {}).get('customInputs') or {}).get(
        'batteryDischargeWindow') or BATTERY_DISCHARGE_WINDOW).lower()
    if mode == 'clock':
        return is_grid_peak
    live = (np.asarray(peak_band, dtype=float) > 0).astype(float)
    return np.maximum(np.asarray(is_grid_peak, dtype=float), live)


@jit(nopython=True)
def _shared_battery_loop(solar_excess, wind_excess, served_deficit, is_grid_peak,
                         market_price, q, is_market_peak,
                         Pmax, battery_size, charge_eff, discharge_eff,
                         min_SOC, max_SOC, initial_SOC, allow_market_charge=True):
    """One SOC trajectory for the single shared BESS.

    ``served_deficit`` is the deficit the battery is *allowed* to serve
    (peak + mid-merit + mid-base, never base). The BESS is SINGLE-DIRECTION
    each hour: DISCHARGE takes priority -- in a grid peak/shoulder hour with a
    deficit it discharges and does NOT charge (so a single asset is never
    charging and discharging in the same hour); in every other hour it charges
    from surplus solar, then surplus wind, then cheap off-peak market.
    """
    n = len(served_deficit)
    discharge = np.zeros(n)
    charge = np.zeros(n)
    soc_out = np.zeros(n)
    solar_charge_arr = np.zeros(n)
    wind_charge_arr = np.zeros(n)
    solar_charge_tot = 0.0
    wind_charge_tot = 0.0
    market_charge_tot = 0.0
    market_charge_cost = 0.0
    SOC = initial_SOC
    for i in range(n):
        s_charge = 0.0
        w_charge = 0.0
        m_charge = 0.0
        d = 0.0
        if battery_size > 0:
            want_discharge = (is_grid_peak[i] == 1) and (served_deficit[i] > 0)
            if want_discharge:
                # DISCHARGE priority: serve top-of-stack deficit, no charge this hour
                avail = SOC * battery_size - min_SOC * battery_size
                d = served_deficit[i]
                if d > Pmax:
                    d = Pmax
                if d > avail:
                    d = avail
                if d < 0:
                    d = 0.0
                SOC -= d / (battery_size * discharge_eff)
                if SOC < min_SOC:
                    SOC = min_SOC
            else:
                # otherwise CHARGE: surplus solar, then wind, then cheap market
                if solar_excess[i] > 0:
                    s_charge = min(solar_excess[i], Pmax, battery_size * (max_SOC - SOC))
                if wind_excess[i] > 0:
                    w_charge = min(wind_excess[i], Pmax - s_charge,
                                   battery_size * (max_SOC - SOC) - s_charge)
                if allow_market_charge and (market_price[i] <= q[i]) and (is_market_peak[i] == 0):
                    m_charge = min(Pmax - (s_charge + w_charge),
                                   battery_size * (max_SOC - SOC) - (s_charge + w_charge))
                c = s_charge + w_charge + m_charge
                SOC += (c * charge_eff) / battery_size
                if SOC > max_SOC:
                    SOC = max_SOC
            discharge[i] = d
            charge[i] = s_charge + w_charge + m_charge
            solar_charge_arr[i] = s_charge
            wind_charge_arr[i] = w_charge
            solar_charge_tot += s_charge
            wind_charge_tot += w_charge
            market_charge_tot += m_charge
            market_charge_cost += m_charge * market_price[i]   # pay for grid charging
        soc_out[i] = SOC
    return (discharge, charge, soc_out, solar_charge_tot, wind_charge_tot,
            market_charge_tot, solar_charge_arr, wind_charge_arr, market_charge_cost)


def _allocate_discharge(discharge, zone_deficit, zones=BATTERY_SERVED_ZONES):
    """Split the single battery's hourly discharge across zones by priority.

    Peak is served first, then mid-merit, then mid-base (then base, when the
    firmed-hybrid base is active) — an accounting split of one asset. Returns
    per-zone battery-served arrays that sum, hour by hour, back to ``discharge``.
    """
    remaining = discharge.copy()
    out = {}
    for zone in zones:
        take = np.minimum(remaining, zone_deficit[zone])
        take = np.clip(take, 0.0, None)
        out[zone] = take
        remaining = remaining - take
    return out


def _split_upper_firm(zone, options, fcost, fsrc, upper_clean_frac, tier_fn):
    """Split non-base firm residual between clean and fossil firm if requested."""
    if upper_clean_frac is None or zone == 'base':
        return [(fsrc, fcost, 1.0)]
    clean = [(c, s) for (c, s) in options if not _is_fossil(s)]
    fossil = [(c, s) for (c, s) in options if _is_fossil(s)]
    if not clean or not fossil:
        return [(fsrc, fcost, 1.0)]
    key = lambda t: (tier_fn(t[1], zone), t[0])
    cc, cs = min(clean, key=key)
    fc, fs = min(fossil, key=key)
    frac = float(min(1.0, max(0.0, upper_clean_frac)))
    return [(cs, cc, frac), (fs, fc, 1.0 - frac)]


def dispatch_zone_system(sizes_by_zone, battery_size, arrays, bess, firm,
                         layer_zones, unit_costs, green_base_load=False,
                         psp_size=0.0, firm_cap=None, cfe_target=0.0,
                         green_available=False, base_clean_frac=None,
                         upper_clean_frac=None):
    """Run the independent-zone + shared-battery dispatch.

    Parameters
    ----------
    sizes_by_zone : {zone: {'solar': mw, 'wind': mw}}
        Independently sized solar/wind per zone (same profiles, different MW).
    battery_size : float   -- the ONE shared BESS energy capacity (MWh).
    arrays : dict of numpy arrays -- solar_p, wind_p, demand, market_price,
             q, is_market_peak, hour.
    bess : dict -- battery config (Pmax etc. resolved by the caller).
    firm : float -- flat firm base level serving the base band.
    unit_costs : dict -- solarPPA, windPPA, batteryFixed, firmCost.

    Returns a dict with per-zone energy/cost/cfe, the aggregate metric
    tuple, and the single-battery bookkeeping.
    """
    solar_p = arrays['solar_p']
    wind_p = arrays['wind_p']
    demand = arrays['demand']
    market_price = arrays['market_price']
    q = arrays['q']
    is_market_peak = arrays['is_market_peak']
    n = len(demand)
    total_demand = float(demand.sum())

    # deck rule: a hybrid-only source (bare solar in mid-base) may not serve a
    # zone without its hybrid partner -- zero it if the pairing is missing
    payload = unit_costs.get('payload')
    if payload is not None:
        sizes_by_zone = enforce_hybrid_pairing(
            {z: dict(s) for z, s in sizes_by_zone.items()}, payload)

    edges = resolve_edges(demand, layer_zones)
    bands = band_demand(demand, edges)
    # Discharge window: the tariff clock, widened to the hours the peak band is
    # actually live (see battery_discharge_window). Charging is unaffected.
    _batt_window = battery_discharge_window(
        (unit_costs or {}).get('metricsConfig'),
        arrays['is_grid_peak'], bands.get('peak'))

    # ---- per-zone independent generation (no cross-zone spill) -----------
    zone_used = {z: {} for z in ECONOMIC_ZONES}      # energy served, by source
    zone_deficit = {}
    zone_gen = {z: {} for z in ECONOMIC_ZONES}       # gross generation, by source
    surplus_solar = np.zeros(n)
    surplus_wind = np.zeros(n)
    firm_costs = unit_costs.get('firmCosts', {})
    firm_name_base = unit_costs.get('firmName')
    firm_cost_base = unit_costs.get('firmCost')
    firm_options = unit_costs.get('firmOptions', {})   # {zone: [qualified firm sources]}
    firmed_hybrid_base = bool(unit_costs.get('firmedHybridBase'))
    # firm sources are contracted PPAs (take-or-pay) by default, not merchant
    # tolls: they deliver their block at the contract price irrespective of spot
    take_or_pay = bool(unit_costs.get('firmTakeOrPay', True))
    # no-open-market scenario: the spot market is removed entirely. Firm must
    # cover the whole residual (force take-or-pay, ignore any firm-capacity cap),
    # and the battery/PSP may charge only from RE surplus (handled at the loop).
    no_market = bool(unit_costs.get('noOpenMarket'))
    if no_market:
        take_or_pay = True
    # residual backstop price: in the no-open-market world the GRID serves the
    # residual (guaranteed supply, ₹grid_cost) — cost ignored, it just has to be
    # met. Otherwise the open market (spot MCP) prices the residual.
    _grid_price = arrays.get('grid_price')
    residual_price = (_grid_price if (no_market and _grid_price is not None) else market_price)
    grid_cost_fossil_val = _grid_price
    market_price_delivered = market_price

    def _external_dispatch(residual, zone=None):
        """Split residual demand between the open (DAM) market and the fossil grid
        by hour, taking the CHEAPER of the two each hour. The market side uses the
        DELIVERED price (MCP + T&D) so the pick is like-for-like with the fossil
        grid tariff, which already bundles T&D. No grid tariff -> all market."""
        residual = np.clip(residual, 0.0, None)
        if zone == 'base' or no_market:
            if grid_cost_fossil_val is None:
                return residual, np.zeros(n)   # no grid tariff -> market
            return np.zeros(n), residual        # grid tariff exists -> 100% grid
        if grid_cost_fossil_val is None:
            return residual, np.zeros(n)
        # fossil grid wins only when strictly cheaper than the delivered market;
        # ties (and everything else) go to the market.
        fossil_wins = grid_cost_fossil_val < market_price_delivered
        fossil = np.where(fossil_wins, residual, 0.0)
        return residual - fossil, fossil

    # zones that size RE (base included under the firmed-hybrid flag) and zones
    # the shared battery firms (base last — bottom of the stack)
    re_zones = _re_zones(firmed_hybrid_base)
    battery_served = tuple(BATTERY_SERVED_ZONES) + (('base',) if firmed_hybrid_base else ())

    base_clean_src = unit_costs.get('baseCleanSource')
    base_clean_cost = unit_costs.get('baseCleanCost')
    base_fossil_src = unit_costs.get('baseFossilSource')
    base_fossil_cost = unit_costs.get('baseFossilCost')
    base_blend_ok = (base_clean_frac is not None and not firmed_hybrid_base
                     and base_clean_src and base_fossil_src)
    if base_blend_ok:
        frac = float(min(1.0, max(0.0, base_clean_frac)))
        base_components = [
            (base_clean_src, float(base_clean_cost), frac),
            (base_fossil_src, float(base_fossil_cost), 1.0 - frac),
        ]
    elif firm_name_base:
        base_components = [(firm_name_base, float(firm_cost_base or 0.0), 1.0)]
    else:
        base_components = []
    base_component_frac = {}
    for src, _cost, frac in base_components:
        base_component_frac[src] = base_component_frac.get(src, 0.0) + frac

    if firmed_hybrid_base or not base_components:
        # No flat take-or-pay firm floor. For a firmed hybrid, base may be served
        # by its own RE + shared battery; with no firm source at all, the base
        # band falls through to the hourly grid/market residual backstop.
        base_used = np.zeros(n)
        zone_deficit['base'] = bands['base'].copy()
    else:
        # base: the firm floor at the p10 level, take-or-pay. With a CFE floor,
        # it may be a clean/fossil blend so the target can be met without forcing
        # every other zone to overbuild renewables.
        base_used = np.minimum(bands['base'], firm)
        for src, _cost, frac in base_components:
            if frac <= 0:
                continue
            zone_used['base'][src] = zone_used['base'].get(src, np.zeros(n)) + frac * base_used
        zone_deficit['base'] = bands['base'] - base_used

    # Variable RE is SIZED per zone (capacity decision); firm sources are
    # dispatchable and handled on marginal cost after the battery (below).
    #
    # Bottom-up, qualification-gated CARRY-FORWARD: a lower zone's RE excess is
    # carried up and serves a higher zone's band FIRST (it is already generated
    # and paid for), but only for a source the matrix qualifies in that zone --
    # so solar/wind can flow mid-base -> mid-merit, but never into peak or base
    # (RE misfits). zone_gen holds each zone's OWN generation (what it pays PPA
    # on); zone_used holds what actually served the zone (own + carried).
    var_qual = unit_costs.get('zoneVarQualified', {})
    carried = {s: np.zeros(n) for s in VARIABLE_SOURCES}
    # MWh each zone spilled into the carry pool -- the causal record of who
    # over-contracted, used below to bill RE that ends up serving nobody.
    zone_re_spill = {z: {s: 0.0 for s in VARIABLE_SOURCES} for z in ECONOMIC_ZONES}
    for zone in re_zones:
        band = bands[zone]
        remaining = band.copy()
        sizes = sizes_by_zone.get(zone, {})
        qualified = var_qual.get(zone, set())
        # 1) carried excess from lower zones (free energy), only where qualified
        for s in VARIABLE_SOURCES:
            if s not in qualified:
                continue
            take = np.minimum(carried[s], remaining)
            remaining = remaining - take
            carried[s] = carried[s] - take
            zone_used[zone][s] = zone_used[zone].get(s, np.zeros(n)) + take
        # 2) this zone's own sized RE; its excess joins the carry-forward pool
        order = ('solar', 'wind') if zone == 'midMerit' else ('wind', 'solar')
        for source in order:
            mw = sizes.get(source, 0.0) or 0.0
            if mw <= 0:
                continue
            profile = solar_p if source == 'solar' else wind_p
            gen = profile * mw
            used = np.minimum(gen, remaining)
            remaining = remaining - used
            zone_gen[zone][source] = zone_gen[zone].get(source, np.zeros(n)) + gen
            zone_used[zone][source] = zone_used[zone].get(source, np.zeros(n)) + used
            carried[source] = carried[source] + (gen - used)
            zone_re_spill[zone][source] += float((gen - used).sum())
        zone_deficit[zone] = remaining

    # whatever RE remains after the cascade is the true surplus -> battery/export
    surplus_solar = carried['solar']
    surplus_wind = carried['wind']

    # ---- single shared battery ------------------------------------------
    served_deficit = sum(zone_deficit[z] for z in battery_served)
    Pmax = bess['Pmax']
    if battery_size and battery_size > 0:
        (discharge, charge, soc, solar_charge, wind_charge, market_charge,
         solar_charge_arr, wind_charge_arr, battery_market_charge_cost) = _shared_battery_loop(
            surplus_solar, surplus_wind, served_deficit, _batt_window,
            market_price, q, is_market_peak,
            Pmax, float(battery_size),
            bess['charge_eff'], bess['discharge_eff'],
            bess['min_SOC'], bess['max_SOC'], bess['initial_SOC'],
            not no_market,
        )
    else:
        discharge = np.zeros(n)
        charge = np.zeros(n)
        soc = np.zeros(n)
        solar_charge_arr = np.zeros(n)
        wind_charge_arr = np.zeros(n)
        solar_charge = wind_charge = market_charge = 0.0
        battery_market_charge_cost = 0.0

    battery_by_zone = _allocate_discharge(discharge, zone_deficit, battery_served)
    for zone in battery_served:
        zone_used[zone]['battery'] = battery_by_zone[zone]
        zone_deficit[zone] = zone_deficit[zone] - battery_by_zone[zone]

    # ---- optional second shared storage: pumped storage (PSP) -----------
    # Same single-shared-asset dispatch as the battery, one tier behind it:
    # charges from the RE surplus the battery didn't take, discharges the
    # top-of-stack deficit the battery didn't cover. No-op unless a pspConfig
    # (via unit_costs['pspBase']) and a psp_size are supplied.
    psp_base = unit_costs.get('pspBase')
    psp_fixed = 0.0
    psp_discharge = np.zeros(n)
    psp_soc = np.zeros(n)
    psp_solar_charge = psp_wind_charge = 0.0
    psp_market_charge_cost = 0.0
    psp_by_zone = {z: np.zeros(n) for z in ECONOMIC_ZONES}
    if psp_base and psp_size and psp_size > 0:
        psp_backup = psp_base['backup_hours']
        Pmax_psp = psp_size / psp_backup
        psp_fixed = psp_base.get('cost', 0.0) * Pmax_psp
        psp_surplus_solar = np.clip(surplus_solar - solar_charge_arr, 0.0, None)
        psp_surplus_wind = np.clip(surplus_wind - wind_charge_arr, 0.0, None)
        served_deficit_psp = sum(zone_deficit[z] for z in battery_served)
        (psp_discharge, _psp_charge, psp_soc, psp_solar_charge, psp_wind_charge,
         _psp_mc, psp_scharge_arr, psp_wcharge_arr, psp_market_charge_cost) = _shared_battery_loop(
            psp_surplus_solar, psp_surplus_wind, served_deficit_psp, _batt_window,
            market_price, q, is_market_peak, Pmax_psp, float(psp_size),
            psp_base['charge_eff'], psp_base['discharge_eff'],
            psp_base['min_SOC'], psp_base['max_SOC'], psp_base['initial_SOC'],
            not no_market,
        )
        psp_by_zone = _allocate_discharge(psp_discharge, zone_deficit, battery_served)
        for zone in battery_served:
            zone_used[zone]['pumpedStorage'] = psp_by_zone[zone]
            zone_deficit[zone] = zone_deficit[zone] - psp_by_zone[zone]
        # PSP charging also soaks RE that would otherwise export (hourly arrays);
        # battery/PSP charge scalars stay separate for per-asset CFE crediting
        solar_charge_arr = solar_charge_arr + psp_scharge_arr
        wind_charge_arr = wind_charge_arr + psp_wcharge_arr
    psp_discharge_tot = float(psp_discharge.sum())
    psp_rte = (psp_base or {}).get('rte', 0.8)
    # grid-charged portion of PSP discharge (clamped >= 0: clean discharge is
    # capped at the RE that charged it, so this can't go negative)
    psp_market_discharge = max(0.0, psp_discharge_tot - (psp_solar_charge + psp_wind_charge) * psp_rte)

    # ---- India energy banking: bank surplus RE, withdraw to cover deficit ------
    # Sits between storage and firm/market/grid: a free (config-charged), slot-
    # segregated, monthly time-shift of the customer's OWN surplus RE. Enabled
    # only for banking states (STATE_BANKING_MODES == 1).
    banking_mode = int(unit_costs.get('bankingMode', 0) or 0)
    bank_discharge_tot = 0.0
    bank_recovery_cost = 0.0
    bank_deposit_h = np.zeros(n)
    if banking_mode == 1:
        surplus_h = (np.clip(surplus_solar - solar_charge_arr, 0.0, None)
                     + np.clip(surplus_wind - wind_charge_arr, 0.0, None))
        served_deficit_bank = sum(zone_deficit[z] for z in battery_served)
        if str(unit_costs.get('state') or '') == 'IN-MH':      # 3-slot scheme
            hr = arrays['hour']
            slot_arr = np.where(hr < 9, 0, np.where(hr < 17, 1, 2)).astype(np.int64)
        else:                                                  # 2-slot: peak vs off-peak
            slot_arr = (arrays['banking_peak'] * 2).astype(np.int64)
        bank_deposit_h, bank_withdraw_h, lapsed = _banking_loop(
            surplus_h, served_deficit_bank, arrays['month'], slot_arr,
            unit_costs.get('bankingChargePct', 0.0))
        bank_by_zone = _allocate_discharge(bank_withdraw_h, zone_deficit, battery_served)
        for zone in battery_served:
            zone_used[zone]['bank'] = bank_by_zone[zone]
            zone_deficit[zone] = zone_deficit[zone] - bank_by_zone[zone]
        bank_discharge_tot = float(bank_withdraw_h.sum())
        bank_recovery_cost = float(lapsed) * float(unit_costs.get('recoveryBankCost', 0.0))

    # ---- residual per zone -> firm dispatch (merit) then market ---------
    # Dispatchable firm sources compete to serve a zone's residual, but the
    # QUALIFICATION TIER leads the marginal cost: a source the matrix rates a
    # natural fit (fit) for the zone beats a merely conditional one even if the
    # conditional source is cheaper. This is why cheap gas ($70, only conditional
    # in base) does NOT displace fit hydro ($80) from the base floor — gas is a
    # mid-layer fit, not a base source. Within the same tier the cheapest $/MWh
    # wins, and clean beats fossil on a price tie. A firm source is a CONTRACTED
    # PPA: you pay a capacity charge ($/MW-yr, after this loop) on the MW you
    # contract, PLUS the energy at the CONTRACT price ($/MWh) on what it delivers.
    # Under take-or-pay (default) it delivers its block regardless of the spot
    # price — the DAM price only prices the leftover import / surplus export.
    matrix = (payload or {}).get('qualificationMatrix', {})
    firm_capacity_costs = unit_costs.get('firmCapacityCosts', {})

    def _firm_tier(source, zone):
        level = matrix.get(source, {}).get(zone)
        return 0 if level == 'fit' else (1 if level == 'conditional' else 2)

    total_market_cost = 0.0
    market_energy_total = 0.0
    total_grid_cost = 0.0        # India grid backstop (guaranteed supply)
    grid_energy_total = 0.0
    firm_dispatch_cost = 0.0
    firm_source_dispatch_energy = {}   # source -> MWh served across zones
    firm_source_dispatch_cost = {}
    zone_firm_dispatch_cost = {z: 0.0 for z in ECONOMIC_ZONES}
    # same figures split by SOURCE, so a zone can report what each firm cost it
    zone_firm_source_cost = {z: {} for z in ECONOMIC_ZONES}
    firm_source_hourly = {}            # source -> hourly MW dispatched (ALL zones)
    firm_source_hourly_upper = {}      # source -> hourly MW dispatched (upper zones only, for capacity cost)
    firm_zone_source_energy = {}       # source -> {zone: MWh} (for capacity split)
    # Seed base-zone firm hourly so peak MW covers all zones.
    if not firmed_hybrid_base:
        for _src, _cost, _frac in base_components:
            if _frac <= 0:
                continue
            firm_source_hourly[_src] = firm_source_hourly.get(_src, np.zeros(n)) + _frac * base_used
    # optional CONTRACTED firm-capacity caps (MW), sized independently PER LAYER:
    # {'midBase': mw, 'midMerit': mw}. A capped band's take-or-pay firm may serve
    # only up to its OWN cap each hour; the rest of that band's residual spills to
    # the spot market (kept in check by the market-cap penalty). Base floor and
    # peak stay fully firm unless a cap is supplied for them.
    if isinstance(firm_cap, dict):
        firm_caps = firm_cap
    elif firm_cap is None:
        firm_caps = {}
    else:                                    # legacy scalar -> apply to both mids
        firm_caps = {'midBase': firm_cap, 'midMerit': firm_cap}
    if no_market:
        firm_caps = {}                       # firm covers everything; no spill to spot
    _firm_fill_order = ('base', 'midBase', 'midMerit', 'peak')
    for zone in _firm_fill_order:
        residual = np.clip(zone_deficit[zone], 0.0, None)
        # qualified firm sources for this zone
        options = [(firm_costs[s], s) for s in firm_options.get(zone, [])
                   if s in firm_costs]
        if options:
            # base uses the SELECTED (priority-preferred, e.g. geothermal) base
            # source when it is an option there — so the firm floor is the source
            # the deck/priority chose, not just the cheapest. Other zones dispatch
            # on merit: fit-tier leads, then cheapest $/MWh, clean beats fossil on a tie.
            base_pick = [(c, s) for (c, s) in options if s == firm_name_base]
            if zone == 'base' and base_pick:
                fcost, fsrc = base_pick[0]
            else:
                fcost, fsrc = min(options, key=lambda t: (_firm_tier(t[1], zone), t[0], _is_fossil(t[1])))
            # TAKE-OR-PAY firm PPA (default): you contract the block at a fixed
            # price, so it is dispatched to cover the residual REGARDLESS of the
            # spot price — a PPA is a hedge against the market, not a bet on it.
            # The DAM price only prices the leftover you import / surplus you
            # export, never whether your contracted firm runs. (Set a source to
            # toll mode to restore the old "run only when <= DAM price" merit.)
            if take_or_pay or (firmed_hybrid_base and zone == 'base'):
                firm_e = residual
            else:
                firm_e = np.where(fcost <= market_price, residual, 0.0)
            # per-layer contracted-capacity cap: this band's firm limited to its
            # own cap MW/hr; the rest of the band's residual spills to spot
            cap_z = firm_caps.get(zone)
            if cap_z is not None:
                firm_e = np.minimum(firm_e, float(cap_z))
            mkt = residual - firm_e
            for psrc, pcost, pfrac in _split_upper_firm(
                    zone, options, fcost, fsrc, upper_clean_frac, _firm_tier):
                if pfrac <= 0:
                    continue
                part_e = pfrac * firm_e
                zone_used[zone][psrc] = zone_used[zone].get(psrc, np.zeros(n)) + part_e
                e = float(part_e.sum())
                firm_source_dispatch_energy[psrc] = firm_source_dispatch_energy.get(psrc, 0.0) + e
                firm_source_dispatch_cost[psrc] = firm_source_dispatch_cost.get(psrc, 0.0) + e * pcost
                firm_dispatch_cost += e * pcost
                zone_firm_dispatch_cost[zone] += e * pcost
                zone_firm_source_cost[zone][psrc] = (
                    zone_firm_source_cost[zone].get(psrc, 0.0) + e * pcost)
                # Track ALL-zone hourly for peak MW reporting
                firm_source_hourly[psrc] = firm_source_hourly.get(psrc, np.zeros(n)) + part_e
                # capacity is contracted for UPPER-zone dispatch (the base floor's
                # flat run already amortises its capacity via $/MWh x hours)
                if zone != 'base':
                    firm_source_hourly_upper[psrc] = firm_source_hourly_upper.get(psrc, np.zeros(n)) + part_e
                    firm_zone_source_energy.setdefault(psrc, {})[zone] = \
                        firm_zone_source_energy.get(psrc, {}).get(zone, 0.0) + e
        else:
            mkt = residual
        # buy the residual externally from the CHEAPER of open market vs GRID each
        # hour via _external_dispatch (base zone / noOpenMarket -> grid only).
        market_e, grid_e = _external_dispatch(mkt, zone=zone)
        zone_used[zone]['market'] = market_e
        zone_used[zone]['grid'] = grid_e
        total_market_cost += float((market_e * market_price).sum())
        if _grid_price is not None:
            total_grid_cost += float((grid_e * _grid_price).sum())
        market_energy_total += float(market_e.sum())
        grid_energy_total += float(grid_e.sum())

    # ---- firm CAPACITY charge: contract each dispatchable for its peak MW ----
    # peak MW = the most that source ever supplied simultaneously (across upper
    # zones) in any hour; you pay $/MW-yr on that whether it ran 30 h or 3,000 h.
    # Peak MW from ALL zones (for reporting)
    firm_source_peak_mw = {}
    for src, hourly in firm_source_hourly.items():
        firm_source_peak_mw[src] = float(hourly.max()) if hourly.size else 0.0

    # Capacity COST from upper zones only (base already pays take-or-pay)
    firm_capacity_cost = 0.0
    firm_source_capacity_cost = {}
    zone_firm_capacity_cost = {z: 0.0 for z in ECONOMIC_ZONES}
    for src, hourly in firm_source_hourly_upper.items():
        peak_mw = float(hourly.max()) if hourly.size else 0.0
        rate = firm_capacity_costs.get(src, 0.0)
        if rate <= 0 or peak_mw <= 0:
            continue
        cap = rate * peak_mw
        firm_source_capacity_cost[src] = cap
        firm_capacity_cost += cap
        zenergy = firm_zone_source_energy.get(src, {})
        tot_e = sum(zenergy.values())
        for z, ez in zenergy.items():
            if tot_e > 0:
                zone_firm_capacity_cost[z] += cap * (ez / tot_e)
                zone_firm_source_cost[z][src] = (
                    zone_firm_source_cost[z].get(src, 0.0) + cap * (ez / tot_e))

    # the battery/PSP also BUY energy from the market when they charge off-peak
    # — that energy must be paid for (matches the flat model), else the optimiser
    # would treat cheap-market charging as free and oversize storage
    total_market_cost += float(battery_market_charge_cost) + float(psp_market_charge_cost)

    # ---- energy roll-ups -------------------------------------------------
    def src_total(source):
        return float(sum(zone_used[z].get(source, np.zeros(n)).sum() for z in ECONOMIC_ZONES))

    solar_used_tot = src_total('solar')
    wind_used_tot = src_total('wind')
    # clean vs fossil firm energy + emissions, per source (data-driven)
    _payload = unit_costs.get('payload') or {}
    clean_firm_used_tot = sum(src_total(s) for s in CLEAN_FIRM_SOURCES)
    if firm_name_base and firm_name_base not in FIRM_SOURCES and not _is_fossil(firm_name_base):
        clean_firm_used_tot += src_total(firm_name_base)   # e.g. firmedHybrid base
    gas_used_tot = src_total('naturalGas')
    fossil_used_tot = sum(src_total(s) for s in FOSSIL_FIRM_SOURCES)
    fossil_emissions = sum(src_total(s) * emission_factor(s, _payload)
                           for s in FOSSIL_FIRM_SOURCES)
    firm_used_tot = clean_firm_used_tot + fossil_used_tot   # all firm energy served
    discharge_tot = float(discharge.sum())

    solar_gen_tot = float(sum(zone_gen[z].get('solar', np.zeros(n)).sum() for z in ECONOMIC_ZONES))
    wind_gen_tot = float(sum(zone_gen[z].get('wind', np.zeros(n)).sum() for z in ECONOMIC_ZONES))
    re_gen_tot = solar_gen_tot + wind_gen_tot
    # storage RE charge = battery + PSP (both reduce export)
    solar_charge_all = solar_charge + psp_solar_charge
    wind_charge_all = wind_charge + psp_wind_charge
    solar_exported = max(0.0, solar_gen_tot - solar_used_tot - solar_charge_all)
    wind_exported = max(0.0, wind_gen_tot - wind_used_tot - wind_charge_all)
    re_exported = solar_exported + wind_exported
    # hourly export (surplus not absorbed by the battery) valued at the DAM price
    solar_export_h = np.clip(surplus_solar - solar_charge_arr, 0.0, None)
    wind_export_h = np.clip(surplus_wind - wind_charge_arr, 0.0, None)
    # surplus banked (bank_deposit_h) is not exported — it went to the bank
    export_h = np.clip(solar_export_h + wind_export_h - bank_deposit_h, 0.0, None)
    export_revenue = (0.0 if bool(unit_costs.get('noExport'))
                      else float((export_h * market_price).sum()))

    # ---- per-zone economics (independent P&L) ---------------------------
    firm_cost = unit_costs['firmCost']
    # An absent PPA means the source is not procurable (zone_sizing_plan
    # refuses to size it), so its generation is zero and 0.0 is the honest
    # price. Without this, MWh * None raises TypeError mid-dispatch.
    solar_ppa = unit_costs['solarPPA'] or 0.0
    wind_ppa = unit_costs['windPPA'] or 0.0
    battery_fixed = unit_costs['batteryFixed']
    # base firm is take-or-pay (firm MW x cost x hours); upper-zone firm is
    # dispatched pay-per-use on marginal cost. Under the firmed-hybrid base
    # there is NO flat floor — base firm comes from the merit backstop instead.
    firm_annual_cost = (0.0 if firmed_hybrid_base
                        else sum(frac * firm * cost * n for _src, cost, frac in base_components))
    firm_total_cost = firm_annual_cost + firm_dispatch_cost + firm_capacity_cost

    # per firm-source served energy + cost (variable + capacity), base + upper zones
    firm_source_energy = {}
    firm_source_cost = {}
    base_used_tot = float(base_used.sum())
    if not firmed_hybrid_base:
        for src, cost, frac in base_components:
            if frac <= 0:
                continue
            firm_source_energy[src] = firm_source_energy.get(src, 0.0) + frac * base_used_tot
            firm_source_cost[src] = firm_source_cost.get(src, 0.0) + frac * firm * cost * n
    elif firm_name_base:
        firm_source_energy[firm_name_base] = 0.0
        firm_source_cost[firm_name_base] = 0.0
    for s, e in firm_source_dispatch_energy.items():
        firm_source_energy[s] = firm_source_energy.get(s, 0.0) + e
        firm_source_cost[s] = (firm_source_cost.get(s, 0.0)
                               + firm_source_dispatch_cost.get(s, 0.0)
                               + firm_source_capacity_cost.get(s, 0.0))

    max_demand = float(demand.max())
    zone_bounds = {
        'base': [0.0, edges[0]],
        'midBase': [edges[0], edges[1]],
        'midMerit': [edges[1], edges[2]],
        'peak': [edges[2], max_demand],
    }
    zone_reports = {}
    layer_wise_mw = {}      # per-zone installed/effective capacity by source (MW)
    # Who to bill for RE that nobody consumed: the zone that spilled it.
    # Falls back to share of generation if nothing was spilled at all.
    _re_waste_share = {}
    for _s in VARIABLE_SOURCES:
        _tot = sum(zone_re_spill[z][_s] for z in ECONOMIC_ZONES)
        if _tot > 0:
            _re_waste_share[_s] = {z: zone_re_spill[z][_s] / _tot for z in ECONOMIC_ZONES}
        else:
            _g = {z: float(zone_gen[z].get(_s, np.zeros(n)).sum()) for z in ECONOMIC_ZONES}
            _gt = sum(_g.values())
            _re_waste_share[_s] = ({z: _g[z] / _gt for z in ECONOMIC_ZONES} if _gt > 0
                                   else {z: 0.0 for z in ECONOMIC_ZONES})
    for zone in ECONOMIC_ZONES:
        used = zone_used[zone]
        energy = {src: float(arr.sum()) for src, arr in used.items()}
        e_solar = energy.get('solar', 0.0)
        e_wind = energy.get('wind', 0.0)
        e_batt = energy.get('battery', 0.0)
        e_psp = energy.get('pumpedStorage', 0.0)
        e_clean_firm = sum(energy.get(s, 0.0) for s in CLEAN_FIRM_SOURCES)
        mkt_cost = float((used.get('market', np.zeros(n)) * market_price).sum())
        grid_cost_z = float((used.get('grid', np.zeros(n)) * (_grid_price if _grid_price is not None else 0.0)).sum())
        # shared storage fixed cost apportioned to zones by discharge share
        batt_share = (e_batt / discharge_tot) if discharge_tot > 0 else 0.0
        psp_share = (e_psp / psp_discharge_tot) if psp_discharge_tot > 0 else 0.0
        # base carries its flat firm floor (non-hybrid) AND any merit firm it
        # dispatched (firmed-hybrid base) — both, so per-zone P&L sums to aggregate
        firm_component = ((firm_annual_cost if zone == 'base' else 0.0)
                          + zone_firm_dispatch_cost[zone] + zone_firm_capacity_cost[zone])
        # RE cost is charged on the zone's OWN generation (take-or-pay PPA),
        # not on carried-in energy (that was paid by the zone that generated it)
        # RE cost follows CONSUMPTION, and whatever nobody consumed falls back
        # on the zone that contracted it.
        #
        # It used to be charged on the zone's OWN generation, carried-in energy
        # being free to the receiver. That made the ratio meaningless: the cost
        # was a zone's generation, the denominator its consumption, and those
        # are different quantities the moment a zone is a net donor. A zone
        # generating 4x what it used showed ~4x its true unit cost, while the
        # zones eating its spill showed a price for energy they never paid for.
        #
        # Each zone now pays for what it CONSUMED (own + carried), its share of
        # the RE that went into shared storage (by discharge share, as the
        # storage fixed cost already was), and its share of the RE that served
        # nobody (by who spilled it). Every generated MWh is charged exactly
        # once, so the zone costs still sum to the same site total.
        own_solar_gen = float(zone_used[zone].get('solar', np.zeros(n)).sum())
        own_wind_gen = float(zone_used[zone].get('wind', np.zeros(n)).sum())
        for _s, _chg, _waste in (('solar', solar_charge_all, solar_exported),
                                 ('wind', wind_charge_all, wind_exported)):
            _add = _chg * batt_share + _waste * _re_waste_share.get(_s, {}).get(zone, 0.0)
            if _s == 'solar':
                own_solar_gen += _add
            else:
                own_wind_gen += _add
        # Same arithmetic, kept SPLIT BY SOURCE so a layer can show what each source
        # cost it rather than one blended figure. Every term below already existed --
        # they were simply summed on one line. cost stays the sum of these parts, so
        # cost_musd and blended_cost_per_mwh are bit-for-bit unchanged.
        cost_by_source = {}
        if own_solar_gen:
            cost_by_source['solar'] = own_solar_gen * solar_ppa
        if own_wind_gen:
            cost_by_source['wind'] = own_wind_gen * wind_ppa
        if mkt_cost:
            cost_by_source['market'] = mkt_cost
        if grid_cost_z:
            cost_by_source['grid'] = grid_cost_z
        if battery_fixed * batt_share:
            cost_by_source['battery'] = battery_fixed * batt_share
        if psp_fixed * psp_share:
            cost_by_source['pumpedStorage'] = psp_fixed * psp_share
        for _fs, _fc in zone_firm_source_cost.get(zone, {}).items():
            if _fc:
                cost_by_source[_fs] = cost_by_source.get(_fs, 0.0) + _fc
        # The base floor's flat take-or-pay block, split across its blend components
        # at each component's OWN contracted price (frac x firm MW x $/MWh x hours) --
        # the same arithmetic firm_annual_cost is built from. Splitting by the energy
        # fraction instead would hand every component the identical $/MWh and hide the
        # very price difference the blend exists to trade off.
        if zone == 'base' and firm_annual_cost:
            if base_components:
                for _s, _c, _fr in base_components:
                    if _fr > 0:
                        cost_by_source[_s] = (cost_by_source.get(_s, 0.0)
                                              + _fr * firm * _c * n)
            else:
                cost_by_source['firm'] = cost_by_source.get('firm', 0.0) + firm_annual_cost

        cost = (own_solar_gen * solar_ppa + own_wind_gen * wind_ppa + firm_component
                + mkt_cost + grid_cost_z + battery_fixed * batt_share + psp_fixed * psp_share)
        # never let the split silently disagree with the number that is actually used
        _split = sum(cost_by_source.values())
        if abs(_split - cost) > max(0.01, abs(cost) * 1e-9):
            logging.warning("zone %s: per-source cost split %.2f != zone cost %.2f; "
                            "reporting the blend only", zone, _split, cost)
            cost_by_source = None
        band_total = float(bands[zone].sum())
        # clean = solar + wind + battery + PSP + clean firm; gas + market not clean
        clean = e_solar + e_wind + e_batt + e_psp + e_clean_firm
        zone_reports[zone] = {
            'band_mw': [round(zone_bounds[zone][0], 3), round(zone_bounds[zone][1], 3)],
            'demand_share_pct': (band_total / total_demand * 100) if total_demand else 0.0,
            'energy_mwh': energy,
            'source_mix_pct_of_zone': {
                k: (v / band_total * 100 if band_total else 0.0) for k, v in energy.items()
            },
            'cost_musd': cost / 1e7,
            'cost_by_source_musd': ({k: v / 1e7 for k, v in cost_by_source.items()}
                                    if cost_by_source else None),
            'cost_per_mwh_by_source': ({k: (v / energy[k]) for k, v in cost_by_source.items()
                                        if energy.get(k)} if cost_by_source else None),
            'blended_cost_per_mwh': (cost / band_total) if band_total else 0.0,
            'cfe_pct': (clean / band_total * 100) if band_total else 0.0,
            'sized_mw': {k: round((sizes_by_zone.get(zone, {}).get(k) or 0.0)
                                  / (get_firm_source_cuf(k, unit_costs.get('metricsConfig'))
                                     if k in FIRM_SOURCES else 1.0), 4)
                         for k in (VARIABLE_SOURCES + FIRM_SOURCES)
                         if (sizes_by_zone.get(zone, {}).get(k) or 0) > 0},
        }
        # capacity per source in this zone, unambiguous: nameplate_mw = installed
        # capacity (only sized RE + the firm floor have one); peak_delivered_mw =
        # max MW that source actually supplied to the zone in any hour (all sources)
        zmw = {}
        for src, arr in used.items():
            peak = float(np.max(arr)) if arr.size else 0.0
            entry = {'peak_delivered_mw': round(peak, 4)}
            if src in VARIABLE_SOURCES:
                entry['nameplate_mw'] = round(float(sizes_by_zone.get(zone, {}).get(src, 0) or 0), 4)
            elif zone == 'base' and not firmed_hybrid_base and src in base_component_frac:
                _cuf = get_firm_source_cuf(src, unit_costs.get('metricsConfig'))
                _base_req = float(firm) * base_component_frac[src]
                entry['nameplate_mw'] = round(_base_req / _cuf if _cuf > 0 else _base_req, 4)
            if peak > 1e-9 or entry.get('nameplate_mw', 0) > 1e-9:
                zmw[src] = entry
        layer_wise_mw[zone] = zmw

    # ---- aggregate metric tuple (matches compute_metrics contract) ------
    total_energy_cost = (solar_gen_tot * solar_ppa + wind_gen_tot * wind_ppa
                         + firm_total_cost + battery_fixed + psp_fixed
                         + total_market_cost + total_grid_cost + bank_recovery_cost)
    total_cost = (total_energy_cost / total_demand) if total_demand > 0 else 1e12

    RE_utilization = ((re_gen_tot - re_exported) / re_gen_tot * 100) if re_gen_tot > 0 else 0.0
    rte = bess.get('rte', bess['charge_eff'] * bess['discharge_eff'])
    dod = bess.get('dod', 1.0)
    battery_utilization = (discharge_tot / (battery_size * 365 * rte * dod) * 100) \
        if battery_size and battery_size > 0 else 0.0
    # peak-demand-met measured over the ToD window the battery actually serves
    peak_window_deficit = float((served_deficit * _batt_window).sum())
    peak_demand_met = (discharge_tot / peak_window_deficit * 100) if peak_window_deficit > 0 else 100.0
    annual_dc_demand_meet = ((re_gen_tot - re_exported) / total_demand * 100) if total_demand > 0 else 0.0
    solar_meet = (solar_used_tot / total_demand * 100) if total_demand else 0.0
    wind_meet = (wind_used_tot / total_demand * 100) if total_demand else 0.0
    market_meet = (market_energy_total / total_demand * 100) if total_demand else 0.0
    base_meet = (firm_used_tot / total_demand * 100) if total_demand else 0.0
    battery_meet = (discharge_tot / total_demand * 100) if total_demand else 0.0

    re_charged = solar_charge + wind_charge                 # battery RE charge
    batt_market_discharge = max(0.0, discharge_tot - re_charged * rte)
    # both storages' grid-charged discharge counts as market draw for emissions
    market_discharge = batt_market_discharge + psp_market_discharge
    # clean energy the BUILD itself delivers: solar + wind + clean firm + battery +
    # PSP, each net of the portion charged from the grid (which is not carbon-free).
    clean_build = (solar_used_tot + wind_used_tot + clean_firm_used_tot
                   + discharge_tot - batt_market_discharge
                   + psp_discharge_tot - psp_market_discharge)
    # AUTOMATIC MINIMAL GREEN. When green procurement is available (green open
    # access / G-DAM), buy ONLY the green energy needed to reach the CFE floor --
    # zero if the build already clears it -- from the non-clean residual (market +
    # grid + the storage they charged; fossil firm is a contracted PPA, not
    # greenable). This lifts CFE toward the target WITHOUT over-building RE, and
    # pays the green premium only on what is actually greened. Green is priced at
    # the green-grid rate, displacing the dirty residual it replaces.
    greenable = max(0.0, total_demand - clean_build - fossil_used_tot)
    green_needed = (float(cfe_target) / 100.0) * total_demand - clean_build
    green_fill = min(max(0.0, green_needed), greenable) if green_available else 0.0
    green_price = float(unit_costs.get('greenGridPriceMean') or 0.0)
    avg_dirty_price = ((total_market_cost + total_grid_cost) / greenable) if greenable > 1e-9 else 0.0
    green_premium = green_fill * (green_price - avg_dirty_price)
    total_energy_cost += green_premium
    total_cost = (total_energy_cost / total_demand) if total_demand > 0 else 1e12
    # the grid's own carbon-free share counts toward 24/7 CFE: an imported
    # MWh from a ~99% clean grid is not the same as one from a coal grid.
    # Defaults to 0, so a payload without gridCfePercentage is unchanged.
    _grid_cfe = get_grid_cfe_fraction(unit_costs.get('metricsConfig'))
    cfe_numerator = clean_build + green_fill + grid_energy_total * _grid_cfe
    cfe_score = (cfe_numerator / total_demand * 100) if total_demand else 0.0

    # reliability = demand-weighted average of EVERY source's reliability. Each
    # source that actually served demand must be mapped to its reliability, or
    # its slice is silently treated as zero-reliability. Storage (BESS + PSP)
    # rides with solar as 'solarAndBess'; the firm sources each map to their own
    # key so hydro / biomass / geothermal / nuclear are counted (previously they
    # were dropped — geothermal was hardcoded 0 — which understated the score for
    # any firm-heavy stack). Normalised by the share that has a reliability score
    # so it is a true weighted average, not a partial sum.
    reliability_scores = unit_costs.get('reliabilityScores', {}) or {}
    gas_meet = (gas_used_tot / total_demand * 100) if total_demand else 0.0

    def _frac(mwh):
        return (mwh / total_demand) if total_demand else 0.0

    energy_mix = {
        'solarAndBess': _frac(solar_used_tot + discharge_tot + psp_discharge_tot),
        'wind': _frac(wind_used_tot),
        'market': _frac(market_energy_total),
        'hydro': _frac(src_total('hydro')),
        'biomass': _frac(src_total('biomass')),
        'geothermal': _frac(src_total('geothermal')),
        'nuclear': _frac(src_total('nuclear')),
        'naturalGas': _frac(gas_used_tot),
    }
    _weighted = {s: w for s, w in energy_mix.items() if s in reliability_scores}
    _wsum = sum(_weighted.values())
    reliability_score = (
        sum(w * reliability_scores[s] for s, w in _weighted.items()) / _wsum
    ) if _wsum > 0 else 0.0

    metric_tuple = (
        RE_utilization, battery_utilization, peak_demand_met, annual_dc_demand_meet,
        solar_meet, wind_meet, market_meet, reliability_score, total_cost, cfe_score,
    )

    # single-asset bookkeeping — each storage's per-zone split sums to one asset
    allocated = float(sum(battery_by_zone[z].sum() for z in battery_served))
    partition_ok = abs(discharge_tot - allocated) <= 1e-6 * max(1.0, discharge_tot)
    psp_allocated = float(sum(psp_by_zone[z].sum() for z in battery_served))
    psp_partition_ok = abs(psp_discharge_tot - psp_allocated) <= 1e-6 * max(1.0, psp_discharge_tot)
    psp_backup_hours = (psp_base or {}).get('backup_hours', 8)

    # ---- hourly simulation sheet -----------------------------------------
    # Sum each source's hourly dispatch across all zones (zone_used holds a
    # length-n array per source per zone), then add storage + demand, so the
    # zone path emits a full 8760-row sheet like the flat simulation does.
    hourly_sources = {}
    for _z in ECONOMIC_ZONES:
        for _s, _arr in zone_used[_z].items():
            if _arr is None:
                continue
            hourly_sources[_s] = hourly_sources.get(_s, np.zeros(n)) + _arr
    hourly = {
        'day': arrays.get('day', np.zeros(n)),
        'month': arrays.get('month', np.zeros(n)),
        'hour': arrays.get('hour', np.arange(n)),
        'demand': demand,
    }
    for _s in sorted(hourly_sources):
        hourly[f'{_s}_mw'] = hourly_sources[_s]
    hourly['battery_charge_mw'] = charge
    hourly['soc'] = soc
    if psp_base and psp_size and psp_size > 0:
        hourly['psp_discharge_mw'] = psp_discharge
        hourly['psp_soc'] = psp_soc

    base_blend = {
        'active': bool(base_blend_ok),
        'clean_source': base_clean_src if base_blend_ok else None,
        'fossil_source': base_fossil_src if base_blend_ok else None,
        'clean_fraction': (base_components[0][2] if base_blend_ok else None),
        'components': [
            {
                'source': src,
                'cost_per_mwh': cost,
                'fraction': frac,
                'energy_mwh': frac * base_used_tot,
            }
            for src, cost, frac in base_components if frac > 0
        ],
    }

    return {
        'metric_tuple': metric_tuple,
        'hourly': hourly,
        'edges': edges,
        'zones': zone_reports,
        'layer_wise_mw': layer_wise_mw,
        'base_blend': base_blend,
        'aggregate': {
            'solar_used': solar_used_tot, 'wind_used': wind_used_tot,
            'firm_used': firm_used_tot, 'battery_discharge': discharge_tot,
            'market_energy': market_energy_total, 'total_market_cost': total_market_cost,
            'grid_energy': grid_energy_total, 'total_grid_cost': total_grid_cost,
            'bank_discharge': bank_discharge_tot, 'bank_recovery_cost': bank_recovery_cost,
            'solar_exported': solar_exported, 'wind_exported': wind_exported,
            'solar_charge': solar_charge_all, 'wind_charge': wind_charge_all,
            'solar_charge_battery': solar_charge, 'wind_charge_battery': wind_charge,
            'market_charge': market_charge, 'total_cost_per_mwh': total_cost,
            'green_fill': green_fill, 'green_premium': green_premium,
            'greenable': greenable, 'green_available': bool(green_available),
            'export_revenue': export_revenue, 'total_energy_cost': total_energy_cost,
            'total_demand': total_demand, 're_gen': re_gen_tot,
            'solar_gen': solar_gen_tot, 'wind_gen': wind_gen_tot,
            'market_discharge': market_discharge, 'cfe_score': cfe_score,
            'firm_annual_cost': firm_annual_cost,
            # cost of charging the shared storage FROM THE GRID. It is inside
            # total_market_cost but belongs to no single band, so the reporting
            # layer attributes it by discharge share, like the storage fixed cost.
            # Reporting only -- total_market_cost already carries it (see above).
            'battery_market_charge_cost': float(battery_market_charge_cost),
            'psp_market_charge_cost': float(psp_market_charge_cost),
            'battery_fixed': battery_fixed,
            'psp_fixed': psp_fixed,
            'solar_energy_cost': solar_gen_tot * solar_ppa,
            'wind_energy_cost': wind_gen_tot * wind_ppa,
            'gas_used': gas_used_tot, 'clean_firm_used': clean_firm_used_tot,
            'fossil_emissions': fossil_emissions,
            'firm_total_cost': firm_total_cost,
            'firm_source_energy': firm_source_energy,
            'firm_source_cost': firm_source_cost,
            'firm_capacity_cost': firm_capacity_cost,
            'firm_source_capacity_cost': firm_source_capacity_cost,
            'firm_source_peak_mw': firm_source_peak_mw,
        },
        'battery': {
            'shared_single_asset': True,
            'per_zone_sizing': False,
            'storage_capacity_mwh': battery_size,
            'storage_capacity_mw': (battery_size / bess['backup_hours']) if battery_size else 0.0,
            'annual_discharge_mwh': discharge_tot,
            'zone_energy_mwh': {z: float(battery_by_zone.get(z, np.zeros(n)).sum())
                                for z in ECONOMIC_ZONES},
            'partition_conserves_energy': partition_ok,
            'note': 'One BESS shared across all zones; zone figures are an accounting '
                    'split of the same asset, not separate batteries.',
        },
        'pumped_storage': {
            'active': bool(psp_base and psp_size and psp_size > 0),
            'shared_single_asset': True,
            'storage_capacity_mwh': psp_size or 0.0,
            'storage_capacity_mw': (psp_size / psp_backup_hours) if psp_size else 0.0,
            'annual_discharge_mwh': psp_discharge_tot,
            'annual_cost_musd': psp_fixed / 1e7,
            'zone_energy_mwh': {z: float(psp_by_zone.get(z, np.zeros(n)).sum())
                                for z in ECONOMIC_ZONES},
            'partition_conserves_energy': psp_partition_ok,
            'note': 'Second shared storage, dispatched one tier behind the BESS '
                    '(charges the RE surplus it left, discharges the deficit it did not cover).',
        },
        'series': {'discharge': discharge, 'charge': charge, 'soc': soc},
    }
