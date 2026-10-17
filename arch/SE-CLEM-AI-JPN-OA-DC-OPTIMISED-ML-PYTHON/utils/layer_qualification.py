"""Right Source, Right Layer — source-to-layer qualification and allocation.

Implements the deck's framework: 14 sources qualified into Base · Mid-Base ·
Mid-Merit · Peak (+ Emergency as a fifth, non-economic layer) and the hourly
simulated dispatch mapped into those layer bands.

The layer bands are demand-quantile zones, same mechanism as the existing
`customInputs.zones` (which stays untouched for `zone_wise_percentages`):
    base      -> p10  (level served every hour)
    midBase   -> p40  (band present 60-90% of hours)
    midMerit  -> p90  (band present 10-60% of hours)
    peak      -> p100 (band present <10% of hours)

The payload carrying the deck data (sources, attributes, qualification
matrix, merit orders, emergency layer) lives locally in
data/right_source_right_layer_payload.json for now; once the backend sends
it inside configurationInputs (like the rest of the inputs) it is picked
from metricsConfig first.
"""

import json
import logging
import os

from config import config
from utils.get_cheap_source import cheap_source
from utils.source_feasibility import parse_feasible_sources, source_available

logger = logging.getLogger(__name__)

DEFAULT_PAYLOAD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'right_source_right_layer_payload.json'
)

ECONOMIC_LAYERS = ('base', 'midBase', 'midMerit', 'peak')
QUALIFYING_LEVELS = ('fit', 'conditional')

# The battery is a SINGLE shared asset across every layer — it is never sized
# or duplicated per zone. Solar and wind, by contrast, may be distributed
# across zones. This set is the contract enforced by build_rsrl_result.
SINGLE_ASSET_SOURCES = ('battery',)
DISTRIBUTABLE_SOURCES = ('solar', 'wind', 'baseSource', 'market')
FIT_SYMBOLS = {'fit': '✓', 'conditional': '~', 'misfit': '✗', 'legacy': '~*'}

BASE_LOAD_CANDIDATE_COSTS = {
    'hydro': 'hydroCost',
    'biomass': 'biomassCost',
    'geothermal': 'geothermalCost',
    'nuclear': 'nuclearCost',
    'naturalGas': 'naturalGasCost',
    # RTC / contractually-firmed solar-wind+storage hybrid — the deck's modern
    # clean C&I base product. Competes for the base floor when a cost is given.
    'firmedHybrid': 'firmedHybridCost',
}

# Base-layer qualification for candidates the matrix doesn't list directly.
# firmedHybrid is a FIT clean base product (RTC), carbon-free (never dropped
# under greenBaseLoad).
BASE_CANDIDATE_QUAL = {'firmedHybrid': 'fit'}

# Base-source PREFERENCE order (firmest baseload first). Base needs endurance,
# not just the lowest sticker price: geothermal and nuclear are the textbook
# clean base (80-92% CUF, non-seasonal, self-firming), so they lead — even over
# a slightly cheaper but variability-exposed hydro. Within the same preference a
# cheaper source still wins. Override per request via customInputs.baseSourcePriority
# or payload.baseSourcePriority.
BASE_SOURCE_PRIORITY = ['naturalGas','geothermal', 'nuclear', 'hydro', 'firmedHybrid', 'biomass', ]


def base_source_priority(metricsConfig):
    """Ordered base-source preference (firmest first); request/payload can override."""
    ci = (metricsConfig or {}).get('customInputs') or {}
    if ci.get('baseSourcePriority'):
        return ci['baseSourcePriority']
    try:
        p = get_payload(metricsConfig).get('baseSourcePriority')
        if p:
            return p
    except Exception:
        pass
    return BASE_SOURCE_PRIORITY

_payload_cache = None
_payload_cache_mtime = None


def load_rsrl_payload(path=None):
    """Load the Right-Source-Right-Layer payload from the local JSON file.

    Cached by the file's modification time: the cache is reused while the file
    is unchanged, but a fresh edit to the JSON is picked up automatically on the
    next call — no process restart needed. A custom ``path`` always reads fresh
    and never touches the default-path cache.
    """
    global _payload_cache, _payload_cache_mtime
    file_path = path or DEFAULT_PAYLOAD_PATH
    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        mtime = None
    if (path is None and _payload_cache is not None
            and mtime is not None and mtime == _payload_cache_mtime):
        return _payload_cache
    with open(file_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if path is None:
        _payload_cache = payload
        _payload_cache_mtime = mtime
    return payload


def get_payload(metricsConfig=None):
    """Payload from metricsConfig when the backend sends it, else the local file."""
    if metricsConfig:
        payload = (
            metricsConfig.get('rsrlPayload')
            or (metricsConfig.get('customInputs') or {}).get('rightSourceRightLayer')
        )
        if payload:
            return payload
    return load_rsrl_payload()


def qualified_shortlist(payload, layer, green_base_load=False, feasible=None):
    """Sources matrix-qualified for a layer AND procurable at the location.

    fit first, then conditional. Sources not listed in the input feasibility
    whitelist are dropped — 'take the matrix, then shortlist by location, since
    we only get what's possible there'.
    """
    matrix = payload.get('qualificationMatrix', {})
    carbon_disqualified = set(payload.get('carbonDisqualifiedWhenGreenBaseLoad', []))
    shortlist = []
    for level in QUALIFYING_LEVELS:
        for source, fits in matrix.items():
            if fits.get(layer) != level:
                continue
            if not source_available(source, feasible):
                continue
            entry = {
                'source': source,
                'name': payload.get('sources', {}).get(source, {}).get('name', source),
                'fit': level,
                'symbol': FIT_SYMBOLS[level],
                'note': fits.get('note', ''),
            }
            # emergency is the fifth, non-economic layer — the deck keeps
            # diesel legitimate there even in a CFE-constrained stack
            if green_base_load and layer != 'emergency' and source in carbon_disqualified:
                entry['fit'] = 'carbon-disqualified'
                entry['symbol'] = '✗'
            shortlist.append(entry)
    return shortlist


def select_base_load_source(metricsConfig):
    """Pick the base-load source by layer qualification, then cost.

    Only sources the matrix qualifies for the base layer (natural fits first,
    conditionals as fallback) compete on price — 'a source fits a layer when
    its attributes match the layer's hours, control needs and economics'.
    Falls back to the plain cheapest-source rule if the payload is missing.
    """
    try:
        payload = get_payload(metricsConfig)
        matrix = payload['qualificationMatrix']
    except Exception as e:
        logger.warning(f"RSRL payload unavailable, falling back to cheapest source: {e}")
        return cheap_source(metricsConfig)

    candidates = {
        source: metricsConfig.get(cost_key)
        for source, cost_key in BASE_LOAD_CANDIDATE_COSTS.items()
    }
    if metricsConfig.get('greenBaseLoad'):
        for fossil in payload.get('carbonDisqualifiedWhenGreenBaseLoad', []):
            candidates.pop(fossil, None)
    # location whitelist: only sources procurable at the site can be the base source
    feasible = parse_feasible_sources(metricsConfig)
    available = {k: v for k, v in candidates.items()
                 if v not in (None, 0) and source_available(k, feasible)}
    if not available:
        return None, 0.0

    def base_qual(source):
        return BASE_CANDIDATE_QUAL.get(source) or matrix.get(source, {}).get('base')

    # Cost is the primary driver for the base source: every source the matrix
    # qualifies for the base layer (fit OR conditional) competes on price, and
    # the cheapest wins. The base-source priority order only breaks exact cost
    # ties (e.g. two sources quoted at the same $/MWh). Override the tie-break
    # via customInputs.baseSourcePriority or payload.baseSourcePriority.
    priority = base_source_priority(metricsConfig)

    def rank(source):
        return priority.index(source) if source in priority else len(priority)

    qualified = {
        source: cost for source, cost in available.items()
        if base_qual(source) in QUALIFYING_LEVELS
    }
    if not qualified:
        return None, 0.0
    chosen = min(qualified, key=lambda s: (qualified[s], rank(s)))
    return chosen, qualified[chosen]


def select_base_load_blend(metricsConfig):
    """Cheapest qualified clean and fossil base candidates for a CFE blend.

    Returns ``(clean, fossil)`` where each item is ``(source, cost)`` or None.
    This does not choose the base floor by itself; it only exposes the two
    carbon classes so the optimizer can tune their shares when a CFE target
    requires greener baseload than the cheapest single source.
    """
    try:
        payload = get_payload(metricsConfig)
        matrix = payload['qualificationMatrix']
    except Exception as e:
        logger.warning(f"RSRL payload unavailable, no base blend candidates: {e}")
        return None, None

    candidates = {
        source: metricsConfig.get(cost_key)
        for source, cost_key in BASE_LOAD_CANDIDATE_COSTS.items()
    }
    feasible = parse_feasible_sources(metricsConfig)
    available = {k: v for k, v in candidates.items()
                 if v not in (None, 0) and source_available(k, feasible)}

    def base_qual(source):
        return BASE_CANDIDATE_QUAL.get(source) or matrix.get(source, {}).get('base')

    qualified = {
        source: float(cost) for source, cost in available.items()
        if base_qual(source) in QUALIFYING_LEVELS
    }
    if not qualified:
        return None, None

    fossil_sources = set(payload.get('carbonDisqualifiedWhenGreenBaseLoad', []))
    fossil_sources.update({'naturalGas', 'ocgtNaturalGas', 'diesel', 'coal'})
    clean = [(cost, source) for source, cost in qualified.items() if source not in fossil_sources]
    fossil = [(cost, source) for source, cost in qualified.items() if source in fossil_sources]
    clean_pick = min(clean, default=None)
    fossil_pick = min(fossil, default=None)
    return (
        (clean_pick[1], clean_pick[0]) if clean_pick else None,
        (fossil_pick[1], fossil_pick[0]) if fossil_pick else None,
    )


def _parse_zone(value, demand):
    if str(value).startswith('p'):
        return float(demand.quantile(float(str(value)[1:]) / 100.0))
    return float(value)


def _layer_bounds(demand, layer_zones):
    p_base = _parse_zone(layer_zones.get('base', 'p10'), demand)
    p_mid_base = _parse_zone(layer_zones.get('midBase', 'p40'), demand)
    p_mid_merit = _parse_zone(layer_zones.get('midMerit', 'p90'), demand)
    p_peak = _parse_zone(layer_zones.get('peak', 'p100'), demand)
    return {
        'base': (0.0, p_base),
        'midBase': (p_base, max(p_mid_base, p_base)),
        'midMerit': (max(p_mid_base, p_base), max(p_mid_merit, p_mid_base, p_base)),
        'peak': (max(p_mid_merit, p_mid_base, p_base), p_peak),
    }


def build_rsrl_result(df_sim, metricsConfig, sizes=None, bessType=None):
    """Map the simulated hourly dispatch into the deck's layer stack.

    The battery is a single shared BESS: it is sized once for the whole stack
    and only its hourly discharge is *partitioned* across zones for reporting —
    the per-zone battery energy always sums back to that one asset (guarded
    below). Solar and wind may be distributed across zones freely.

    Splits demand into the four layer bands, fills each band with the
    simulated sources in the layer's merit order (clean natural fits first,
    per the qualification matrix), and returns the full deck-style block:
    layer bounds, qualified shortlist, selection, disqualifications and the
    emergency fifth layer. Returns None instead of raising so the main
    result pipeline never breaks.
    """
    try:
        payload = get_payload(metricsConfig)

        demand = df_sim['demand']
        total_demand = float(demand.sum())
        if total_demand <= 0:
            return None

        custom_inputs = metricsConfig.get('customInputs') or {}
        layer_zones = (
            custom_inputs.get('layerZones')
            or payload.get('customInputs', {}).get('layerZones')
            or {"base": "p10", "midBase": "p40", "midMerit": "p90", "peak": "p100"}
        )
        bounds = _layer_bounds(demand, layer_zones)

        band_demand = {
            'base': demand.clip(upper=bounds['base'][1]),
            'midBase': (demand - bounds['midBase'][0]).clip(
                lower=0, upper=bounds['midBase'][1] - bounds['midBase'][0]),
            'midMerit': (demand - bounds['midMerit'][0]).clip(
                lower=0, upper=bounds['midMerit'][1] - bounds['midMerit'][0]),
            'peak': (demand - bounds['peak'][0]).clip(lower=0),
        }

        base_source_name = config.BASE_LOAD_NAME or 'baseLoad'
        supply = {
            'baseSource': df_sim['base_load_used'].astype(float).copy(),
            'wind': df_sim['wind_used'].astype(float).copy(),
            'solar': df_sim['solar_used'].astype(float).copy(),
            'battery': df_sim['battery_discharge_mw'].astype(float).copy(),
            'market': df_sim['market_for_demand'].astype(float).copy(),
        }
        display_names = {
            'baseSource': base_source_name,
            'wind': 'Wind',
            'solar': 'Solar',
            'battery': 'Battery Storage',
            'market': 'Market',
        }

        merit_order = payload.get('meritOrder', {})
        # top-down so each layer's specialists are claimed by their own band
        # first (BESS by peak, solar by mid-merit, wind by mid-base) and base
        # is left to the firm source + leftovers + grid anchor, as in the deck
        allocation_order = payload.get('allocationOrder', list(reversed(ECONOMIC_LAYERS)))
        layer_energy = {}
        for layer in allocation_order:
            remaining = band_demand[layer].astype(float).copy()
            layer_energy[layer] = {}
            for source in merit_order.get(layer, list(supply)):
                if source not in supply:
                    continue
                take = supply[source].clip(lower=0).clip(upper=remaining.clip(lower=0))
                supply[source] -= take
                remaining -= take
                layer_energy[layer][source] = float(take.sum())

        layers_meta = payload.get('layers', {})
        green_base_load = bool(metricsConfig.get('greenBaseLoad'))
        feasible = parse_feasible_sources(metricsConfig)
        layer_wise_percentages = {}
        layer_details = {}
        for layer in ECONOMIC_LAYERS:
            meta = layers_meta.get(layer, {})
            label = meta.get('label', layer)
            band_total = float(band_demand[layer].sum())
            layer_wise_percentages[label] = {
                display_names[source]: (energy / total_demand) * 100
                for source, energy in layer_energy[layer].items()
            }
            layer_details[layer] = {
                'label': label,
                'hours': meta.get('hours'),
                'requirement': meta.get('requirement'),
                'band_mw': [round(bounds[layer][0], 3), round(bounds[layer][1], 3)],
                'demand_share_pct': (band_total / total_demand) * 100,
                'source_mix_pct_of_layer': {
                    display_names[source]: (energy / band_total) * 100 if band_total > 0 else 0
                    for source, energy in layer_energy[layer].items()
                },
                'qualified_shortlist': qualified_shortlist(payload, layer, green_base_load, feasible),
                'disqualified': meta.get('disqualifiedReason'),
            }

        # ---- Single-battery guarantee -------------------------------------
        # The BESS is one shared asset. Its discharge is partitioned across
        # zones above; here we verify the partition sums back to the single
        # asset and never split it into per-zone batteries. Solar/wind carry
        # no such constraint — they may be distributed across zones.
        sizes = sizes or {}
        backup_hours = (bessType or {}).get('batteryBackupHours') or 1
        total_battery_energy = float(df_sim['battery_discharge_mw'].sum())
        allocated_battery_energy = sum(
            layer_energy[layer].get('battery', 0.0) for layer in ECONOMIC_LAYERS
        )
        battery_partition_ok = (
            abs(total_battery_energy - allocated_battery_energy)
            <= 1e-6 * max(1.0, total_battery_energy)
        )
        if not battery_partition_ok:
            logger.warning(
                "Single-battery partition mismatch: discharged %.4f MWh but "
                "allocated %.4f MWh across zones",
                total_battery_energy, allocated_battery_energy,
            )
        battery_mwh = sizes.get('battery_size')
        assets = {
            'battery': {
                'shared_single_asset': True,
                'per_zone_sizing': False,
                'storage_capacity_mwh': battery_mwh,
                'storage_capacity_mw': (battery_mwh / backup_hours) if battery_mwh else None,
                'annual_discharge_mwh': total_battery_energy,
                'zone_energy_mwh': {
                    layers_meta.get(layer, {}).get('label', layer):
                        layer_energy[layer].get('battery', 0.0)
                    for layer in ECONOMIC_LAYERS
                },
                'partition_conserves_energy': battery_partition_ok,
                'note': 'One BESS shared across all zones; zone figures are an '
                        'accounting split of the same asset, not separate batteries.',
            },
            'solar': {
                'shared_single_asset': False,
                'distributable_across_zones': True,
                'plant_capacity_mw': sizes.get('solar_size'),
            },
            'wind': {
                'shared_single_asset': False,
                'distributable_across_zones': True,
                'plant_capacity_mw': sizes.get('wind_size'),
            },
        }

        # Emergency — the fifth, non-economic layer: standby insurance, not energy
        emergency_meta = layers_meta.get('emergency', {})
        emergency_cfg = payload.get('emergency', {})
        max_demand = float(demand.max())
        emergency_layer = {
            'label': emergency_meta.get('label', 'Emergency (5th layer)'),
            'hours': emergency_meta.get('hours'),
            'requirement': emergency_meta.get('requirement'),
            'strategy': emergency_cfg.get('strategy'),
            'diesel_units': emergency_cfg.get('dieselRedundancy', 2),
            'diesel_unit_mw': round(max_demand * emergency_cfg.get('dieselUnitSizingFactor', 1.12), 2),
            'bess_reserve_soc_floor_pct': emergency_cfg.get('bessSocFloorPct', 25),
            'role': emergency_cfg.get('role'),
            'qualified_shortlist': qualified_shortlist(payload, 'emergency', green_base_load, feasible),
        }

        return {
            'qualification_rule': payload.get('meta', {}).get('qualificationRule'),
            'layer_zones': layer_zones,
            'base_load_source': base_source_name,
            'assets': assets,
            'layer_wise_percentages': layer_wise_percentages,
            'layers': layer_details,
            'emergency_layer': emergency_layer,
        }
    except Exception as e:
        logger.warning(f"Right-Source-Right-Layer block skipped: {e}")
        return None
