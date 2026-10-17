"""Input-driven source feasibility helpers.

Generation sources are eligible only when the request explicitly lists them in
``resources.feasibleEnergySources``. Battery and market are system/backstop
mechanisms, so they remain outside the procurable-generation whitelist.
"""

# Order matters ('pumped' before 'hydro', 'natural gas' before bare 'gas').
FEASIBLE_KEYWORDS = [
    ('firmed hybrid', 'firmedHybrid'),
    ('firmedhybrid', 'firmedHybrid'),
    ('rtc', 'firmedHybrid'),
    ('pumped', 'pumpedStorage'),
    ('solar', 'solar'),
    ('wind', 'wind'),
    ('geothermal', 'geothermal'),
    ('biomass', 'biomass'),
    ('nuclear', 'nuclear'),
    ('hydrogen', 'h2Turbine'),
    ('h2', 'h2Turbine'),
    ('ocgt', 'ocgtNaturalGas'),
    ('diesel', 'diesel'),
    ('coal', 'coal'),
    ('natural gas', 'naturalGas'),
    ('ccgt', 'naturalGas'),
    ('gas', 'naturalGas'),
    ('battery', 'battery'),
    ('bess', 'battery'),
    ('hydro', 'hydro'),
]

ALWAYS_AVAILABLE_SOURCES = {'battery', 'market'}


def parse_feasible_sources(metricsConfig):
    """Return source keys explicitly listed in resources.feasibleEnergySources."""
    res = (metricsConfig or {}).get('resources') or {}
    fes = res.get('feasibleEnergySources') or {}
    names = list(fes.get('fossilSources') or []) + list(fes.get('nonFossilSources') or [])
    feasible = set()
    for name in names:
        low = str(name).lower()
        for kw, key in FEASIBLE_KEYWORDS:
            if kw in low:
                feasible.add(key)
                break
    return feasible


def source_available(source, feasible):
    """True when a source is explicitly feasible, except system backstops."""
    if source in ALWAYS_AVAILABLE_SOURCES:
        return True
    return source in (feasible or set())
