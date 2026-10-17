from utils.source_feasibility import parse_feasible_sources, source_available


def cheap_source(metricsConfig):
    # Firm-source costs come from the request, but a source is eligible only when
    # resources.feasibleEnergySources explicitly lists it for this location.
    resource_cost_map = {
        'hydro': metricsConfig.get('hydroCost'),
        'biomass': metricsConfig.get('biomassCost'),
        'geothermal': metricsConfig.get('geothermalCost'),
        'nuclear': metricsConfig.get('nuclearCost'),
        'naturalGas': metricsConfig.get('naturalGasCost'),
        'ocgtNaturalGas': metricsConfig.get('ocgtNaturalGasCost'),
        'firmedHybrid': metricsConfig.get('firmedHybridCost'),
    }
    if metricsConfig.get('greenBaseLoad'):
        resource_cost_map.pop('naturalGas', None)
    feasible = parse_feasible_sources(metricsConfig)
    filtered_costs = {
        k: v for k, v in resource_cost_map.items()
        if v not in (None, 0) and source_available(k, feasible)
    }
    if not filtered_costs:
        return None, 0.0
    cheapest_name = min(filtered_costs, key=filtered_costs.get)
    cheapest_cost = filtered_costs[cheapest_name]
    return cheapest_name, cheapest_cost
