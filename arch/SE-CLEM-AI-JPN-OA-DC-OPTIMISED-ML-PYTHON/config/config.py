STATE = None  # default state
SOLAR_PPA = 0    # Yen/MW
WIND_PPA = 0  
FALLBACK_WIND_COST = 0
FALLBACK_SOLAR_COST = 0
# Battery charging threshold
BATTERY_CHARGING_THRESHOLD = 10  # $/MWh

MCP_PERCENTILE = 0.15  # 10th percentile of MCP for charging battery
HYDRO_COST = 0
NATURAL_GAS_COST = 0
BIOMASS_COST = 0
GEOTHERMAL_COST = 0
# Constants
# POWER_FACTOR = 0

CD_THRESHOLD = 0.90      # 90% of contract demand
DEMAND_CHARGE_PER_MW = 0   # Yen/MW
TRANSMISSION_COST = 0  # Yen/MW
BASIC_CHARGE = 0  # Yen/MW/month
# Can be taken from user
CD_PERCENTAGE = 0.40      # 40% of the full grid CD case
CONTRACT_DEMAND = 0
POWER_FACTOR = 0
GRID_PERCENTAGE = 0.70
# base_load_percent = 0.001 # will be updated based on user input
base_load_percent = 0.70
BASE_LOAD = 25
CFE_SCORE = 60
# Battery cost parameters
BATTERY_FIXED_COST_FACTOR = 0  # Annual fixed cost factor
BATTERY_BACKUP_HOURS = 4  # For backup duration
INITIAL_SOC = 0.4

# Constraints for optimization
RE_utilization = 90    # % to be met
RE_utilization_WB = 100   #
Battery_utilization = 100  # % to be met
Peak_demand_meet = 50   # % to be met
Annual_dc_demand_meet = 40  # % to be met
reliabilityScores = {}
BESS_TYPE = {}
RE_Score = 90
CFE_SCORE = 50
market_meet_percent = 20
# solar_meet_percent = 17
# wind_meet_percent = 25


MIN_SOC = 0.00001
MAX_SOC = 1.0
dod = MAX_SOC -MIN_SOC

# # Battery charging and discharging efficiency
BATTERY_CHARGING_EFFICIENCY = 0.95
BATTERY_DISCHARGING_EFFICIENCY = 0.95
RTE = BATTERY_CHARGING_EFFICIENCY * \
    BATTERY_DISCHARGING_EFFICIENCY  # Round trip efficiency


# Optimization parameters
OPT_MAXITER = 100
OPT_POPSIZE = 5
OPT_TOL = 1e-3
# Base load is fixed at the 10th percentile (p10) of demand. The optimizer is
# only allowed to search a small window [p10 - OPT_BASE_LOAD_DELTA * p10, p10],
# so the base load effectively stays at p10.
OPT_BASE_LOAD_DELTA = 0.05

# Emission Factors (in kg CO2e per MWh)
GRID_EMISSION_FACTOR = 367  # Average grid emission factor
NATURAL_GAS_EMISSION_FACTOR = 400  # Natural gas plant emission factor


# Default Firm Sources Capacity Utilization Factor (CUF). 1 = report the
# delivered MW as-is; the payload (cuf / firmCuf / <source>Cuf) overrides per
# source. Matches the models that already carry get_firm_source_cuf, so an
# absent CUF never silently derates.
HYDRO_CUF = 0.37
BIOMASS_CUF = 0.58
GEOTHERMAL_CUF = 0.74
NUCLEAR_CUF = 0.80
H2_TURBINE_CUF = 0.80
NATURAL_GAS_CUF = 0.85
OCGT_CUF = 0.05
DIESEL_CUF = 0.06
COAL_CUF = 0.61
FIRMED_HYBRID_CUF = 1

FIRM_SOURCE_CUF = {
    'hydro': HYDRO_CUF,
    'biomass': BIOMASS_CUF,
    'geothermal': GEOTHERMAL_CUF,
    'nuclear': NUCLEAR_CUF,
    'h2Turbine': H2_TURBINE_CUF,
    'naturalGas': NATURAL_GAS_CUF,
    'ocgtNaturalGas': OCGT_CUF,
    'ocgt': OCGT_CUF,
    'diesel': DIESEL_CUF,
    'coal': COAL_CUF,
    'firmedHybrid': FIRMED_HYBRID_CUF,
}

# Carbon-free share of the imported grid (fraction, or percent >1 which main.py
# normalises). 0 = grid contributes nothing to the 24/7 CFE score, the previous
# behaviour.
GRID_CFE_PERCENT = 0.0
