
class JPNConfig:
    SOLAR_PPA = 0
    WIND_PPA = 0
    YEAR = 0
    POWER_FACTOR = 0
    BATTERY_COST = 0
    DEMAND_CHARGE = 0
    OPEN_ACCESS_CHARGES = 0
    CONTRACT_DEMAND = 0
    HYDRO_COST = 0
    NATURAL_GAS_COST = 0
    BIOMASS_COST = 0
    REC_COST = {}

# STATE_CLASS_CONFIG = {
#     "aus": AUSConfig
# }

JPN_STATES = [
    'JP-HK',
    'JP-CB',
    'JP-KS',
    'JP-KT',
    'JP-TH',
    'JP-CB',
    'JP-CG',
    'JP-HK',
    'JP-HR',
    'JP-KS',
    'JP-KT',
    'JP-KY',
    'JP-ON',
    'JP-SK',
    'JP-TH'
]

STATE_CLASS_CONFIG = {state: JPNConfig for state in JPN_STATES}