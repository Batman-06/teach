import pandas as pd
from config import config


def compute_full_grid_scenrio(df):
    max_grid_demand_full_grid_per_month = {}
    for _, row in df.iterrows():
        demand = row['demand']
        month = row['month']
        max_grid_demand_full_grid_per_month[month] = max(
            max_grid_demand_full_grid_per_month.get(month, 0), demand)
    full_grid_demands = list(max_grid_demand_full_grid_per_month.values())
    Q3_full_grid = pd.Series(full_grid_demands).quantile(0.75)
    cd_full_grid_MVA = Q3_full_grid / config.POWER_FACTOR
    demand_charge_total_full_grid = sum(
        max(config.CD_THRESHOLD * cd_full_grid_MVA, max_demand /
            config.POWER_FACTOR) * config.DEMAND_CHARGE_PER_MW
        for max_demand in max_grid_demand_full_grid_per_month.values()
    ) / 1e7

    total_energy_charge_full_fossil_grid = sum(
        row['demand'] * row['grid_cost_fossil_val'] for _, row in df.iterrows()) / 1e7
    total_grid_cost_full_fossil_grid = demand_charge_total_full_grid + \
        total_energy_charge_full_fossil_grid
    fossil_grid_avg_cost_per_unit = (
        total_grid_cost_full_fossil_grid / (df['demand'].sum() * 1000)) * 10**7
    return (cd_full_grid_MVA,
            demand_charge_total_full_grid, total_grid_cost_full_fossil_grid, fossil_grid_avg_cost_per_unit, Q3_full_grid,  total_energy_charge_full_fossil_grid)


# def compute_clem_demand_charges(max_grid_demand_per_month):
#     monthly_demands = list(max_grid_demand_per_month.values())
#     Q3 = pd.Series(monthly_demands).quantile(0.75)
#     new_cd_MVA = Q3 / config.POWER_FACTOR

#     demand_charge_total = 0.0
#     demand_charge_per_month = {}
#     effective_demand_per_month = {}

#     for month, max_demand in max_grid_demand_per_month.items():
#         max_demand_MVA = max_demand / config.POWER_FACTOR
#         effective_demand = max(config.CD_THRESHOLD *
#                                new_cd_MVA, max_demand_MVA)
#         demand_charge = effective_demand * config.DEMAND_CHARGE_PER_MW
#         demand_charge_total += demand_charge
#         demand_charge_per_month[month] = demand_charge / 1e7
#         effective_demand_per_month[month] = effective_demand
#     return demand_charge_total, demand_charge_per_month, effective_demand_per_month, new_cd_MVA

def basic_charge(df):
    max_demand_per_month = df.groupby('month')['demand'].max()
    total_basic_charge = (max_demand_per_month * config.BASIC_CHARGE).sum()
    return total_basic_charge

def compute_clem_demand_charges(df):


    max_demand_per_month = df.groupby('month')['demand'].max()

    new_cd_MW = 36     # for Japan demand charge on basis MW not MVA
    clem_CD = 36   # clem demand is taken as 40 MVA

    demand_charge_total = 0.0
    penalty = 0.0
    demand_charge_per_month = {}
    effective_demand_per_month = {}

    # Create a list to hold the data for the DataFrame
    results = []
    # print("---------------------------------------------------------------------------!!!!!!!!!!!!!!!!!!!!-")
    # print("max_demand_per_month:", max_demand_per_month)
    for month, max_demand in max_demand_per_month.items():
        # max_demand_MVA = max_demand / config.POWER_FACTOR
        max_demand_MW = max_demand 
        # effective_demand = max(config.CONTRACT_DEMAND*config.POWER_FACTOR, max_demand_MW)
        demand_charge = max_demand_MW * config.DEMAND_CHARGE_PER_MW
        demand_charge_total += demand_charge

        # if max_demand_MW > new_cd_MW:
        #     penalty = (max_demand_MW - new_cd_MW) * config.DEMAND_CHARGE_PER_MW*2
        #     demand_charge_total += penalty
        # else:
        #     penalty = 0.0
        
        # print('-----------82---------monthly demnad charge is :', demand_charge_total/1e7)
        demand_charge_per_month[month] = demand_charge / 1e6

        # effective_demand_per_month[month] = effective_demand
    # print("---------------------------------------------------------------------------!!!!!!!!!!!!!!!!!!!!-")
    # print("demand_charge_per_month:", demand_charge_per_month)

    return demand_charge_total, demand_charge_per_month, effective_demand_per_month, new_cd_MW

