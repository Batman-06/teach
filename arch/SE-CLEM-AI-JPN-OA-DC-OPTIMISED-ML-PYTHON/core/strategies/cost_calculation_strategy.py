from abc import ABC, abstractmethod
from typing import Dict

class CostCalculationStrategy(ABC):
    """Base class for state-specific cost calculation strategies."""
    
    @abstractmethod
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        """Calculate the total cost with demand charges and selling."""
        pass

    def get_base_cost(self, cost_dict: Dict[str, float], cost_key: str) -> float:
        """Helper method to safely get cost values."""
        return cost_dict.get(cost_key, 0.0)

    def get_market_and_revenue(self, total_market_cost: float, total_export_revenue: float) -> float:
        """Calculate market and revenue component."""
        return total_market_cost - total_export_revenue

class NYISOCostStrategy(CostCalculationStrategy):
    """Cost calculation strategy for NYISO."""
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        return (self.get_base_cost(base_costs, 'total_solar_cost') +
                self.get_base_cost(base_costs, 'total_wind_cost') +
                self.get_base_cost(base_costs, 'total_battery_cost') +
                self.get_base_cost(base_costs, 'total_cd_delivery_charge') +
                self.get_base_cost(base_costs, 'total_daily_demand_charge') +
                self.get_market_and_revenue(total_market_cost, total_export_revenue))

class TexasCostStrategy(CostCalculationStrategy):
    """Cost calculation strategy for Texas."""
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        return (self.get_base_cost(base_costs, 'total_solar_cost') +
                self.get_base_cost(base_costs, 'total_wind_cost') +
                self.get_base_cost(base_costs, 'total_battery_cost') +
                self.get_base_cost(base_costs, 'total_distribution_charge') +
                self.get_base_cost(base_costs, 'total_tcrf_charge') +
                self.get_market_and_revenue(total_market_cost, total_export_revenue))

class PennsylvaniaCostStrategy(CostCalculationStrategy):
    """Cost calculation strategy for Pennsylvania."""
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        return (self.get_base_cost(base_costs, 'total_solar_cost') +
                self.get_base_cost(base_costs, 'total_wind_cost') +
                self.get_base_cost(base_costs, 'total_battery_cost') +
                self.get_base_cost(base_costs, 'total_distribution_charge') +
                self.get_base_cost(base_costs, 'total_acr4_charge') +
                self.get_market_and_revenue(total_market_cost, total_export_revenue))

class IllinoisCostStrategy(CostCalculationStrategy):
    """Cost calculation strategy for Illinois."""
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        return (self.get_base_cost(base_costs, 'total_solar_cost') +
                self.get_base_cost(base_costs, 'total_wind_cost') +
                self.get_base_cost(base_costs, 'total_battery_cost') +
                self.get_base_cost(base_costs, 'total_transmission_distribution_charge') +
                self.get_market_and_revenue(total_market_cost, total_export_revenue))

class OhioCostStrategy(CostCalculationStrategy):
    """Cost calculation strategy for Ohio."""
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        return (self.get_base_cost(base_costs, 'total_solar_cost') +
                self.get_base_cost(base_costs, 'total_wind_cost') +
                self.get_base_cost(base_costs, 'total_battery_cost') +
                self.get_base_cost(base_costs, 'total_customer_charge') +
                self.get_base_cost(base_costs, 'total_bad_debt_rider_charge') +
                self.get_base_cost(base_costs, 'total_grid_smart_rider_charge') +
                self.get_base_cost(base_costs, 'total_sder_charge') +
                self.get_market_and_revenue(total_market_cost, total_export_revenue))


class MarylandCostStrategy(CostCalculationStrategy):
    """Cost calculation strategy for Maryland."""
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        return (self.get_base_cost(base_costs, 'total_solar_cost') +
                self.get_base_cost(base_costs, 'total_wind_cost') +
                self.get_base_cost(base_costs, 'total_battery_cost') +
                self.get_base_cost(base_costs, 'total_delivery_service_demand_charge') +
                self.get_base_cost(base_costs, 'total_transmission_demand_charge') +
                self.get_base_cost(base_costs, 'total_administrative_charge') +
                self.get_base_cost(base_costs, 'total_delivery_service_energy_charge') +
                self.get_market_and_revenue(total_market_cost, total_export_revenue))

class MassachusettsCostStrategy(CostCalculationStrategy):
    """Cost calculation strategy for Massachusetts."""
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        return (self.get_base_cost(base_costs, 'total_solar_cost') +
                self.get_base_cost(base_costs, 'total_wind_cost') +
                self.get_base_cost(base_costs, 'total_battery_cost') +
                self.get_base_cost(base_costs, 'total_customer_charge') +
                self.get_base_cost(base_costs, 'total_distribution_demand_charge') +
                self.get_base_cost(base_costs, 'total_transmission_charge') +
                self.get_base_cost(base_costs, 'total_energy_efficiency_charge') +
                self.get_base_cost(base_costs, 'total_distributed_solar_charge') +
                self.get_base_cost(base_costs, 'total_net_metering_charge') +
                self.get_base_cost(base_costs, 'total_electric_vehicle_prog_charge') +
                self.get_base_cost(base_costs, 'total_renewable_energy_charge') +
                self.get_base_cost(base_costs, 'total_transmission_demand_charge') +
                self.get_base_cost(base_costs, 'total_transmission_upgrades_cost_recovery_charge') +
                self.get_base_cost(base_costs, 'total_scheduling_system_control_and_dispatch_service') +
                self.get_market_and_revenue(total_market_cost, total_export_revenue))

class ConnecticutCostStrategy(CostCalculationStrategy):
    """Cost calculation strategy for Connecticut."""
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        return (self.get_base_cost(base_costs, 'total_solar_cost') +
                self.get_base_cost(base_costs, 'total_wind_cost') +
                self.get_base_cost(base_costs, 'total_battery_cost') +
                self.get_base_cost(base_costs, 'total_customer_charge') +
                self.get_base_cost(base_costs, 'total_distribution_demand_charge') +
                self.get_base_cost(base_costs, 'total_electric_system_improvement_charge') +
                self.get_base_cost(base_costs, 'total_revenue_adjustment_charge') +
                self.get_base_cost(base_costs, 'total_transmission_demand_charge') +
                self.get_base_cost(base_costs, 'total_combined_public_benefits_charge') +
                self.get_base_cost(base_costs, 'total_transition_demand_charge') +
                self.get_market_and_revenue(total_market_cost, total_export_revenue))

class NewJerseyCostStrategy(CostCalculationStrategy):
    """Cost calculation strategy for New Jersey."""
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        return (self.get_base_cost(base_costs, 'total_solar_cost') +
                self.get_base_cost(base_costs, 'total_wind_cost') +
                self.get_base_cost(base_costs, 'total_battery_cost') +
                self.get_base_cost(base_costs, 'total_customer_charge') +
                self.get_base_cost(base_costs, 'total_distribution_demand_charge') +
                self.get_base_cost(base_costs, 'total_electric_system_improvement_charge') +
                self.get_base_cost(base_costs, 'total_revenue_adjustment_charge') +
                self.get_base_cost(base_costs, 'total_transmission_demand_charge') +
                self.get_base_cost(base_costs, 'total_combined_public_benefits_charge') +
                self.get_base_cost(base_costs, 'total_transition_demand_charge') +
                self.get_market_and_revenue(total_market_cost, total_export_revenue))

class CaliforniaCostStrategy(CostCalculationStrategy):
    """Cost calculation strategy for California."""
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        return (self.get_base_cost(base_costs, 'total_solar_cost') +
                self.get_base_cost(base_costs, 'total_wind_cost') +
                self.get_base_cost(base_costs, 'total_battery_cost') +
                self.get_base_cost(base_costs, 'total_customer_charge') +
                self.get_base_cost(base_costs, 'total_distribution_demand_charge') +
                self.get_base_cost(base_costs, 'total_electric_system_improvement_charge') +
                self.get_base_cost(base_costs, 'total_revenue_adjustment_charge') +
                self.get_base_cost(base_costs, 'total_transmission_demand_charge') +
                self.get_base_cost(base_costs, 'total_combined_public_benefits_charge') +
                self.get_base_cost(base_costs, 'total_transition_demand_charge') +
                self.get_market_and_revenue(total_market_cost, total_export_revenue))
    
class DelawareCostStrategy(CostCalculationStrategy):
    """Cost calculation strategy for Delaware."""
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        return (self.get_base_cost(base_costs, 'total_solar_cost') +
                self.get_base_cost(base_costs, 'total_wind_cost') +
                self.get_base_cost(base_costs, 'total_battery_cost') +
                self.get_base_cost(base_costs, 'total_customer_charge') +
                self.get_base_cost(base_costs, 'total_distribution_demand_charge') +
                self.get_base_cost(base_costs, 'total_electric_system_improvement_charge') +
                self.get_base_cost(base_costs, 'total_revenue_adjustment_charge') +
                self.get_base_cost(base_costs, 'total_transmission_demand_charge') +
                self.get_base_cost(base_costs, 'total_combined_public_benefits_charge') +
                self.get_base_cost(base_costs, 'total_transition_demand_charge') +
                self.get_market_and_revenue(total_market_cost, total_export_revenue))
    
class MaineCostStrategy(CostCalculationStrategy):
    """Cost calculation strategy for Maine."""
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        return (self.get_base_cost(base_costs, 'total_solar_cost') +
                self.get_base_cost(base_costs, 'total_wind_cost') +
                self.get_base_cost(base_costs, 'total_battery_cost') +
                self.get_base_cost(base_costs, 'total_customer_charge') +
                self.get_base_cost(base_costs, 'total_distribution_demand_charge') +
                self.get_base_cost(base_costs, 'total_electric_system_improvement_charge') +
                self.get_base_cost(base_costs, 'total_revenue_adjustment_charge') +
                self.get_base_cost(base_costs, 'total_transmission_demand_charge') +
                self.get_base_cost(base_costs, 'total_combined_public_benefits_charge') +
                self.get_base_cost(base_costs, 'total_transition_demand_charge') +
                self.get_market_and_revenue(total_market_cost, total_export_revenue))

class DefaultCostStrategy(CostCalculationStrategy):
    """Default cost calculation strategy."""
    def calculate(self, base_costs: Dict[str, float], 
                 total_market_cost: float, 
                 total_export_revenue: float) -> float:
        return (self.get_base_cost(base_costs, 'total_solar_cost') +
                self.get_base_cost(base_costs, 'total_wind_cost') +
                self.get_base_cost(base_costs, 'total_battery_cost') +
                self.get_market_and_revenue(total_market_cost, total_export_revenue))
