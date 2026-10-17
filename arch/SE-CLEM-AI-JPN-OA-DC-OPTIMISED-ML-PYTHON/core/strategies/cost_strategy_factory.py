from enum import Enum
from typing import Dict
from .cost_calculation_strategy import CostCalculationStrategy
from .cost_calculation_strategy import (
    NYISOCostStrategy,
    TexasCostStrategy,
    PennsylvaniaCostStrategy,
    IllinoisCostStrategy,
    OhioCostStrategy,
    MarylandCostStrategy,
    MassachusettsCostStrategy,
    ConnecticutCostStrategy,
    NewJerseyCostStrategy,
    CaliforniaCostStrategy,
    DefaultCostStrategy
)

class State(Enum):
    """Enumeration of supported states."""
    NYISO = "US-NY"
    TEXAS = "US-TX"
    PENNSYLVANIA = "US-PA"
    ILLINOIS = "US-IL"
    OHIO = "US-OH"
    MARYLAND = "US-MD"
    MASSACHUSETTS = "US-MA"
    CONNECTICUT = "US-CT"
    NEW_JERSEY = "US-NJ"
    CALIFORNIA = "US-CA"
    DELAWARE = "US-DE"
    MAINE = "US-ME"
    NEW_HAMPSHIRE = "US-NH"
    DEFAULT = "default"

class CostStrategyFactory:
    """Factory class for creating cost calculation strategies."""
    
    _strategies: Dict[State, CostCalculationStrategy] = {
        State.NYISO: NYISOCostStrategy(),
        State.TEXAS: TexasCostStrategy(),
        State.PENNSYLVANIA: PennsylvaniaCostStrategy(),
        State.ILLINOIS: IllinoisCostStrategy(),
        State.OHIO: OhioCostStrategy(),
        State.MARYLAND: MarylandCostStrategy(),
        State.MASSACHUSETTS: MassachusettsCostStrategy(),
        State.CONNECTICUT: ConnecticutCostStrategy(),
        State.NEW_JERSEY: NewJerseyCostStrategy(),
        State.CALIFORNIA: CaliforniaCostStrategy(),
        State.DELAWARE: DefaultCostStrategy(),
        State.MAINE: DefaultCostStrategy(),
        State.NEW_HAMPSHIRE: DefaultCostStrategy(),
        State.DEFAULT: DefaultCostStrategy()
    }
    
    @staticmethod
    def get_strategy(state: str) -> CostCalculationStrategy:
        """
        Get the appropriate cost calculation strategy for the given state.
        
        Args:
            state: The state identifier (case-insensitive)
            
        Returns:
            CostCalculationStrategy: The appropriate strategy for the state
        """
        try:
            state_enum = State(state )
            return CostStrategyFactory._strategies[state_enum]
        except ValueError:
            return CostStrategyFactory._strategies[State.DEFAULT]
