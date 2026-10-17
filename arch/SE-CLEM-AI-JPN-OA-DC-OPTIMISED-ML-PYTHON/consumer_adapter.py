"""
The JPN model's adapter — the ONLY new file that model repo needs.

This sits beside `main.py`. It is `main.py`'s
`run_optimization` with everything that is not the optimisation taken out: no Redis
read, no Redis write, no SQS publish, no try/except ladder. Those are the library's
job now. `core/`, `utils/` and `config/` are not touched at all.

    .venv/bin/python consumer_adapter.py
    # or, without an entrypoint of your own:
    CLEM_HANDLER=consumer_adapter:handle python -m clem_ml_consumer

Keep `main.py` deployed alongside it during the transition. The HTTP path stays the
rollback: removing the lane from SQS_DIRECT_LANES on the backend sends those jobs back
through the router with no deploy on this side.
"""

from __future__ import annotations

from typing import Any, Dict

from clem_ml_consumer import JobContext, ModelRejected, run_consumer

# The one import line that touches anything that already exists.
from config import config
from config.aws_redis import data_loader_from_json
from utils.energy_system_optimizer import process_configurations

CONFIGURATIONS = {"Solar+Wind+Battery": ["Solar", "Wind", "Battery"]}

JPN_STATES = {
    "JP-HK", "JP-CB", "JP-KS", "JP-KT", "JP-TH", "JP-CG",
    "JP-HR", "JP-KY", "JP-ON", "JP-SK",
}


def _merge_custom_inputs(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fold the top-level customInputs into configurationInputs, as main.py does.
    in:  payload - the decompressed Redis blob
    out: the configurationInputs dict, with customInputs merged and bessType resolved
    """
    configuration_inputs = dict(payload.get("configurationInputs") or {})
    custom = payload.get("customInputs") or {}
    if custom:
        merged = dict(custom)
        merged.update(configuration_inputs.get("customInputs") or {})
        configuration_inputs["customInputs"] = merged

    bess_type = configuration_inputs.get("bessType") or configuration_inputs.get("batteryConfig") or {}
    configuration_inputs["bessType"] = bess_type
    configuration_inputs["batteryCost"] = bess_type.get("batteryCost", 0)
    return configuration_inputs


def _apply_process_config(state: str, inputs: Dict[str, Any]) -> None:
    """
    Copy the run's inputs onto the module-level `config`, exactly as main.py does.

    Safe here for the same reason it is safe there: each worker is its own process and
    handles one job at a time, so nothing else is reading these globals mid-run.

    in:  state - the JPN state code; inputs - the merged configurationInputs
    out: nothing; mutates the config module
    """
    open_access_charges = inputs.get("openAccessCharges") or 0
    config.STATE = state
    config.SOLAR_PPA = inputs.get("solarPPA")
    config.FALLBACK_SOLAR_COST = inputs.get("solarPPA")
    config.WIND_PPA = inputs.get("windPPA")
    config.FALLBACK_WIND_COST = inputs.get("windPPA")
    config.DEMAND_CHARGE_PER_MW = inputs.get("demandCharge")
    config.TRANSMISSION_COST = open_access_charges
    config.BATTERY_FIXED_COST_FACTOR = inputs.get("batteryCost", 0)
    config.POWER_FACTOR = inputs.get("powerFactor")
    config.CONTRACT_DEMAND = inputs.get("contractDemand")
    config.HYDRO_COST = (inputs.get("hydroCost") or 0) + open_access_charges
    config.NATURAL_GAS_COST = inputs.get("naturalGasCost", 0)
    config.BIOMASS_COST = inputs.get("biomassCost", 0)
    config.GEOTHERMAL_COST = inputs.get("geothermalCost", 0)
    config.BASIC_CHARGE = inputs.get("basicCharge", 0)
    config.reliabilityScores = inputs.get("reliabilityScores") or {}
    config.BESS_TYPE = inputs.get("bessType") or {}


def handle(job: JobContext) -> Dict[str, Any]:
    """
    Run one optimisation.
    in:  job - the JobContext; job.payload is the decompressed Redis blob
    out: the Redis hash fields to store — the same three the model writes today.
         Raises ModelRejected for an input this model cannot run, which the library
         parks immediately instead of retrying an outcome that cannot change
    """
    configuration_inputs = _merge_custom_inputs(job.payload)
    state = configuration_inputs.get("state")
    if state not in JPN_STATES:
        raise ModelRejected(f"Invalid state: {state}")

    _apply_process_config(state, configuration_inputs)
    df = data_loader_from_json(job.payload)

    results, _results_json, results_be_json = process_configurations(
        df, CONFIGURATIONS, state, configuration_inputs["bessType"], configuration_inputs)

    job.logger.info("[jpn] optimisation finished for state %s", state)
    return {
        "outcome": results,
        "simulation": results_be_json,
        "year": int(df["year"].iloc[0]),
    }


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    raise SystemExit(run_consumer(handle))
