import os
from core.optimization import optimize_system_zone
import time
import json
import logging
from config import config
from utils.layer_qualification import select_base_load_source

def process_configurations(df, configurations, state, bessType, metricsConfig=None):
    """Independent-zone + shared-battery optimisation for Japan."""
    from utils.zone_dispatch import build_zone_result_entry
    import pandas as pd

    if metricsConfig is None:
        metricsConfig = {}
        
    metricsConfig['state'] = state
    metricsConfig['bessType'] = bessType

    results = []
    hourly = {}
    for config_name, config_list in configurations.items():
        start_time = time.time()
        logging.info(f"\nRunning ZONE optimization for {config_name}...")
        try:
            cheapest_name, cheapest_cost = select_base_load_source(metricsConfig)
        except Exception:
            from utils.get_cheap_source import cheap_source
            cheapest_name, cheapest_cost = cheap_source(metricsConfig)
            
        config.BASE_LOAD_NAME = cheapest_name
        config.BASE_LOAD_COST = cheapest_cost
        config.BASE_LOAD = metricsConfig.get('base_load', 0)
        
        opt_result = optimize_system_zone(df, metricsConfig)
        entry, hourly = build_zone_result_entry(opt_result, df, metricsConfig)
        
        # Append config_name to entry for main_local filtering
        entry['energy_mix_combination'] = config_name
        results.append(entry)
        logging.info(f"✅ ZONE optimization for {config_name} complete in "
                     f"{round(time.time() - start_time, 2)} sec")

    hourly_records = []
    if hourly:
        hourly_df = pd.DataFrame(
            {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in hourly.items()})
        hourly_records = hourly_df.to_dict(orient='records')
        
    return results, json.dumps(hourly_records), "{}" # Return a dummy 3rd arg for Japan's main_local.py signature
