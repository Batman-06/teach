import time
import pandas as pd
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from utils.energy_system_optimizer import process_configurations
from data_loader.data_loader_db import load_data_local
import traceback
from config import config
import random
import json
import pandas as pd
from pydantic import BaseModel
from config.config_utils import update_state_config
from config.state_config import STATE_CLASS_CONFIG
import uvicorn
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class OptimizeRequest(BaseModel):
    state: str

def load_and_process_data(data_file):
    df = load_data_local(data_file)
    if df is None:
        raise ValueError("Failed to load demand profile")
    return df

@app.get("/")
async def welcome_message():
    return {"message": "Welcome to the Optimization API"}

@app.post("/optimize")
async def optimize(request: Request):
    body = await request.json()
    # state = body.get("state", "") 
   


    configurationInputs = body.get("configurationInputs", {})
    payload_custom_inputs = body.get("customInputs") or {}
    if payload_custom_inputs:
        merged_custom_inputs = dict(payload_custom_inputs)
        merged_custom_inputs.update(configurationInputs.get("customInputs") or {})
        configurationInputs["customInputs"] = merged_custom_inputs

    state = configurationInputs.get("state")
    solarPPA = configurationInputs.get("solarPPA")
    windPPA = configurationInputs.get("windPPA")
    demandCharge = configurationInputs.get("demandCharge")
    openAccessCharges = configurationInputs.get("openAccessCharges")
    powerFactor = configurationInputs.get("powerFactor")
    contractDemand = configurationInputs.get("contractDemand")
    recCost = configurationInputs.get("recCost", {})
    hydroCost = configurationInputs.get("hydroCost", 0)
    naturalGasCost = configurationInputs.get("naturalGasCost", 0)
    biomassCost = configurationInputs.get("biomassCost", 0)
    geothermalCost = configurationInputs.get("geothermalCost", 0)
    basicCharge = configurationInputs.get("basicCharge", 0)
    reliabilityScores = configurationInputs.get("reliabilityScores", {})
    bessType = configurationInputs.get("bessType", configurationInputs.get("batteryConfig", {}))
    batteryCost = bessType.get("batteryCost", 0)
    configurationInputs["batteryCost"] = batteryCost
    configurationInputs["bessType"] = bessType
    

    config.STATE = state
    config.SOLAR_PPA = solarPPA
    config.FALLBACK_SOLAR_COST = solarPPA
    config.WIND_PPA = windPPA
    config.FALLBACK_WIND_COST = windPPA
    config.DEMAND_CHARGE_PER_MW = demandCharge
    config.TRANSMISSION_COST = openAccessCharges
    config.BATTERY_FIXED_COST_FACTOR = batteryCost
    config.POWER_FACTOR = powerFactor
    config.CONTRACT_DEMAND = contractDemand
    config.HYDRO_COST=hydroCost+ openAccessCharges
    config.NATURAL_GAS_COST=naturalGasCost
    config.BIOMASS_COST=biomassCost
    config.GEOTHERMAL_COST=geothermalCost 
    config.BASIC_CHARGE = basicCharge
    config.reliabilityScores = reliabilityScores
    # config.POWER_FACTOR = powerFactor
    



    try:
        start_time = time.time()

        # data_file = 'data/usa_new_hampshire.xlsx'
        data_file = f'data/tohoku.xlsx'

        df = load_and_process_data(data_file)
        print("Data loaded successfully from local.")

        if df is not None and not df.empty:
            configurationInputs["base_load"] = float(df["demand"].quantile(0.10))

        configurations = {
             "Solar+Wind+Battery": ["Solar", "Wind", "Battery"]
            # "Solar+Wind": ["Solar", "Wind"]
        }         

        excel_output_dir = "results/local_excel"
        os.makedirs(excel_output_dir, exist_ok=True)
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
        if state not in JPN_STATES:
            logging.error(f"State {state} is not configured in STATE_CLASS_CONFIG.")
            raise ValueError(f"Invalid state: {state}")
        
        results, results_json , results_be_json = process_configurations(df, configurations, state, bessType, configurationInputs)

        try:
            # STEP 1: Parse JSON string if needed
            if isinstance(results_json, str):
                results_json_parsed = json.loads(results_json)
            else:
                results_json_parsed = results_json

            #STEP 2: Convert to DataFrame
            if isinstance(results_json_parsed, list):
                df_excel = pd.DataFrame(results_json_parsed)
            elif isinstance(results_json_parsed, dict):
                df_excel = pd.json_normalize(results_json_parsed)
            else:
                raise ValueError("Parsed results_json is not a list or dict")

            # STEP 3: Save to Excel
            excel_path = os.path.join(
                excel_output_dir, f"simulation_{config.STATE}.csv"
            )
            df_excel.to_csv(excel_path, index=False)

            print(f"📊 Saved results_json to Excel at: {excel_path}")

        except Exception as e:
            print(f"❌ Failed to export results_json to Excel: {e}")
 

        #Save consolidated results
        a = random.randint(1, 1000)
        consolidated_path = f"results/{state}.csv"
        consolidated_json_path = f"results/{state}.json"
        os.makedirs(os.path.dirname(consolidated_path), exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(consolidated_path, index=False)
        results_df.to_json(consolidated_json_path, orient='records', indent=4)
        print(f"\nConsolidated results saved to {consolidated_path} and {consolidated_json_path}")

        #Save simulation sheets
        # print(f"Simulation sheets created successfully in {base_output_dir}")

        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

        return JSONResponse(content={
            "status": "success",
            "message": "Optimization completed successfully",
            "results_path": consolidated_path
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"An error occurred: {str(e)}")
        print("Error details:")
        print(error_details)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
