import pandas as pd
from dotenv import load_dotenv
from . import config

# Load environment variables
load_dotenv()

def data_loader_from_json(data: dict):
    if data:
        model_input = data.get("modelInput", [])
        if not model_input:
            raise ValueError("Missing 'modelInput' in the input data.")
        df = pd.DataFrame(model_input)
        df.rename(columns={
            'day': 'day', 'hour': 'hour', 'year': 'year', 'month': 'month',
            'demand': 'demand', 'damMCP': 'MCP (Rs/MWh)', 'referenceSolarPerMw': 'Solar Power Per MW',
            'referenceWindPerMw': 'Wind Power Per MW', 'greenGridCost': 'grid_cost_val',
            'fossilGridCost': 'grid_cost_fossil_val'
        }, inplace=True)
        # required_columns = ['day', 'hour', 'year', 'month', 'demand', 'MCP (Rs/MWh)', 
        #                     'Solar Power Per MW', 'Wind Power Per MW', 'grid_cost_val']
        
        # missing_columns = [col for col in required_columns if col not in df.columns]
        # if missing_columns:
        #     raise ValueError(f"Missing columns in the data: {', '.join(missing_columns)}")
        
        if df.isnull().values.any():
            raise ValueError("Null values found in the DataFrame. Please ensure all data is complete.")
        
        # quantile_df = df.groupby(['Day', 'Month'], as_index=False)['MCP (Rs/MWh)'].quantile(config.MCP_PERCENTILE)
        quantile_df = df.groupby(['day', 'month'], as_index=False)['MCP (Rs/MWh)'].quantile(config.MCP_PERCENTILE)
        quantile_df.rename(columns={
            'MCP (Rs/MWh)': f'MCP_{int(config.MCP_PERCENTILE * 100)}th_Percentile'
        }, inplace=True)

        df = df.merge(quantile_df, on=['day', 'month'], how='left')
        return df  
    else:
        raise ValueError("Empty or invalid model input JSON.")

def column_check(df):
    required_columns = df.columns.tolist()
    return required_columns
    