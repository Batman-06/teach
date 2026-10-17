import boto3
import pandas as pd
from dotenv import load_dotenv
from boto3.dynamodb.conditions import Key
import os
from typing import Optional
from decimal import Decimal
from config import config

# Load from Local
def load_data_local(file_path):
    df = pd.read_excel(file_path)
    df['demand'] = df['Actual Demand']
    # df['network_charge'] = df['networkCharge']

    # if 'Month' not in df.columns:
    #     df['Month'] = pd.to_datetime(df['Date'], format='%Y-%m-%d').dt.month
        
    df['hour'] = df['Hour']
    df['day'] = df['Day']
    df['month'] = df['Month']
    df['year'] = df['Year']
    # df['network_charge'] = df['networkCharge']
    # df['MCP (Rs/MWh)'] = df['DAM_MCP (Rs/MWh)']
    quantile_df = df.groupby(['Day', 'Month'], as_index=False)['MCP (Rs/MWh)'].quantile(config.MCP_PERCENTILE)
    quantile_df.rename(columns={
        'MCP (Rs/MWh)': f'MCP_{int(config.MCP_PERCENTILE * 100)}th_Percentile'
    }, inplace=True)

    df = df.merge(quantile_df, on=['Day', 'Month'], how='left')
    df.to_excel('temp.xlsx')
    # print(df[:24])

    return df