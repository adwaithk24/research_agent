from enum import auto
import pandas as pd
from scipy.sparse import data
from yahoo_fin import stock_info as si
from datetime import datetime
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import os
from dotenv import load_dotenv

load_dotenv()

# Get NVIDIA stock information
nvidia_data = pd.DataFrame(si.get_stats_valuation("NVDA"))

# Function to convert date to year_Q format
def convert_date_to_quarter(date_str):
    if date_str == "Current":
        year = datetime.now().year
        return f"{year}_Q{datetime.now().month // 3 + 1}"

    month, day, year = map(int, date_str.split('/'))
    quarter = (month - 1) // 3 + 1
    return f"{year}_Q{quarter}"

# Function to convert 'T' values to numeric
def convert_t_to_numeric(value):
    if isinstance(value, str) and value.endswith('T'):
        return float(value[:-1]) * 1e12
    return value

def upload_to_snowflake(df, table_name):
    conn = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USERNAME'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA')
    )

    cursor = conn.cursor()

    # Create database if it does not exist
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {os.getenv('SNOWFLAKE_DATABASE')}")
    cursor.execute(f"USE DATABASE {os.getenv('SNOWFLAKE_DATABASE')}")

    # Create schema if it does not exist
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {os.getenv('SNOWFLAKE_SCHEMA')}")
    cursor.execute(f"USE SCHEMA {os.getenv('SNOWFLAKE_SCHEMA')}")

    cols = []
    for col in df.columns:
        cols.append(col)
        
    drop_table_sql = f"DROP TABLE IF EXISTS {table_name}"
    cursor.execute(drop_table_sql)

    create_table_sql = f"""CREATE TABLE {table_name} (
    {', '.join([f'{col} VARCHAR(50)' if col == 'METRICS' else f'{col} FLOAT' for col in cols])}
    )"""

    print(f"Creating table with SQL:\n{create_table_sql}")
    cursor.execute(create_table_sql)
    print(f"DataFrame columns: {cols}")
    
    success, num_chunks, num_rows, output = write_pandas(
        conn=conn,
        df=df,
        table_name=table_name,
        quote_identifiers=True,
        auto_create_table=False,
        index=False
    )
    
    print(f"Data uploaded to Snowflake: {success}, Rows: {num_rows}")
    conn.close()

if __name__ == "__main__":
    # Apply the conversion to all cells except the "METRICS" column
    for col in nvidia_data.columns[1:]:
        nvidia_data[col] = nvidia_data[col].apply(convert_t_to_numeric)

    # Convert all columns except "METRICS" to numeric type
    for col in nvidia_data.columns[1:]:
        nvidia_data[col] = pd.to_numeric(nvidia_data[col], errors='coerce')
        
    # Rename the first column to "METRICS"
    nvidia_data = nvidia_data.rename(columns={str(nvidia_data.columns[0]): "METRICS"})

    # Convert date columns to year_Q format and prefix with 'Q'
    new_columns = ["METRICS"] + [f"Y{convert_date_to_quarter(col)}" for col in nvidia_data.columns[1:]]
    nvidia_data.columns = new_columns

    print(nvidia_data)
    # Upload the DataFrame to Snowflake
    upload_to_snowflake(nvidia_data, "NVIDIA_VALUATION_MEASURES")