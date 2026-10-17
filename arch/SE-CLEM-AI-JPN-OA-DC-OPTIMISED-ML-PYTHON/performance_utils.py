import time
import functools
import pandas as pd
import os
from collections import defaultdict

# Dictionary to store total timing and count for each function
performance_data = defaultdict(lambda: {"total_time": 0.0, "calls": 0})

def timeit(func):
    """
    Decorator that times a function and aggregates the total execution time
    and call count for each unique function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time

        # Aggregate execution time and call count
        performance_data[func.__name__]["total_time"] += elapsed
        performance_data[func.__name__]["calls"] += 1

        return result
    return wrapper

def export_performance_data_to_excel(filename = 'results/time.xlsx'):
    """
    Exports the aggregated performance data to a single Excel file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if performance_data:
        records = []
        for func_name, data in performance_data.items():
            records.append({
                "function": func_name,
                "total_time_seconds": round(data["total_time"], 4),
                "call_count": data["calls"],
                "average_time_per_call_seconds": round(data["total_time"] / data["calls"], 4)
            })

        df = pd.DataFrame(records)
        # print (df)
        return df.to_excel('results/time.xlsx', index=False)
        
        # print(f"[TIMEIT] Aggregated performance data exported to {filename}")
    else:
        print("[TIMEIT] No performance data to export.")
 