import numpy as np

def grid_peak(hour):
    peak_hours = [5,6,7,8,9,17,18,19,20,21,22]
    return 1 if hour in peak_hours else 0

def market_peak(hour, deficit):
    peak_hours = [5,6,7,8,9,17,18,19,20,21,22]
    return 1 if (hour in peak_hours) and (deficit > 0) else 0