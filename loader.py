import csv
import os
import pandas as pd

DB_INDIVS_FILE = "Data/db_patterns(TouchLp).csv"
DB_RESULTS_FILE = "Data/db_results(TouchLp).csv"
TRADING_PARAMS_FILE = "Data/trading params.csv"
NON_TRADING_PARAMS_FILE = "Data/non-trading params.csv"

def load_indivs():
    return pd.read_csv(DB_INDIVS_FILE, sep=';', header=None)

def load_results():
    with open(DB_RESULTS_FILE) as results_file:
        FIRST_QUOTE_INDEX = 5
        results = list(csv.reader(results_file, delimiter=';', quotechar='\n'))
        for result in results:
            if (len(result) - FIRST_QUOTE_INDEX) % 2 != 0: 
                raise ValueError("Different number of highs and lows in quotes")
        return results

def load_trading_params():
    return pd.read_csv(TRADING_PARAMS_FILE, delimiter=',', quotechar='\n')

def load_nontrading_params():
    return pd.read_csv(NON_TRADING_PARAMS_FILE, delimiter=',', quotechar='\n')

