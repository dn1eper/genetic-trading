import csv
import os
import pandas as pd

SYMBOL_PROPERTIES_FILE = "Data/symbol_properties(TouchLp).csv"
DB_INDIVS_FILE = "Data/db_patterns(TouchLp).csv"
DB_RESULTS_FILE = "Data/db_results(TouchLp).csv"
DB_QUOTES_TIME = "Data/db_results_time(TouchLp).csv"
DB_QUOTES_TIME_STRING = "Data/db_results_time_string(TouchLp).csv"
TRADING_PARAMS_FILE = "Data/trading params.csv"
NON_TRADING_PARAMS_FILE = "Data/non-trading params.csv"
TRADING_REPORT_FILE = "Data/FractureLpTestReport.csv"


def load_symbol_properties():
    chart_prop_file = open(SYMBOL_PROPERTIES_FILE)
    digits = int(chart_prop_file.readline())
    spread = float(chart_prop_file.readline())
    return digits, spread

def load_indivs():
    return pd.read_csv(DB_INDIVS_FILE, sep=';', header=None)

def load_results():
    with open(DB_RESULTS_FILE) as results_file:
        FIRST_QUOTE_INDEX = 6
        results = list(csv.reader(results_file, delimiter=';', quotechar='\n'))
        for result in results:
            if (len(result) - FIRST_QUOTE_INDEX) % 2 != 0:
                raise ValueError("Different number of highs and lows in quotes")
        return results

def load_quotes_time():
    with open(DB_QUOTES_TIME) as results_file:
        return list(csv.reader(results_file, delimiter=';', quotechar='\n'))

def load_quotes_time_string():
    with open(DB_QUOTES_TIME_STRING) as results_file:
        return list(csv.reader(results_file, delimiter=';', quotechar='\n'))

def load_trading_params():
    return pd.read_csv(TRADING_PARAMS_FILE, delimiter=',', quotechar='\n')

def load_nontrading_params():
    return pd.read_csv(NON_TRADING_PARAMS_FILE, delimiter=',', quotechar='\n')

def load_trading_report():
    return pd.read_csv(TRADING_REPORT_FILE, delimiter=';', quotechar='\n')



