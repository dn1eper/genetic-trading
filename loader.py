import csv
import os
import pandas as pd
from quotes import Quotes

DB_INDIVS_FILE = "Data/db_patterns(TouchLp).csv"
DB_RESULTS_FILE = "Data/db_results(TouchLp).csv"
TRADING_PARAMS_FILE = "Data/trading params.csv"
NON_TRADING_PARAMS_FILE = "Data/non-trading params.csv"


def convert():
    header = [HEADER_PREFIX + str(i + 1) for i in range(N_PARAMS)]

    write_file = open(CONVERTED_FILE, 'w', newline='')
    with open(DATA_FILE, 'r', newline='') as read_file:
        reader = csv.reader(read_file, delimiter=";", quotechar="\n")
        writer = csv.writer(write_file, delimiter=',', quotechar='\n', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for row in reader:
            writer.writerow(row[:N_PARAMS])

    write_file.close()

def load_indivs():
    with open(DB_INDIVS_FILE) as db_indivs:
        return pd.read_csv(DB_INDIVS_FILE, sep=';', header=None)

def load_results():
    with open(DB_RESULTS_FILE) as results_file:
        results = list(csv.reader(results_file, delimiter=';', quotechar='\n'))
        return list(Quotes(result) for result in results)
    # with open(DB_RESULTS_FILE) as db_indivs:
    #     return pd.read_csv(DB_RESULTS_FILE, sep=';', header=None)

# def load_results():
#     return pd.read_csv(DB_RESULTS_FILE)

def load_trading_params():
    return pd.read_csv(TRADING_PARAMS_FILE, delimiter=',', quotechar='\n')

def load_nontrading_params():
    return pd.read_csv(NON_TRADING_PARAMS_FILE, delimiter=',', quotechar='\n')

