import csv
import os
import pandas as pd

N_PARAMS = 10
DATA_FILE = "Data/bd(TouchLp).csv"
PARAMS_FILE = "Data/params.csv"
CONVERTED_FILE = "Data/data.csv"
HEADER_PREFIX = "COLUMN"

def convert():
    header = [HEADER_PREFIX+str(i+1) for i in range(N_PARAMS)]

    write_file = open(CONVERTED_FILE, 'w', newline='')
    with open(DATA_FILE, 'r', newline='') as read_file:
        reader = csv.reader(read_file, delimiter=";", quotechar="\n")
        writer = csv.writer(write_file, delimiter=',', quotechar='\n', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for row in reader:
            writer.writerow(row[:N_PARAMS])
            
    write_file.close()


def load_data():
    if not os.path.isfile(CONVERTED_FILE):
        convert()
    return pd.read_csv(CONVERTED_FILE)

def load_params():
    return pd.read_csv(PARAMS_FILE)