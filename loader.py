import csv
import os
import pandas as pd

N_PARAMS = 9
INPUT_FILE = "Data/bd(TouchLp).csv"
CONVERTED_FILE_NAME = "Data/converted.csv"
HEADER_PREFIX = "COLUMN"

def convert():
    header = [HEADER_PREFIX+str(i) for i in range(N_PARAMS)]

    write_file = open(CONVERTED_FILE_NAME, 'w', newline='')
    with open(INPUT_FILE, 'r', newline='') as read_file:
        reader = csv.reader(read_file, delimiter=";", quotechar="\n")
        writer = csv.writer(write_file, delimiter=',', quotechar='\n', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for row in reader:
            index = row.index("BD") + 1
            writer.writerow(row[index:index+N_PARAMS])
            
    write_file.close()


def load():
    if not os.path.isfile(CONVERTED_FILE_NAME):
        convert()
    return pd.read_csv(CONVERTED_FILE_NAME)