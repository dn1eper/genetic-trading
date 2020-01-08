import random as rand
import sys

def print_row(row, cell_size = 10):
    row_format = ("|{:>" + str(cell_size) +  "}") * (len(row)) + "|"
    print(row_format.format(*row))

def print_header(header, cell_size = 10):
    print_row(header)
    print(("+"+("-"*cell_size))* (len(header)) + "|")

def print_flush(text):
    sys.stdout.write("\r" + text + " " * 100)
    sys.stdout.flush()

def round05(param):
    a = param % 1
    if a < 0.25:
        return int(param)
    elif a >= 0.75:
        return int(param) + 1
    else:
        return int(param) + 0.5

