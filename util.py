import random as rand

def print_row(row, cell_size = 10):
    row_format = ("|{:>" + str(cell_size) +  "}") * (len(row)) + "|"
    print(row_format.format(*row))

def print_header(header, cell_size = 10):
    print_row(header)
    print(("+"+("-"*cell_size))* (len(header)) + "|")

def floor05(param):
    a = param % 1
    if a < 0.25:
        return int(param)
    elif a >= 0.75:
        return int(param) + 1
    else:
        return int(param) + 0.5

