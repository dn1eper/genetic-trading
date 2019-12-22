def print_row(row, cell_size = 10):
    row_format = ("|{:>" + str(cell_size) +  "}") * (len(row)) + "|"
    print(row_format.format(*row))

def print_header(header, cell_size = 10):
    print_row(header)
    print(("+"+("-"*cell_size))* (len(header)) + "|")