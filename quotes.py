class Quotes:
    def __init__(self, result_row:list):
        self.model_number = int(result_row[0])
        self.time = int(result_row[1])
        self.level = float(result_row[2])
        self.atr = float(result_row[3])
        self.direction = bool(result_row[4])
        quotes = list(float(quote) for quote in result_row[5:])
        if len(quotes) % 2 != 0:
            print(result_row)
            raise ValueError("Different number of highs and lows in quotes")
        self.highs = quotes[:int(len(quotes)/2)]
        self.lows = quotes[int(len(quotes)/2):]


