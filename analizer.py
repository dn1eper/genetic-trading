import loader
from fitness import FitnessResult
from fitness_test import FitnessTest, ModelID
from collections import namedtuple
from loader import load_trading_report


class Trade:
    def __init__(self, model_id: ModelID, stops: int, take: int, bu: int):
        self.model_id = model_id
        self.stops = stops
        self.take = take
        self.bu = bu

    def __str__(self):
        return "stops:take:bu = {}:{}:{}".format(self.stops, self.take, self.bu)


def compare_trading_reports(fitness: FitnessTest, fitness_result: FitnessResult):
    report = []
    unequal_trades = 0
    optimizer_trading_report = build_trading_report_from_fitness_result(fitness_result)
    terminal_trading_report = build_trading_report_from_file()

    db_models_id = []
    for mod_idx in range(len(fitness.db_results)):
        qt = fitness.db_results[mod_idx]
        db_models_id.append(ModelID(qt.model_time, qt.level_time, qt.direction))

    for idx, real_trade in enumerate(terminal_trading_report):
        compare_result = "#" + str(idx + 1) + " real: " + str(real_trade)
        shift = (len(compare_result) - len("emulated: ")) * " "
        compare_result += "\n" + shift + "emulated: "
        virt_trade = None
        for trade in optimizer_trading_report:
            if trade.model_id == real_trade.model_id:
                virt_trade = trade
                break
        if virt_trade is None:
            compare_result += "no trade"
        else:
            if virt_trade.stops == real_trade.stops and virt_trade.take == real_trade.take and \
               virt_trade.bu == real_trade.bu:
                continue
            compare_result += str(virt_trade)
        if real_trade.model_id in db_models_id:
            compare_result += ", model line " + str(db_models_id.index(real_trade.model_id) + 1)
        else:
            compare_result += ", no model in db"

        report.append(compare_result)
        unequal_trades += 1

    print("unequal trades: " + str(unequal_trades) + "\n")
    for string in report:
        print(string)


def build_trading_report_from_file():
    trading_report_file = load_trading_report()
    trading_report = []
    for index, row in trading_report_file.iterrows():
        model_time = int(row["MODEL_TIME"])
        level_time = int(row["LEVEL_TIME"])
        direction = bool(int(row["DIRECTION"]))
        model_id = ModelID(model_time, level_time, direction)
        stops = int(row["STOPS"])
        take = int(row["TAKE"])
        bu = int(row["BU"])
        trading_report.append(Trade(model_id=model_id, stops=stops, take=take, bu=bu))

    return trading_report


def build_trading_report_from_fitness_result(fitness: FitnessResult):
    trading_report = []
    for idx, model_id in enumerate(fitness.models_id):
        stops = fitness.stops[idx]
        take = fitness.takes[idx]
        bu = fitness.bus[idx]
        trading_report.append(Trade(model_id, stops, take, bu))

    return trading_report
