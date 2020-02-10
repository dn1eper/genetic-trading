from typing import List

from loader import load_indivs, load_results, load_quotes_time, load_quotes_time_string, load_symbol_properties
import pandas as pd
import numpy as np
import math
import csv
from numba import jitclass, float64, int64, boolean, types, jit
from numba import cuda
from collections import namedtuple
from multiprocessing import Pool

REPORT_FILE_NAME = "trading_report.csv"

Quote = namedtuple("Quote", ["model_number", "model_time", "level_time", "level", "atr", "direction", "highs", "lows",
                             "time", "time_string"])
OpenModel = namedtuple("OpenModel", ["range_start", "range_end", "close_time", "model_line"])
Model = namedtuple("Model", ["model_index", "model_id", "stop", "take", "bu", "insufficient_history"])
ModelID = namedtuple("ModelID", ["model_time", "level_time", "direction"])
Trade = namedtuple("Trade", ["modelID", "stops", "take", "bu"])


class TradeReportUnit:
    def __init__(self, trade_number, model_line, time, order_type, order, price, stop_loss, take_profit):
        self._trade = [trade_number, model_line, time, order_type, order, round(price, 5), round(stop_loss, 5),
                       round(take_profit, 5)]

    def __iter__(self):
        for trade_param in self._trade:
            yield trade_param


class FitnessResult:
    def __init__(self, results, ext_info=False):
        self._ext_info = ext_info
        if ext_info:
            self.fitness, self.ratio, self.models, self.max_drawdown, self.max_account, self.models, self.fitness_list, \
            self.trade_history = results  # debugging parameters
        else:
            self.fitness, self.ratio, self.models = results

    def __str__(self):
        if self._ext_info:
            return "fitness: {} ratio: {} models: {} max drawdown: {} max account value: {} ".format(
                self.fitness, self.ratio, self.models, self.max_drawdown, self.max_account)
        else:
            return "fitness: {}, ratio: {} models: {}".format(self.fitness, self.ratio, self.models)

    def fitness_list(self):
        if self._ext_info:
            return self.fitness_list()
        return [self.fitness]


class FitnessTest:
    def __init__(self):
        self.db_indivs = load_indivs().to_numpy()
        self.digits, self.spread = load_symbol_properties()
        self.db_results = []
        quotes_time = list(list(int(time) for time in quotes) for quotes in load_quotes_time())
        quotes_time_string = load_quotes_time_string()

        for idx, result in enumerate(load_results()):
            quotes = np.array([float(quote) for quote in result[6:]])

            self.db_results.append(Quote(
                int(result[0]),
                int(result[1]),
                int(result[2]),
                float(result[3]),
                float(result[4]),
                bool(int(result[5])),
                quotes[:int(len(quotes) / 2)],
                quotes[int(len(quotes) / 2):],
                quotes_time[idx],
                quotes_time_string[idx]
            ))

        # self.db_results = np.array(self.db_results)

    def calc(self, indivs: list, gpu=False, multithreaded=True, ext_info=False, test_model_line=-1):
        params = [
            (
                self.db_indivs,
                self.db_results,
                np.array([gene.value() for gene in indiv.get("trading")]),  # trading params
                np.array([(gene.value() - gene.radius(), gene.value() + gene.radius()) for gene in  # nontrading params
                          indiv.get("trading", has_tag=False)]),
                self.spread,
                self.digits
            ) for indiv in indivs
        ]

        results = []

        if gpu:
            for indiv_idx in range(len(indivs)):
                results.append(evaluate_gpu(params[indiv_idx]))
        elif multithreaded and not ext_info and test_model_line is None:
            pool = Pool()
            results = pool.map(evaluate, params)
            pool.close()
            pool.join()
        else:
            for indiv_idx in range(len(indivs)):
                results.append(evaluate(params[indiv_idx], ext_info=ext_info, test_model_line=test_model_line))

        for indiv_idx, indiv in enumerate(indivs):
            indiv.fitness = FitnessResult(results[indiv_idx], ext_info=ext_info)


def evaluate(params, ext_info=False, test_model_line=-1):
    db_indivs, db_results, trading_params, non_trading_params, spread, digits = params
    entry_dist, stop, stop_out, take, bu, bu_cond, min_dist_betw_orders, max_stops, exp, min_tim_betw_odrs = trading_params
    test_mod_idx = test_model_line - 1
    trade_list = []
    use_bu = True
    if bu > bu_cond:
        use_bu = False
    # setting parameters
    result = 0
    result_list = []
    orders = 0
    trades = 0
    models = 0
    models_list = []
    model_indexes = []
    models_id = []
    model_stops = []
    model_takes = []
    model_bus = []
    max_drawdown = 0
    max_account_value = 0

    open_models = []  # list with OpenModel objects
    recently_opened_models = []  # list of model numbers
    for mod_idx in range(len(db_indivs)):
        model = db_indivs[mod_idx]
        qt = db_results[mod_idx]
        insufficient_history = False

        fits = True
        atr, level, direction, model_number, model_time, level_time = \
            qt.atr, qt.level, qt.direction, qt.model_number, qt.model_time, qt.level_time

        open_ord_cond_shift = -spread if direction else 0  # (-spread) if buy, (0) otherwise
        close_ord_cond_shift = 0 if direction else -spread  # (0) if buy, (-spread) otherwise

        placed_order = False
        is_opened = False  # order is in trade now
        exp_time = math.inf
        stops = 0
        placed_bu = False
        have_take = False
        expired = False
        closed_by_bu = False
        closed_by_stop_out = False
        time_after_prev_order = math.inf  # initiates with math.inf to pass (time_after_prev_order < min_tim_betw_odrs) condition and place first order
        time_after_order = 0  # time passed after placing an order and before opening it
        if direction:  # long model
            entry_lvl = round(level - entry_dist * atr, digits)
            bu_lvl = round(entry_lvl + bu * atr, digits)
            bu_condition_level = entry_lvl + bu_cond * atr
            take_lvl = round(entry_lvl + take * atr, digits)
            stop_lvl = round(entry_lvl - stop * atr, digits)
            stop_out_lvl = entry_lvl - stop_out * atr
        else:  # short model
            entry_lvl = round(level + entry_dist * atr, digits)
            bu_lvl = round(entry_lvl - bu * atr, digits)
            bu_condition_level = entry_lvl - bu_cond * atr
            take_lvl = round(entry_lvl - take * atr, digits)
            stop_lvl = round(entry_lvl + stop * atr, digits)
            stop_out_lvl = entry_lvl + stop_out * atr

        # bringing buffers up to date
        for open_model in open_models:
            if model_time >= open_model.close_time:
                open_models.remove(open_model)

        if mod_idx == test_mod_idx:
            result_string = ""
            if not (stops == 0 and not have_take and not closed_by_bu):
                result_string += "active "
            else:
                result_string += "passive "
            print(
                "active model index = {}, direction = {}, level = {}, entry_lvl = {}, bu_lvl = {}, bu_condition_level = {}, take_level = {}, stop_lvl = {}, stop_out_lvl = {}, bottom_boundary = {}, top_boundary = {}, atr = {}".format(
                    (mod_idx + 1), direction, level, entry_lvl, bu_lvl, bu_condition_level,
                    take_lvl, stop_lvl, stop_out_lvl, entry_lvl - min_dist_betw_orders * atr,
                                                      entry_lvl + min_dist_betw_orders * atr, atr))

        # checking if this model should be counted
        for opened_model in recently_opened_models:
            if model_number == opened_model:
                if mod_idx == test_mod_idx:
                    print(
                        "{} model rejected cause model with the same number is ALREADY OPENED".format(test_model_line))
                fits = False

        for open_model in open_models:
            if open_model.range_start <= entry_lvl <= open_model.range_end:
                if mod_idx == test_mod_idx:
                    print("{} model rejected cause ANOTHER OPENED model {} ".format(test_model_line,
                                                                                    open_model.model_line))
                fits = False

        for gene_idx in range(len(model)):
            attr = model[gene_idx]
            if non_trading_params[gene_idx][0] > attr or non_trading_params[gene_idx][1] < attr:
                if mod_idx == test_mod_idx:
                    print("{} model rejected cause parameters are NOT RIGHT".format(test_model_line))
                fits = False
                break

        if not fits:
            continue

        # evaluating model
        models += 1
        price_is_up = False
        if qt.highs[0] > entry_lvl + open_ord_cond_shift and qt.lows[0] > entry_lvl + open_ord_cond_shift:
            price_is_up = True
        current_time = qt.time[0]

        for idx in range(len(qt.highs)):
            current_time = qt.time[idx]
            time_after_prev_order += 1

            if (direction and qt.lows[idx] <= stop_out_lvl) or (not direction and qt.highs[idx] >= stop_out_lvl):
                closed_by_stop_out = True
                if is_opened:
                    stops += 1
                    if ext_info:
                        trades += 1
                        order_type = "s/l(sell)" if direction else "s/l(buy)"
                        trade_list.append(
                            TradeReportUnit(trade_number=trades, model_line=(mod_idx + 1),
                                            time=qt.time_string[idx],
                                            order_type=order_type, order=orders, price=stop_lvl, stop_loss=stop_lvl,
                                            take_profit=take_lvl))
                elif ext_info:
                    trades += 1
                    order_type = "delete"
                    trade_list.append(
                        TradeReportUnit(trade_number=trades, model_line=(mod_idx + 1),
                                        time=qt.time_string[idx],
                                        order_type=order_type, order=orders, price=entry_lvl, stop_loss=stop_lvl,
                                        take_profit=take_lvl))

                if mod_idx == test_mod_idx:
                    print("closed by stop out")
                break

            if is_opened:
                if use_bu and not placed_bu and \
                        ((direction and qt.highs[idx] >= bu_condition_level) or (
                                not direction and qt.lows[idx] <= bu_condition_level)):
                    stop_lvl = bu_lvl
                    placed_bu = True
                elif (direction and qt.lows[idx] <= stop_lvl + close_ord_cond_shift) or (
                        not direction and qt.highs[idx] >= stop_lvl + close_ord_cond_shift):
                    if ext_info:
                        trades += 1
                        order_type = "s/l(sell)" if direction else "s/l(buy)"
                        trade_list.append(
                            TradeReportUnit(trade_number=trades, model_line=(mod_idx + 1),
                                            time=qt.time_string[idx],
                                            order_type=order_type, order=orders, price=stop_lvl, stop_loss=stop_lvl,
                                            take_profit=take_lvl))
                    if placed_bu:
                        closed_by_bu = True
                        break
                    stops += 1
                    if stops >= max_stops:
                        break

                    is_opened = False
                    time_after_order = 0
                    placed_order = False
                    if qt.highs[idx] > entry_lvl + open_ord_cond_shift and qt.lows[
                        idx] > entry_lvl + open_ord_cond_shift:
                        price_is_up = True
                    else:
                        price_is_up = False
                elif (direction and qt.highs[idx] >= take_lvl + close_ord_cond_shift) or (
                        not direction and qt.lows[idx] <= take_lvl + close_ord_cond_shift):
                    if ext_info:
                        trades += 1
                        order_type = "t/p(sell)" if direction else "t/p(buy)"
                        trade_list.append(
                            TradeReportUnit(trade_number=trades, model_line=(mod_idx + 1),
                                            time=qt.time_string[idx],
                                            order_type=order_type, order=orders, price=take_lvl, stop_loss=stop_lvl,
                                            take_profit=take_lvl))
                    have_take = True
                    break

            if not is_opened:
                if not placed_order and time_after_prev_order < min_tim_betw_odrs:  # if 'True' then we can't open new order right now
                    continue
                if not placed_order:
                    time_after_prev_order = 0
                    placed_order = True
                    if ext_info:
                        orders += 1
                        trades += 1
                        exp_time = current_time + exp * 60
                        order_type = "buy limit" if direction else "sell limit"
                        trade_list.append(
                            TradeReportUnit(trade_number=trades, model_line=(mod_idx + 1), time=qt.time_string[idx],
                                            order_type=order_type, order=orders, price=entry_lvl, stop_loss=stop_lvl,
                                            take_profit=take_lvl))

                if current_time >= exp_time:
                    expired = True
                    if ext_info:
                        trades += 1
                        trade_list.append(
                            TradeReportUnit(trade_number=trades, model_line=(mod_idx + 1),
                                            time=qt.time_string[idx],
                                            order_type="expiration", order=orders, price=entry_lvl, stop_loss=stop_lvl,
                                            take_profit=take_lvl))
                    break

                if (price_is_up and qt.lows[idx] <= entry_lvl + open_ord_cond_shift) or \
                        (not price_is_up and qt.highs[idx] >= entry_lvl + open_ord_cond_shift):
                    is_opened = True
                    if ext_info:
                        trades += 1
                        order_type = "buy" if direction else "sell"
                        trade_list.append(
                            TradeReportUnit(trade_number=trades, model_line=(mod_idx + 1),
                                            time=qt.time_string[idx],
                                            order_type=order_type, order=orders, price=entry_lvl, stop_loss=stop_lvl,
                                            take_profit=take_lvl))
                else:
                    time_after_order += 1

        if not have_take and not stops >= max_stops and not expired and not closed_by_bu and not closed_by_stop_out:
            insufficient_history = True

        result -= stops * stop
        if have_take:
            result += take
        elif closed_by_bu:
            result += bu
        result = round(result, 2)

        models_list.append(Model(mod_idx + 1, ModelID(model_time, level_time, direction), stops, int(have_take),
                                 int(closed_by_bu), insufficient_history))

        if result > max_account_value:
            max_account_value = result

        drawdown = round(max_account_value - result, 2)
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        open_models.append(OpenModel(
            entry_lvl - min_dist_betw_orders * atr,
            entry_lvl + min_dist_betw_orders * atr,
            current_time,
            mod_idx + 1
        ))

        if len(recently_opened_models) >= 10:
            recently_opened_models.pop(0)
        recently_opened_models.append(model_number)

        if ext_info:
            if len(result_list) == 0 or result != result_list[-1]:
                result_list.append(result)
            model_indexes.append(mod_idx + 1)
            models_id.append(ModelID(model_time, level_time, direction))
            model_stops.append(stops)
            model_takes.append(int(have_take))
            model_bus.append(int(closed_by_bu))

    ratio = result / max_drawdown if max_drawdown != 0 else 0
    fitness = result

    if ext_info:
        ret = fitness, ratio, models, max_drawdown, max_account_value, models_list, result_list, trade_list
    else:
        ret = [fitness, ratio, models]
    return ret


@cuda.jit(debug=True)
def evaluate_gpu(params):
    db_indivs, db_results, spread, trading_params, non_trading_params = params
    entry_dist, stop, stop_out, take, bu, bu_cond, min_dist_betw_orders, max_stops, exp, min_tim_betw_odrs = trading_params
    # setting parameters
    result = 0
    models = 0
    max_drawdown = 0
    max_account_value = 0

    open_models_index = 0
    open_models = []  # list with OpenModel objects
    recently_opened_models_index = 0
    recently_opened_models = []  # list of model numbers

    # cuda.syncthreads()
    for mod_idx in range(len(db_indivs)):
        model = db_indivs[mod_idx]
        qt = db_results[mod_idx]

        fits = True
        atr, level, direction, model_number, model_time = qt.atr, qt.level, qt.direction, qt.model_number, qt.time
        open_ord_cond_shift = -spread if direction else 0  # (-spread) if buy, (0) otherwise
        close_ord_cond_shift = 0 if direction else -spread  # (0) if buy, (-spread) otherwise

        is_pending = True
        is_opened = False
        stops = 0
        placed_bu = False
        have_take = False
        closed_by_bu = False
        time_after_stop = math.inf  # initiates with math.inf to pass (time_after_stop < min_tim_betw_odrs) condition and place first order
        time_after_order = 0  # time passed after placing an order and before opening it
        if direction:  # long model
            entry_lvl = round(level + entry_dist * atr, 5)
            bu_lvl = entry_lvl + bu * atr
            bu_condition_level = entry_lvl + bu_cond * atr
            take_lvl = entry_lvl + take * atr
            stop_lvl = entry_lvl - stop * atr
            stop_out_lvl = entry_lvl - stop_out * atr
        else:  # short model
            entry_lvl = level - entry_dist * atr
            bu_lvl = entry_lvl - bu * atr
            bu_condition_level = entry_lvl - bu_cond * atr
            take_lvl = entry_lvl - take * atr
            stop_lvl = entry_lvl + stop * atr
            stop_out_lvl = entry_lvl + stop_out * atr

        # bringing buffers up to date
        for i in range(open_models_index):
            if model_time >= open_models[i].close_time:
                open_models_index -= 1

        # checking if this model should be counted
        for i in range(recently_opened_models_index):
            if model_number == recently_opened_models[i]:
                fits = False

        for i in range(open_models_index):
            if open_models[i].range_start <= entry_lvl <= open_models[i].range_end:
                fits = False

        for gene_idx in range(len(model)):
            attr = model[gene_idx]
            if non_trading_params[gene_idx][0] > attr or non_trading_params[gene_idx][1] < attr:
                fits = False
                break

        if not fits:
            continue

        models += 1
        # evaluating model
        close_time = model_time  # time when all model orders are closed
        for idx in range(len(qt.highs)):
            close_time += 60
            if is_opened:
                if not placed_bu and qt.lows[idx] <= bu_condition_level <= qt.highs[idx]:
                    stop_lvl = bu_lvl
                    placed_bu = True
                elif qt.lows[idx] <= stop_lvl + close_ord_cond_shift <= qt.highs[idx]:
                    if placed_bu:
                        closed_by_bu = True
                        break
                    stops += 1
                    if stops >= max_stops:
                        break
                    is_opened = False
                    time_after_stop = 0
                    time_after_order = 0
                elif qt.lows[idx] <= take_lvl + close_ord_cond_shift <= qt.highs[idx]:
                    have_take = True
                    break
            else:
                if time_after_stop < min_tim_betw_odrs:  # if 'True' then we can't open new order right now
                    time_after_stop += 1
                    continue

                if time_after_order >= exp:
                    break

                if qt.lows[idx] <= entry_lvl + open_ord_cond_shift <= qt.highs[idx]:
                    is_opened = True
                else:
                    time_after_order += 1

        result -= stops * stop
        if have_take:
            result += take
        elif closed_by_bu:
            result += bu
        result = round(result, 1)

        if result > max_account_value:
            max_account_value = result

        drawdown = round(max_account_value - result, 2)
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        open_models[open_models_index] = OpenModel(
            entry_lvl - min_dist_betw_orders * atr,
            entry_lvl + min_dist_betw_orders * atr,
            close_time,
            mod_idx + 1
        )
        open_models_index += 1

        if recently_opened_models_index >= 10:
            raise Exception("recently_opened_models len >= 10")
            # np.delete(recently_opened_models, 0)
        recently_opened_models[recently_opened_models_index] = model_number
        recently_opened_models_index += 1
    fitness = result / max_drawdown if max_drawdown != 0 else 0

    # cuda.syncthreads()
    return result, models, max_drawdown, max_account_value, fitness
