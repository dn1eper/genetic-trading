from loader import load_indivs, load_results, load_quotes_time, load_symbol_properties
import numpy as np
import pandas as pd
import math
from numba import float64, int64, boolean, types, jit, njit
from numba import cuda
from collections import namedtuple
from multiprocessing import Pool

Quote = namedtuple("Quote", ["model_number", "model_time", "level_time", "level", "atr", "direction", "highs", "lows", "time"])
Model = namedtuple("Model", ["range_start", "range_end", "close_time", "model_line"])

class FitnessResult:
    def __init__(self, results):
        self.fitness, self.ratio, self.models = results

    def __str__(self):
        return "fitness: {}, ratio: {} models: {}".format(self.fitness, self.ratio, self.models)


class Fitness:
    def __init__(self):
        self.digits, self.spread = load_symbol_properties()
        self.db_indivs = load_indivs().to_numpy()
        self.db_results = []
        self.quotes = []
        self.params = None
        self.quotes_time = [np.resize(np.array(quotes).astype(int), 400) for quotes in load_quotes_time()]
        
        for result in load_results():
            self.quotes.append(np.array(result[6:]).astype(float))
            self.db_results.append(np.array(result[:6]).astype(float))

        self.db_results = np.array(self.db_results)
        self.quotes = np.array(self.quotes)
        self.quotes_time = np.array(self.quotes_time)

    def calc(self, indivs: list, gpu, multithreaded, compile):
        trading_params = []
        nontrading_params = []
        for indiv in indivs:
            trading_params.append( np.array([gene.value() for gene in indiv.get("trading")]) )
            nontrading_params.append( np.array([[gene.value() - gene.radius(), gene.value() + gene.radius()] for gene in indiv.get("trading", has_tag=False)]) )
        
        results = []

        if compile:
            for indiv_idx in range(len(indivs)):
                results.append(evaluate_compile(self.db_indivs, self.db_results, self.quotes, self.quotes_time, trading_params[indiv_idx], nontrading_params[indiv_idx], self.spread, self.digits))
        elif multithreaded:
            pool = Pool()
            results = pool.map(evaluate, [self.db_indivs, self.db_results, self.quotes, self.quotes_time, trading_params[indiv_idx][indiv_idx], nontrading_params[indiv_idx], self.spread, self.digits])
            pool.close()
            pool.join()
        elif gpu:
            raise Exception("Gpu not implemented")
        else:
            for indiv_idx in range(len(indivs)):
                results.append(evaluate([self.db_indivs, self.db_results, self.quotes, self.quotes_time, trading_params[indiv_idx], nontrading_params[indiv_idx], self.spread, self.digits]))

        for indiv_idx, indiv in enumerate(indivs):
            indiv.fitness = FitnessResult(results[indiv_idx])


def evaluate(params):
    db_indivs, db_results, quotes, quotes_time, trading_params, non_trading_params, spread, digits = params
    entry_dist, stop, stop_out, take, bu, bu_cond, min_dist_betw_orders, max_stops, exp, min_tim_betw_odrs = trading_params
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
        model_number, model_time, level_time, level, atr, direction = db_results[mod_idx]
        quote = quotes[mod_idx]
        highs = quote[:int(len(quote) / 2)]
        lows = quote[int(len(quote) / 2):]
        time = quotes_time[mod_idx]
        fits = True
        

        open_ord_cond_shift = -spread if direction else 0  # (-spread) if buy, (0) otherwise
        close_ord_cond_shift = 0 if direction else -spread  # (0) if buy, (-spread) otherwise

        placed_order = False
        is_opened = False  # order is in trade now
        exp_time = math.inf
        stops = 0
        placed_bu = False
        have_take = False
        closed_by_bu = False
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

        # checking if this model should be counted
        for opened_model in recently_opened_models:
            if model_number == opened_model:
                fits = False

        for open_model in open_models:
            if open_model.range_start <= entry_lvl <= open_model.range_end:
                fits = False

        for gene_idx in range(len(model)):
            attr = model[gene_idx]
            if non_trading_params[gene_idx][0] > attr or non_trading_params[gene_idx][1] < attr:
                fits = False
                break

        if not fits:
            continue

        # evaluating model
        models += 1
        price_is_up = False
        if highs[0] > entry_lvl + open_ord_cond_shift and lows[0] > entry_lvl + open_ord_cond_shift:
            price_is_up = True
        current_time = time[0]

        for idx in range(len(highs)):
            current_time = time[idx]
            time_after_prev_order += 1

            if (direction and lows[idx] <= stop_out_lvl) or (not direction and highs[idx] >= stop_out_lvl):
                if is_opened:
                    stops += 1
                break

            if is_opened:
                if use_bu and not placed_bu and \
                        ((direction and highs[idx] >= bu_condition_level) or (
                                not direction and lows[idx] <= bu_condition_level)):
                    stop_lvl = bu_lvl
                    placed_bu = True
                elif (direction and lows[idx] <= stop_lvl + close_ord_cond_shift) or (
                        not direction and highs[idx] >= stop_lvl + close_ord_cond_shift):
                    if placed_bu:
                        closed_by_bu = True
                        break
                    stops += 1
                    if stops >= max_stops:
                        break

                    is_opened = False
                    time_after_order = 0
                    placed_order = False
                    if highs[idx] > entry_lvl + open_ord_cond_shift and lows[idx] > entry_lvl + open_ord_cond_shift:
                        price_is_up = True
                    else:
                        price_is_up = False
                elif (direction and highs[idx] >= take_lvl + close_ord_cond_shift) or (
                        not direction and lows[idx] <= take_lvl + close_ord_cond_shift):
                    have_take = True
                    break

            if not is_opened:
                if not placed_order and time_after_prev_order < min_tim_betw_odrs:  # if 'True' then we can't open new order right now
                    continue
                if not placed_order:
                    time_after_prev_order = 0
                    placed_order = True

                if current_time >= exp_time:
                    break

                if (price_is_up and lows[idx] <= entry_lvl + open_ord_cond_shift) or \
                        (not price_is_up and highs[idx] >= entry_lvl + open_ord_cond_shift):
                    is_opened = True
                else:
                    time_after_order += 1

        result -= stops * stop
        if have_take:
            result += take
        elif closed_by_bu:
            result += bu
        result = round(result, 2)

        if result > max_account_value:
            max_account_value = result

        drawdown = round(max_account_value - result, 2)
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        open_models.append(Model(
            entry_lvl - min_dist_betw_orders * atr,
            entry_lvl + min_dist_betw_orders * atr,
            current_time,
            mod_idx + 1
        ))

        if len(recently_opened_models) >= 10:
            recently_opened_models.pop(0)
        recently_opened_models.append(model_number)

    ratio = result / max_drawdown if max_drawdown != 0 else 0
    fitness = result

    return fitness, ratio, models


@njit
def evaluate_compile(db_indivs, db_results, quotes, quotes_time, trading_params, non_trading_params, spread, digits):
    entry_dist, stop, stop_out, take, bu, bu_cond, min_dist_betw_orders, max_stops, exp, min_tim_betw_odrs = trading_params
    #trade_list = []
    use_bu = True
    if bu > bu_cond:
        use_bu = False
    # setting parameters
    result = 0
    # result_list = []
    # orders = 0
    # trades = 0
    models = 0
    # model_indexes = []
    # models_id = []
    # model_stops = []
    # model_takes = []
    # model_bus = []
    max_drawdown = 0
    max_account_value = 0

    open_models = []  # list with OpenModel objects
    recently_opened_models = []  # list of model numbers
    for mod_idx in range(len(db_indivs)):
        model = db_indivs[mod_idx]
        model_number, model_time, level_time, level, atr, direction = db_results[mod_idx]
        quote = quotes[mod_idx]
        highs = quote[:int(len(quote) / 2)]
        lows = quote[int(len(quote) / 2):]
        time = quotes_time[mod_idx]
        fits = True
        

        open_ord_cond_shift = -spread if direction else 0  # (-spread) if buy, (0) otherwise
        close_ord_cond_shift = 0 if direction else -spread  # (0) if buy, (-spread) otherwise

        placed_order = False
        is_opened = False  # order is in trade now
        exp_time = math.inf
        stops = 0
        placed_bu = False
        have_take = False
        closed_by_bu = False
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

        # checking if this model should be counted
        for opened_model in recently_opened_models:
            if model_number == opened_model:
                fits = False

        for open_model in open_models:
            if open_model.range_start <= entry_lvl <= open_model.range_end:
                fits = False

        for gene_idx in range(len(model)):
            attr = model[gene_idx]
            if non_trading_params[gene_idx][0] > attr or non_trading_params[gene_idx][1] < attr:
                fits = False
                break

        if not fits:
            continue

        # evaluating model
        models += 1
        price_is_up = False
        if highs[0] > entry_lvl + open_ord_cond_shift and lows[0] > entry_lvl + open_ord_cond_shift:
            price_is_up = True
        current_time = time[0]

        for idx in range(len(highs)):
            current_time = time[idx]
            time_after_prev_order += 1

            if (direction and lows[idx] <= stop_out_lvl) or (not direction and highs[idx] >= stop_out_lvl):
                if is_opened:
                    stops += 1
                break

            if is_opened:
                if use_bu and not placed_bu and \
                        ((direction and highs[idx] >= bu_condition_level) or (
                                not direction and lows[idx] <= bu_condition_level)):
                    stop_lvl = bu_lvl
                    placed_bu = True
                elif (direction and lows[idx] <= stop_lvl + close_ord_cond_shift) or (
                        not direction and highs[idx] >= stop_lvl + close_ord_cond_shift):
                    if placed_bu:
                        closed_by_bu = True
                        break
                    stops += 1
                    if stops >= max_stops:
                        break

                    is_opened = False
                    time_after_order = 0
                    placed_order = False
                    if highs[idx] > entry_lvl + open_ord_cond_shift and lows[idx] > entry_lvl + open_ord_cond_shift:
                        price_is_up = True
                    else:
                        price_is_up = False
                elif (direction and highs[idx] >= take_lvl + close_ord_cond_shift) or (
                        not direction and lows[idx] <= take_lvl + close_ord_cond_shift):
                    have_take = True
                    break

            if not is_opened:
                if not placed_order and time_after_prev_order < min_tim_betw_odrs:  # if 'True' then we can't open new order right now
                    continue
                if not placed_order:
                    time_after_prev_order = 0
                    placed_order = True

                if current_time >= exp_time:
                    break

                if (price_is_up and lows[idx] <= entry_lvl + open_ord_cond_shift) or \
                        (not price_is_up and highs[idx] >= entry_lvl + open_ord_cond_shift):
                    is_opened = True
                else:
                    time_after_order += 1

        result -= stops * stop
        if have_take:
            result += take
        elif closed_by_bu:
            result += bu
        result = round(result, 2)

        if result > max_account_value:
            max_account_value = result

        drawdown = round(max_account_value - result, 2)
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        open_models.append(Model(
            entry_lvl - min_dist_betw_orders * atr,
            entry_lvl + min_dist_betw_orders * atr,
            current_time,
            mod_idx + 1
        ))

        if len(recently_opened_models) >= 10:
            recently_opened_models.pop(0)
        recently_opened_models.append(model_number)

    ratio = result / max_drawdown if max_drawdown != 0 else 0
    fitness = result

    return fitness, ratio, models
