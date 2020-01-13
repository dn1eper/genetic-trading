from loader import load_indivs, load_results
import pandas as pd
import numpy as np
import math
from numba import jitclass, float64, int64, boolean, types, jit
from numba import cuda
from collections import namedtuple
from multiprocessing import Pool

Quote = namedtuple("Quote", ["model_number", "time", "level", "atr", "direction", "highs", "lows"])
Model = namedtuple("Model", ["range_start", "range_end", "close_time"])
#FitnessResults = namedtuple("FitnessResult", ["score", "models", "max_drawdown", "max_account"])

class FitnessResult():
    def __init__(self, results):
        self.score = results[0]
        self.models = results[1]
        self.max_drawdown = results[2]
        self.max_account = results[3]
    
    def __str__(self):
        return "fitness: " + str(self.score) + " models: " + str(self.models) + " max account value: " + \
              str(self.max_account) + " max drawdown: " + str(self.max_drawdown)

class Fitness():
    def __init__(self, spread):
        self.db_indivs = load_indivs().to_numpy()
        self.spread = spread
        self.db_results = []

        for result in load_results():
            quotes = np.array([float(quote) for quote in result[5:]])

            self.db_results.append(Quote(
                int(result[0]),
                int(result[1]),
                float(result[2]), 
                float(result[3]),
                bool(result[4]), 
                quotes[:int(len(quotes)/2)], 
                quotes[int(len(quotes)/2):]
            ))
        
        #self.db_results = np.array(self.db_results)

    def calc(self, indivs: list, gpu=False, multithread=True):
        params = [ 
            (
                self.db_indivs,
                self.db_results,
                self.spread,
                np.array([gene.value() for gene in indiv.get("trading")]), 
                np.array([(gene.value() - gene.radius(), gene.value() + gene.radius()) for gene in indiv.get("trading", has_tag=False)])
            ) for indiv in indivs
        ]

        results = []

        if gpu:
            for indiv_idx in range(len(indivs)):
                results.append(evaluate_gpu(params[indiv_idx]))
        elif multithread:
            pool = Pool()
            results = pool.map(evaluate, params)
            pool.close()
            pool.join()
        else:
            for indiv_idx in range(len(indivs)):
                results.append(evaluate(params[indiv_idx]))
        

        for indiv_idx, indiv in enumerate(indivs):
            indiv.fitness = FitnessResult(results[indiv_idx])

def evaluate(params):
    db_indivs, db_results, spread, trading_params, non_trading_params = params
    entry_dist, stop, stop_out, take, bu, bu_cond, min_dist_betw_orders, max_stops, exp, min_tim_betw_odrs = trading_params
    # setting parameters
    result = 0
    models = 0
    max_drawdown = 0
    max_account_value = 0

    open_models = []            # list with OpenModel objects
    recently_opened_models = [] # list of model numbers
    for mod_idx in range(len(db_indivs)):
        model = db_indivs[mod_idx]
        qt = db_results[mod_idx]

        fits = True
        atr, level, direction, model_number, model_time = qt.atr, qt.level, qt.direction, qt.model_number, qt.time
        open_ord_cond_shift = -spread if direction else 0   # (-spread) if buy, (0) otherwise
        close_ord_cond_shift = 0 if direction else -spread  # (0) if buy, (-spread) otherwise

        is_pending = True
        is_opened = False
        stops = 0
        placed_bu = False
        have_take = False
        closed_by_bu = False
        time_after_stop = math.inf  # initiates with math.inf to pass (time_after_stop < min_tim_betw_odrs) condition and place first order
        time_after_order = 0        # time passed after placing an order and before opening it
        if direction:               # long model
            entry_lvl = level + entry_dist * atr
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
        
        open_models.append(Model(
            entry_lvl - min_dist_betw_orders,                         
            entry_lvl + min_dist_betw_orders, 
            close_time
        ))

        if len(recently_opened_models) >= 10:
            recently_opened_models.pop(0)
        recently_opened_models.append(model_number)

    return (result, models, max_drawdown, max_account_value)

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
    open_models = []            # list with OpenModel objects
    recently_opened_models_index = 0
    recently_opened_models = [] # list of model numbers

    #cuda.syncthreads()
    for mod_idx in range(len(db_indivs)):
        model = db_indivs[mod_idx]
        qt = db_results[mod_idx]

        fits = True
        atr, level, direction, model_number, model_time = qt.atr, qt.level, qt.direction, qt.model_number, qt.time
        open_ord_cond_shift = -spread if direction else 0   # (-spread) if buy, (0) otherwise
        close_ord_cond_shift = 0 if direction else -spread  # (0) if buy, (-spread) otherwise

        is_pending = True
        is_opened = False
        stops = 0
        placed_bu = False
        have_take = False
        closed_by_bu = False
        time_after_stop = math.inf  # initiates with math.inf to pass (time_after_stop < min_tim_betw_odrs) condition and place first order
        time_after_order = 0        # time passed after placing an order and before opening it
        if direction:               # long model
            entry_lvl = level + entry_dist * atr
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

        open_models[open_models_index] = Model(
            entry_lvl - min_dist_betw_orders,                         
            entry_lvl + min_dist_betw_orders, 
            close_time
        )
        open_models_index += 1

        if recently_opened_models_index >= 10:
            raise Exception("recently_opened_models len >= 10")
            #np.delete(recently_opened_models, 0)
        recently_opened_models[recently_opened_models_index] = model_number
        recently_opened_models_index += 1

        #cuda.syncthreads()
    return (result, models, max_drawdown, max_account_value)