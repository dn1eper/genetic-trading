from loader import load_indivs, load_results
from queue import Queue
import pandas as pd
import math

class OpenModel:
    def __init__(self, range_start, range_end, close_time):
        """
        :param range_start: start of range in witch other models can't be opened till 'close_term'
        :param range_end: end of range in witch other models can't be opened till 'close_term'
        """
        self.range_start = range_start
        self.range_end = range_end
        self.close_time = close_time

class Evaluation:
    def __init__(self, spread):
        self._def_db_indivs = load_indivs()
        self._def_db_results = load_results()
        if not isinstance(self._def_db_indivs, pd.DataFrame) or not isinstance(self._def_db_results, list):
            raise ValueError("Problem with loading indivs or results file")
        if self._def_db_indivs.empty or not self._def_db_results:
            raise ValueError("No items in indivs or results object")
        self._spread = spread

    def evaluate(self, individual):
        # print('entered evaluation.evaluate')
        result = 0
        db_indivs = self._def_db_indivs
        db_results = self._def_db_results
        spread = self._spread
        # setting parameters
        models = 0
        max_drawdown = 0
        max_account_value = 0

        open_models = []  # list with OpenModel objects
        recently_opened_models = Queue(maxsize=10)  # list of model numbers
        for mod_idx, model in db_indivs.iterrows():
            fits = True
            qt = db_results[mod_idx]
            atr, level, direction, model_number = qt.atr, qt.level, qt.direction, qt.model_number
            model_time, model_number = qt.time, qt.model_number
            open_ord_cond_shift = -spread if direction else 0  # (-spread) if buy, (0) otherwise
            close_ord_cond_shift = 0 if direction else -spread  # (0) if buy, (-spread) otherwise
            entry_dist, stop, stop_out, take, bu, bu_cond, min_dist_betw_orders, max_stops, exp, min_tim_betw_odrs = \
                [gene.value() for gene in individual.get("trading")]

            is_pending = True
            is_opened = False
            stops = 0
            placed_bu = False
            have_take = False
            closed_by_bu = False
            time_after_stop = math.inf  # initiates with math.inf to pass (time_after_stop < min_tim_betw_odrs) condition and place first order
            time_after_order = 0  # time passed after placing an order and before opening it
            if direction:  # long model
                # print('level =', level, 'entry_dist =', entry_dist, 'atr =', atr)
                # print('level =', type(level), 'entry_dist =', type(entry_dist), 'atr =', type(atr))

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
                    # print("Removed old open model")

            # checking if this model should be counted
            for opened_model in list(recently_opened_models.queue):
                if model_number == opened_model:
                    fits = False
                    # print('model number is present in recently_opened_models')

            for open_model in open_models:
                if open_model.range_start <= entry_lvl <= open_model.range_end:
                    fits = False
                    # print('another model is to close to the open one')

            for gene_idx, attr in enumerate(model):
                if (individual[gene_idx].type is bool and attr != individual[gene_idx]) or \
                    (individual[gene_idx].type is not bool and
                    (attr < individual[gene_idx].range_start() or attr > individual[gene_idx].range_end())):
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

            open_models.append(OpenModel(range_start=entry_lvl - min_dist_betw_orders,
                                         range_end=entry_lvl + min_dist_betw_orders, close_time=close_time))
            if recently_opened_models.full():
                recently_opened_models.get(timeout=1)
            recently_opened_models.put(model_number)

        # print('exited evaluation.evaluate')
        individual.score = result
        individual.models = models
        individual.max_drawdown = max_drawdown
        individual.max_account = max_account_value
        return result


def evaluate(individual, db_indivs, db_results, spread):
    """
    :param db_indivs: pandas object with the list of historical individuals
    :param db_results: pandas object with 'Quotes' objects should be in global scope
    """
    result = 0
    # setting parameters
    open_models = []  # list with OpenModel objects
    recently_opened_models = Queue(maxsize=10)  # list of model numbers
    for mod_idx, model in db_indivs.iterrows():
        fits = True
        qt = db_results[mod_idx]
        atr, level, direction, model_number = qt.atr, qt.level, qt.direction, qt.model_number
        model_time, model_number = qt.time, qt.model_number
        open_ord_cond_shift = -spread if direction else 0  # (-spread) if buy, (0) otherwise
        close_ord_cond_shift = 0 if direction else -spread  # (0) if buy, (-spread) otherwise
        entry_dist, stop, stop_out, take, bu, bu_cond, min_dist_betw_orders, max_stops, exp, min_tim_betw_odrs = \
            [gene.value() for gene in individual.get("trading")]

        is_pending = True
        is_opened = False
        stops = 0
        placed_bu = False
        have_take = False
        closed_by_bu = False
        time_after_stop = math.inf  # initiates with math.inf to pass (time_after_stop < min_tim_betw_odrs) condition and place first order
        time_after_order = 0  # time passed after placing an order and before opening it
        if direction:  # long model
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
        for opened_model in list(recently_opened_models.queue):
            if model_number == opened_model:
                fits = False
                # print('model number is present in recently_opened_models')

        for open_model in open_models:
            if open_model.range_start <= entry_lvl <= open_model.range_end:
                fits = False
                # print('model is to close to the open one')

        for gene_idx, attr in enumerate(model):
            if (individual[gene_idx].type is bool and attr != individual[gene_idx]) or \
                    (individual[gene_idx].type is not bool and
                     (attr < individual[gene_idx].range_start() or attr > individual[gene_idx].range_end())):
                fits = False
                break
        if not fits:
            continue

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

        open_models.append(OpenModel(range_start=entry_lvl - min_dist_betw_orders,
                                     range_end=entry_lvl + min_dist_betw_orders, close_time=close_time))
        if recently_opened_models.full():
            recently_opened_models.get(timeout=1)
        recently_opened_models.put(model_number)

    return result


