from pandas import read_hdf
from hashlib import md5
from os.path import join as p_join
from os.path import exists as os_exists
from tqdm import tqdm
from hyperopt import fmin as hp_fmin
from hyperopt import tpe as hp_tpe
from hyperopt import space_eval as hp_space_eval

from configure import *
from feature_generators import *
from learning_models import *
from iterators import *
from judges import *


class ModelSelector(object):

    DATA_FOLDER = "./data"
    CACHE_FOLDER = "./cache"

    def __init__(self, param_range_obj, max_evals=MAX_EVALS):
        self._param_range_obj = param_range_obj
        self._max_evals = max_evals

    def optimize(self):
        best = hp_fmin(
            self.run,
            space=self._param_range_obj,
            algo=hp_tpe.suggest,
            max_evals=self._max_evals,
            verbose=2,
        )
        optimized_param_dct = self._param_range_obj.copy()
        best = hp_space_eval(self._param_range_obj, best)
        optimized_param_dct["learning_model"] = best["learning_model"]

        ModelSelector.run(optimized_param_dct, False)

    def run(self, config_dct):
        raise NotImplementedError()


class MPModelSelector(ModelSelector):

    from os import cpu_count
    from multiprocessing import Queue, Event, Process
    from multiprocessing import queues as mpq
    from time import time
    from numpy import inf as np_inf
    from functools import reduce

    def __init__(self, param_range_obj, process_count=None, q_size=None, max_evals=MAX_EVALS, single_fit_max_time=SINGLE_FIT_MAX_TIME):
        super(MPModelSelector, self).__init__(param_range_obj, max_evals)
        self._p_count = MPModelSelector.cpu_count() - 1 if process_count is None else process_count
        self._q_size = self._p_count if q_size is None else q_size
        self._max_time = single_fit_max_time

    def optimize(self):
        best = hp_fmin(
            self.run,
            space=self._param_range_obj,
            algo=hp_tpe.suggest,
            max_evals=self._max_evals,
            verbose=2,
        )
        best = hp_space_eval(self._param_range_obj, best)
        return best

    @staticmethod
    def _get_learning_curve(model, X_train, y_train, X_test, y_test, step=2):
        total_len = len(y_train)
        rst_lst = []
        for i in range(1, total_len + step, step):
            model.fit(X_train[:i, :], y_train[:i])
            rst_lst.append((i, model.score(X_train[:i, :], y_train[:i]), model.score(X_test, y_test)))
        return rst_lst

    @staticmethod
    def _work_func(in_queue, out_queue, close_event, curve_event):
        while not close_event.is_set() or not in_queue.empty():
            try:
                idx, X_train, y_train, X_test, y_test, tkr_name, lm_dct, is_curve_epoch = in_queue.get_nowait()
            except MPModelSelector.mpq.Empty:
                continue

            this_lm = eval(lm_dct["type"])(config_dct=lm_dct["config"])

            if is_curve_epoch:
                curve_event.set()
                try:
                    rst_lst = MPModelSelector._get_learning_curve(this_lm, X_train, y_train, X_test, y_test)
                except ValueError: # problems in fitting
                    out_queue.put(ValueError("fit failed"))
                    continue

            else:
                rst_lst = []

            try:
                this_lm.fit(X_train, y_train)
            except ValueError:
                out_queue.put(ValueError("fit failed"))
                continue

            y_predict = this_lm.predict(X_test).reshape(-1, 1)

            out_queue.put((idx, y_predict, y_test, tkr_name, rst_lst))
            curve_event.clear()

    def run(self, config_dct):
        t_md5 = md5()
        data_info_dct = {"feature_generators": config_dct["feature_generators"],
                         "iterator__predict_period": config_dct["iterator"]["config"]["predict_period"]
                         }
        t_md5.update(str(data_info_dct).encode("utf-8"))
        cache_file = p_join(
            ModelSelector.CACHE_FOLDER, t_md5.hexdigest() + ".hdf"
        )

        # 1. read cache or generate df from raw_df
        if os_exists(cache_file):
            save_cache = False
            print("Using cached DataFrame...")
            raw_df = read_hdf(cache_file)

        else:
            save_cache = True
            print("Generating feature DataFrame...")
            raw_df = read_hdf(
                p_join(ModelSelector.DATA_FOLDER,
                       config_dct["others"]["raw_data_file"])
            )
            fg_dct = config_dct["feature_generators"]

            # generate customized features
            for fg in tqdm(fg_dct.values()):
                this_fg = eval(fg["type"])(config_dct=fg["config"])
                raw_df = this_fg.generate_all(raw_df)

            raw_df.to_hdf(cache_file, "data")

        # 2. get iterator of data, create training target
        dt_dct = config_dct["iterator"]
        #data_iter = eval(dt_dct["type"])(raw_df, config_dct=dt_dct["config"], generate_target=save_cache)
        data_iter = eval(dt_dct["type"])(raw_df, config_dct=dt_dct["config"], generate_target=True)

        if save_cache:
            data_iter.get_data_df_with_y().to_hdf(cache_file, "data")

        # 3. get judge and learning algorithms; train, predict and evaluate
        jg_dct = config_dct["judge"]

        judge = eval(jg_dct["type"])(
            config_dct=jg_dct["config"]
        )

        in_queue = MPModelSelector.Queue(maxsize=self._q_size)
        out_queue = MPModelSelector.Queue(maxsize=self._q_size)
        close_event = MPModelSelector.Event()
        cur_event_lst = [MPModelSelector.Event() for _ in range(self._p_count)]
        p_lst = [
            MPModelSelector.Process(
                target=MPModelSelector._work_func,
                args=(in_queue, out_queue, close_event, cur_event_lst[i])
            ) for i in range(self._p_count)
        ]
        [p.start() for p in p_lst]

        lm_dct = config_dct["learning_model"]
        arg_lst = ((idx, *ctt, lm_dct, idx in LEARNING_CURVE_LST) for idx, ctt in enumerate(data_iter))

        total_len = 0
        last_start = MPModelSelector.time()
        for arg in tqdm(arg_lst):
            try:
                obj = out_queue.get_nowait()

                if isinstance(obj, ValueError):
                    total_len -= 1
                else:
                    idx, y_predict, y_test, tkr_name, curve_lst = obj
                    judge.add_score(y_predict, y_test, tkr_name, idx, curve_lst)
                    total_len -= 1
                """
                idx, y_predict, y_test, tkr_name, curve_lst = obj
                judge.add_score(y_predict, y_test, tkr_name, idx, curve_lst)
                total_len -= 1
                """
            except MPModelSelector.mpq.Empty:
                pass

            while 1:
                try:
                    in_queue.put_nowait(arg)
                    last_start = MPModelSelector.time()
                    total_len += 1
                    break
                except MPModelSelector.mpq.Full:
                    if MPModelSelector.reduce(lambda x, y: x or y.is_set(), cur_event_lst):
                        last_start = MPModelSelector.time()
                    else:
                        if MPModelSelector.time() - last_start > SINGLE_FIT_MAX_TIME:
                            [p.terminate() for p in p_lst]
                            print("Too slow, terminate: ", config_dct)
                            return -MPModelSelector.np_inf

        close_event.set()

        while total_len > 0:
            try:
                obj = out_queue.get_nowait()

                if isinstance(obj, ValueError):
                    total_len -= 1
                else:
                    idx, y_predict, y_test, tkr_name, curve_lst = obj
                    judge.add_score(y_predict, y_test, tkr_name, idx, curve_lst)
                    total_len -= 1
                """
                idx, y_predict, y_test, tkr_name, curve_lst = obj
                judge.add_score(y_predict, y_test, tkr_name, idx, curve_lst)
                total_len -= 1
                """
            except MPModelSelector.mpq.Empty:
                pass

        [p.join() for p in p_lst]

        judge.save_result(config_dct)
        return judge.get_result()
