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

    def __init__(self, param_range_obj=PIPELINE_OBJ, max_evals=MAX_EVALS):
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

    @staticmethod
    def run(config_dct, test=True):
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
                this_fg.generate_all(raw_df)

            raw_df.to_hdf(cache_file, "data")

        print("Generating feature DataFrame...")
        raw_df = read_hdf(
            p_join(ModelSelector.DATA_FOLDER,
                   config_dct["others"]["raw_data_file"])
        )
        fg_dct = config_dct["feature_generators"]

        # generate customized features
        for fg in tqdm(fg_dct.values()):
            this_fg = eval(fg["type"])(config_dct=fg["config"])
            this_fg.generate_all(raw_df)
        
        # 2. get iterator of data, create training target
        dt_dct = config_dct["iterator"]
        data_iter = eval(dt_dct["type"])(raw_df, config_dct=dt_dct["config"], generate_target=True)

        if save_cache:
            data_iter.get_data_df_with_y().to_hdf(cache_file, "data")

        # 3. get judge and learning algorithms; train, predict and evaluate
        jg_dct = config_dct["judge"]
        lm_dct = config_dct["learning_model"]
        this_lm = eval(lm_dct["type"])(config_dct=lm_dct["config"])

        if test:
            judge = eval(jg_dct["test_type"])(
                test_id=str(config_dct["test_id"]),
                config_dct=jg_dct["config"]
            )
        else:
            judge = eval(jg_dct["product_type"])(
                test_id=str(config_dct["test_id"]),
                config_dct=jg_dct["config"]
            )
        for x_train, y_train, x_test, y_test in tqdm(data_iter):
            this_lm.fit(x_train, y_train)
            y_predict = this_lm.predict(x_test)
            judge.score(y_predict, y_test)

        return judge.get_result()


class MPModelSelector(ModelSelector):

    from os import cpu_count
    from multiprocessing import Queue, Event, Process
    from multiprocessing import queues as mpq
    from time import time
    from numpy import inf as np_inf

    def __init__(self, process_count=None, q_size=None, param_range_obj=PIPELINE_OBJ, max_evals=MAX_EVALS, single_fit_max_time=SINGLE_FIT_MAX_TIME):
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
    def _work_func(in_queue, out_queue, close_event):
        while not close_event.is_set() or not in_queue.empty():
            try:
                idx, X_train, y_train, X_test, y_test, tkr_name, lm_dct = in_queue.get_nowait()
            except MPModelSelector.mpq.Empty:
                continue

            this_lm = eval(lm_dct["type"])(config_dct=lm_dct["config"])
            this_lm.fit(X_train, y_train)

            y_predict = this_lm.predict(X_test).reshape(-1, 1)

            out_queue.put((idx, y_predict, y_test, tkr_name))

    def run(self, config_dct, test=True):
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
                this_fg.generate_all(raw_df)

            raw_df.to_hdf(cache_file, "data")

        # 2. get iterator of data, create training target
        dt_dct = config_dct["iterator"]
        data_iter = eval(dt_dct["type"])(raw_df, config_dct=dt_dct["config"], generate_target=save_cache)

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
        p_lst = [
            MPModelSelector.Process(
                target=MPModelSelector._work_func,
                args=(in_queue, out_queue, close_event)
            ) for _ in range(self._p_count)
        ]
        [p.start() for p in p_lst]

        lm_dct = config_dct["learning_model"]
        arg_lst = ((idx, *ctt, lm_dct) for idx, ctt in enumerate(data_iter))

        total_len = 0
        last_start = MPModelSelector.time()
        for arg in tqdm(arg_lst):
            try:
                idx, y_predict, y_test, tkr_name = out_queue.get_nowait()
                judge.add_score(y_predict, y_test, tkr_name, idx)
                total_len -= 1
            except MPModelSelector.mpq.Empty:
                pass

            while 1:
                try:
                    in_queue.put_nowait(arg)
                    total_len += 1
                    last_start = MPModelSelector.time()
                    break
                except MPModelSelector.mpq.Full:
                    if MPModelSelector.time() - last_start > self._max_time:
                        [p.terminate() for p in p_lst]
                        print("Too slow, terminate: ", config_dct)
                        return -MPModelSelector.np_inf

        close_event.set()

        while total_len > 0:
            try:
                idx, y_predict, y_test, tkr_name = out_queue.get_nowait()
                judge.add_score(y_predict, y_test, tkr_name, idx)
                total_len -= 1
            except MPModelSelector.mpq.Empty:
                pass

        [p.join() for p in p_lst]

        judge.save_result(config_dct)
        return judge.get_result()
