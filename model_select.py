from typing import Dict
from pandas import read_msgpack
from hashlib import md5
from os.path import join as p_join
from os.path import exists as os_exists
from tqdm import tqdm
from hyperopt import fmin as hp_fmin
from hyperopt import tpe as hp_tpe
from hyperopt import space_eval as hp_space_eval
#from hyperopt.mongoexp import MongoTrials

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
        #trials = MongoTrials("mongo://localhost:27017/local/jobs")
        best = hp_fmin(
            self.run,
            space=self._param_range_obj,
            algo=hp_tpe.suggest,
            max_evals=self._max_evals,
            verbose=2,
           #trials=trials
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
            ModelSelector.CACHE_FOLDER, t_md5.hexdigest() + ".msg"
        )

        # 1. read cache or generate df from raw_df
        '''
        if os_exists(cache_file):
            save_cache = False
            print("Using cached DataFrame...")
            raw_df = read_msgpack(cache_file)

        else:
            save_cache = True
            print("Generating feature DataFrame...")
            raw_df = read_msgpack(
                p_join(ModelSelector.DATA_FOLDER,
                       config_dct["others"]["raw_data_file"])
            )
            fg_dct = config_dct["feature_generators"]

            # generate customized features
            for fg in tqdm(fg_dct.values()):
                this_fg = eval(fg["type"])(config_dct=fg["config"])
                this_fg.generate_all(raw_df)

            raw_df.to_msgpack(cache_file)
        '''
        print("Generating feature DataFrame...")
        raw_df = read_msgpack(
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

        #if save_cache:
        #    data_iter.get_data_df_with_y().to_msgpack(cache_file)

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
