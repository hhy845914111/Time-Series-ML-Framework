from pandas import read_msgpack
from hashlib import md5
from os.path import join as p_join
from os.path import exists as os_exists

from configure import PIPELINE_CONFIG
from feature_generators import *
from learning_models import *
from iterators import *
from judges import *
from tqdm import tqdm


class ModelSelector(object):

    DATA_FOLDER = "./data"
    CACHE_FOLDER = "./cache"

    def __init__(self, config_dct=PIPELINE_CONFIG):
        self._config_dct = config_dct
        self._raw_df = None

        self._m = md5()
        self._m.update(str(self._config_dct["feature_generators"]).encode("utf-8"))
        self._cache_file = p_join(
            ModelSelector.CACHE_FOLDER, self._m.hexdigest() + ".msg"
        )

    def run(self):
        # 1. read cache or generate df from raw_df
        if os_exists(self._cache_file):
            print("Using cached DataFrame...")
            self._raw_df = read_msgpack(self._cache_file)

        else:
            print("Generating feature DataFrame...")
            self._raw_df = read_msgpack(
                p_join(ModelSelector.DATA_FOLDER,
                       self._config_dct["others"]["raw_data_file"])
            )
            fg_dct = self._config_dct["feature_generators"]

            # generate customized features
            for fg in fg_dct.values():
                this_fg = eval(fg["type"])(config_dct=fg["config"])
                this_fg.generate_all(self._raw_df)

            self._raw_df.to_msgpack(self._cache_file)

        # 2. get iterator of data, create training target
        dt_dct = self._config_dct["iterator"]
        data_iter = eval(dt_dct["type"])(self._raw_df, config_dct=dt_dct["config"])
        date_ar = data_iter.get_dates()

        # 3. get judge and learning algorithms; train, predict and evaluate
        jg_dct = self._config_dct["judge"]
        lm_dct = self._config_dct["learning_models"]
        for idx, lm in lm_dct.items():
            this_lm = eval(lm["type"])(config_dct=lm["config"])

            judge = eval(jg_dct["type"])(
                date_ar=date_ar,
                test_id=str(self._config_dct["test_id"]) + '_' + str(idx),
                config_dct=jg_dct["config"]
            )
            for x_train, y_train, x_test, y_test in tqdm(data_iter):
                this_lm.fit(x_train, y_train)
                y_predict = this_lm.predict(x_test)
                judge.score(y_predict, y_test)
            judge.save_result()
