from pandas import read_msgpack
from numpy import zeros as np_zeros

from configure import PIPELINE_CONFIG
from feature_generators import *
from learning_models import *
from iterators import *



class ModelSelector(object):

    def __init__(self, config_dct=PIPELINE_CONFIG):
        self._config_dct = config_dct
        self._raw_df = None

    def _load_data(self):
        tdf = read_msgpack(self._config_dct["others"]["raw_data_file"])
        if isinstance(tdf, list):
            raise IOError("load raw data error")
        return tdf

    def run(self):
        # TODO: use cache to speed up
        self._raw_df = self._load_data()

        fg_dct = self._config_dct["feature_generators"]

        # generate customized features
        for fg in fg_dct.values():
            this_fg = eval(fg["type"])(config_dct=fg["config"])
            this_fg.generate_all(self._raw_df)

        dt_dct = self._config_dct["iterator"]
        data_iter = eval(dt_dct["type"])(config_dct=dt_dct["config"])

        jg_dct = self._config_dct["judge"]
        lm_dct = self._config_dct["learning_models"]
        for lm in lm_dct.values():
            this_lm = eval(lm["type"])(config_dct=lm["config"])

            judge = eval(jg_dct["type"])(config_dct=jg_dct["config"])
            for x_train, y_train, x_test, y_test in data_iter:
                this_lm.fit(x_train, y_train)
                y_predict = this_lm.predict(x_test)
                judge.score(y_predict, y_test)
            judge.save_report()
