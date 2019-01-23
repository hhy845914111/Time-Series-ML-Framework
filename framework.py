from configure import PIPELINE_CONFIG
from pandas import DataFrame, Series

from feature_generators import *
from estimators import *


class ModelSelector(object):
    config = PIPELINE_CONFIG

    def __init__(self, x: DataFrame, y: Series):
        self._x = x
        self._y = y

    def run(self):
        fg_dct = ModelSelector.config["feature_generators"]

        # generate customized features
        for fg in fg_dct.values():
            this_fg = eval(fg["type"])(config_dct=fg["config"])
            this_fg.generate_all(self._x)

        # score models
        est_dct = ModelSelector.config["estimator"]
        estimator = eval(est_dct["type"])(est_dct["config"])
        estimator.fit(self._x, self._y)





