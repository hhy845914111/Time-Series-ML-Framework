from typing import Dict
from numpy import ndarray as np_ndarray
from numpy import float as np_float
from numpy import nan as np_nan

from configure import DATE_COL_NAME, TKR_COL_NAME

from numpy import corrcoef as np_corrcoef
from pandas import DataFrame as pd_DataFrame
from os.path import join as p_join
from os.path import exists as os_exists
from os import makedirs as os_makedirs
from json import dumps as js_dump
from numpy import hstack as np_hstack
from matplotlib import pyplot as plt


class Judge(object):

    def __init__(self, config_dct: Dict):
        self._config_dct = config_dct

    def calc_score(self, y_predict: np_ndarray, y: np_ndarray) -> np_float:
        raise NotImplementedError()

    def add_score(self):
        raise NotImplementedError()

    def get_result(self):
        raise NotImplementedError()

    def save_result(self):
        raise NotImplementedError()


class ICJudge(Judge):

    REPORT_PATH = "./results"
    TEST_COUNT = 0

    def __init__(self, config_dct):
        super(ICJudge, self).__init__(config_dct)
        self._ic_lag = self._config_dct["ic_lag"]
        self._tdf = pd_DataFrame(data=None, columns=["ticker", "y_predict", "y_test", "date", "score"])
        ICJudge.TEST_COUNT += 1

        self._this_path = p_join(ICJudge.REPORT_PATH, str(ICJudge.TEST_COUNT))
        if not os_exists(self._this_path):
            os_makedirs(self._this_path)

    def calc_score(self, y_predict: np_ndarray, y: np_ndarray) -> np_float:
        if self._ic_lag == 0:
            try:
                return np_float(np_corrcoef(y_predict.T, y.T)[0, 1])
            except AttributeError: # only 1 predict
                return np_nan
        else:
            try:
                return np_float(np_corrcoef(y_predict[:-self._ic_lag].T, y[self._ic_lag:].T)[0, 1])
            except AttributeError:
                return np_nan

    def add_score(self, y_predict, y_test, ticker_ar, date, curve_lst):
        tdf = pd_DataFrame(
            data=np_hstack((ticker_ar, y_predict, y_test)),
            columns=["ticker", "y_predict", "y_test"]
        )
        tdf["date"] = date
        tdf["score"] = self.calc_score(y_predict, y_test)
        self._tdf = self._tdf.append(tdf)

        if len(curve_lst) > 0:
            tdf2 = pd_DataFrame(data=curve_lst, columns=["sample_size", "train", "validate"])
            tdf2.to_csv(p_join(ICJudge.REPORT_PATH, str(ICJudge.TEST_COUNT), f"curve_{date}.csv"), index=False)
            fig = plt.figure(figsize=self._config_dct["figsize"])
            ax = fig.add_subplot(111)
            ax.plot(tdf2["sample_size"].values, tdf2["train"].values, "r")
            ax.plot(tdf2["sample_size"].values, tdf2["validate"].values, "b")
            fig.savefig(p_join(ICJudge.REPORT_PATH, "pics", f"{ICJudge.TEST_COUNT}_curve_{date}.png"))

    def get_result(self):
        return self._tdf["score"].mean()

    def save_result(self, total_config_dct):
        self._tdf.sort_values(["date", "ticker"], inplace=True)
        self._tdf.to_csv(p_join(self._this_path, "results.csv"))

        with open(p_join(self._this_path, "config.json"), "w") as fp:
            fp.write(js_dump(total_config_dct))
