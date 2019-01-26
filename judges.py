from typing import Dict
from numpy import float64 as np_float64
from numpy import ndarray as np_ndarray


class Judge(object):

    def __init__(self, config_dct: Dict):
        self._config_dct = config_dct

    def score(self, y_predict: np_ndarray, y: np_ndarray) -> np_float64:
        pass


class ICJudge(Judge):

    from numpy import corrcoef as np_corrcoef

    def __init__(self, config_dct):
        super(ICJudge, self).__init__(config_dct)
        self._ic_lag = self._config_dct["ic_lag"]

    def score(self, y_predict, y):
        return ICJudge.np_corrcoef(y_predict[:-self._ic_lag].reshape(1, -1), y[self._ic_lag:].reshape(1, -1))[0, 1]


class WinRatioJudge(Judge):

    from numpy import where as np_where
    from numpy import sum as np_sum

    def score(self, y_predict, y):
        return WinRatioJudge.np_sum(WinRatioJudge.np_where(y * y_predict > 0, 1, 0)) / len(y)


class MeanSquareJudge(Judge):

    def __init__(self, config_dct):
        super(MeanSquareJudge, self).__init__(config_dct)
        self._judge_norm = self._config_dct["judge_norm"]

    def score(self, y_predict, y):
        diff = y_predict - y
        return np.float64(diff.T.dot(diff))


if __name__ == "__main__":
    import numpy as np

    y_target = np.random.rand(10000, 1)
    y_test = y_target[1:]
    y_target = y_target[:-1]
    ic_judge = ICJudge({"ic_lag": 1})
    print(ic_judge.score(y_test, y_target))

    y_target = y_target
    y_test = -y_target
    wr_judge = WinRatioJudge({})
    print(wr_judge.score(y_test, y_target))

    y_target = y_target
    y_test = y_target / 2
    ms_judge = MeanSquareJudge({"judge_norm": 2})
    print(ms_judge.score(y_test, y_target))
