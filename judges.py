from typing import Dict
from numpy import float64 as np_float64
from numpy import ndarray as np_ndarray


class Judge(object):

    from os.path import join as p_join
    import sqlite3

    def __init__(self, config_dct: Dict):
        self._config_dct = config_dct

        self._report_path = self._config_dct["report_path"]

        self._test_id = self._config_dct["test_id"]
        self._result_lst = []

        self._conn = Judge.sqlite3.connect(Judge.p_join(self._report_path, "result.db"))
        self._cursor = self._conn.cursor()
        self.create_db_if_not_exists()

    def score(self, y_predict: np_ndarray, y: np_ndarray) -> None:
        pass

    def create_db_if_not_exists(self):
        raise NotImplementedError()

    def add_one_account(self):
        raise NotImplementedError()

    def save_result(self) -> None:
        self._conn.close()


class ICJudge(Judge):

    from numpy import corrcoef as np_corrcoef
    from numpy import array as np_array
    from os.path import exists as os_exists
    from os import makedirs as os_makedirs
    from matplotlib import pyplot as plt

    def __init__(self, config_dct):
        super(ICJudge, self).__init__(config_dct)
        self._ic_lag = self._config_dct["ic_lag"]
        self._date_ar = self._config_dct["date_ar"]

    def calc_score(self):
        raise 

    def score(self, y_predict, y):
        self._result_lst.append(ICJudge.np_corrcoef(
            y_predict[:-self._ic_lag].reshape(1, -1),
            y[self._ic_lag:].reshape(1, -1))[0, 1]
                                )

    def create_db_if_not_exists(self):
        self._cursor.execute(
            """CREATE TABLE IF NOT EXISTS mean_win_ratio
            (id INT PRIMARY KEY NOT NULL,
            ratio FLOAT NOT NULL);"""
        )
        self._conn.commit()

    def add_one_account(self, account_id, value):
        self._cursor.execute(
            f"""INSERT INTO mean_win_ratio (id, ratio) VALUES({account_id}, {value});"""
        )
        self._conn.commit()

    def save_result(self):
        this_path = ICJudge.p_join(self._report_path, "pics")
        if not ICJudge.os_exists(this_path):
            ICJudge.os_makedirs(this_path)

        result_ar = ICJudge.np_array(self._result_lst)
        value = ICJudge.np_mean(result_ar)

        self.add_one_account(self._test_id, value)

        fig = ICJudge.plt.figure(figsize=self._size)
        ICJudge.plt.plot(self._date_ar, result_ar)
        fig.savefig(ICJudge.p_join(this_path, f"{self._test_id}_{value}.png"))

        self._conn.close()


class WinRatioJudge(Judge):

    from numpy import where as np_where
    from numpy import sum as np_sum

    def calc_score(self, y_predict, y):
        return WinRatioJudge.np_sum(WinRatioJudge.np_where(y * y_predict > 0, 1, 0)) / len(y)


class MeanSquareJudge(Judge):

    from numpy import float64 as np_float64

    def __init__(self, config_dct):
        super(MeanSquareJudge, self).__init__(config_dct)
        self._judge_norm = self._config_dct["judge_norm"]

    def score(self, y_predict, y):
        diff = y_predict - y
        return MeanSquareJudge.np_float64(diff.T.dot(diff))
