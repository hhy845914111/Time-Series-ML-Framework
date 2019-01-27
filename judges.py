from typing import Dict
from numpy import ndarray as np_ndarray
from numpy import float as np_float


class Judge(object):

    from os.path import join as p_join
    import sqlite3
    from os.path import exists as os_exists
    from os import makedirs as os_makedirs
    REPORT_PATH = "./results"

    def __init__(self, test_id: str, config_dct: Dict):
        """
        :param config_dct: required parametes:
        None
        """
        self._config_dct = config_dct

        self._test_id = test_id
        self._result_lst = []

        self._this_path = Judge.p_join(Judge.REPORT_PATH, self._test_id)
        if not Judge.os_exists(self._this_path):
            Judge.os_makedirs(self._this_path)

        self._conn = Judge.sqlite3.connect(Judge.p_join(self._this_path, "result.db"))
        self._cursor = self._conn.cursor()
        self._create_db_if_not_exists()

    def _calc_score(self, y_predict: np_ndarray, y: np_ndarray) -> np_float:
        raise NotImplementedError()

    def _create_db_if_not_exists(self) -> None:
        raise NotImplementedError()

    def _add_one_account(self) -> None:
        raise NotImplementedError()

    def score(self, y_predict: np_ndarray, y: np_ndarray) -> None:
        self._result_lst.append(self._calc_score(y_predict, y))

    def save_result(self) -> None:
        self._conn.close()


class ICJudge(Judge):

    from numpy import corrcoef as np_corrcoef
    from numpy import array as np_array
    from numpy import mean as np_mean
    from matplotlib import pyplot as plt

    def __init__(self, date_ar: np_ndarray, test_id: str, config_dct: Dict):
        """
        :param date_ar:
        :param config_dct: required paramiters:
        figsize: Tuple
        ic_lag: int
        """
        super(ICJudge, self).__init__(test_id, config_dct)
        self._date_ar = date_ar
        self._ic_lag = self._config_dct["ic_lag"]
        self._fig = ICJudge.plt.figure(figsize=self._config_dct["figsize"])

    def _calc_score(self, y_predict: np_ndarray, y: np_ndarray) -> np_float:
        return ICJudge.np_corrcoef(
            y_predict[:-self._ic_lag].reshape(1, -1),
            y[self._ic_lag:].reshape(1, -1))[0, 1]

    def _create_db_if_not_exists(self) -> None:
        self._cursor.execute(
            """CREATE TABLE IF NOT EXISTS mean_win_ratio
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL,
            ratio REAL NOT NULL);"""
        )
        self._conn.commit()

    def _add_one_account(self, account_id: str, value: np_float) -> None:
        self._cursor.execute(
            f"""INSERT INTO mean_win_ratio (account_id, ratio) VALUES('{account_id}', {value});"""
        )
        self._conn.commit()

    def save_result(self):
        result_ar = ICJudge.np_array(self._result_lst)
        value = ICJudge.np_mean(result_ar)

        self._add_one_account(self._test_id, value)
        self._fig.clear()
        ICJudge.plt.plot(self._date_ar, result_ar)
        self._fig.savefig(ICJudge.p_join(ICJudge.REPORT_PATH, "pics", f"{self._test_id}_{value}.png"))
        self._conn.close()
