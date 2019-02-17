from typing import Dict
from numpy import ndarray as np_ndarray
from numpy import float as np_float


class Judge(object):

    def __init__(self, test_id, config_dct: Dict):
        self._config_dct = config_dct
        self._test_id = test_id
        self._result_lst = []

    def _calc_score(self, y_predict: np_ndarray, y: np_ndarray) -> np_float:
        raise NotImplementedError()

    def score(self, y_predict: np_ndarray, y: np_ndarray) -> None:
        self._result_lst.append(self._calc_score(y_predict, y))

    def get_result(self):
        raise NotImplementedError()


class Saving(object):

    import sqlite3
    from os.path import exists as os_exists
    from os import makedirs as os_makedirs
    from os.path import join as p_join
    REPORT_PATH = "./results"

    def __init__(self, test_id: str):
        """
        :param config_dct: required parametes:
        None
        """
        self._test_id = test_id

        self._this_path = Saving.p_join(Saving.REPORT_PATH, self._test_id)
        if not Saving.os_exists(self._this_path):
            Saving.os_makedirs(self._this_path)

        self._conn = Saving.sqlite3.connect(Saving.p_join(self._this_path, "result.db"))
        self._cursor = self._conn.cursor()
        self._create_db_if_not_exists()

    def _create_db_if_not_exists(self) -> None:
        raise NotImplementedError()

    def _add_one_account(self) -> None:
        raise NotImplementedError()

    def get_result(self) -> None:
        self._conn.close()


class ICJudge(Judge):

    from numpy import corrcoef as np_corrcoef
    from numpy import float as np_float
    from numpy import mean as np_mean

    def __init__(self, test_id, config_dct):
        super(ICJudge, self).__init__(test_id, config_dct)
        self._ic_lag = self._config_dct["ic_lag"]

    def _calc_score(self, y_predict: np_ndarray, y: np_ndarray) -> np_float:
        return ICJudge.np_float(ICJudge.np_corrcoef(
            y_predict[:-self._ic_lag].reshape(1, -1),
            y[self._ic_lag:].reshape(1, -1))[0, 1])

    def get_result(self):
        return ICJudge.np_float(ICJudge.np_mean(self._result_lst))


class ICJudgeAndSave(ICJudge, Saving):

    from numpy import array as np_array
    from numpy import mean as np_mean
    from matplotlib import pyplot as plt

    def __init__(self, test_id: str, config_dct: Dict):
        """
        :param date_ar:
        :param config_dct: required paramiters:
        figsize: Tuple
        ic_lag: int
        """
        ICJudge.__init__(self, test_id, config_dct)
        Saving.__init__(self, test_id)

        self._ic_lag = self._config_dct["ic_lag"]
        self._fig = ICJudgeAndSave.plt.figure(figsize=self._config_dct["figsize"])

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

    def get_result(self):
        value = super(ICJudgeAndSave, self).get_result()

        self._add_one_account(self._test_id, value)
        self._fig.clear()
        ICJudgeAndSave.plt.plot(self._result_lst)
        self._fig.savefig(ICJudgeAndSave.p_join(ICJudgeAndSave.REPORT_PATH, "pics", f"{self._test_id}_{value}.png"))
        self._conn.close()
