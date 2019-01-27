from typing import Dict


class Reporter(object):

    from os.path import join as p_join
    from os.path import exists as os_exists
    from os import makedirs as os_makedirs
    import sqlite3

    def __init__(self, config_dct: Dict):
        self._config_dct = config_dct
        self._reporter_path = self._config_dct["reporter_path"]

        self._conn = Reporter.sqlite3.connect(Reporter.p_join(self._reporter_path, "result.db"))
        self._cursor = self._conn.cursor()
        self.create_db_if_not_exists()

    def create_db_if_not_exists(self):
        raise NotImplementedError()

    def add_one_account(self):
        raise NotImplementedError()

    def save_result(self, result_obj: Dict) -> None:
        self._conn.close()


class MeanWinRatioReporter(Reporter):

    from matplotlib import pyplot as plt
    from numpy import mean as np_mean

    def __init__(self, config_dct):
        super(MeanWinRatioReporter, self).__init__(config_dct)
        self._size = self._config_dct["pic_size"]

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

    def save_result(self, result_obj):
        win_ratio_ar = result_obj["ratio_ar"]
        date_ar = result_obj["date_ar"]
        test_id = result_obj["test_id"]
        this_path = MeanWinRatioReporter.p_join(self._reporter_path, "pics")

        if not MeanWinRatioReporter.os_exists(this_path):
            MeanWinRatioReporter.os_makedirs(this_path)

        value = MeanWinRatioReporter.np_mean(win_ratio_ar)

        self.add_one_account(test_id, value)

        fig = MeanWinRatioReporter.plt.figure(figsize=self._size)
        MeanWinRatioReporter.plt.plot(date_ar, win_ratio_ar)
        fig.savefig(MeanWinRatioReporter.p_join(this_path, f"{test_id}_{value}.png"))

        self._conn.close()


if __name__ == "__main__":
    import numpy as np

    config_dct = {"reporter_path": "./results/mean_win_ratio", "pic_size": (20, 10)}

    mw_reporter = MeanWinRatioReporter(config_dct)
    mw_reporter.save_result({"ratio_ar": np.random.rand(1000, 1),
                             "date_ar": np.arange(1000),
                             "test_id": 1
        }
    )
