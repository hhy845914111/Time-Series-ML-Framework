from typing import Dict, Iterable, Tuple
from numpy import ndarray as np_ndarray
from pandas import DataFrame as pd_DataFrame


class DataIterator(object):

    def __init__(self, config_dct: Dict, generate_target: bool):
        self._config_dct = config_dct
        self._generate_target = generate_target

    def __iter__(self) -> Iterable:
        pass

    def __next__(self) -> Tuple[np_ndarray, np_ndarray, np_ndarray, np_ndarray]:
        pass

    def get_data_df_with_y(self) -> pd_DataFrame:
        pass

    def get_dates(self) -> np_ndarray:
        pass


class DFIterator(DataIterator):

    from pandas import concat as pd_concat
    from tqdm import tqdm

    def __init__(self, df, config_dct, generate_target):
        super(DFIterator, self).__init__(config_dct, generate_target)
        self._sample_lag = self._config_dct["sample_lag"]
        self._predict_period = self._config_dct["predict_period"]
        self._sample_generator = None
        self._df = df

        print("Generating training target...")
        if self._generate_target:
            self._g_target()
        print("Done.")

    def _g_target(self):
        df_tpl = [i[1] for i in self._df.groupby("tkr")]
        for tdf in DFIterator.tqdm(df_tpl):
            yield_ar = tdf["yield"]
            tdf["y"] = yield_ar.shift(self._predict_period)

        self._df = DFIterator.pd_concat(df_tpl).dropna()


    def _get_sample_generator(self):
        df_lst = [i[1] for i in self._df.groupby("date")]

        len_ = len(df_lst)
        for i in range(self._sample_lag, len_ - 1, 1):
            tdf = DFIterator.pd_concat(objs=df_lst[i - self._sample_lag:i])
            x_train = tdf.iloc[:, :-1].values
            y_train = tdf["y"].values.reshape(-1, 1)

            test_df = df_lst[i + 1]
            x_test = test_df.iloc[:, :-1].values
            y_test = test_df["y"].values.reshape(-1, 1)
            tkr_name = test_df["tkr"].values.reshape(-1, 1)

            yield x_train, y_train, x_test, y_test, tkr_name

    def __iter__(self):
        self._sample_generator = self._get_sample_generator()
        return self

    def __next__(self):
        return next(self._sample_generator)

    def get_data_df_with_y(self) -> pd_DataFrame:
        return self._df
