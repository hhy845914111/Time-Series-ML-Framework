from typing import Dict, Iterable, Tuple
from numpy import ndarray as np_ndarray
from pandas import DataFrame as pd_DataFrame
from pandas import concat as pd_concat
from tqdm import tqdm
from configure import Y_COL_NAME, DATE_COL_NAME, TKR_COL_NAME


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

    def __init__(self, df, config_dct, generate_target):
        super(DFIterator, self).__init__(config_dct, generate_target)
        self._sample_lag = self._config_dct["sample_lag"]
        self._predict_period = self._config_dct["predict_period"]
        self._sample_generator = None
        self._df = df
        self._df["y"] = None
        self._feature_lst = []

        print("Generating training target...")
        if self._generate_target:
            self._g_target()
        print("Done.")

    def _g_target(self):
        df_tpl = [i[1] for i in self._df.groupby(TKR_COL_NAME)]
        for tdf in tqdm(df_tpl):
            yield_ar = tdf[Y_COL_NAME]
            tdf["y"] = yield_ar.shift(self._predict_period)

        self._df = pd_concat(df_tpl).dropna()
        self._feature_lst = self._df.columns.tolist()
        self._feature_lst.remove(TKR_COL_NAME)
        self._feature_lst.remove("y")
        self._feature_lst.remove(DATE_COL_NAME)

    def _get_sample_generator(self):
        df_lst = [i[1] for i in self._df.groupby(DATE_COL_NAME)]

        len_ = len(df_lst)
        for i in range(self._sample_lag, len_ - 1, 1):
            tdf = pd_concat(objs=df_lst[i - self._sample_lag:i])
            x_train = tdf.loc[:, self._feature_lst].values
            y_train = tdf.loc[:, "y"].values.reshape(-1, 1)

            test_df = df_lst[i + 1]
            x_test = test_df.loc[:, self._feature_lst].values
            y_test = test_df.loc[:, "y"].values.reshape(-1, 1)
            tkr_name = test_df[TKR_COL_NAME].values.reshape(-1, 1)

            yield x_train, y_train, x_test, y_test, tkr_name

    def __iter__(self):
        self._sample_generator = self._get_sample_generator()
        return self

    def __next__(self):
        return next(self._sample_generator)

    def get_data_df_with_y(self) -> pd_DataFrame:
        return self._df
