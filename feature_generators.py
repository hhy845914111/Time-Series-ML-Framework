from numpy import ndarray
from typing import Dict
from pandas import DataFrame
from configure import DATE_COL_NAME
from tqdm import tqdm


class FeatureGenerator(object):

    def __init__(self, config_dct: Dict):
        self._config_dct = config_dct

    def generate_all(self, raw_df: DataFrame) -> None:
        """
        :param raw_df: raw_dataframe to be added with new columns of features
        :return: nothing, add features inplace
        """
        pass

    def generate_one(self, x: ndarray) -> ndarray:
        """
        To generate one feature to be added to the raw_df
        :param x: ndarray, can be one column or multiple columns
        :return: ndarray, dimension 1
        """
        pass


class EMAGenerator(FeatureGenerator):

    from talib import EMA as ta_EMA

    def __init__(self, config_dct):
        super(EMAGenerator, self).__init__(config_dct)
        self._lag = self._config_dct["lag"]

    def generate_all(self, raw_df):
        for col_name in raw_df:
            if str(col_name)[:2] != "g_":  # to prevent generate feature on automated features
                try:
                    raw_df[f"g_{col_name}_EMA_{self._lag}"] = self.generate_one(raw_df[col_name].values)
                except:
                    continue

    def generate_one(self, x):
        return EMAGenerator.ta_EMA(x, self._lag)


class AbnormalDataFiller(FeatureGenerator):

    def __init__(self, config_dct):
        super(AbnormalDataFiller, self).__init__(config_dct)

    def generate_all(self, raw_df):
        return raw_df.dropna()

    def generate_one(self, x):
        pass


class RollingMinMaxScaler(FeatureGenerator):

    from sklearn.preprocessing import MinMaxScaler

    def __init__(self, config_dct):
        super(RollingMinMaxScaler, self).__init__(config_dct)
        self._window_size = config_dct["window_size"]
        self._target_lst = config_dct["target_lst"]
        self._model = RollingMinMaxScaler.MinMaxScaler(copy=True)

    def generate_all(self, raw_df):
        date_lst = raw_df[DATE_COL_NAME].unique()
        date_lst.sort()

        for i in tqdm(range(self._window_size, len(date_lst), 1)):
            train_mat = raw_df.loc[
                (raw_df[DATE_COL_NAME] >= date_lst[i - self._window_size]) & (raw_df[DATE_COL_NAME] < date_lst[i]), self._target_lst].values
            self._model.fit(train_mat)

            change_set = raw_df.loc[raw_df[DATE_COL_NAME] == date_lst[i], self._target_lst].values
            raw_df.loc[raw_df[DATE_COL_NAME] == date_lst[i], self._target_lst] = self._model.transform(change_set)

        return raw_df


class OneHotTransformer(FeatureGenerator):

    from sklearn.preprocessing import OneHotEncoder
    from pandas import DataFrame as pd_dataframe
    from pandas import concat as pd_concat

    def __init__(self, config_dct):
        super(OneHotTransformer, self).__init__(config_dct)
        self._model = OneHotTransformer.OneHotEncoder(sparse=False)
        self._target_lst = config_dct["target_lst"]

    def generate_all(self, raw_df):
        for col_n in tqdm(self._target_lst):
            x = raw_df[col_n].astype(str).values
            x_trans = self._model.fit_transform(x.reshape(-1, 1))
            tdf = OneHotTransformer.pd_dataframe(data=x_trans, columns=self._model.get_feature_names((col_n, )), index=raw_df.index)
            raw_df = OneHotTransformer.pd_concat((raw_df, tdf), axis=1)
            del raw_df[col_n]

        return raw_df

