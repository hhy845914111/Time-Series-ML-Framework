from numpy import ndarray
from typing import Dict
from pandas import DataFrame


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
        raw_df.fillna(method="ffill", inplace=True)
        raw_df.fillna(0.0, inplace=True)

    def generate_one(self, x):
        pass
