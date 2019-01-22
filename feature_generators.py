from numpy import ndarray
from pandas import DataFrame
from typing import Dict


class FeatureGenerator(object):

    def __init__(self, config_dct: Dict):
        self._config_dct = config_dct

    def generate_all(self, raw_df: DataFrame) -> None:
        """
        :param raw_df: raw_dataframe to be added with new columns of features
        :return: nothing, add features inplace
        """

    def generate_one(self, x: ndarray) -> ndarray:
        """
        To generate one feature to be added to the raw_df
        :param x: ndarray, can be one column or multiple columns
        :return: ndarray, dimension 1
        """
        pass


if __name__ == "__main__":
    from talib import EMA
    import numpy as np

    class EMAGenerator(FeatureGenerator):
        """
        one example of features
        """
        def __init__(self, config_dct):
            super(EMAGenerator, self).__init__(config_dct)

        def generate_all(self, raw_df):
            for col_name in raw_df:
                raw_df[f"{col_name}_EMA"] = self.generate_one(raw_df[col_name].values)

        def generate_one(self, x):
            return EMA(x, self._config_dct["lag"])

    test_data = DataFrame(np.random.rand(2000, 10))

    ema_generator = EMAGenerator({"lag": 10})
    ema_generator.generate_all(test_data)
    print(test_data.shape)  # -> (2000, 20)
