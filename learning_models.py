from typing import Dict
from numpy import ndarray


class LearningModel(object):

    def __init__(self, config_dct: Dict):
        self._config_dct = config_dct

    def fit(self, x: ndarray, y: ndarray) -> None:
        pass

    def predict(self, x: ndarray) -> ndarray:
        pass


class SVR(LearningModel):

    from sklearn.svm import SVR as sl_SVR
    from sklearn.decomposition import PCA as sl_PCA

    def __init__(self, config_dct):
        super(SVR, self).__init__(config_dct)
        self._pca = SVR.sl_PCA(10)
        self.model = SVR.sl_SVR(**self._config_dct)

    def fit(self, x, y):
        x = self._pca.fit_transform(x)
        self.model.fit(x, y)

    def predict(self, x):
        x = self._pca.transform(x)
        return self.model.predict(x)


class LR(LearningModel):

    from sklearn.linear_model import LinearRegression as sl_LR

    def __init__(self, config_dct):
        super(LR, self).__init__(config_dct)

        self.model = LR.sl_LR(**self._config_dct)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class AdaBoostRegress(LearningModel):

    from sklearn.ensemble import AdaBoostRegressor as sl_abr

    def __init__(self, config_dct):
        super(AdaBoostRegress, self).__init__(config_dct)

        self.model = AdaBoostRegress.sl_abr(**self._config_dct)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class BaggingRegress(LearningModel):

    from sklearn.ensemble import BaggingRegressor as sl_bgr

    def __init__(self, config_dct):
        super(BaggingRegress, self).__init__(config_dct)

        self.model = BaggingRegress.sl_bgr(**self._config_dct)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class ExtraTreesRegress(LearningModel):

    from sklearn.ensemble import ExtraTreesRegressor as sl_etr

    def __init__(self, config_dct):
        super(ExtraTreesRegress, self).__init__(config_dct)

        self.model = ExtraTreesRegress.sl_etr(**self._config_dct)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class GradientBoostingRegress(LearningModel):

    from sklearn.ensemble import GradientBoostingRegressor as sl_gbr

    def __init__(self, config_dct):
        super(GradientBoostingRegress, self).__init__(config_dct)

        self.model = GradientBoostingRegress.sl_gbr(**self._config_dct)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class RandomForestRegress(LearningModel):

    from sklearn.ensemble import RandomForestRegressor as sl_rfr

    def __init__(self, config_dct):
        super(RandomForestRegress, self).__init__(config_dct)

        self.model = RandomForestRegress.sl_rfr(**self._config_dct)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


if __name__ == "__main__":
    pass
