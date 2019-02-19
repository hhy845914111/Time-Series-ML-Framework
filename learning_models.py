from typing import Dict
from numpy import ndarray


class LearningModel(object):

    def __init__(self, config_dct: Dict):
        self._config_dct = config_dct
        self._model = None

    def fit(self, x: ndarray, y: ndarray) -> None:
        self._model.fit(x, y)

    def predict(self, x: ndarray) -> ndarray:
        return self._model.predict(x)


class SVR(LearningModel):

    from sklearn.svm import SVR as model

    def __init__(self, config_dct):
        super(SVR, self).__init__(config_dct)
        self._model = SVR.model(**self._config_dct)


class LR(LearningModel):

    from sklearn.linear_model import LinearRegression as model

    def __init__(self, config_dct):
        super(LR, self).__init__(config_dct)
        self._model = LR.model(**self._config_dct)


class SGDRegress(LearningModel):

    from sklearn.linear_model import SGDRegressor as model

    def __init__(self, config_dct):
        super(SGDRegress, self).__init__(config_dct)
        self._model = SGDRegress.model(**self._config_dct)


class KNRegress(LearningModel):

    from sklearn.neighbors import KNeighborsRegressor as model

    def __init__(self, config_dct):
        super(KNRegress, self).__init__(config_dct)
        self._model = KNRegress.model(**self._config_dct)


class GPRegress(LearningModel):

    from sklearn.gaussian_process import GaussianProcessRegressor as model

    def __init__(self, config_dct):
        super(GPRegress, self).__init__(config_dct)
        self._model = GPRegress.model(**self._config_dct)


class AdaBoostRegress(LearningModel):

    from sklearn.ensemble import AdaBoostRegressor as model

    def __init__(self, config_dct):
        super(AdaBoostRegress, self).__init__(config_dct)
        self._model = AdaBoostRegress.model(**self._config_dct)


class BaggingRegress(LearningModel):

    from sklearn.ensemble import BaggingRegressor as model

    def __init__(self, config_dct):
        super(BaggingRegress, self).__init__(config_dct)
        self._model = BaggingRegress.model(**self._config_dct)


class ExtraTreesRegress(LearningModel):

    from sklearn.ensemble import ExtraTreesRegressor as model

    def __init__(self, config_dct):
        super(ExtraTreesRegress, self).__init__(config_dct)
        self._model = ExtraTreesRegress.model(**self._config_dct)


class GradientBoostingRegress(LearningModel):

    from sklearn.ensemble import GradientBoostingRegressor as model

    def __init__(self, config_dct):
        super(GradientBoostingRegress, self).__init__(config_dct)
        self._model = GradientBoostingRegress.model(**self._config_dct)


class RandomForestRegress(LearningModel):

    from sklearn.ensemble import RandomForestRegressor as model

    def __init__(self, config_dct):
        super(RandomForestRegress, self).__init__(config_dct)
        self._model = RandomForestRegress.model(**self._config_dct)


if __name__ == "__main__":
    pass
