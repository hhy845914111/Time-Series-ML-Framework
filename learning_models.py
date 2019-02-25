from typing import Dict
from numpy import ndarray
from numpy import float as np_float
from sklearn.utils import shuffle


class LearningModel(object):

    def __init__(self, config_dct: Dict):
        self._config_dct = config_dct
        self._model = None

    def fit(self, x: ndarray, y: ndarray) -> None:
        x, y = shuffle(x, y)
        self._model.fit(x, y)

    def predict(self, x: ndarray) -> ndarray:
        return self._model.predict(x)

    def score(self, x: ndarray, y: ndarray) -> np_float:
        return self._model.score(x, y)


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


class RidgeCVRegress(LearningModel):

    from sklearn.linear_model import RidgeCV as model

    def __init__(self, config_dct):
        super(RidgeCVRegress, self).__init__(config_dct)
        self._model = RidgeCVRegress.model(**self._config_dct)


class LassoCVRegress(LearningModel):

    from sklearn.linear_model import LassoCV as model

    def __init__(self, config_dct):
        super(LassoCVRegress, self).__init__(config_dct)
        self._model = LassoCVRegress.model(**self._config_dct)


class ElasticNetCVRegress(LearningModel):

    from sklearn.linear_model import ElasticNetCV as model

    def __init__(self, config_dct):
        super(ElasticNetCVRegress, self).__init__(config_dct)
        self._model = ElasticNetCVRegress.model(**self._config_dct)


class OMPCVRegress(LearningModel):

    from sklearn.linear_model import OrthogonalMatchingPursuitCV as model

    def __init__(self, config_dct):
        super(OMPCVRegress, self).__init__(config_dct)
        self._model = OMPCVRegress.model(**self._config_dct)


class BRidgeRegress(LearningModel):

    from sklearn.linear_model import BayesianRidge as model

    def __init__(self, config_dct):
        super(BRidgeRegress, self).__init__(config_dct)
        self._model = BRidgeRegress.model(**self._config_dct)


class ARDRegression(LearningModel):

    from sklearn.linear_model import ARDRegression as model

    def __init__(self, config_dct):
        super(ARDRegression, self).__init__(config_dct)
        self._model = ARDRegression.model(**self._config_dct)


class ARDRegress(LearningModel):

    from sklearn.linear_model import ARDRegression as model

    def __init__(self, config_dct):
        super(ARDRegress, self).__init__(config_dct)
        self._model = ARDRegress.model(**self._config_dct)


class PARegress(LearningModel):

    from sklearn.linear_model import PassiveAggressiveRegressor as model

    def __init__(self, config_dct):
        super(PARegress, self).__init__(config_dct)
        self._model = PARegress.model(**self._config_dct)


class HuberRegress(LearningModel):

    from sklearn.linear_model import HuberRegressor as model

    def __init__(self, config_dct):
        super(HuberRegress, self).__init__(config_dct)
        self._model = HuberRegress.model(**self._config_dct)


class RANSACRegress(LearningModel):

    from sklearn.linear_model import RANSACRegressor as model

    def __init__(self, config_dct):
        super(RANSACRegress, self).__init__(config_dct)
        self._model = RANSACRegress.model(**self._config_dct)


class TheilSenRegress(LearningModel):

    from sklearn.linear_model import TheilSenRegressor as model

    def __init__(self, config_dct):
        super(TheilSenRegress, self).__init__(config_dct)
        self._model = TheilSenRegress.model(**self._config_dct)
