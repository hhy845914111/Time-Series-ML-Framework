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


if __name__ == "__main__":
    pass
