from typing import Dict
from numpy import ndarray, float64


class Estimator(object):

    def __init__(self, config_dct: Dict):
        self._config_dct = config_dct

    def fit(self, x: ndarray, y: ndarray) -> None:
        pass

    def predict(self, x: ndarray) -> ndarray:
        pass

    def score(self, x: ndarray, y: ndarray) -> float64:
        pass


if __name__ == "__main__":
    from sklearn import svm
    from sklearn.datasets import samples_generator
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    from sklearn.pipeline import Pipeline

    # generate data to play with
    x, y = samples_generator.make_classification(
        n_informative=5, n_redundant=0, random_state=42)

    class AnovaSVM(Estimator):

        def __init__(self, config_dct):
            super(AnovaSVM, self).__init__(config_dct)

            anova_filter = SelectKBest(f_regression, k=5)
            clf = svm.SVC(kernel=self._config_dct["kernel"])
            self.model = Pipeline([('anova', anova_filter), ('svc', clf)], memory=None)
            self.model.set_params(anova__k=self._config_dct["anova__k"], svc__C=self._config_dct["svc__C"])

        def fit(self, x, y):
            self.model.fit(x, y)

        def predict(self, x):
            return self.model.predict(x)

        def score(self, x, y):
            return self.model.score(x, y)


    a_svm = AnovaSVM({"kernel": "linear", "anova__k": 10, "svc__C": 0.1})
    a_svm.fit(x, y)
    print(a_svm.predict(x))
    print(a_svm.score(x, y))

