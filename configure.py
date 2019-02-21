"""

Notice: python>=3.6, dicts are ordered

"""
from hyperopt import hp
from sklearn.gaussian_process import kernels as gk

MAX_EVALS = 200
SINGLE_FIT_MAX_TIME = 60 * 1
Y_COL_NAME = "Yield_real"
TKR_COL_NAME = "ISIN number"

PIPELINE_OBJ_1 = {
    "test_id": 1,
    "feature_generators": {
        1: {
            "type": "RollingMinMaxScaler", "config": {"window_size": 100, "target_lst": []}
        },
        2: {
            "type": "OneHotTransformer", "config": {"target_lst": []}
        },
        3: {
            "type": "AbnormalDataFiller", "config": {}
        }
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 20, "predict_period": 1}

    },
    "learning_model": hp.choice("learning_model", (
        {
            "type": "SVR", "config": {"kernel": hp.choice("svr_kernel", ("linear", "poly", "rbf", "sigmoid")), "degree": hp.choice("degree", range(1, 5)), "max_iter": 1000000}
        },
        {
            "type": "AdaBoostRegress", "config": {
                "n_estimators": hp.choice("adr_n_estimators", range(50, 850, 50)),
                "loss": hp.choice("adr_loss", ("linear", "square", "exponential")),
            }
        },
        {
            "type": "BaggingRegress", "config": {
                "n_estimators": hp.choice("br_n_estimators", range(10, 850, 50)),
            }
        },
        {
            "type": "ExtraTreesRegress", "config": {
                "n_estimators": hp.choice("etr_n_estimators", range(10, 850, 50)),
                "max_depth": hp.choice("etr_max_depth", range(3, 20, 2))
            }
        },
    )),
    "judge": {
        "type": "ICJudge", "config": {"ic_lag": 0, "figsize": (20, 10)},
    },
    "others": {"raw_data_file": "toy_data.hdf"}
}

PIPELINE_OBJ_2 = {
    "test_id": 2,
    "feature_generators": {
        1: {
            "type": "RollingMinMaxScaler", "config": {"window_size": 100, "target_lst": []}
        },
        2: {
            "type": "OneHotTransformer", "config": {"target_lst": []}
        },
        3: {
            "type": "AbnormalDataFiller", "config": {}
        }
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 20, "predict_period": 1}

    },
    "learning_model": hp.choice("learning_model", (
        {
            "type": "GPRegress", "config": {
                "kernel": hp.choice("gp_kernel", (
                        gk.ConstantKernel(), gk.DotProduct(), gk.ExpSineSquared(), gk.Matern(), gk.PairwiseKernel(),
                        gk.RBF(), gk.RationalQuadratic(), gk.WhiteKernel(),
                    )
                )
            }
        },
    )),
    "judge": {
        "type": "ICJudge", "config": {"ic_lag": 0, "figsize": (20, 10)},
    },
    "others": {"raw_data_file": "toy_data.hdf"}
}

PIPELINE_OBJ_3 = {
    "test_id": 3,
    "feature_generators": {
        1: {
            "type": "RollingMinMaxScaler", "config": {"window_size": 100, "target_lst": []}
        },
        2: {
            "type": "OneHotTransformer", "config": {"target_lst": []}
        },
        3: {
            "type": "AbnormalDataFiller", "config": {}
        }
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 20, "predict_period": 1}

    },
    "learning_model": hp.choice("learning_model", (
        {
            "type": "KNRegress", "config": {
                "weights": hp.choice("kn_weights", ("uniform", "distance")),
                "n_neighbors": hp.choice("kn_n_neighbors", range(5, 55, 5)),
            }
        },
        {
            "type": "SGDRegress", "config": {
                "loss": hp.choice("sdg_loss",
                              ("squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive")),
                "penalty": hp.choice("sdg_penalty", ("none", "l2", "l1", "elasticnet"))
            }
        },
    )),
    "judge": {
        "type": "ICJudge", "config": {"ic_lag": 0, "figsize": (20, 10)},
    },
    "others": {"raw_data_file": "toy_data.hdf"}
}

PIPELINE_OBJ_4 = {
    "test_id": 4,
    "feature_generators": {
        1: {
            "type": "RollingMinMaxScaler", "config": {"window_size": 100, "target_lst": []}
        },
        2: {
            "type": "OneHotTransformer", "config": {"target_lst": []}
        },
        3: {
            "type": "AbnormalDataFiller", "config": {}
        }
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 20, "predict_period": 1}

    },
    "learning_model": hp.choice("learning_model", (
        {
            "type": "GradientBoostingRegress", "config": {
                "loss": hp.choice("gdr_loss", ("ls", "lad", "huber", "quantile")),
                "n_estimators": hp.choice("gdr_n_estimators", range(100, 850, 50)),
                "max_depth": hp.choice("gdr_max_depth", range(3, 20, 2))
            }
        },
        {
            "type": "RandomForestRegress", "config": {
                "n_estimators": hp.choice("rfr_n_estimators", range(10, 850, 50)),
                "max_depth": hp.choice("rfr_max_depth", range(3, 20, 2)),
            }
        },
    )),
    "judge": {
        "type": "ICJudge", "config": {"ic_lag": 0, "figsize": (20, 10)},
    },
    "others": {"raw_data_file": "toy_data.hdf"}
}

PIPELINE_OBJ_5 = {
    "test_id": 5,
    "feature_generators": {
        1: {
            "type": "RollingMinMaxScaler", "config": {"window_size": 100, "target_lst": []}
        },
        2: {
            "type": "OneHotTransformer", "config": {"target_lst": []}
        },
        3: {
            "type": "AbnormalDataFiller", "config": {}
        }
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 20, "predict_period": 1}

    },
    "learning_model": hp.choice("learning_model", (
        {
            "type": "RidgeCVRegress", "config": {}
        },
        {
            "type": "LassoCVRegress", "config": {}
        },
        {
            "type": "ElasticNetCVRegress", "config": {}
        },
        {
            "type": "OMPRegressCV", "config": {}
        },
        {
            "type": "BRidgeRegress", "config": {}
        },
    )),
    "judge": {
        "type": "ICJudge", "config": {"ic_lag": 0, "figsize": (20, 10)},
    },
    "others": {"raw_data_file": "toy_data.hdf"}
}

PIPELINE_OBJ_6 = {
    "test_id": 6,
    "feature_generators": {
        1: {
            "type": "RollingMinMaxScaler", "config": {"window_size": 100, "target_lst": []}
        },
        2: {
            "type": "OneHotTransformer", "config": {"target_lst": []}
        },
        3: {
            "type": "AbnormalDataFiller", "config": {}
        }
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 20, "predict_period": 1}

    },
    "learning_model": hp.choice("learning_model", (
        {
            "type": "ARDRegress", "config": {}
        },
        {
            "type": "PARegress", "config": {}
        },
        {
            "type": "HuberRegress", "config": {}
        },
        {
            "type": "RANSACRegress", "config": {}
        },
        {
            "type": "TheilSenRegress", "config": {}
        },
    )),
    "judge": {
        "type": "ICJudge", "config": {"ic_lag": 0, "figsize": (20, 10)},
    },
    "others": {"raw_data_file": "toy_data.hdf"}
}
