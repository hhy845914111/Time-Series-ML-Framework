"""

Notice: python>=3.6, dicts are ordered

"""
from hyperopt import hp

MAX_EVALS = 200
SINGLE_FIT_MAX_TIME = 60 * 1

PIPELINE_OBJ = {
    "test_id": 1,
    "feature_generators": {
        1: {
            "type": "EMAGenerator", "config": {"lag": 5}
        },
        2: {
            "type": "EMAGenerator", "config": {"lag": 10}
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
            "type": "LR", "config": {}
        },
        {
            "type": "SGDRegress", "config": {
                "loss": hp.choice("sdg_loss", ("squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive")),
                "penalty": hp.choice("sdg_penalty", ("none", "l2", "l1", "elasticnet"))
            }
        },
        {
            "type": "KNRegress", "config": {
                "n_neighbors": hp.choice("kn_n_neighbors", range(5, 55, 5)),
            }
        },
        {
            "type": "GPRegress", "config": {}
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
