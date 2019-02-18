"""

Notice: python>=3.6, dicts are ordered

"""
from hyperopt import hp

MAX_EVALS = 20

PIPELINE_OBJ = {
    "test_id": 1,
    "feature_generators": {
        1: {
            "type": "EMAGenerator", "config": {"lag": 5}
        },
        2: {
            "type": "EMAGenerator", "config": {"lag": 10}
        },
        1: {
            "type": "AbnormalDataFiller", "config": {}
        }
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 20, "predict_period": 1}

    },
    "learning_model": hp.choice("learning_model", (
        {
            "type": "SVR", "config": {"kernel": hp.choice("kernel", ("linear", "poly",)), "degree": hp.choice("degree", range(1, 3)), "max_iter": 1000000}
        },
        {
            "type": "AdaBoostRegress", "config": {}
        },
        {
            "type": "BaggingRegress", "config": {}
        },
        {
            "type": "ExtraTreesRegress", "config": {}
        },
        {
            "type": "GradientBoostingRegress", "config": {}
        },
        {
            "type": "RandomForestRegress", "config": {}
        },
    )),
    "judge": {
        "type": "ICJudge", "config": {"ic_lag": 0, "figsize": (20, 10)},
    },
    "others": {"raw_data_file": "toy_data.hdf"}
}
