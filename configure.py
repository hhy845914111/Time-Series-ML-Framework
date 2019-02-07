"""

Notice: python>=3.6, dicts are ordered

"""
from hyperopt import hp

MAX_EVALS = 10

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
        "type": "DFIterator", "config": {"sample_lag": 2, "predict_period": 1}

    },
    "learning_model": hp.choice("learning_model", (
        {
            "type": "SVR", "config": {"kernel": hp.choice("kernel", ("linear", "rbf", "poly")), "degree": hp.choice("degree", range(1, 3)), "max_iter": 100000}
        },
    )),
    "judge": {
        "test_type": "ICJudge", "config": {"ic_lag": 1},
        "product_type": "ICJudgeAndSave", "config": {"ic_lag": 1, "figsize": (20, 10)}
    },
    "others": {"raw_data_file": "toy_data.msg"}
}
