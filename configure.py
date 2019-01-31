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
        "type": "DFIterator", "config": {"sample_lag": 10, "predict_period": 1}

    },
    "learning_model": hp.choice("model", (
        {
            "type": "SVR", "config": {"kernel": hp.choice("SVR_kernel", ("rbf", "linear", "poly")), "degree": hp.choice("SRV_degree", range(5))}
        },
    )),
    "judge": {
        "test_type": "ICJudge", "config": {"ic_lag": 1},
        "product_type": "ICJudgeAndSave", "config": {"ic_lag": 1, "figsize": (20, 10)}
    },
    "others": {"raw_data_file": "toy_data.msg"}
}
