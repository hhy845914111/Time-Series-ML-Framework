"""

Notice: python>=3.6, dicts are ordered

"""

PIPELINE_CONFIG = {
    "test_id": 1,
    "feature_generators": {
        1: {
            "type": "EMAGenerator", "config": {"lag": 5}
        },
        2: {
            "type": "EMAGenerator", "config": {"lag": 10}
        },
        3: {
            "type": "AbnormalDataFiler", "config": {}
        }
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 10, "predict_period": 1}

    },
    "learning_models": {
        1: {
            "type": "LR", "config": {"n_jobs": -1}
        },
        2: {
            "type": "LR", "config": {"n_jobs": -1}
        },
    },
    "judge": {
        "type": "ICJudge", "config": {"ic_lag": 1, "figsize": (20, 10)}
    },
    "others": {"raw_data_file": "toy_data.msg"}
}
