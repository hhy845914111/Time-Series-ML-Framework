"""

Notice: python>=3.6, dicts are ordered

"""

PIPELINE_CONFIG = {
    "pipeline_name": "toy_pipeline",
    "feature_generators": {
        1: {
            "type": "EMAGenerator", "config": {"lag": 10}
        },
        2: {
            "type": "EMAGenerator", "config": {"lag": 20}
        },
        3: {
            "type": "AutoEncoder", "config": {}
        }
    },
    "iterator": {
        "type": "DayByDay", "config": {}

    },
    "estimators": {
        1: {
            "type": "BayesianRidgeModel", "config": {"verbose": 1}
        },
        2: {
            "type": "BayesianRidgeModel", "config": {"verbose": 11}
        },
    },
    "judge": {
        "type": "IC_judge", "config": {}
    },
    "others": {}
}
