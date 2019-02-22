"""

Notice: python>=3.6, dicts are ordered

"""
from hyperopt import hp
from sklearn.gaussian_process import kernels as gk

MAX_EVALS = 200
SINGLE_FIT_MAX_TIME = 60 * 1
Y_COL_NAME = "Yield-lag-1"
TKR_COL_NAME = "ISIN number"
DATE_COL_NAME = "Date"
LEARNING_CURVE_LST = [i for i in range(100, 2904 + 100, 200)]


one_hot_lst = ["ISO Country Code", 'Industry Lvl 2 Desc', 'Industry Lvl 3 Desc', 'Industry Lvl 4 Desc', 'Subordination Type']

min_max_lst = ['Par Wtd Coupon',  'Face Value_LOC', 'Price',
       'Accrued Interest', '% Mkt Value', 'OAS', 'Asset Swap Spread',
       'Effective Yield', 'Yield to Worst (s.a.)',
       'Yield to Maturity (s.a.)', 'Effective Duration',
       'Mod. Dur to Worst (s.a.)', 'Modified Duration (s.a.)',
       'Effective Convexity', 'Convexity to Worst (s.a.)',
       'Convexity (s.a.)', '美元指数', 'M2增速',
       'CPI', 'PPI', 'repo rate（回购利率）', 'S&P 500', 'TED', 'GDP(对数）',
       'Bond Volatility(%)', '30 天波动率', '60天波动率', '90天波动率', '10天波动率',
       '200天波动率', '260天波动率', '180天波动率', '360天波动率', '20天波动率', '120天波动率',
       '150天波动率', '10年期国债指数', '5年期国债指数', '3年期国债指数', '2年期国债指数', '1年期国债指数',
       '工业增加值(Monthly%)', '美元指数-lag1', '美元指数-lag2', '美元指数-lag5',
       '美元指数-lag10', '美元指数-lag20', 'M2增速-lag1', 'M2增速-lag2', 'M2增速-lag5',
       'M2增速-lag10', 'M2增速-lag20', 'CPI-lag1', 'CPI-lag2', 'CPI-lag5',
       'CPI-lag10', 'CPI-lag20', 'PPI-lag1', 'PPI-lag2', 'PPI-lag5',
       'PPI-lag10', 'PPI-lag20', 'repo rate（回购利率）-lag1',
       'repo rate（回购利率）-lag2', 'repo rate（回购利率）-lag5',
       'repo rate（回购利率）-lag10', 'repo rate（回购利率）-lag20', 'S&P 500-lag1',
       'S&P 500-lag2', 'S&P 500-lag5', 'S&P 500-lag10', 'S&P 500-lag20',
       'TED-lag1', 'TED-lag2', 'TED-lag5', 'TED-lag10', 'TED-lag20',
       'GDP(对数）-lag1', 'GDP(对数）-lag2', 'GDP(对数）-lag5', 'GDP(对数）-lag10',
       'GDP(对数）-lag20', 'Bond Volatility(%)-lag1',
       'Bond Volatility(%)-lag2', 'Bond Volatility(%)-lag5',
       'Bond Volatility(%)-lag10', 'Bond Volatility(%)-lag20',
       '30 天波动率-lag1', '30 天波动率-lag2', '30 天波动率-lag5', '30 天波动率-lag10',
       '30 天波动率-lag20', '60天波动率-lag1', '60天波动率-lag2', '60天波动率-lag5',
       '60天波动率-lag10', '60天波动率-lag20', '90天波动率-lag1', '90天波动率-lag2',
       '90天波动率-lag5', '90天波动率-lag10', '90天波动率-lag20', '10天波动率-lag1',
       '10天波动率-lag2', '10天波动率-lag5', '10天波动率-lag10', '10天波动率-lag20',
       '200天波动率-lag1', '200天波动率-lag2', '200天波动率-lag5', '200天波动率-lag10',
       '200天波动率-lag20', '260天波动率-lag1', '260天波动率-lag2', '260天波动率-lag5',
       '260天波动率-lag10', '260天波动率-lag20', '180天波动率-lag1', '180天波动率-lag2',
       '180天波动率-lag5', '180天波动率-lag10', '180天波动率-lag20', '360天波动率-lag1',
       '360天波动率-lag2', '360天波动率-lag5', '360天波动率-lag10', '360天波动率-lag20',
       '20天波动率-lag1', '20天波动率-lag2', '20天波动率-lag5', '20天波动率-lag10',
       '20天波动率-lag20', '120天波动率-lag1', '120天波动率-lag2', '120天波动率-lag5',
       '120天波动率-lag10', '120天波动率-lag20', '150天波动率-lag1', '150天波动率-lag2',
       '150天波动率-lag5', '150天波动率-lag10', '150天波动率-lag20', '10年期国债指数-lag1',
       '10年期国债指数-lag2', '10年期国债指数-lag5', '10年期国债指数-lag10',
       '10年期国债指数-lag20', '5年期国债指数-lag1', '5年期国债指数-lag2', '5年期国债指数-lag5',
       '5年期国债指数-lag10', '5年期国债指数-lag20', '3年期国债指数-lag1', '3年期国债指数-lag2',
       '3年期国债指数-lag5', '3年期国债指数-lag10', '3年期国债指数-lag20', '2年期国债指数-lag1',
       '2年期国债指数-lag2', '2年期国债指数-lag5', '2年期国债指数-lag10', '2年期国债指数-lag20',
       '1年期国债指数-lag1', '1年期国债指数-lag2', '1年期国债指数-lag5', '1年期国债指数-lag10',
       '1年期国债指数-lag20', '工业增加值(Monthly%)-lag1', '工业增加值(Monthly%)-lag2',
       '工业增加值(Monthly%)-lag5', '工业增加值(Monthly%)-lag10',
       '工业增加值(Monthly%)-lag20', 'BAMLC0A1CAAAEY', 'DGS10', 'Spread',
       'BAMLC0A1CAAAEY-lag1', 'DGS10-lag1', 'BAMLC0A1CAAAEY-lag2',
       'DGS10-lag2', 'BAMLC0A1CAAAEY-lag5', 'DGS10-lag5',
       'BAMLC0A1CAAAEY-lag10', 'DGS10-lag10', 'BAMLC0A1CAAAEY-lag20',
       'DGS10-lag20', 'RETURN_ON_ASSET', 'RETURN_COM_EQY', 'FNCL_LVRG',
       'CUR_MKT_CAP', 'CUR_RATIO', 'QUICK_RATIO', 'BEST_ANALYST_RATING',
       'EBIT', 'RETURN_ON_ASSET-lag1', 'RETURN_ON_ASSET-lag2',
       'RETURN_ON_ASSET-lag5', 'RETURN_ON_ASSET-lag10',
       'RETURN_ON_ASSET-lag20', 'RETURN_COM_EQY-lag1',
       'RETURN_COM_EQY-lag2', 'RETURN_COM_EQY-lag5',
       'RETURN_COM_EQY-lag10', 'RETURN_COM_EQY-lag20', 'FNCL_LVRG-lag1',
       'FNCL_LVRG-lag2', 'FNCL_LVRG-lag5', 'FNCL_LVRG-lag10',
       'FNCL_LVRG-lag20', 'CUR_MKT_CAP-lag1', 'CUR_MKT_CAP-lag2',
       'CUR_MKT_CAP-lag5', 'CUR_MKT_CAP-lag10', 'CUR_MKT_CAP-lag20',
       'CUR_RATIO-lag1', 'CUR_RATIO-lag2', 'CUR_RATIO-lag5',
       'CUR_RATIO-lag10', 'CUR_RATIO-lag20', 'QUICK_RATIO-lag1',
       'QUICK_RATIO-lag2', 'QUICK_RATIO-lag5', 'QUICK_RATIO-lag10',
       'QUICK_RATIO-lag20', 'BEST_ANALYST_RATING-lag1',
       'BEST_ANALYST_RATING-lag2', 'BEST_ANALYST_RATING-lag5',
       'BEST_ANALYST_RATING-lag10', 'BEST_ANALYST_RATING-lag20',
       'EBIT-lag1', 'EBIT-lag2', 'EBIT-lag5', 'EBIT-lag10', 'EBIT-lag20',
       'Yield_real', 'Yield-lag1', 'Yield-lag2', 'Yield-lag5',
       'Yield-lag10', 'Yield-lag20', 'till maturity']

PIPELINE_OBJ_1 = {
    "test_id": 1,
    "feature_generators": {
        1: {
            "type": "RollingMinMaxScaler", "config": {"window_size": 100, "target_lst": min_max_lst}
        },
        2: {
            "type": "OneHotTransformer", "config": {"target_lst": one_hot_lst}
        },
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 30, "predict_period": 1}

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
    "others": {"raw_data_file":  "real_data.hdf"}
}

PIPELINE_OBJ_2 = {
    "test_id": 2,
    "feature_generators": {
        1: {
            "type": "RollingMinMaxScaler", "config": {"window_size": 100, "target_lst": min_max_lst}
        },
        2: {
            "type": "OneHotTransformer", "config": {"target_lst": one_hot_lst}
        },
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 30, "predict_period": 1}

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
    "others": {"raw_data_file":  "real_data.hdf"}
}

PIPELINE_OBJ_3 = {
    "test_id": 3,
    "feature_generators": {
        1: {
            "type": "RollingMinMaxScaler", "config": {"window_size": 100, "target_lst": min_max_lst}
        },
        2: {
            "type": "OneHotTransformer", "config": {"target_lst": one_hot_lst}
        },
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 30, "predict_period": 1}

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
    "others": {"raw_data_file":  "real_data.hdf"}
}

PIPELINE_OBJ_4 = {
    "test_id": 4,
    "feature_generators": {
        1: {
            "type": "RollingMinMaxScaler", "config": {"window_size": 100, "target_lst": min_max_lst}
        },
        2: {
            "type": "OneHotTransformer", "config": {"target_lst": one_hot_lst}
        },
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 30, "predict_period": 1}

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
    "others": {"raw_data_file":  "real_data.hdf"}
}

PIPELINE_OBJ_5 = {
    "test_id": 5,
    "feature_generators": {
        1: {
            "type": "RollingMinMaxScaler", "config": {"window_size": 100, "target_lst": min_max_lst}
        },
        2: {
            "type": "OneHotTransformer", "config": {"target_lst": one_hot_lst}
        },
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 30, "predict_period": 1}

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
    "others": {"raw_data_file":  "real_data.hdf"}
}

PIPELINE_OBJ_6 = {
    "test_id": 6,
    "feature_generators": {
        1: {
            "type": "RollingMinMaxScaler", "config": {"window_size": 100, "target_lst": min_max_lst}
        },
        2: {
            "type": "OneHotTransformer", "config": {"target_lst": one_hot_lst}
        },
    },
    "iterator": {
        "type": "DFIterator", "config": {"sample_lag": 30, "predict_period": 1}

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
    "others": {"raw_data_file": "real_data.hdf"}
}
