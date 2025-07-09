“””
PHYSBO + MLflow統合の設定ファイル
“””

# MLflow設定

MLFLOW_CONFIG = {
“experiment_name”: “physbo_bayesian_optimization”,
“tracking_uri”: “sqlite:///mlflow.db”,  # ローカルSQLiteデータベース
“artifact_location”: “./mlruns”
}

# PHYSBO設定

PHYSBO_CONFIG = {
“default”: {
“num_search_each_probe”: 1,
“score”: “TS”,  # Thompson Sampling
“interval”: 10,
“num_rand_basis”: 5000
},
“fast”: {
“num_search_each_probe”: 1,
“score”: “EI”,  # Expected Improvement
“interval”: 5,
“num_rand_basis”: 1000
},
“thorough”: {
“num_search_each_probe”: 5,
“score”: “TS”,
“interval”: 20,
“num_rand_basis”: 10000
}
}

# 最適化設定

OPTIMIZATION_CONFIG = {
“max_iterations”: 100,
“early_stopping_patience”: 10,
“convergence_threshold”: 0.001,
“random_seed”: 42
}

# 探索空間の例（問題に応じて修正）

SEARCH_SPACE_EXAMPLES = {
“ml_hyperparameters”: {
“learning_rate”: [0.001, 0.1],
“batch_size”: [16, 128],
“dropout_rate”: [0.0, 0.5],
“num_layers”: [1, 10]
},
“experimental_parameters”: {
“temperature”: [20.0, 100.0],
“pressure”: [1.0, 10.0],
“concentration”: [0.1, 1.0]
}
}

# 獲得関数の設定

ACQUISITION_FUNCTIONS = [“EI”, “PI”, “TS”]

# 可視化設定

VISUALIZATION_CONFIG = {
“figure_size”: (12, 8),
“dpi”: 300,
“save_format”: “png”
}