“””
PHYSBO + MLflow統合のユーティリティ関数
“””

import numpy as np
import matplotlib.pyplot as plt
import mlflow
import json
from typing import Dict, List, Any, Tuple
from config import VISUALIZATION_CONFIG

def action_to_params(action: np.ndarray, search_space: Dict[str, List[float]]) -> Dict[str, float]:
“””
PHYSBOのactionを実際のパラメータに変換

```
Args:
    action: PHYSBOから取得したアクション配列
    search_space: パラメータの探索空間 {"param_name": [min, max]}

Returns:
    実際のパラメータ値の辞書
"""
params = {}
param_names = list(search_space.keys())

for i, (param_name, param_range) in enumerate(search_space.items()):
    if i < len(action):
        # 正規化された値(0-1)を実際の範囲に変換
        normalized_value = action[i] if isinstance(action[i], (int, float)) else action[i][0]
        params[param_name] = param_range[0] + normalized_value * (param_range[1] - param_range[0])

return params
```

def params_to_action(params: Dict[str, float], search_space: Dict[str, List[float]]) -> np.ndarray:
“””
実際のパラメータをPHYSBOのactionに変換

```
Args:
    params: パラメータ値の辞書
    search_space: パラメータの探索空間

Returns:
    正規化されたアクション配列
"""
action = []
for param_name, param_range in search_space.items():
    if param_name in params:
        # 実際の値を正規化(0-1)
        normalized = (params[param_name] - param_range[0]) / (param_range[1] - param_range[0])
        action.append(normalized)
    else:
        action.append(0.5)  # デフォルト値

return np.array(action)
```

def log_physbo_internal_state(policy, iteration: int):
“””
PHYSBOの内部状態をMLflowに記録

```
Args:
    policy: PHYSBOのポリシーオブジェクト
    iteration: 現在の反復回数
"""
try:
    # 周辺尤度の記録
    if hasattr(policy, 'get_marginal_likelihood'):
        marginal_likelihood = policy.get_marginal_likelihood()
        if marginal_likelihood is not None:
            mlflow.log_metric("marginal_likelihood", marginal_likelihood, step=iteration)
    
    # 事後分布の統計情報
    if hasattr(policy, 'get_post_fmean') and hasattr(policy, 'get_post_fcov'):
        post_mean = policy.get_post_fmean()
        post_cov = policy.get_post_fcov()
        
        if post_mean is not None and post_cov is not None:
            mlflow.log_metric("mean_prediction", np.mean(post_mean), step=iteration)
            mlflow.log_metric("mean_uncertainty", np.mean(np.diag(post_cov)), step=iteration)
            mlflow.log_metric("max_uncertainty", np.max(np.diag(post_cov)), step=iteration)
    
    # 探索・活用のバランス
    if hasattr(policy, 'get_acquisition_values'):
        acq_values = policy.get_acquisition_values()
        if acq_values is not None:
            mlflow.log_metric("mean_acquisition_value", np.mean(acq_values), step=iteration)
            mlflow.log_metric("max_acquisition_value", np.max(acq_values), step=iteration)
            
except Exception as e:
    # エラーが発生しても実験は継続
    mlflow.log_param(f"internal_state_error_{iteration}", str(e))
```

def save_optimization_plots(optimization_history: List[float],
best_score_history: List[float],
file_prefix: str = “optimization”):
“””
最適化の進行状況を可視化してMLflowに保存

```
Args:
    optimization_history: 各試行の目的関数値
    best_score_history: 各時点での最良スコア
    file_prefix: 保存ファイル名のプレフィックス
"""
fig, axes = plt.subplots(2, 2, figsize=VISUALIZATION_CONFIG["figure_size"])
fig.suptitle('Bayesian Optimization Progress', fontsize=16)

# 1. 目的関数の履歴
axes[0, 0].plot(optimization_history, 'b-', alpha=0.7, label='Objective Score')
axes[0, 0].plot(best_score_history, 'r-', linewidth=2, label='Best Score So Far')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Optimization Progress')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. 改善量の履歴
if len(optimization_history) > 1:
    improvements = [optimization_history[i] - optimization_history[i-1] 
                   for i in range(1, len(optimization_history))]
    axes[0, 1].plot(improvements, 'g-', alpha=0.7)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Improvement')
    axes[0, 1].set_title('Score Improvement per Iteration')
    axes[0, 1].grid(True)

# 3. 最良スコアの収束
axes[1, 0].plot(best_score_history, 'r-', linewidth=2)
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Best Score')
axes[1, 0].set_title('Best Score Convergence')
axes[1, 0].grid(True)

# 4. スコアの分布
axes[1, 1].hist(optimization_history, bins=min(20, len(optimization_history)//2), 
                alpha=0.7, color='skyblue', edgecolor='black')
axes[1, 1].axvline(x=max(optimization_history), color='r', linestyle='--', 
                   label=f'Best: {max(optimization_history):.4f}')
axes[1, 1].set_xlabel('Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Score Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()

# 保存
filename = f"{file_prefix}_progress.{VISUALIZATION_CONFIG['save_format']}"
plt.savefig(filename, dpi=VISUALIZATION_CONFIG["dpi"])
mlflow.log_artifact(filename)
plt.close()
```

def calculate_convergence_metrics(optimization_history: List[float],
window_size: int = 10) -> Dict[str, float]:
“””
収束に関するメトリクスを計算

```
Args:
    optimization_history: 最適化履歴
    window_size: 収束判定の窓サイズ

Returns:
    収束メトリクスの辞書
"""
if len(optimization_history) < window_size * 2:
    return {}

# 最近の改善量
recent_scores = optimization_history[-window_size:]
previous_scores = optimization_history[-window_size*2:-window_size]

recent_improvement = max(recent_scores) - max(previous_scores)

# 分散の変化
recent_variance = np.var(recent_scores)
previous_variance = np.var(previous_scores)

return {
    "recent_improvement": recent_improvement,
    "recent_variance": recent_variance,
    "variance_change": recent_variance - previous_variance,
    "improvement_rate": recent_improvement / window_size,
    "stability_score": 1.0 / (1.0 + recent_variance)  # 分散が小さいほど安定
}
```

def save_experiment_summary(optimization_history: List[float],
best_params: Dict[str, float],
config: Dict[str, Any],
execution_time: float):
“””
実験のサマリーを保存

```
Args:
    optimization_history: 最適化履歴
    best_params: 最良パラメータ
    config: 実験設定
    execution_time: 実行時間
"""
summary = {
    "experiment_config": config,
    "results": {
        "best_score": max(optimization_history),
        "best_params": best_params,
        "total_iterations": len(optimization_history),
        "execution_time": execution_time,
        "convergence_metrics": calculate_convergence_metrics(optimization_history)
    },
    "optimization_history": optimization_history
}

# JSON形式で保存
with open("experiment_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

mlflow.log_artifact("experiment_summary.json")
```

def setup_mlflow_environment(config: Dict[str, str]):
“””
MLflow環境のセットアップ

```
Args:
    config: MLflow設定
"""
if config.get("tracking_uri"):
    mlflow.set_tracking_uri(config["tracking_uri"])

# 実験の作成または取得
try:
    experiment = mlflow.get_experiment_by_name(config["experiment_name"])
    if experiment is None:
        mlflow.create_experiment(
            name=config["experiment_name"],
            artifact_location=config.get("artifact_location")
        )
except Exception as e:
    print(f"Warning: Could not set up MLflow experiment: {e}")

mlflow.set_experiment(config["experiment_name"])
```