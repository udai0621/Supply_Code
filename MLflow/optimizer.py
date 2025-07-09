“””
PHYSBO + MLflow統合のメイン最適化クラス
“””

import physbo
import mlflow
import numpy as np
import time
from typing import Dict, List, Callable, Any, Tuple, Optional
from mlflow.tracking import MlflowClient

from config import PHYSBO_CONFIG, OPTIMIZATION_CONFIG, MLFLOW_CONFIG
from utils import (
action_to_params, params_to_action, log_physbo_internal_state,
save_optimization_plots, calculate_convergence_metrics,
save_experiment_summary, setup_mlflow_environment
)

class PHYSBOMLflowOptimizer:
“””
PHYSBOとMLflowを統合したベイズ最適化クラス
“””

```
def __init__(self, 
             search_space: Dict[str, List[float]],
             objective_function: Callable[[Dict[str, float]], float],
             config_type: str = "default",
             experiment_name: Optional[str] = None):
    """
    初期化
    
    Args:
        search_space: 探索空間 {"param_name": [min, max]}
        objective_function: 目的関数
        config_type: PHYSBO設定のタイプ ("default", "fast", "thorough")
        experiment_name: MLflow実験名
    """
    self.search_space = search_space
    self.objective_function = objective_function
    self.config = PHYSBO_CONFIG[config_type].copy()
    self.optimization_config = OPTIMIZATION_CONFIG.copy()
    
    # MLflow環境のセットアップ
    mlflow_config = MLFLOW_CONFIG.copy()
    if experiment_name:
        mlflow_config["experiment_name"] = experiment_name
    setup_mlflow_environment(mlflow_config)
    
    # 最適化履歴
    self.optimization_history = []
    self.best_score_history = []
    self.best_params = None
    self.best_score = float('-inf')
    
    # PHYSBO policy
    self.policy = None
    
def _create_physbo_policy(self) -> physbo.search.discrete.policy:
    """
    PHYSBOポリシーの作成
    
    Returns:
        PHYSBOポリシーオブジェクト
    """
    # 探索空間に基づいて離散的な候補点を生成
    test_X = self._generate_candidate_points()
    
    return physbo.search.discrete.policy(test_X=test_X, config=self.config)

def _generate_candidate_points(self, num_points: int = 1000) -> np.ndarray:
    """
    探索空間に基づいて候補点を生成
    
    Args:
        num_points: 生成する候補点数
        
    Returns:
        候補点の配列 (num_points, num_dimensions)
    """
    num_dims = len(self.search_space)
    
    # 各次元で均等に分割した格子点を生成
    grid_size = int(num_points ** (1.0 / num_dims)) + 1
    
    # 各次元の格子点を作成
    dim_grids = []
    param_names = list(self.search_space.keys())
    
    for param_name in param_names:
        param_range = self.search_space[param_name]
        # 0-1の範囲で正規化された格子点
        grid = np.linspace(0, 1, grid_size)
        dim_grids.append(grid)
    
    # 格子点の組み合わせを生成
    mesh_grids = np.meshgrid(*dim_grids)
    
    # フラット化して候補点行列を作成
    candidate_points = []
    for i in range(len(mesh_grids[0].flat)):
        point = []
        for j in range(num_dims):
            point.append(mesh_grids[j].flat[i])
        candidate_points.append(point)
    
    candidate_array = np.array(candidate_points)
    
    # 指定された数になるように調整
    if len(candidate_array) > num_points:
        # ランダムサンプリングで削減
        indices = np.random.choice(len(candidate_array), num_points, replace=False)
        candidate_array = candidate_array[indices]
    elif len(candidate_array) < num_points:
        # ランダム点で補完
        additional_points = num_points - len(candidate_array)
        random_points = np.random.rand(additional_points, num_dims)
        candidate_array = np.vstack([candidate_array, random_points])
    
    return candidate_array

def _normalized_to_params(self, normalized_point: np.ndarray) -> Dict[str, float]:
    """
    正規化された点を実際のパラメータに変換
    
    Args:
        normalized_point: 正規化された点 (0-1の範囲)
        
    Returns:
        実際のパラメータ値
    """
    params = {}
    param_names = list(self.search_space.keys())
    
    for i, param_name in enumerate(param_names):
        param_range = self.search_space[param_name]
        # 0-1の値を実際の範囲に変換
        params[param_name] = param_range[0] + normalized_point[i] * (param_range[1] - param_range[0])
    
    return params

def _params_to_normalized(self, params: Dict[str, float]) -> np.ndarray:
    """
    実際のパラメータを正規化された点に変換
    
    Args:
        params: 実際のパラメータ値
        
    Returns:
        正規化された点 (0-1の範囲)
    """
    normalized = []
    param_names = list(self.search_space.keys())
    
    for param_name in param_names:
        param_range = self.search_space[param_name]
        # 実際の値を0-1の範囲に正規化
        normalized_value = (params[param_name] - param_range[0]) / (param_range[1] - param_range[0])
        normalized.append(normalized_value)
    
    return np.array(normalized)

def _evaluate_objective(self, params: Dict[str, float], iteration: int) -> float:
    """
    目的関数の評価とMLflowへの記録
    
    Args:
        params: 評価するパラメータ
        iteration: 現在の反復回数
        
    Returns:
        目的関数の値
    """
    start_time = time.time()
    
    try:
        score = self.objective_function(params)
        execution_time = time.time() - start_time
        
        # MLflowに記録
        mlflow.log_metric("objective_score", score, step=iteration)
        mlflow.log_metric("execution_time", execution_time, step=iteration)
        
        # 最良スコアの更新
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            mlflow.log_metric("best_score_improvement", score - self.best_score, step=iteration)
        
        return score
        
    except Exception as e:
        execution_time = time.time() - start_time
        mlflow.log_param(f"evaluation_error_{iteration}", str(e))
        mlflow.log_metric("execution_time", execution_time, step=iteration)
        return float('-inf')  # エラー時は最悪スコア

def _check_convergence(self, iteration: int) -> bool:
    """
    収束判定
    
    Args:
        iteration: 現在の反復回数
        
    Returns:
        収束したかどうか
    """
    if len(self.optimization_history) < self.optimization_config["early_stopping_patience"]:
        return False
    
    # 最近の改善量をチェック
    recent_window = self.optimization_config["early_stopping_patience"]
    recent_best = max(self.optimization_history[-recent_window:])
    previous_best = max(self.optimization_history[-recent_window*2:-recent_window])
    
    improvement = recent_best - previous_best
    
    if improvement < self.optimization_config["convergence_threshold"]:
        mlflow.log_param("convergence_reason", "small_improvement")
        mlflow.log_param("converged_at_iteration", iteration)
        mlflow.log_metric("final_improvement", improvement)
        return True
    
    return False

def optimize(self, max_iterations: Optional[int] = None) -> Tuple[Dict[str, float], float]:
    """
    ベイズ最適化の実行
    
    Args:
        max_iterations: 最大反復回数
        
    Returns:
        (最良パラメータ, 最良スコア)
    """
    if max_iterations is None:
        max_iterations = self.optimization_config["max_iterations"]
    
    # 親実験の開始
    with mlflow.start_run(run_name="physbo_optimization_session"):
        parent_run_id = mlflow.active_run().info.run_id
        
        # 設定の記録
        mlflow.log_params(self.config)
        mlflow.log_params(self.optimization_config)
        mlflow.log_param("search_space_dims", len(self.search_space))
        mlflow.log_param("max_iterations", max_iterations)
        mlflow.set_tag("library", "PHYSBO")
        mlflow.set_tag("acquisition_function", self.config["score"])
        
        # 探索空間の記録
        for param_name, param_range in self.search_space.items():
            mlflow.log_param(f"space_{param_name}_min", param_range[0])
            mlflow.log_param(f"space_{param_name}_max", param_range[1])
        
        # PHYSBOポリシーの初期化
        self.policy = self._create_physbo_policy()
        
        # 最適化ループ
        start_time = time.time()
        
        for iteration in range(max_iterations):
            # 子実験として各試行を記録
            with mlflow.start_run(run_name=f"trial_{iteration+1}", nested=True):
                # 次の候補点を取得
                if iteration == 0:
                    # 初回はランダム探索
                    action = self.policy.random_search(max_num_probes=1, simulator=None)
                else:
                    # 2回目以降はベイズ最適化
                    action = self.policy.bayes_search(max_num_probes=1, simulator=None)
                
                # actionがインデックスの場合は実際の値に変換
                if isinstance(action, (int, np.integer)):
                    # actionは候補点のインデックス
                    selected_point = self.policy.test_X[action]
                    params = self._normalized_to_params(selected_point)
                elif isinstance(action, np.ndarray) and action.shape == ():
                    # スカラーの場合
                    selected_point = self.policy.test_X[int(action)]
                    params = self._normalized_to_params(selected_point)
                else:
                    # 直接的な値の場合
                    params = action_to_params(action, self.search_space)
                
                # パラメータの記録
                mlflow.log_params(params)
                mlflow.log_param("iteration", iteration + 1)
                mlflow.log_param("action_index", int(action) if isinstance(action, (int, np.integer, np.ndarray)) else "direct")
                
                # 目的関数の評価
                score = self._evaluate_objective(params, iteration)
                
                # 履歴の更新
                self.optimization_history.append(score)
                current_best = max(self.optimization_history)
                self.best_score_history.append(current_best)
                
                # 進行状況の記録
                mlflow.log_metric("best_score_so_far", current_best, step=iteration)
                improvement = score - (self.optimization_history[-2] if len(self.optimization_history) > 1 else 0)
                mlflow.log_metric("improvement", improvement, step=iteration)
                
                # PHYSBOの内部状態を記録
                log_physbo_internal_state(self.policy, iteration)
                
                # 収束メトリクスの計算と記録
                if iteration > 10:
                    conv_metrics = calculate_convergence_metrics(self.optimization_history)
                    for key, value in conv_metrics.items():
                        mlflow.log_metric(f"convergence_{key}", value, step=iteration)
            
            # PHYSBOにフィードバック
            if isinstance(action, (int, np.integer)):
                self.policy.write(action, score)
            elif isinstance(action, np.ndarray) and action.shape == ():
                self.policy.write(int(action), score)
            else:
                # actionが候補点のインデックスでない場合の処理
                # 最も近い候補点を探す
                if hasattr(self.policy, 'test_X'):
                    normalized_params = self._params_to_normalized(params)
                    distances = np.sum((self.policy.test_X - normalized_params) ** 2, axis=1)
                    closest_index = np.argmin(distances)
                    self.policy.write(closest_index, score)
            
            # 収束判定
            if self._check_convergence(iteration):
                break
        
        # 最適化完了
        total_time = time.time() - start_time
        best_iteration = np.argmax(self.optimization_history)
        
        # 最終結果の記録
        mlflow.log_metric("total_optimization_time", total_time)
        mlflow.log_metric("final_best_score", self.best_score)
        mlflow.log_metric("best_iteration", best_iteration + 1)
        mlflow.log_metric("total_iterations", len(self.optimization_history))
        
        # 最良パラメータの記録
        if self.best_params:
            for param_name, param_value in self.best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)
        
        # 可視化の保存
        save_optimization_plots(self.optimization_history, self.best_score_history)
        
        # 実験サマリーの保存
        save_experiment_summary(
            self.optimization_history,
            self.best_params or {},
            self.config,
            total_time
        )
        
        return self.best_params or {}, self.best_score

def get_optimization_history(self) -> List[float]:
    """最適化履歴を取得"""
    return self.optimization_history.copy()

def get_best_results(self) -> Tuple[Dict[str, float], float]:
    """最良結果を取得"""
    return self.best_params or {}, self.best_score
```

class MultiAcquisitionOptimizer:
“””
複数の獲得関数を比較するオプティマイザー
“””

```
def __init__(self, 
             search_space: Dict[str, List[float]],
             objective_function: Callable[[Dict[str, float]], float],
             acquisition_functions: List[str] = ["EI", "PI", "TS"]):
    """
    初期化
    
    Args:
        search_space: 探索空間
        objective_function: 目的関数
        acquisition_functions: 比較する獲得関数のリスト
    """
    self.search_space = search_space
    self.objective_function = objective_function
    self.acquisition_functions = acquisition_functions
    self.results = {}

def compare_acquisition_functions(self, max_iterations: int = 50) -> Dict[str, Dict]:
    """
    複数の獲得関数の比較実験
    
    Args:
        max_iterations: 各獲得関数での最大反復回数
        
    Returns:
        各獲得関数の結果
    """
    with mlflow.start_run(run_name="acquisition_function_comparison"):
        
        for acq_func in self.acquisition_functions:
            print(f"Testing acquisition function: {acq_func}")
            
            # 各獲得関数での最適化
            optimizer = PHYSBOMLflowOptimizer(
                search_space=self.search_space,
                objective_function=self.objective_function,
                config_type="default"
            )
            
            # 獲得関数の設定
            optimizer.config["score"] = acq_func
            
            # 最適化実行
            best_params, best_score = optimizer.optimize(max_iterations)
            
            # 結果の保存
            self.results[acq_func] = {
                "best_params": best_params,
                "best_score": best_score,
                "history": optimizer.get_optimization_history()
            }
            
            # 比較用メトリクスの記録
            mlflow.log_metric(f"{acq_func}_best_score", best_score)
            mlflow.log_metric(f"{acq_func}_convergence_speed", 
                            len(optimizer.get_optimization_history()))
    
    return self.results
```