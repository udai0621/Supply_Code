“””
PHYSBOとMLflowを組み合わせたベイズ最適化サンプルコード

このコードは、PHYSBOのベイズ最適化機能とMLflowの実験追跡機能を統合し、
最適化プロセスの全体を効率的に管理・記録するサンプルです。

料理のレシピ最適化に例えると：

- PHYSBO：シェフが次に試すべき調味料の組み合わせを提案
- MLflow：各料理の味、材料、調理法を詳細に記録するレシピノート
  “””

import numpy as np
import mlflow
import mlflow.sklearn
import physbo
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import pickle
import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings(‘ignore’)

# MLflowの設定

mlflow.set_tracking_uri(“file:./mlruns”)  # ローカルファイルシステムを使用
mlflow.set_experiment(“PHYSBO_Bayesian_Optimization”)

class OptimizationTarget:
“””
最適化対象の目的関数を定義するクラス

```
ここでは多峰性を持つ複雑な関数を例として使用
実際の用途では、機械学習モデルの性能、物理シミュレーション結果、
実験データなどを対象とすることができます
"""

def __init__(self, target_function: str = "rastrigin"):
    self.target_function = target_function
    self.evaluation_count = 0
    self.evaluation_history = []
    
def __call__(self, action: np.ndarray) -> float:
    """
    目的関数の評価
    
    Args:
        action: 探索候補のインデックス
        
    Returns:
        評価値（PHYSBOは最大化するため、最小化問題では-1を乗算）
    """
    self.evaluation_count += 1
    
    if self.target_function == "rastrigin":
        # Rastrigin関数（多峰性関数の代表例）
        result = self._rastrigin_function(action)
    elif self.target_function == "sphere":
        # Sphere関数（シンプルな凸関数）
        result = self._sphere_function(action)
    else:
        # カスタム関数
        result = self._custom_function(action)
    
    # 評価履歴を記録
    self.evaluation_history.append({
        'evaluation_id': self.evaluation_count,
        'action': action,
        'result': result,
        'timestamp': datetime.now().isoformat()
    })
    
    return result

def _rastrigin_function(self, action: np.ndarray) -> float:
    """Rastrigin関数の実装"""
    action_idx = action[0]
    x = X[action_idx]  # グローバル変数Xから座標を取得
    
    # 2次元Rastrigin関数
    A = 10
    n = len(x)
    result = A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])
    
    return -result  # 最小化のため負の値を返す

def _sphere_function(self, action: np.ndarray) -> float:
    """Sphere関数の実装"""
    action_idx = action[0]
    x = X[action_idx]
    result = sum([xi**2 for xi in x])
    return -result

def _custom_function(self, action: np.ndarray) -> float:
    """カスタム関数の実装（元のチュートリアルの関数）"""
    action_idx = action[0]
    x = X[action_idx][0]
    result = 3.0 * x**4 + 4.0 * x**3 + 1.0
    return -result
```

class MLflowOptimizationTracker:
“””
MLflowを使用した最適化プロセスの追跡クラス

```
最適化の各ステップで以下を記録：
- パラメータ：探索候補、獲得関数設定など
- メトリクス：目的関数値、ベストスコア、収束状況
- アーティファクト：グラフ、モデル、データファイル
- データ：探索候補、評価値、探索空間の詳細データ
"""

def __init__(self, experiment_name: str = "PHYSBO_Optimization", 
             data_format: str = "parquet"):
    self.experiment_name = experiment_name
    self.data_format = data_format.lower()  # "csv" or "parquet"
    self.run_id = def run_optimization_with_mlflow(config: Dict[str, Any]):
"""
MLflowを使用したベイズ最適化の実行

Args:
    config: 最適化設定辞書
"""
# MLflowトラッカーの初期化（データ形式を指定）
data_format = config.get('data_format', 'parquet')
tracker = MLflowOptimizationTracker(data_format=data_format)
tracker.start_run(config.get('run_name'))

try:
    # 設定のログ記録
    tracker.log_optimization_setup(config)
    
    # 探索空間の生成
    global X  # グローバル変数として設定（目的関数で使用）
    X = create_search_space(
        dimensions=config['dimensions'],
        points_per_dim=config['points_per_dim'],
        bounds=config['bounds']
    )
    
    print(f"探索空間: {X.shape[0]} 個の候補点, {X.shape[1]} 次元")
    
    # 探索空間データの初期化
    tracker.initialize_search_space_data(X)
    
    # 目的関数の設定
    objective = OptimizationTarget(config['target_function'])
    
    # PHYSBOポリシーの初期化
    policy = physbo.search.discrete.policy(test_X=X)
    policy.set_seed(config['seed'])
    
    # ランダムサーチフェーズ
    print("ランダムサーチフェーズ開始...")
    random_steps = config['random_steps']
    
    for step in range(random_steps):
        # 次の候補を選択
        action = policy.random_action()
        
        # 目的関数を評価
        score = objective(action)
        
        # 結果を記録
        policy.write(action, score)
        
        # MLflowにログ（データも含む）
        tracker.log_step("random", step, score, action[0], X)
        
        print(f"ランダムサーチ {step+1}/{random_steps}: "
              f"score={score:.4f}, action={action[0]}")
    
    self.step_count = 0
    self.best_score_history = []
    self.evaluation_history = []
    
    # データ保存用のDataFrame
    self.optimization_data = pd.DataFrame()
    self.search_space_data = pd.DataFrame()
    self.acquisition_data = pd.DataFrame()
    
    # データ形式のバリデーション
    if self.data_format not in ["csv", "parquet"]:
        raise ValueError("data_format must be 'csv' or 'parquet'")

def start_run(self, run_name: str = None):
    """MLflowランの開始"""
    if run_name is None:
        run_name = f"optimization_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    mlflow.start_run(run_name=run_name)
    self.run_id = mlflow.active_run().info.run_id
    
    # 実験の基本情報をログ
    mlflow.log_param("experiment_start_time", datetime.now().isoformat())
    mlflow.log_param("run_id", self.run_id)
    mlflow.log_param("data_format", self.data_format)

def initialize_search_space_data(self, X: np.ndarray):
    """
    探索空間データの初期化
    
    Args:
        X: 探索候補点の配列
    """
    # 探索空間データの作成
    space_data = []
    for idx, point in enumerate(X):
        row = {'candidate_id': idx}
        
        # 各次元の座標を記録
        for dim in range(point.shape[0]):
            row[f'dim_{dim}'] = point[dim]
        
        # 距離関連の特徴量
        row['distance_from_origin'] = np.linalg.norm(point)
        row['sum_of_coordinates'] = np.sum(point)
        row['max_coordinate'] = np.max(point)
        row['min_coordinate'] = np.min(point)
        
        space_data.append(row)
    
    self.search_space_data = pd.DataFrame(space_data)
    
    # 探索空間の統計情報をパラメータとして記録
    mlflow.log_param("search_space_size", len(X))
    mlflow.log_param("search_space_dimensions", X.shape[1])
    mlflow.log_param("search_space_bounds_min", X.min(axis=0).tolist())
    mlflow.log_param("search_space_bounds_max", X.max(axis=0).tolist())

def log_optimization_setup(self, config: Dict[str, Any]):
    """最適化設定のログ記録"""
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)):
            mlflow.log_param(key, value)
        else:
            mlflow.log_param(key, str(value))

def log_step(self, step_type: str, step_num: int, score: float, 
             action_id: int, X: np.ndarray, additional_metrics: Dict[str, float] = None):
    """
    各最適化ステップのログ記録（データも含む）
    
    Args:
        step_type: ステップタイプ（"random" or "bayes"）
        step_num: ステップ番号
        score: 評価値
        action_id: 選択されたアクションID
        X: 探索空間
        additional_metrics: 追加メトリクス
    """
    self.step_count += 1
    
    # 基本メトリクスのログ
    mlflow.log_metric(f"{step_type}_score", score, step=step_num)
    mlflow.log_metric(f"{step_type}_action_id", action_id, step=step_num)
    
    # ベストスコアの更新
    current_best = max(self.best_score_history) if self.best_score_history else float('-inf')
    if score > current_best:
        self.best_score_history.append(score)
        mlflow.log_metric("best_score", score, step=step_num)
        mlflow.log_metric("best_action_id", action_id, step=step_num)
    else:
        self.best_score_history.append(current_best)
        mlflow.log_metric("best_score", current_best, step=step_num)
    
    # 追加メトリクスのログ
    if additional_metrics:
        for key, value in additional_metrics.items():
            mlflow.log_metric(key, value, step=step_num)
    
    # 最適化データの記録
    selected_point = X[action_id]
    optimization_row = {
        'step': step_num,
        'step_type': step_type,
        'action_id': action_id,
        'score': score,
        'is_best': score > current_best,
        'cumulative_best_score': max(score, current_best),
        'timestamp': datetime.now().isoformat()
    }
    
    # 選択されたポイントの座標を追加
    for dim in range(selected_point.shape[0]):
        optimization_row[f'selected_dim_{dim}'] = selected_point[dim]
    
    # 探索履歴に関する統計
    optimization_row['total_steps'] = self.step_count
    optimization_row['distance_from_origin'] = np.linalg.norm(selected_point)
    
    # DataFrameに追加
    new_row_df = pd.DataFrame([optimization_row])
    self.optimization_data = pd.concat([self.optimization_data, new_row_df], 
                                     ignore_index=True)
    
    # 評価履歴の記録（既存）
    self.evaluation_history.append({
        'step': step_num,
        'step_type': step_type,
        'score': score,
        'action_id': action_id,
        'timestamp': datetime.now().isoformat()
    })

def log_acquisition_function_data(self, policy, X: np.ndarray, step_num: int):
    """
    獲得関数の値を全候補点について記録
    
    Args:
        policy: PHYSBOのポリシー
        X: 探索空間
        step_num: 現在のステップ番号
    """
    try:
        # 獲得関数の値を計算
        acquisition_scores = policy.get_score(mode="EI", xs=X)
        
        # 事後分布の平均と分散も計算
        posterior_mean = policy.get_post_fmean(X)
        posterior_var = policy.get_post_fcov(X)
        
        acquisition_step_data = []
        for idx, (acq_score, post_mean, post_var) in enumerate(
            zip(acquisition_scores, posterior_mean, posterior_var)):
            
            row = {
                'step': step_num,
                'candidate_id': idx,
                'acquisition_score': acq_score,
                'posterior_mean': post_mean,
                'posterior_variance': post_var,
                'posterior_std': np.sqrt(post_var),
                'timestamp': datetime.now().isoformat()
            }
            
            # 座標情報も追加
            point = X[idx]
            for dim in range(point.shape[0]):
                row[f'dim_{dim}'] = point[dim]
            
            acquisition_step_data.append(row)
        
        # DataFrameに追加
        step_df = pd.DataFrame(acquisition_step_data)
        self.acquisition_data = pd.concat([self.acquisition_data, step_df], 
                                        ignore_index=True)
        
        # 獲得関数の統計をメトリクスとしてログ
        mlflow.log_metric("acquisition_max", np.max(acquisition_scores), step=step_num)
        mlflow.log_metric("acquisition_mean", np.mean(acquisition_scores), step=step_num)
        mlflow.log_metric("acquisition_std", np.std(acquisition_scores), step=step_num)
        mlflow.log_metric("posterior_mean_avg", np.mean(posterior_mean), step=step_num)
        mlflow.log_metric("posterior_var_avg", np.mean(posterior_var), step=step_num)
        
    except Exception as e:
        print(f"獲得関数データの記録でエラー: {e}")

def save_data_files(self):
    """
    蓄積されたデータをファイルに保存してMLflowにログ
    """
    file_extension = self.data_format
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    saved_files = []
    
    # 1. 最適化データの保存
    if not self.optimization_data.empty:
        opt_filename = f"optimization_data_{timestamp}.{file_extension}"
        self._save_dataframe(self.optimization_data, opt_filename)
        saved_files.append(opt_filename)
    
    # 2. 探索空間データの保存
    if not self.search_space_data.empty:
        space_filename = f"search_space_data_{timestamp}.{file_extension}"
        self._save_dataframe(self.search_space_data, space_filename)
        saved_files.append(space_filename)
    
    # 3. 獲得関数データの保存
    if not self.acquisition_data.empty:
        acq_filename = f"acquisition_data_{timestamp}.{file_extension}"
        self._save_dataframe(self.acquisition_data, acq_filename)
        saved_files.append(acq_filename)
    
    # 4. 統合データセットの作成と保存
    self._create_and_save_integrated_dataset(timestamp)
    saved_files.append(f"integrated_dataset_{timestamp}.{file_extension}")
    
    # MLflowにファイルをログ
    for filename in saved_files:
        if os.path.exists(filename):
            mlflow.log_artifact(filename)
            print(f"保存完了: {filename}")
    
    return saved_files

def _save_dataframe(self, df: pd.DataFrame, filename: str):
    """DataFrameを指定形式で保存"""
    if self.data_format == "csv":
        df.to_csv(filename, index=False, encoding='utf-8')
    elif self.data_format == "parquet":
        df.to_parquet(filename, index=False, engine='pyarrow')

def _create_and_save_integrated_dataset(self, timestamp: str):
    """
    統合データセットの作成
    最適化プロセスの全体像を把握できる包括的なデータセット
    """
    if self.optimization_data.empty:
        return
    
    # 最適化データをベースに統合
    integrated_data = self.optimization_data.copy()
    
    # 探索空間データとマージ
    if not self.search_space_data.empty:
        integrated_data = integrated_data.merge(
            self.search_space_data, 
            left_on='action_id', 
            right_on='candidate_id', 
            how='left'
        )
    
    # 獲得関数データから最新の情報を追加
    if not self.acquisition_data.empty:
        # 各ステップでの選択されたポイントの獲得関数値を取得
        latest_acq_data = self.acquisition_data.groupby(['step', 'candidate_id']).last().reset_index()
        integrated_data = integrated_data.merge(
            latest_acq_data[['step', 'candidate_id', 'acquisition_score', 
                           'posterior_mean', 'posterior_variance', 'posterior_std']],
            left_on=['step', 'action_id'],
            right_on=['step', 'candidate_id'],
            how='left'
        )
    
    # 追加の統計情報を計算
    integrated_data['step_rank'] = integrated_data['step'].rank()
    integrated_data['score_rank'] = integrated_data['score'].rank(ascending=False)
    integrated_data['score_percentile'] = integrated_data['score'].rank(pct=True)
    
    # 移動平均の計算
    window_size = min(5, len(integrated_data))
    integrated_data['score_rolling_mean'] = (
        integrated_data['score'].rolling(window=window_size, min_periods=1).mean()
    )
    
    # 改善率の計算
    integrated_data['score_improvement'] = integrated_data['score'].diff()
    integrated_data['cumulative_improvement'] = (
        integrated_data['score'] - integrated_data['score'].iloc[0]
    )
    
    # ファイル保存
    filename = f"integrated_dataset_{timestamp}.{self.data_format}"
    self._save_dataframe(integrated_data, filename)
    
    # データセットの統計をメトリクスとしてログ
    mlflow.log_metric("dataset_total_rows", len(integrated_data))
    mlflow.log_metric("dataset_total_columns", len(integrated_data.columns))
    mlflow.log_metric("dataset_score_range", integrated_data['score'].max() - integrated_data['score'].min())

def log_convergence_plot(self, scores: List[float], title: str = "Convergence Plot"):
    """収束プロットの生成とログ"""
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.grid(True)
    
    plot_path = f"convergence_plot_{self.step_count}.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()
    
    # ファイルを削除
    if os.path.exists(plot_path):
        os.remove(plot_path)

def log_search_space_analysis(self, policy, X: np.ndarray):
    """探索空間の分析結果をログ"""
    try:
        # 事後分布の平均と分散を計算
        mean = policy.get_post_fmean(X)
        var = policy.get_post_fcov(X)
        std = np.sqrt(var)
        
        # 獲得関数の値を計算
        score = policy.get_score(mode="EI", xs=X)
        
        # 可視化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 事後分布のプロット
        if X.shape[1] == 1:  # 1次元の場合
            x_vals = X[:, 0]
            ax1.plot(x_vals, mean, label='Posterior Mean', color='blue')
            ax1.fill_between(x_vals, mean - std, mean + std, 
                           alpha=0.3, color='blue', label='Uncertainty')
            ax1.set_title('Posterior Distribution')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Predicted Value')
            ax1.legend()
            ax1.grid(True)
            
            # 獲得関数のプロット
            ax2.plot(x_vals, score, label='Acquisition Function', color='red')
            ax2.set_title('Acquisition Function')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Acquisition Score')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        analysis_path = f"search_space_analysis_{self.step_count}.png"
        plt.savefig(analysis_path)
        mlflow.log_artifact(analysis_path)
        plt.close()
        
        # ファイルを削除
        if os.path.exists(analysis_path):
            os.remove(analysis_path)
            
    except Exception as e:
        print(f"探索空間分析でエラー: {e}")

def log_final_results(self, policy, best_action: int, best_score: float, 
                     total_evaluations: int):
    """最終結果のログ記録"""
    # 最終メトリクス
    mlflow.log_metric("final_best_score", best_score)
    mlflow.log_metric("final_best_action", best_action)
    mlflow.log_metric("total_evaluations", total_evaluations)
    
    # 評価履歴をJSONファイルとして保存
    history_path = "evaluation_history.json"
    with open(history_path, 'w') as f:
        json.dump(self.evaluation_history, f, indent=2)
    mlflow.log_artifact(history_path)
    
    # データファイルの保存
    saved_files = self.save_data_files()
    
    # PHYSBOの結果オブジェクトを保存
    try:
        model_path = "physbo_policy.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(policy, f)
        mlflow.log_artifact(model_path)
    except Exception as e:
        print(f"モデル保存でエラー: {e}")
    
    # データ保存の統計をログ
    mlflow.log_param("saved_data_files", len(saved_files))
    mlflow.log_param("data_files_list", saved_files)
    
    # ファイルを削除
    cleanup_files = [history_path, model_path] + saved_files
    for path in cleanup_files:
        if os.path.exists(path):
            os.remove(path)

def end_run(self):
    """MLflowランの終了"""
    mlflow.end_run()
```

def create_search_space(dimensions: int = 2, points_per_dim: int = 100,
bounds: List[Tuple[float, float]] = None) -> np.ndarray:
“””
探索空間の生成

```
Args:
    dimensions: 次元数
    points_per_dim: 各次元あたりのグリッド点数
    bounds: 各次元の範囲 [(min1, max1), (min2, max2), ...]

Returns:
    探索候補のグリッド点配列
"""
if bounds is None:
    bounds = [(-2.0, 2.0)] * dimensions

# 各次元のグリッド点を生成
grids = []
for i in range(dimensions):
    min_val, max_val = bounds[i]
    grid = np.linspace(min_val, max_val, points_per_dim)
    grids.append(grid)

# メッシュグリッドを作成
mesh_grids = np.meshgrid(*grids, indexing='ij')

# 候補点の配列に変換
candidate_points = np.column_stack([grid.ravel() for grid in mesh_grids])

return candidate_points
```

def run_optimization_with_mlflow(config: Dict[str, Any]):
“””
MLflowを使用したベイズ最適化の実行

```
Args:
    config: 最適化設定辞書
"""
# MLflowトラッカーの初期化
tracker = MLflowOptimizationTracker()
tracker.start_run(config.get('run_name'))

try:
    # 設定のログ記録
    tracker.log_optimization_setup(config)
    
    # 探索空間の生成
    global X  # グローバル変数として設定（目的関数で使用）
    X = create_search_space(
        dimensions=config['dimensions'],
        points_per_dim=config['points_per_dim'],
        bounds=config['bounds']
    )
    
    print(f"探索空間: {X.shape[0]} 個の候補点, {X.shape[1]} 次元")
    
    # 目的関数の設定
    objective = OptimizationTarget(config['target_function'])
    
    # PHYSBOポリシーの初期化
    policy = physbo.search.discrete.policy(test_X=X)
    policy.set_seed(config['seed'])
    
    # ランダムサーチフェーズ
    print("ランダムサーチフェーズ開始...")
    random_steps = config['random_steps']
    
    for step in range(random_steps):
        # 次の候補を選択
        action = policy.random_action()
        
        # 目的関数を評価
        score = objective(action)
        
        # 結果を記録
        policy.write(action, score)
        
        # MLflowにログ
        tracker.log_step("random", step, score, action[0])
        
        print(f"ランダムサーチ {step+1}/{random_steps}: "
              f"score={score:.4f}, action={action[0]}")
    
    # ベイズ最適化フェーズ
    print("ベイズ最適化フェーズ開始...")
    bayes_steps = config['bayes_steps']
    
    for step in range(bayes_steps):
        # 次の候補を選択（ベイズ最適化）
        action = policy.bayes_action(
            score=config['acquisition_function'],
            interval=config['hyperparameter_update_interval'],
            num_rand_basis=config['num_rand_basis']
        )
        
        # 目的関数を評価
        score = objective(action)
        
        # 結果を記録
        policy.write(action, score)
        
        # MLflowにログ
        tracker.log_step("bayes", random_steps + step, score, action[0])
        
        print(f"ベイズ最適化 {step+1}/{bayes_steps}: "
              f"score={score:.4f}, action={action[0]}")
        
        # 定期的に探索空間分析をログ
        if step % 10 == 0:
            tracker.log_search_space_analysis(policy, X)
    
    # 最終結果の取得
    history = policy.history
    best_score = max(history.fx)
    best_action_idx = np.argmax(history.fx)
    best_action = history.chosen_actions[best_action_idx]
    
    print(f"\n最適化完了!")
    print(f"最良スコア: {best_score:.4f}")
    print(f"最良アクション: {best_action}")
    print(f"最良パラメータ: {X[best_action]}")
    print(f"総評価回数: {objective.evaluation_count}")
    
    # 収束プロットの生成
    tracker.log_convergence_plot(history.fx, "Optimization Convergence")
    
    # 最終的な探索空間分析
    tracker.log_search_space_analysis(policy, X)
    
    # 最終結果のログ
    tracker.log_final_results(policy, best_action, best_score, 
                             objective.evaluation_count)
    
    return policy, objective, tracker
    
except Exception as e:
    print(f"最適化中にエラーが発生: {e}")
    mlflow.log_param("error", str(e))
    raise
finally:
    # MLflowランの終了
    tracker.end_run()
```

def main():
“”“メイン実行関数”””
# 最適化設定
config = {
‘run_name’: ‘rastrigin_optimization_demo’,
‘dimensions’: 2,  # 2次元最適化
‘points_per_dim’: 50,  # 各次元50点（総候補数: 50^2 = 2500）
‘bounds’: [(-2.0, 2.0), (-2.0, 2.0)],  # 各次元の範囲
‘target_function’: ‘rastrigin’,  # 目的関数の種類
‘seed’: 42,  # 再現性のためのシード
‘random_steps’: 10,  # ランダムサーチのステップ数
‘bayes_steps’: 30,  # ベイズ最適化のステップ数
‘acquisition_function’: ‘EI’,  # 獲得関数 (EI, PI, TS)
‘hyperparameter_update_interval’: 5,  # ハイパーパラメータ更新間隔
‘num_rand_basis’: 100,  # ランダム基底の数
}

```
print("PHYSBOとMLflowを使用したベイズ最適化を開始します...")
print("=" * 60)

# 最適化の実行
policy, objective, tracker = run_optimization_with_mlflow(config)

print("=" * 60)
print("最適化が完了しました！")
print(f"MLflow UIで結果を確認: mlflow ui --backend-store-uri file:./mlruns")
print(f"実行ID: {tracker.run_id}")
```

if **name** == “**main**”:
main()