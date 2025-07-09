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
"""

def __init__(self, experiment_name: str = "PHYSBO_Optimization"):
    self.experiment_name = experiment_name
    self.run_id = None
    self.step_count = 0
    self.best_score_history = []
    self.evaluation_history = []
    
def start_run(self, run_name: str = None):
    """MLflowランの開始"""
    if run_name is None:
        run_name = f"optimization_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    mlflow.start_run(run_name=run_name)
    self.run_id = mlflow.active_run().info.run_id
    
    # 実験の基本情報をログ
    mlflow.log_param("experiment_start_time", datetime.now().isoformat())
    mlflow.log_param("run_id", self.run_id)
    
def log_optimization_setup(self, config: Dict[str, Any]):
    """最適化設定のログ記録"""
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)):
            mlflow.log_param(key, value)
        else:
            mlflow.log_param(key, str(value))

def log_step(self, step_type: str, step_num: int, score: float, 
             action_id: int, additional_metrics: Dict[str, float] = None):
    """各最適化ステップのログ記録"""
    self.step_count += 1
    
    # 基本メトリクスのログ
    mlflow.log_metric(f"{step_type}_score", score, step=step_num)
    mlflow.log_metric(f"{step_type}_action_id", action_id, step=step_num)
    
    # ベストスコアの更新
    if not self.best_score_history or score > max(self.best_score_history):
        self.best_score_history.append(score)
        mlflow.log_metric("best_score", score, step=step_num)
    else:
        self.best_score_history.append(max(self.best_score_history))
        mlflow.log_metric("best_score", max(self.best_score_history), step=step_num)
    
    # 追加メトリクスのログ
    if additional_metrics:
        for key, value in additional_metrics.items():
            mlflow.log_metric(key, value, step=step_num)
    
    # 評価履歴の記録
    self.evaluation_history.append({
        'step': step_num,
        'step_type': step_type,
        'score': score,
        'action_id': action_id,
        'timestamp': datetime.now().isoformat()
    })

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
    
    # PHYSBOの結果オブジェクトを保存
    try:
        model_path = "physbo_policy.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(policy, f)
        mlflow.log_artifact(model_path)
    except Exception as e:
        print(f"モデル保存でエラー: {e}")
    
    # ファイルを削除
    for path in [history_path, model_path]:
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