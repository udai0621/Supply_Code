“””
PHYSBOMLflowOptimizerの使用例
“””

import numpy as np
import time
from typing import Dict
from optimizer import PHYSBOMLflowOptimizer, MultiAcquisitionOptimizer
from config import SEARCH_SPACE_EXAMPLES

def objective_function_example1(params: Dict[str, float]) -> float:
“””
機械学習のハイパーパラメータ最適化の例

```
Args:
    params: ハイパーパラメータ
    
Returns:
    検証精度（模擬）
"""
# 実際の機械学習の訓練をシミュレート
time.sleep(0.1)  # 訓練時間のシミュレート

# 仮の目的関数（実際にはモデル訓練→検証精度）
lr = params["learning_rate"]
batch_size = params["batch_size"]
dropout = params["dropout_rate"]
layers = params["num_layers"]

# 複雑な非線形関数で精度をシミュレート
score = 0.8 + 0.15 * np.exp(-((lr - 0.01) / 0.02) ** 2)
score += 0.1 * np.exp(-((batch_size - 64) / 32) ** 2)
score -= 0.2 * dropout  # ドロップアウトが高いと精度低下
score += 0.05 * np.exp(-((layers - 3) / 2) ** 2)

# ノイズを追加して現実的に
score += np.random.normal(0, 0.02)

return min(1.0, max(0.0, score))  # 0-1の範囲にクリップ
```

def objective_function_example2(params: Dict[str, float]) -> float:
“””
物理実験パラメータ最適化の例

```
Args:
    params: 実験パラメータ
    
Returns:
    実験結果（模擬）
"""
# 実験の実行時間をシミュレート
time.sleep(0.05)

temp = params["temperature"]
pressure = params["pressure"]
conc = params["concentration"]

# 化学反応の収率をシミュレート
# 最適値: temp=60, pressure=5, conc=0.5
optimal_temp = 60.0
optimal_pressure = 5.0
optimal_conc = 0.5

# ガウス型の目的関数
score = np.exp(-((temp - optimal_temp) / 20) ** 2)
score *= np.exp(-((pressure - optimal_pressure) / 2) ** 2)
score *= np.exp(-((conc - optimal_conc) / 0.2) ** 2)

# 実験ノイズを追加
score += np.random.normal(0, 0.05)

return max(0.0, score)
```

def simple_optimization_example():
“””
シンプルな最適化の例
“””
print(”=== シンプルな最適化の例 ===”)

```
# 探索空間の定義
search_space = SEARCH_SPACE_EXAMPLES["ml_hyperparameters"]

# オプティマイザーの初期化
optimizer = PHYSBOMLflowOptimizer(
    search_space=search_space,
    objective_function=objective_function_example1,
    config_type="default",
    experiment_name="simple_ml_hyperparameter_optimization"
)

# 最適化の実行
best_params, best_score = optimizer.optimize(max_iterations=30)

# 結果の表示
print(f"最良スコア: {best_score:.4f}")
print("最良パラメータ:")
for param, value in best_params.items():
    print(f"  {param}: {value:.4f}")

print(f"総試行回数: {len(optimizer.get_optimization_history())}")

return optimizer
```

def physical_experiment_example():
“””
物理実験パラメータ最適化の例
“””
print(”\n=== 物理実験パラメータ最適化の例 ===”)

```
# 探索空間の定義
search_space = SEARCH_SPACE_EXAMPLES["experimental_parameters"]

# オプティマイザーの初期化
optimizer = PHYSBOMLflowOptimizer(
    search_space=search_space,
    objective_function=objective_function_example2,
    config_type="thorough",  # より徹底的な探索
    experiment_name="physical_experiment_optimization"
)

# 最適化の実行
best_params, best_score = optimizer.optimize(max_iterations=50)

# 結果の表示
print(f"最良収率: {best_score:.4f}")
print("最適実験条件:")
for param, value in best_params.items():
    print(f"  {param}: {value:.2f}")

return optimizer
```

def acquisition_function_comparison_example():
“””
獲得関数の比較実験の例
“””
print(”\n=== 獲得関数比較実験の例 ===”)

```
# 探索空間の定義（シンプルな2次元問題）
search_space = {
    "x": [-5.0, 5.0],
    "y": [-5.0, 5.0]
}

def simple_2d_objective(params):
    """シンプルな2次元目的関数"""
    x, y = params["x"], params["y"]
    # Himmelblau関数の変形
    return -((x**2 + y - 11)**2 + (x + y**2 - 7)**2) / 1000 + 1

# 複数獲得関数での比較
comparator = MultiAcquisitionOptimizer(
    search_space=search_space,
    objective_function=simple_2d_objective,
    acquisition_functions=["EI", "PI", "TS"]
)

# 比較実験の実行
results = comparator.compare_acquisition_functions(max_iterations=25)

# 結果の表示
print("獲得関数別の結果:")
for acq_func, result in results.items():
    print(f"  {acq_func}: 最良スコア = {result['best_score']:.4f}")
    print(f"       収束速度 = {len(result['history'])} 試行")

return comparator
```

def custom_objective_example():
“””
カスタム目的関数の例（より複雑なケース）
“””
print(”\n=== カスタム目的関数の例 ===”)

```
def complex_objective(params):
    """より複雑な目的関数の例"""
    # 複数の変数を使った複雑な関数
    lr = params["learning_rate"]
    batch_size = params["batch_size"]
    dropout = params["dropout_rate"]
    layers = params["num_layers"]
    
    # 実行時間のシミュレート（バッチサイズに依存）
    time.sleep(batch_size / 1000)  # バッチサイズが大きいほど時間がかかる
    
    # 複雑な非線形関数
    score = 0.9  # ベースライン
    
    # 学習率の影響（最適値周辺でピーク）
    lr_effect = np.exp(-((np.log10(lr) + 2) / 0.5) ** 2) * 0.1
    
    # バッチサイズの影響（64周辺が最適）
    batch_effect = np.exp(-((batch_size - 64) / 32) ** 2) * 0.05
    
    # ドロップアウトの影響（0.3周辺が最適）
    dropout_effect = np.exp(-((dropout - 0.3) / 0.1) ** 2) * 0.03
    
    # レイヤー数の影響（少なすぎても多すぎてもダメ）
    layer_effect = np.exp(-((layers - 4) / 2) ** 2) * 0.02
    
    # 相互作用効果
    interaction = (lr * 1000) * (1 - dropout) * np.exp(-layers / 5) * 0.01
    
    final_score = score + lr_effect + batch_effect + dropout_effect + layer_effect + interaction
    
    # ノイズを追加
    final_score += np.random.normal(0, 0.01)
    
    return max(0.0, min(1.0, final_score))

# カスタム探索空間
custom_search_space = {
    "learning_rate": [0.0001, 0.1],
    "batch_size": [8, 256],
    "dropout_rate": [0.0, 0.8],
    "num_layers": [1, 10]
}

optimizer = PHYSBOMLflowOptimizer(
    search_space=custom_search_space,
    objective_function=complex_objective,
    config_type="thorough",
    experiment_name="complex_hyperparameter_optimization"
)

# 長時間の最適化
best_params, best_score = optimizer.optimize(max_iterations=80)

print(f"複雑な目的関数での最良スコア: {best_score:.4f}")
print("最良パラメータ:")
for param, value in best_params.items():
    print(f"  {param}: {value:.6f}")

return optimizer
```

def analyze_results_example():
“””
MLflowに記録された結果の分析例
“””
print(”\n=== 結果分析の例 ===”)

```
from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()

# 全実験の取得
experiments = client.search_experiments()

print("実行済み実験:")
for exp in experiments:
    if exp.name != "Default":  # デフォルト実験を除く
        print(f"  実験名: {exp.name}")
        
        # 実験内のランを取得
        runs = client.search_runs(exp.experiment_id)
        if runs:
            print(f"    総ラン数: {len(runs)}")
            
            # 最良結果の取得
            best_run = max(runs, key=lambda x: x.data.metrics.get("final_best_score", float('-inf')))
            best_score = best_run.data.metrics.get("final_best_score")
            if best_score:
                print(f"    最良スコア: {best_score:.4f}")
            
            # 獲得関数の分布
            acq_functions = [run.data.tags.get("acquisition_function") for run in runs]
            acq_functions = [af for af in acq_functions if af]  # Noneを除く
            if acq_functions:
                from collections import Counter
                acq_counts = Counter(acq_functions)
                print(f"    使用獲得関数: {dict(acq_counts)}")

# より詳細な分析のための例
try:
    # 特定の実験の詳細分析
    exp = client.get_experiment_by_name("simple_ml_hyperparameter_optimization")
    if exp:
        runs = client.search_runs(exp.experiment_id)
        
        # メトリクスをDataFrameに変換
        metrics_data = []
        for run in runs:
            row = {"run_id": run.info.run_id}
            row.update(run.data.metrics)
            row.update(run.data.params)
            metrics_data.append(row)
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            print(f"\nDataFrame形式での分析（{len(df)}件のラン）:")
            print(df.head())
            
            # 相関分析の例
            if "final_best_score" in df.columns and "total_iterations" in df.columns:
                correlation = df["final_best_score"].astype(float).corr(
                    df["total_iterations"].astype(float)
                )
                print(f"最良スコアと反復回数の相関: {correlation:.3f}")

except Exception as e:
    print(f"詳細分析でエラー: {e}")
```

def main():
“””
メイン実行関数
“””
print(“PHYSBO + MLflow ベイズ最適化の例”)
print(”=” * 50)

```
try:
    # 1. シンプルな最適化
    simple_optimizer = simple_optimization_example()
    
    # 2. 物理実験の例
    physical_optimizer = physical_experiment_example()
    
    # 3. 獲得関数の比較
    acquisition_comparator = acquisition_function_comparison_example()
    
    # 4. 複雑な目的関数
    complex_optimizer = custom_objective_example()
    
    # 5. 結果の分析
    analyze_results_example()
    
    print("\n" + "=" * 50)
    print("すべての実験が完了しました！")
    print("MLflow UIで結果を確認してください:")
    print("  $ mlflow ui")
    print("  ブラウザで http://localhost:5000 を開く")
    
except Exception as e:
    print(f"実行エラー: {e}")
    import traceback
    traceback.print_exc()
```

if **name** == “**main**”:
# 実行前の準備
import warnings
warnings.filterwarnings(“ignore”)  # 警告を抑制

```
# 乱数シードの設定（再現性のため）
np.random.seed(42)

# メイン実行
main()
```