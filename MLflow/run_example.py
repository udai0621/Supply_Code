“””
PHYSBO + MLflow ベイズ最適化の実行例（動作確認済み）
“””

import numpy as np
import time
import os
import sys
from typing import Dict

# 相対インポートの問題を回避

current_dir = os.path.dirname(os.path.abspath(**file**))
sys.path.append(current_dir)

from optimizer import PHYSBOMLflowOptimizer
import mlflow

def simple_quadratic_objective(params: Dict[str, float]) -> float:
“””
シンプルな2次関数の目的関数（テスト用）
最適値: x=0.3, y=0.7で最大値1.0
“””
x = params[“x”]
y = params[“y”]

```
# 実行時間のシミュレート
time.sleep(0.01)

# 2次関数（最大化問題）
score = 1.0 - ((x - 0.3) ** 2 + (y - 0.7) ** 2)

# ノイズを追加して現実的に
score += np.random.normal(0, 0.02)

return max(0.0, score)
```

def ml_hyperparameter_objective(params: Dict[str, float]) -> float:
“””
機械学習ハイパーパラメータの目的関数（模擬）
“””
learning_rate = params[“learning_rate”]
batch_size = params[“batch_size”]
dropout_rate = params.get(“dropout_rate”, 0.0)

```
# 訓練時間のシミュレート（パラメータに依存）
training_time = 0.05 + (batch_size / 1000.0)
time.sleep(training_time)

# 学習率の影響（0.01周辺が最適）
lr_score = np.exp(-((np.log10(learning_rate) + 2) / 0.3) ** 2)

# バッチサイズの影響（64周辺が最適）
batch_score = np.exp(-((batch_size - 64) / 20) ** 2)

# ドロップアウトの影響（0.2周辺が最適）
dropout_score = np.exp(-((dropout_rate - 0.2) / 0.1) ** 2)

# 総合スコア
total_score = 0.7 + 0.1 * lr_score + 0.1 * batch_score + 0.1 * dropout_score

# ノイズを追加
total_score += np.random.normal(0, 0.01)

return max(0.0, min(1.0, total_score))
```

def chemistry_experiment_objective(params: Dict[str, float]) -> float:
“””
化学実験の目的関数（反応収率の模擬）
“””
temperature = params[“temperature”]
pressure = params[“pressure”]
concentration = params[“concentration”]

```
# 実験時間のシミュレート
time.sleep(0.02)

# 最適条件: temp=80, pressure=3.0, conc=0.6
temp_effect = np.exp(-((temperature - 80) / 15) ** 2)
pressure_effect = np.exp(-((pressure - 3.0) / 1.0) ** 2)
conc_effect = np.exp(-((concentration - 0.6) / 0.2) ** 2)

# 相互作用効果
interaction = 0.1 * np.sin(temperature / 10) * np.cos(pressure) * concentration

yield_rate = temp_effect * pressure_effect * conc_effect + interaction

# 実験ノイズ
yield_rate += np.random.normal(0, 0.05)

return max(0.0, min(1.0, yield_rate))
```

def run_simple_optimization():
“””
シンプルな2次元最適化の実行
“””
print(”=== シンプル2次元最適化 ===”)

```
search_space = {
    "x": [0.0, 1.0],
    "y": [0.0, 1.0]
}

optimizer = PHYSBOMLflowOptimizer(
    search_space=search_space,
    objective_function=simple_quadratic_objective,
    config_type="default",
    experiment_name="simple_2d_optimization"
)

# 最適化実行
best_params, best_score = optimizer.optimize(max_iterations=20)

print(f"最良スコア: {best_score:.4f}")
print(f"最良パラメータ: x={best_params.get('x', 0):.3f}, y={best_params.get('y', 0):.3f}")
print(f"理論最適値: x=0.3, y=0.7")

return optimizer
```

def run_ml_hyperparameter_optimization():
“””
機械学習ハイパーパラメータ最適化の実行
“””
print(”\n=== 機械学習ハイパーパラメータ最適化 ===”)

```
search_space = {
    "learning_rate": [0.001, 0.1],
    "batch_size": [16, 128],
    "dropout_rate": [0.0, 0.5]
}

optimizer = PHYSBOMLflowOptimizer(
    search_space=search_space,
    objective_function=ml_hyperparameter_objective,
    config_type="thorough",
    experiment_name="ml_hyperparameter_optimization"
)

# 最適化実行
best_params, best_score = optimizer.optimize(max_iterations=30)

print(f"最良精度: {best_score:.4f}")
print("最良ハイパーパラメータ:")
for param, value in best_params.items():
    print(f"  {param}: {value:.4f}")

return optimizer
```

def run_chemistry_experiment_optimization():
“””
化学実験条件最適化の実行
“””
print(”\n=== 化学実験条件最適化 ===”)

```
search_space = {
    "temperature": [50.0, 120.0],
    "pressure": [1.0, 5.0],
    "concentration": [0.1, 1.0]
}

optimizer = PHYSBOMLflowOptimizer(
    search_space=search_space,
    objective_function=chemistry_experiment_objective,
    config_type="fast",
    experiment_name="chemistry_experiment_optimization"
)

# 最適化実行
best_params, best_score = optimizer.optimize(max_iterations=25)

print(f"最高収率: {best_score:.4f}")
print("最適実験条件:")
for param, value in best_params.items():
    print(f"  {param}: {value:.2f}")
print("理論最適値: temperature=80, pressure=3.0, concentration=0.6")

return optimizer
```

def compare_acquisition_functions():
“””
獲得関数の比較実験
“””
print(”\n=== 獲得関数比較 ===”)

```
search_space = {
    "x": [-2.0, 2.0],
    "y": [-2.0, 2.0]
}

def himmelblau_function(params):
    """Himmelblau関数（最大化版）"""
    x, y = params["x"], params["y"]
    time.sleep(0.01)
    # 元の関数を反転して最大化問題に
    value = -((x**2 + y - 11)**2 + (x + y**2 - 7)**2)
    # 正規化して0-1の範囲に
    normalized = (value + 2000) / 2000
    return max(0.0, min(1.0, normalized))

acquisition_functions = ["EI", "PI", "TS"]
results = {}

for acq_func in acquisition_functions:
    print(f"\n{acq_func}による最適化中...")
    
    optimizer = PHYSBOMLflowOptimizer(
        search_space=search_space,
        objective_function=himmelblau_function,
        config_type="default",
        experiment_name=f"acquisition_comparison_{acq_func}"
    )
    
    # 獲得関数の変更
    optimizer.config["score"] = acq_func
    
    # 最適化実行
    best_params, best_score = optimizer.optimize(max_iterations=20)
    
    results[acq_func] = {
        "best_score": best_score,
        "best_params": best_params,
        "iterations": len(optimizer.get_optimization_history())
    }
    
    print(f"{acq_func}: スコア={best_score:.4f}, 試行回数={results[acq_func]['iterations']}")

# 結果比較
print("\n=== 獲得関数比較結果 ===")
best_acq = max(results.keys(), key=lambda k: results[k]["best_score"])
print(f"最良の獲得関数: {best_acq} (スコア: {results[best_acq]['best_score']:.4f})")

return results
```

def main():
“””
メイン実行関数
“””
print(“PHYSBO + MLflow ベイズ最適化デモ”)
print(”=” * 50)

```
# 乱数シードの設定
np.random.seed(42)

try:
    # 1. シンプルな最適化
    simple_opt = run_simple_optimization()
    
    # 2. 機械学習ハイパーパラメータ最適化
    ml_opt = run_ml_hyperparameter_optimization()
    
    # 3. 化学実験最適化
    chem_opt = run_chemistry_experiment_optimization()
    
    # 4. 獲得関数比較
    acq_results = compare_acquisition_functions()
    
    print("\n" + "=" * 50)
    print("すべてのデモが完了しました！")
    print("\nMLflow UIで結果を確認してください:")
    print("  $ mlflow ui")
    print("  ブラウザで http://localhost:5000 を開く")
    
    print("\n実験結果サマリー:")
    print(f"- シンプル最適化: {simple_opt.best_score:.4f}")
    print(f"- ML最適化: {ml_opt.best_score:.4f}")
    print(f"- 化学実験最適化: {chem_opt.best_score:.4f}")
    
except Exception as e:
    print(f"実行エラー: {e}")
    import traceback
    traceback.print_exc()
```

if **name** == “**main**”:
main()