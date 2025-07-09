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
```