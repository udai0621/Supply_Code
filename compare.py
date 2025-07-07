import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings(‘ignore’)

# 各フレームワークのインポート

try:
import tensorflow as tf
# CPUのみを使用するよう設定
tf.config.set_visible_devices([], ‘GPU’)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
TF_AVAILABLE = True
except ImportError:
TF_AVAILABLE = False
print(“TensorFlow not available”)

try:
import torch
# CPUのみを使用するよう設定
torch.set_num_threads(4)
TORCH_AVAILABLE = True
except ImportError:
TORCH_AVAILABLE = False
print(“PyTorch not available”)

try:
import jax
import jax.numpy as jnp
from flax import linen as nn
# CPUのみを使用するよう設定
jax.config.update(‘jax_platform_name’, ‘cpu’)
JAX_AVAILABLE = True
except ImportError:
JAX_AVAILABLE = False
print(“JAX/Flax not available”)

class BenchmarkResults:
“”“ベンチマーク結果を格納するクラス”””
def **init**(self):
self.results = {}

```
def add_result(self, framework: str, operation: str, time_taken: float):
    if framework not in self.results:
        self.results[framework] = {}
    self.results[framework][operation] = time_taken

def display_results(self):
    """結果を表形式で表示"""
    print("\n" + "="*60)
    print("フレームワーク速度比較結果 (CPU)")
    print("="*60)
    print(f"{'操作':<20} {'TensorFlow':<15} {'PyTorch':<15} {'JAX/Flax':<15}")
    print("-"*60)
    
    operations = set()
    for framework_results in self.results.values():
        operations.update(framework_results.keys())
    
    for op in sorted(operations):
        row = f"{op:<20}"
        for framework in ['TensorFlow', 'PyTorch', 'JAX/Flax']:
            if framework in self.results and op in self.results[framework]:
                time_val = self.results[framework][op]
                row += f"{time_val:.4f}s{'':<8}"
            else:
                row += f"{'N/A':<15}"
        print(row)

def plot_results(self):
    """結果をグラフで可視化"""
    if not self.results:
        print("表示する結果がありません")
        return
    
    operations = set()
    for framework_results in self.results.values():
        operations.update(framework_results.keys())
    
    frameworks = list(self.results.keys())
    x = np.arange(len(operations))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, framework in enumerate(frameworks):
        times = []
        for op in sorted(operations):
            if op in self.results[framework]:
                times.append(self.results[framework][op])
            else:
                times.append(0)
        
        ax.bar(x + i * width, times, width, label=framework, color=colors[i % len(colors)])
    
    ax.set_xlabel('操作')
    ax.set_ylabel('実行時間 (秒)')
    ax.set_title('深層学習フレームワーク速度比較 (CPU)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(sorted(operations))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

def create_sample_data(batch_size: int = 32, input_shape: Tuple[int, int, int] = (32, 32, 3)) -> np.ndarray:
“”“サンプルデータを生成”””
return np.random.randn(batch_size, *input_shape).astype(np.float32)

def benchmark_tensorflow(data: np.ndarray, num_iterations: int = 100) -> Dict[str, float]:
“”“TensorFlowのベンチマーク”””
if not TF_AVAILABLE:
return {}

```
print("TensorFlowベンチマーク実行中...")
results = {}

# シンプルなCNNモデル
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# ウォームアップ
_ = model(data[:1])

# 推論時間測定
start_time = time.time()
for _ in range(num_iterations):
    _ = model(data)
inference_time = (time.time() - start_time) / num_iterations
results['推論'] = inference_time

# 学習時間測定（1エポック）
dummy_labels = np.random.randint(0, 10, size=(data.shape[0],))
start_time = time.time()
model.fit(data, dummy_labels, epochs=1, verbose=0)
training_time = time.time() - start_time
results['学習'] = training_time

return results
```

def benchmark_pytorch(data: np.ndarray, num_iterations: int = 100) -> Dict[str, float]:
“”“PyTorchのベンチマーク”””
if not TORCH_AVAILABLE:
return {}

```
print("PyTorchベンチマーク実行中...")
results = {}

# シンプルなCNNモデル
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

model = SimpleCNN()
model.eval()  # 推論モードに設定

# データをPyTorchテンソルに変換 (NHWC -> NCHW)
torch_data = torch.from_numpy(data.transpose(0, 3, 1, 2))

# ウォームアップ
with torch.no_grad():
    _ = model(torch_data[:1])

# 推論時間測定
start_time = time.time()
with torch.no_grad():
    for _ in range(num_iterations):
        _ = model(torch_data)
inference_time = (time.time() - start_time) / num_iterations
results['推論'] = inference_time

# 学習時間測定
model.train()
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
dummy_labels = torch.randint(0, 10, (data.shape[0],))

start_time = time.time()
optimizer.zero_grad()
output = model(torch_data)
loss = criterion(output, dummy_labels)
loss.backward()
optimizer.step()
training_time = time.time() - start_time
results['学習'] = training_time

return results
```

def benchmark_jax_flax(data: np.ndarray, num_iterations: int = 100) -> Dict[str, float]:
“”“JAX/Flaxのベンチマーク”””
if not JAX_AVAILABLE:
return {}

```
print("JAX/Flaxベンチマーク実行中...")
results = {}

# シンプルなCNNモデル
class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.softmax(x)
        return x

model = SimpleCNN()

# パラメータ初期化
key = jax.random.PRNGKey(42)
jax_data = jnp.array(data)
params = model.init(key, jax_data[:1])

# JITコンパイル
@jax.jit
def predict(params, x):
    return model.apply(params, x)

# ウォームアップ
_ = predict(params, jax_data[:1])

# 推論時間測定
start_time = time.time()
for _ in range(num_iterations):
    _ = predict(params, jax_data)
inference_time = (time.time() - start_time) / num_iterations
results['推論'] = inference_time

# 学習時間測定
def loss_fn(params, x, y):
    pred = model.apply(params, x)
    return jnp.mean(-jnp.sum(y * jnp.log(pred + 1e-8), axis=1))

@jax.jit
def train_step(params, x, y):
    loss_val, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # 簡単な勾配降下法
    new_params = jax.tree_map(lambda p, g: p - 0.01 * g, params, grads)
    return new_params, loss_val

# ダミーラベル（ワンホット形式）
dummy_labels = jax.nn.one_hot(jnp.array(np.random.randint(0, 10, size=(data.shape[0],))), 10)

start_time = time.time()
params, loss_val = train_step(params, jax_data, dummy_labels)
training_time = time.time() - start_time
results['学習'] = training_time

return results
```

def main():
“”“メイン実行関数”””
print(“深層学習フレームワーク速度比較ベンチマーク (CPU)”)
print(”=”*50)

```
# 設定
batch_size = 32
input_shape = (32, 32, 3)
num_iterations = 10  # CPUでは少なめに設定

# サンプルデータ生成
print(f"サンプルデータ生成中... (バッチサイズ: {batch_size}, 入力形状: {input_shape})")
data = create_sample_data(batch_size, input_shape)

# ベンチマーク実行
results = BenchmarkResults()

# TensorFlow
if TF_AVAILABLE:
    tf_results = benchmark_tensorflow(data, num_iterations)
    for op, time_val in tf_results.items():
        results.add_result('TensorFlow', op, time_val)

# PyTorch
if TORCH_AVAILABLE:
    torch_results = benchmark_pytorch(data, num_iterations)
    for op, time_val in torch_results.items():
        results.add_result('PyTorch', op, time_val)

# JAX/Flax
if JAX_AVAILABLE:
    jax_results = benchmark_jax_flax(data, num_iterations)
    for op, time_val in jax_results.items():
        results.add_result('JAX/Flax', op, time_val)

# 結果表示
results.display_results()

# グラフ表示
try:
    results.plot_results()
except Exception as e:
    print(f"グラフ表示でエラーが発生しました: {e}")

# 簡単な分析
print("\n" + "="*60)
print("分析結果")
print("="*60)

if results.results:
    # 推論速度の比較
    inference_times = {}
    for framework, ops in results.results.items():
        if '推論' in ops:
            inference_times[framework] = ops['推論']
    
    if inference_times:
        fastest_framework = min(inference_times, key=inference_times.get)
        print(f"推論速度が最も速い: {fastest_framework} ({inference_times[fastest_framework]:.4f}s)")
    
    # 学習速度の比較
    training_times = {}
    for framework, ops in results.results.items():
        if '学習' in ops:
            training_times[framework] = ops['学習']
    
    if training_times:
        fastest_training = min(training_times, key=training_times.get)
        print(f"学習速度が最も速い: {fastest_training} ({training_times[fastest_training]:.4f}s)")

print("\n注意: CPUでの実行結果です。GPU環境では結果が大きく異なる場合があります。")
```

if **name** == “**main**”:
main()