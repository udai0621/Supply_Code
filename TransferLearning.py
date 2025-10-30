“””
自作モデルで転移学習を実装するサンプルコード

シナリオ:
同じ説明変数(センサーデータ)、異なる目的変数での転移学習

タスクA(事前学習): センサーデータ → 温度異常検知
タスクB(転移学習): 同じセンサーデータ → 振動異常検知
“””

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 再現性のための乱数シード設定

np.random.seed(42)
tf.random.set_seed(42)

print(f”TensorFlow version: {tf.**version**}”)

# ===================================

# 1. データ生成関数

# ===================================

def generate_sensor_data(n_samples, noise_level=0.1):
“””
共通のセンサーデータを生成する関数

```
戻り値:
- X: センサー値(10次元)
- y_temp: 温度異常フラグ(タスクA用)
- y_vibration: 振動異常フラグ(タスクB用)
"""
X = np.zeros((n_samples, 10))  # 10個のセンサー値(説明変数は共通)
y_temp = np.zeros(n_samples)  # タスクA: 温度異常検知
y_vibration = np.zeros(n_samples)  # タスクB: 振動異常検知

for i in range(n_samples):
    # 共通の時系列パターンを生成
    t = np.linspace(0, 2*np.pi, 10)
    base_signal = np.sin(t) + np.cos(2*t) * 0.5
    
    # ランダムな振幅と位相
    amplitude = np.random.uniform(0.5, 1.5)
    phase = np.random.uniform(0, np.pi)
    
    X[i] = amplitude * np.sin(t + phase) + base_signal * 0.3
    X[i] += np.random.normal(0, noise_level, 10)
    
    # タスクA: 温度異常(振幅が大きい場合)
    if amplitude > 1.2:
        y_temp[i] = 1
    else:
        y_temp[i] = 0
    
    # タスクB: 振動異常(位相が特定範囲の場合)
    if phase > np.pi/3 and phase < 2*np.pi/3:
        y_vibration[i] = 1
    else:
        y_vibration[i] = 0

return X, y_temp, y_vibration
```

# ===================================

# 2. タスクA用データ(大量)を生成

# ===================================

print(”\n=== ステップ1: タスクA(温度異常検知)用データを生成 ===”)

# 大量データを生成

X_taskA, y_taskA, _ = generate_sensor_data(n_samples=5000)

# 訓練/検証に分割

split_idx = int(len(X_taskA) * 0.8)
X_taskA_train = X_taskA[:split_idx]
y_taskA_train = y_taskA[:split_idx]
X_taskA_val = X_taskA[split_idx:]
y_taskA_val = y_taskA[split_idx:]

print(f”タスクA訓練データ: {len(X_taskA_train)} サンプル”)
print(f”タスクA検証データ: {len(X_taskA_val)} サンプル”)
print(f”  説明変数: 10次元のセンサー値”)
print(f”  目的変数: 温度異常フラグ(0/1)”)

# ===================================

# 3. 基本モデルの構築

# ===================================

print(”\n=== ステップ2: タスクA用の基本モデルを構築 ===”)

def create_base_model(input_shape=(10,), model_name=‘base_model’):
“””
基本となるニューラルネットワークモデルを構築

```
構造:
- 特徴抽出部: 説明変数から重要な特徴を学習(転移学習で再利用)
- 分類ヘッド: タスク固有の予測(転移学習では置き換え)
"""
model = keras.Sequential([
    # 入力層
    keras.layers.Input(shape=input_shape),
    
    # === 特徴抽出部(転移学習で再利用される) ===
    keras.layers.Dense(64, activation='relu', name='feature_layer_1'),
    keras.layers.BatchNormalization(name='bn_1'),
    keras.layers.Dropout(0.3, name='dropout_1'),
    
    keras.layers.Dense(32, activation='relu', name='feature_layer_2'),
    keras.layers.BatchNormalization(name='bn_2'),
    keras.layers.Dropout(0.3, name='dropout_2'),
    
    keras.layers.Dense(16, activation='relu', name='feature_layer_3'),
    keras.layers.BatchNormalization(name='bn_3'),
    
    # === タスク固有の分類ヘッド(転移学習では置き換え) ===
    keras.layers.Dense(1, activation='sigmoid', name='taskA_classification_head')
], name=model_name)

return model
```

# タスクA用モデルを作成

model_taskA = create_base_model(model_name=‘taskA_model’)
model_taskA.summary()

print(”\n【ポイント】”)
print(”- feature_layer_*: センサーデータから特徴を抽出(転移学習で再利用)”)
print(”- taskA_classification_head: 温度異常を予測(転移学習では置き換え)”)

# ===================================

# 4. タスクAで事前学習の実行

# ===================================

print(”\n=== ステップ3: タスクA(温度異常検知)で事前学習を実行 ===”)

# モデルをコンパイル

model_taskA.compile(
optimizer=keras.optimizers.Adam(learning_rate=0.001),
loss=‘binary_crossentropy’,
metrics=[‘accuracy’]
)

# 事前学習の実行

print(“タスクAで学習中…”)
taskA_history = model_taskA.fit(
X_taskA_train,
y_taskA_train,
validation_data=(X_taskA_val, y_taskA_val),
epochs=20,
batch_size=32,
verbose=1
)

print(”\nタスクAの学習完了!”)
print(f”最終訓練精度: {taskA_history.history[‘accuracy’][-1]:.4f}”)
print(f”最終検証精度: {taskA_history.history[‘val_accuracy’][-1]:.4f}”)

# 事前学習したモデルを保存

model_taskA.save(‘taskA_pretrained_model.h5’)
print(“タスクA学習済みモデルを保存しました”)

# ===================================

# 5. タスクB用データ(少量)を生成

# ===================================

print(”\n=== ステップ4: タスクB(振動異常検知)用データを生成 ===”)

# 少量データを生成(説明変数は同じ、目的変数が異なる)

X_taskB, _, y_taskB = generate_sensor_data(n_samples=200)  # タスクAの1/25!

# 訓練/テストに分割

split_idx = int(len(X_taskB) * 0.7)
X_taskB_train = X_taskB[:split_idx]
y_taskB_train = y_taskB[:split_idx]
X_taskB_test = X_taskB[split_idx:]
y_taskB_test = y_taskB[split_idx:]

print(f”タスクB訓練データ: {len(X_taskB_train)} サンプル”)
print(f”タスクBテストデータ: {len(X_taskB_test)} サンプル”)
print(f”  説明変数: 10次元のセンサー値(タスクAと同じ!)”)
print(f”  目的変数: 振動異常フラグ(0/1) ← タスクAと異なる!”)

# ===================================

# 6. 転移学習モデルの構築

# ===================================

print(”\n=== ステップ5: 転移学習モデルを構築 ===”)

def create_transfer_model(pretrained_model_path, freeze_layers=True):
“””
事前学習済みモデルから転移学習モデルを構築

```
手順:
1. タスクAで学習したモデルをロード
2. 最後の分類ヘッド(タスクA用)を削除
3. 特徴抽出部の重みを固定(freeze_layers=True)
4. 新しい分類ヘッド(タスクB用)を追加
"""
# タスクAで学習したモデルをロード
base_model = keras.models.load_model(pretrained_model_path)

print("\n【転移学習の準備】")
print(f"1. タスクA学習済みモデルをロード: {pretrained_model_path}")

# 最後の分類ヘッド(タスクA用)を削除
base_model.pop()
print("2. タスクA用の分類ヘッドを削除")

# 特徴抽出層を固定するかどうか
if freeze_layers:
    print("3. 特徴抽出層の重みを固定:")
    for layer in base_model.layers:
        layer.trainable = False
        print(f"   - '{layer.name}' を固定(タスクAで学んだ知識を保持)")

# タスクB用の新しい分類ヘッドを追加
x = base_model.output
x = keras.layers.Dropout(0.5, name='taskB_dropout')(x)
output = keras.layers.Dense(1, activation='sigmoid', 
                            name='taskB_classification_head')(x)

# 転移学習モデルを作成
transfer_model = keras.Model(inputs=base_model.input, outputs=output,
                             name='taskB_transfer_model')

print("4. タスクB用の新しい分類ヘッドを追加")

return transfer_model
```

# 転移学習モデルを作成

model_taskB_transfer = create_transfer_model(‘taskA_pretrained_model.h5’,
freeze_layers=True)
print(”\n【転移学習モデルの構造】”)
model_taskB_transfer.summary()

# ===================================

# 7. 転移学習の実行

# ===================================

print(”\n=== ステップ6: タスクB(振動異常検知)で転移学習を実行 ===”)

# モデルをコンパイル

model_taskB_transfer.compile(
optimizer=keras.optimizers.Adam(learning_rate=0.001),
loss=‘binary_crossentropy’,
metrics=[‘accuracy’]
)

# 転移学習の実行

print(“タスクBで転移学習中…”)
print(“→ タスクAで学んだ特徴抽出能力を活用!”)
taskB_transfer_history = model_taskB_transfer.fit(
X_taskB_train,
y_taskB_train,
validation_split=0.2,
epochs=30,
batch_size=8,
verbose=1
)

print(”\n転移学習完了!”)

# ===================================

# 8. 比較用: タスクBをゼロから学習

# ===================================

print(”\n=== ステップ7: 比較のため、タスクBをゼロから学習 ===”)

# 同じ構造の新しいモデル(重みはランダム初期化)

model_taskB_scratch = create_base_model(model_name=‘taskB_scratch_model’)

# 最後の層の名前を変更

model_taskB_scratch.layers[-1]._name = ‘taskB_scratch_head’

model_taskB_scratch.compile(
optimizer=keras.optimizers.Adam(learning_rate=0.001),
loss=‘binary_crossentropy’,
metrics=[‘accuracy’]
)

# ゼロから学習

print(“タスクBをゼロから学習中…”)
print(“→ 事前知識なし、少量データだけで学習”)
taskB_scratch_history = model_taskB_scratch.fit(
X_taskB_train,
y_taskB_train,
validation_split=0.2,
epochs=30,
batch_size=8,
verbose=0
)

print(“ゼロからの学習完了!”)

# ===================================

# 9. 結果の比較

# ===================================

print(”\n” + “=”*60)
print(”=== ステップ8: 結果を比較 ===”)
print(”=”*60)

# テストデータで評価

transfer_loss, transfer_acc = model_taskB_transfer.evaluate(
X_taskB_test, y_taskB_test, verbose=0
)
scratch_loss, scratch_acc = model_taskB_scratch.evaluate(
X_taskB_test, y_taskB_test, verbose=0
)

print(”\n【テストデータでの性能比較】”)
print(f”転移学習(タスクA→B)  - 精度: {transfer_acc:.4f}, 損失: {transfer_loss:.4f}”)
print(f”ゼロから学習(タスクB) - 精度: {scratch_acc:.4f}, 損失: {scratch_loss:.4f}”)
print(f”\n改善率: {(transfer_acc - scratch_acc) / scratch_acc * 100:+.2f}%”)

print(”\n【なぜ転移学習が有効か】”)
print(”- タスクAで「センサーデータの特徴抽出方法」を学習済み”)
print(”- タスクBでは、その知識を活用して「振動異常の判定方法」だけを学習”)
print(”- 少量データでも、ゼロから学習より高精度を実現!”)

# ===================================

# 10. 学習曲線の可視化

# ===================================

print(”\n=== ステップ9: 学習曲線を可視化 ===”)

plt.figure(figsize=(14, 5))

# 精度の比較

plt.subplot(1, 2, 1)
plt.plot(taskB_transfer_history.history[‘accuracy’],
label=‘転移学習(訓練)’, linewidth=2, color=’#2E86AB’)
plt.plot(taskB_transfer_history.history[‘val_accuracy’],
label=‘転移学習(検証)’, linewidth=2, color=’#A23B72’)
plt.plot(taskB_scratch_history.history[‘accuracy’],
label=‘ゼロから(訓練)’, linestyle=’–’, linewidth=2, color=’#F18F01’)
plt.plot(taskB_scratch_history.history[‘val_accuracy’],
label=‘ゼロから(検証)’, linestyle=’–’, linewidth=2, color=’#C73E1D’)
plt.title(‘タスクB: モデル精度の比較’, fontsize=14, fontweight=‘bold’)
plt.xlabel(‘エポック’)
plt.ylabel(‘精度’)
plt.legend()
plt.grid(True, alpha=0.3)

# 損失の比較

plt.subplot(1, 2, 2)
plt.plot(taskB_transfer_history.history[‘loss’],
label=‘転移学習(訓練)’, linewidth=2, color=’#2E86AB’)
plt.plot(taskB_transfer_history.history[‘val_loss’],
label=‘転移学習(検証)’, linewidth=2, color=’#A23B72’)
plt.plot(taskB_scratch_history.history[‘loss’],
label=‘ゼロから(訓練)’, linestyle=’–’, linewidth=2, color=’#F18F01’)
plt.plot(taskB_scratch_history.history[‘val_loss’],
label=‘ゼロから(検証)’, linestyle=’–’, linewidth=2, color=’#C73E1D’)
plt.title(‘タスクB: モデル損失の比較’, fontsize=14, fontweight=‘bold’)
plt.xlabel(‘エポック’)
plt.ylabel(‘損失’)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(‘transfer_learning_taskB_comparison.png’, dpi=150, bbox_inches=‘tight’)
print(“グラフを ‘transfer_learning_taskB_comparison.png’ に保存しました”)

# ===================================

# 11. より高度な転移学習(段階的解凍)

# ===================================

print(”\n=== ステップ10: より高度な転移学習(段階的解凍) ===”)
print(“特徴抽出層の一部を訓練可能にして、さらに微調整…”)

# 後半の特徴抽出層を訓練可能にする

trainable_layers = [‘feature_layer_3’, ‘bn_3’, ‘taskB_dropout’,
‘taskB_classification_head’]

for layer in model_taskB_transfer.layers:
if layer.name in trainable_layers:
layer.trainable = True
print(f”  ✓ ‘{layer.name}’ を訓練可能に設定”)

# より小さな学習率で再コンパイル

model_taskB_transfer.compile(
optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # 学習率を下げる
loss=‘binary_crossentropy’,
metrics=[‘accuracy’]
)

# ファインチューニング

print(”\nファインチューニング中…”)
finetune_history = model_taskB_transfer.fit(
X_taskB_train,
y_taskB_train,
validation_split=0.2,
epochs=20,
batch_size=8,
verbose=0
)

# 最終評価

finetune_loss, finetune_acc = model_taskB_transfer.evaluate(
X_taskB_test, y_taskB_test, verbose=0
)

print(f”\nファインチューニング後の精度: {finetune_acc:.4f}”)
print(f”改善: {(finetune_acc - transfer_acc):+.4f}”)

# ===================================

# まとめ

# ===================================

print(”\n” + “=”*60)
print(”=== 転移学習のサンプル実行完了! ===”)
print(”=”*60)

print(”\n【実験のまとめ】”)
print(f”✓ タスクA(温度異常): {len(X_taskA_train)}サンプルで事前学習”)
print(f”✓ タスクB(振動異常): わずか{len(X_taskB_train)}サンプルで転移学習”)
print(f”✓ 説明変数は共通、目的変数が異なるケース”)
print(f”✓ 転移学習の優位性: ゼロから学習より {(transfer_acc - scratch_acc) / scratch_acc * 100:+.1f}% 性能向上”)

print(”\n【転移学習が有効な理由】”)
print(“1. センサーデータの特徴抽出能力はタスクA/Bで共通”)
print(“2. タスクAで学んだ知識を、タスクBで再利用”)
print(“3. タスクBでは判定ロジックだけを学習すればOK”)
print(“→ 少量データでも高精度を実現!”)