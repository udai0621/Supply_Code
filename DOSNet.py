import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import Conv, Dense, BatchNorm
from typing import Sequence, Optional, Callable
import functools

class DOSFeaturizer(nn.Module):
“”“DOS特徴抽出器 - 状態密度から特徴を自動抽出”””
features: Sequence[int] = (64, 128, 256)
kernel_sizes: Sequence[int] = (3, 3, 3)
strides: Sequence[int] = (2, 2, 2)

```
@nn.compact
def __call__(self, x, train: bool = True):
    # 初期のダウンサンプリング（論文では4倍）
    x = nn.avg_pool(x, window_shape=(4,), strides=(4,))
    
    # 1D畳み込み層のスタック
    for i, (features, kernel_size, stride) in enumerate(
        zip(self.features, self.kernel_sizes, self.strides)
    ):
        x = Conv(
            features=features,
            kernel_size=(kernel_size,),
            strides=(stride,),
            padding='SAME',
            use_bias=False,
            name=f'conv_{i}'
        )(x)
        x = BatchNorm(use_running_average=not train, name=f'bn_{i}')(x)
        x = nn.relu(x)
        
        # 追加のaverage pooling
        if i < len(self.features) - 1:
            x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
    
    return x
```

class DOSNet(nn.Module):
“””
DOSNet - 状態密度から吸着エネルギーを予測

```
論文: "Machine learned features from density of states for 
      accurate adsorption energy prediction" (Nature Communications, 2021)
著者: Victor Fung et al.
"""
num_outputs: int = 1  # 回帰問題（吸着エネルギー予測）
dos_featurizer_features: Sequence[int] = (64, 128, 256)
hidden_dims: Sequence[int] = (512, 256, 128)
multi_adsorbate: bool = False  # 複数の吸着分子を同時に学習するか
num_adsorbates: int = 1  # 吸着分子の種類数

def setup(self):
    self.dos_featurizer = DOSFeaturizer(
        features=self.dos_featurizer_features,
        name='dos_featurizer'
    )
    
    # 全結合層
    self.dense_layers = [
        Dense(features=dim, name=f'dense_{i}')
        for i, dim in enumerate(self.hidden_dims)
    ]
    
    # 出力層
    self.output_layer = Dense(
        features=self.num_outputs,
        name='output'
    )
    
    # 複数吸着分子用の埋め込み
    if self.multi_adsorbate:
        self.adsorbate_embedding = nn.Embed(
            num_embeddings=self.num_adsorbates,
            features=64,
            name='adsorbate_embed'
        )

@nn.compact
def __call__(self, 
             surface_dos, 
             adsorbate_dos=None,
             adsorbate_id=None,
             train: bool = True):
    """
    Args:
        surface_dos: 表面原子のDOS [batch_size, num_atoms, dos_length, num_orbitals]
        adsorbate_dos: 吸着分子のDOS [batch_size, dos_length] (オプション)
        adsorbate_id: 吸着分子のID [batch_size] (multi_adsorbateがTrueの場合)
        train: 訓練モードかどうか
    Returns:
        吸着エネルギーの予測値 [batch_size, num_outputs]
    """
    batch_size = surface_dos.shape[0]
    num_atoms = surface_dos.shape[1]
    
    # 各原子のDOSを独立に処理（重み共有）
    # surface_dos: [batch, atoms, dos_length, orbitals] 
    # -> [batch*atoms, dos_length, orbitals]
    surface_dos_reshaped = surface_dos.reshape(-1, 
                                              surface_dos.shape[2], 
                                              surface_dos.shape[3])
    
    # DOS特徴抽出（重み共有）
    surface_features = self.dos_featurizer(
        surface_dos_reshaped, train=train
    )  # [batch*atoms, feature_length, feature_dims]
    
    # 形状を戻す
    surface_features = surface_features.reshape(
        batch_size, num_atoms, -1
    )  # [batch, atoms, flattened_features]
    
    # 原子レベルの特徴を統合（平均プーリング）
    surface_features = jnp.mean(surface_features, axis=1)  # [batch, features]
    
    # 吸着分子のDOSがある場合の処理
    if adsorbate_dos is not None:
        # 吸着分子のDOS特徴抽出
        adsorbate_features = self.dos_featurizer(
            adsorbate_dos, train=train
        )  # [batch, feature_length, feature_dims]
        
        # 平坦化
        adsorbate_features = adsorbate_features.reshape(
            batch_size, -1
        )  # [batch, flattened_features]
        
        # 表面特徴と結合
        combined_features = jnp.concatenate([
            surface_features, adsorbate_features
        ], axis=-1)
    else:
        combined_features = surface_features
    
    # 複数吸着分子の場合の埋め込み
    if self.multi_adsorbate and adsorbate_id is not None:
        adsorbate_embed = self.adsorbate_embedding(adsorbate_id)
        combined_features = jnp.concatenate([
            combined_features, adsorbate_embed
        ], axis=-1)
    
    # 全結合層による回帰
    x = combined_features
    for dense_layer in self.dense_layers:
        x = dense_layer(x)
        x = nn.relu(x)
        if train:
            x = nn.dropout(x, rate=0.1, deterministic=False)
    
    # 最終出力
    output = self.output_layer(x)
    
    return output
```

def create_dosnet_model(
num_outputs: int = 1,
dos_featurizer_features: Sequence[int] = (64, 128, 256),
hidden_dims: Sequence[int] = (512, 256, 128),
multi_adsorbate: bool = False,
num_adsorbates: int = 12  # H, C, N, O, S, CH, CH2, CH3, NH, OH, SH, etc.
):
“”“DOSNetモデルの作成”””
return DOSNet(
num_outputs=num_outputs,
dos_featurizer_features=dos_featurizer_features,
hidden_dims=hidden_dims,
multi_adsorbate=multi_adsorbate,
num_adsorbates=num_adsorbates
)

def init_dosnet_model(model, key, input_shapes):
“”“モデルの初期化”””
# input_shapes: {‘surface_dos’: (batch, atoms, dos_length, orbitals), …}
batch_size = 1

```
# ダミー入力の作成
dummy_inputs = {}
dummy_inputs['surface_dos'] = jnp.ones((batch_size, *input_shapes['surface_dos']))

if 'adsorbate_dos' in input_shapes:
    dummy_inputs['adsorbate_dos'] = jnp.ones((batch_size, *input_shapes['adsorbate_dos']))

if model.multi_adsorbate:
    dummy_inputs['adsorbate_id'] = jnp.zeros((batch_size,), dtype=jnp.int32)

# 初期化
variables = model.init(key, **dummy_inputs, train=False)
return variables
```

# 損失関数（論文ではLogCosh損失を使用）

def logcosh_loss(predictions, targets):
“”“LogCosh損失 - 外れ値に対してロバスト”””
diff = predictions - targets
return jnp.mean(jnp.log(jnp.cosh(diff)))

def mae_loss(predictions, targets):
“”“平均絶対誤差”””
return jnp.mean(jnp.abs(predictions - targets))

def mse_loss(predictions, targets):
“”“平均二乗誤差”””
return jnp.mean((predictions - targets) ** 2)

# 訓練ステップ

@jax.jit
def train_step(state, batch, loss_fn=logcosh_loss):
“”“単一の訓練ステップ”””
def loss_function(params):
predictions = state.apply_fn(
params,
batch[‘surface_dos’],
adsorbate_dos=batch.get(‘adsorbate_dos’),
adsorbate_id=batch.get(‘adsorbate_id’),
train=True
)
loss = loss_fn(predictions, batch[‘targets’])
return loss, predictions

```
grad_fn = jax.value_and_grad(loss_function, has_aux=True)
(loss, predictions), grads = grad_fn(state.params)

new_state = state.apply_gradients(grads=grads)

metrics = {
    'loss': loss,
    'mae': mae_loss(predictions, batch['targets'])
}

return new_state, metrics
```

@jax.jit
def eval_step(state, batch):
“”“評価ステップ”””
predictions = state.apply_fn(
state.params,
batch[‘surface_dos’],
adsorbate_dos=batch.get(‘adsorbate_dos’),
adsorbate_id=batch.get(‘adsorbate_id’),
train=False
)

```
metrics = {
    'mae': mae_loss(predictions, batch['targets']),
    'mse': mse_loss(predictions, batch['targets'])
}

return predictions, metrics
```

# データ前処理関数

def preprocess_dos_data(surface_dos, adsorbate_dos=None):
“”“DOS データの標準化前処理”””
# 平均0、分散1に標準化（論文の方法）
surface_dos_normalized = (surface_dos - jnp.mean(surface_dos, keepdims=True)) / (
jnp.std(surface_dos, keepdims=True) + 1e-8
)

```
if adsorbate_dos is not None:
    adsorbate_dos_normalized = (adsorbate_dos - jnp.mean(adsorbate_dos, keepdims=True)) / (
        jnp.std(adsorbate_dos, keepdims=True) + 1e-8
    )
    return surface_dos_normalized, adsorbate_dos_normalized

return surface_dos_normalized
```

# 使用例

if **name** == “**main**”:
# モデル作成
model = create_dosnet_model(
num_outputs=1,
multi_adsorbate=True,
num_adsorbates=12
)

```
# 入力形状の定義
input_shapes = {
    'surface_dos': (3, 2000, 9),  # (atoms, dos_length, orbitals)
    'adsorbate_dos': (2000, 1)    # (dos_length, 1)
}

# 初期化
key = jax.random.PRNGKey(42)
variables = init_dosnet_model(model, key, input_shapes)

print("DOSNet モデルが正常に初期化されました！")
print(f"パラメータ数: {sum(x.size for x in jax.tree_leaves(variables['params']))}")

# サンプル予測
batch_size = 4
sample_batch = {
    'surface_dos': jnp.ones((batch_size, 3, 2000, 9)),
    'adsorbate_dos': jnp.ones((batch_size, 2000, 1)),
    'adsorbate_id': jnp.array([0, 1, 2, 0]),  # H, C, N, H
    'targets': jnp.array([[0.1], [-0.5], [1.2], [0.3]])
}

# 前処理
sample_batch['surface_dos'], sample_batch['adsorbate_dos'] = preprocess_dos_data(
    sample_batch['surface_dos'], sample_batch['adsorbate_dos']
)

# 予測
predictions = model.apply(
    variables,
    sample_batch['surface_dos'],
    adsorbate_dos=sample_batch['adsorbate_dos'],
    adsorbate_id=sample_batch['adsorbate_id'],
    train=False
)

print(f"予測結果: {predictions}")
print("形状:", predictions.shape)
```

# 論文で使用された具体的なアーキテクチャ情報：

“””
論文の詳細：

- DOSの解像度: 0.01 eV
- エネルギー範囲: -14 ~ 8 eV (合計2200点)
- 初期ダウンサンプリング: 4倍 (550点に)
- 軌道数: 9 (s, px, py, pz, dxy, dyz, dz2, dxz, dx2-y2)
- 最大原子数: 3 (hollow siteの場合)
- 損失関数: LogCosh (外れ値に対してロバスト)
- オプティマイザー: Adam (lr=0.001)
- エポック数: 60
- バッチサイズ: 16-128
- 平均絶対誤差: 0.116 eV (全吸着分子統合モデル)

データセット:

- 37,000個の吸着エネルギー
- 2,000個のユニークな二元合金表面
- 吸着分子: H, C, N, O, S, CH, CH2, CH3, NH, OH, SH
  “””