# =============================================================================

# UI側（Dashアプリ）- main_app.py

# =============================================================================

import dash
from dash import dcc, html, Input, Output, callback, State
import pandas as pd
import subprocess
import paramiko
import json
import time
import os
from datetime import datetime
import plotly.express as px
import threading

class BayesianOptimizationClient:
def **init**(self, server_config):
self.server_config = server_config
self.job_status = {“status”: “ready”, “job_id”: None, “results”: None}

```
def transfer_and_execute(self, initial_data, candidates, job_config):
    """データ転送 + 実行をバッチで行う"""
    job_id = f"bayesopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # 1. データファイルを作成
        data_package = {
            "initial_data": initial_data,
            "candidates": candidates,
            "config": job_config,
            "job_id": job_id
        }
        
        # ローカルの一時ファイルに保存
        local_data_file = f"/tmp/{job_id}_data.parquet"
        local_config_file = f"/tmp/{job_id}_config.json"
        
        # Parquetファイルとして保存
        pd.DataFrame(data_package["initial_data"]).to_parquet(local_data_file)
        
        # 設定ファイル
        with open(local_config_file, 'w') as f:
            json.dump({
                "candidates": candidates,
                "config": job_config,
                "job_id": job_id
            }, f)
        
        # 2. SCP でファイル転送
        self._scp_transfer(local_data_file, f"/tmp/{job_id}_data.parquet")
        self._scp_transfer(local_config_file, f"/tmp/{job_id}_config.json")
        
        # 3. SSH でバッチジョブ実行
        self._execute_remote_job(job_id)
        
        self.job_status = {"status": "running", "job_id": job_id, "results": None}
        return True, job_id
        
    except Exception as e:
        return False, str(e)

def _scp_transfer(self, local_file, remote_file):
    """SCPでファイル転送"""
    cmd = [
        "scp", 
        "-i", self.server_config["key_path"],
        local_file,
        f"{self.server_config['user']}@{self.server_config['host']}:{remote_file}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"SCP transfer failed: {result.stderr}")

def _execute_remote_job(self, job_id):
    """SSH経由でリモートジョブ実行"""
    cmd = [
        "ssh",
        "-i", self.server_config["key_path"],
        f"{self.server_config['user']}@{self.server_config['host']}",
        f"python3 /path/to/bayesian_server.py {job_id}"
    ]
    
    # 非同期実行
    subprocess.Popen(cmd)

def check_job_status(self, job_id):
    """ジョブの実行状況確認"""
    try:
        # SSH経由でステータスファイル確認
        cmd = [
            "ssh",
            "-i", self.server_config["key_path"],
            f"{self.server_config['user']}@{self.server_config['host']}",
            f"cat /tmp/{job_id}_status.json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            status_data = json.loads(result.stdout)
            return status_data
        else:
            return {"status": "running"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_results(self, job_id):
    """結果ファイルを取得"""
    try:
        # 結果ファイルをSCPで取得
        local_result_file = f"/tmp/{job_id}_results.parquet"
        
        cmd = [
            "scp",
            "-i", self.server_config["key_path"],
            f"{self.server_config['user']}@{self.server_config['host']}:/tmp/{job_id}_results.parquet",
            local_result_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            results_df = pd.read_parquet(local_result_file)
            return results_df
        else:
            return None
            
    except Exception as e:
        print(f"Error getting results: {e}")
        return None
```

# Dashアプリの初期化

app = dash.Dash(**name**)

# サーバー設定

SERVER_CONFIG = {
“host”: “your-compute-server.com”,
“user”: “username”,
“key_path”: “/path/to/private_key”
}

client = BayesianOptimizationClient(SERVER_CONFIG)

app.layout = html.Div([
html.H1(“ベイズ最適化 分散処理システム”),

```
html.Div([
    html.H3("初期データ"),
    dcc.Upload(
        id="upload-initial-data",
        children=html.Div(["初期データファイルをドラッグ&ドロップ"]),
        style={
            "width": "100%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "1px", "borderStyle": "dashed",
            "borderRadius": "5px", "textAlign": "center", "margin": "10px"
        }
    ),
]),

html.Div([
    html.H3("最適化設定"),
    html.Label("候補点数:"),
    dcc.Input(id="n-candidates", type="number", value=10),
    html.Br(),
    html.Label("最適化ステップ数:"),
    dcc.Input(id="n-steps", type="number", value=20),
]),

html.Div([
    html.Button("最適化実行", id="btn-optimize", n_clicks=0),
    html.Button("状況確認", id="btn-status", n_clicks=0),
]),

html.Div(id="status-display"),
html.Div(id="results-display"),

dcc.Interval(id="interval-component", interval=5000, n_intervals=0),
dcc.Store(id="current-job-id"),
```

])

@app.callback(
[Output(“status-display”, “children”),
Output(“current-job-id”, “data”)],
[Input(“btn-optimize”, “n_clicks”)],
[State(“n-candidates”, “value”),
State(“n-steps”, “value”)]
)
def execute_optimization(n_clicks, n_candidates, n_steps):
if n_clicks == 0:
return “待機中…”, None

```
try:
    # サンプルデータ（実際にはファイルアップロードから取得）
    initial_data = [
        {"x1": 1.0, "x2": 2.0, "y": 0.5},
        {"x1": 1.5, "x2": 1.8, "y": 0.3},
    ]
    
    candidates = [
        {"x1": 2.0, "x2": 1.5},
        {"x1": 0.8, "x2": 2.2},
    ]
    
    job_config = {
        "n_candidates": n_candidates,
        "n_steps": n_steps,
        "algorithm": "gaussian_process"
    }
    
    success, result = client.transfer_and_execute(initial_data, candidates, job_config)
    
    if success:
        return f"ジョブ開始: {result}", result
    else:
        return f"エラー: {result}", None
        
except Exception as e:
    return f"実行エラー: {str(e)}", None
```

@app.callback(
Output(“results-display”, “children”),
[Input(“interval-component”, “n_intervals”)],
[State(“current-job-id”, “data”)]
)
def update_results(n_intervals, job_id):
if not job_id:
return “”

```
status = client.check_job_status(job_id)

if status.get("status") == "completed":
    results = client.get_results(job_id)
    if results is not None:
        fig = px.scatter(results, x="x1", y="x2", color="y", 
                       title="ベイズ最適化結果")
        return dcc.Graph(figure=fig)

return f"処理状況: {status.get('status', 'unknown')}"
```

# =============================================================================

# 計算サーバー側 - bayesian_server.py

# =============================================================================

import sys
import json
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import argparse
from datetime import datetime

class BayesianOptimizationServer:
def **init**(self):
self.gp = None

```
def load_job_data(self, job_id):
    """ジョブデータの読み込み"""
    try:
        # データファイル読み込み
        data_file = f"/tmp/{job_id}_data.parquet"
        config_file = f"/tmp/{job_id}_config.json"
        
        initial_data = pd.read_parquet(data_file)
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return initial_data, config
    except Exception as e:
        self.update_status(job_id, "error", str(e))
        return None, None

def update_status(self, job_id, status, message=""):
    """ステータス更新"""
    status_data = {
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"/tmp/{job_id}_status.json", 'w') as f:
        json.dump(status_data, f)

def fit_gaussian_process(self, X, y):
    """ガウシアンプロセスの学習"""
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
    self.gp.fit(X, y)

def acquisition_function(self, X_candidates, xi=0.01):
    """獲得関数（Expected Improvement）"""
    mu, sigma = self.gp.predict(X_candidates, return_std=True)
    
    # 現在の最良値
    y_best = np.max(self.gp.y_train_)
    
    # Expected Improvement計算
    with np.errstate(divide='warn'):
        imp = mu - y_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    
    return ei

def bayesian_optimization_step(self, candidates):
    """ベイズ最適化の1ステップ"""
    X_candidates = np.array([[c["x1"], c["x2"]] for c in candidates])
    
    # 獲得関数で評価
    acquisition_values = self.acquisition_function(X_candidates)
    
    # 最も有望な点を選択
    best_idx = np.argmax(acquisition_values)
    best_candidate = candidates[best_idx]
    
    return best_candidate, acquisition_values[best_idx]

def run_optimization(self, job_id):
    """最適化メイン処理"""
    try:
        self.update_status(job_id, "loading")
        
        # データ読み込み
        initial_data, config = self.load_job_data(job_id)
        if initial_data is None:
            return
        
        self.update_status(job_id, "processing")
        
        # 初期データでGP学習
        X_train = initial_data[["x1", "x2"]].values
        y_train = initial_data["y"].values
        self.fit_gaussian_process(X_train, y_train)
        
        # ベイズ最適化実行
        candidates = config["candidates"]
        n_steps = config["config"]["n_steps"]
        
        results = []
        
        for step in range(n_steps):
            # 次の点を選択
            best_candidate, ei_value = self.bayesian_optimization_step(candidates)
            
            # 実際の評価（ここは問題に応じて実装）
            y_new = self.objective_function(best_candidate["x1"], best_candidate["x2"])
            
            # 結果を記録
            results.append({
                "step": step,
                "x1": best_candidate["x1"],
                "x2": best_candidate["x2"],
                "y": y_new,
                "ei": ei_value
            })
            
            # GPを更新
            X_new = np.array([[best_candidate["x1"], best_candidate["x2"]]])
            y_new_array = np.array([y_new])
            
            X_train = np.vstack([X_train, X_new])
            y_train = np.append(y_train, y_new_array)
            
            self.fit_gaussian_process(X_train, y_train)
            
            self.update_status(job_id, f"processing", f"Step {step+1}/{n_steps}")
        
        # 結果を保存
        results_df = pd.DataFrame(results)
        results_df.to_parquet(f"/tmp/{job_id}_results.parquet")
        
        self.update_status(job_id, "completed")
        
    except Exception as e:
        self.update_status(job_id, "error", str(e))

def objective_function(self, x1, x2):
    """目的関数（例：修正されたガウス関数）"""
    return np.exp(-(x1-1)**2 - (x2-2)**2) + 0.1 * np.random.normal()
```

def main():
if len(sys.argv) != 2:
print(“Usage: python bayesian_server.py <job_id>”)
sys.exit(1)

```
job_id = sys.argv[1]

server = BayesianOptimizationServer()
server.run_optimization(job_id)
```

if **name** == “**main**”:
main()

# =============================================================================

# 設定ファイル - config.py

# =============================================================================

# サーバー設定

SERVER_CONFIG = {
“host”: “your-compute-server.com”,
“user”: “your-username”,
“key_path”: “/path/to/your/private_key”,
“port”: 22
}

# 最適化設定

OPTIMIZATION_CONFIG = {
“default_n_steps”: 20,
“default_n_candidates”: 10,
“acquisition_xi”: 0.01,
“gp_kernel_params”: {
“length_scale”: 1.0,
“alpha”: 1e-6
}
}

# ファイルパス設定

PATH_CONFIG = {
“temp_dir”: “/tmp”,
“server_script_path”: “/path/to/bayesian_server.py”
}

# =============================================================================

# 実行スクリプト - run_app.py

# =============================================================================

if **name** == “**main**”:
# 必要な依存関係をインストール
import subprocess
import sys

```
required_packages = [
    "dash", "plotly", "pandas", "scikit-learn", 
    "numpy", "paramiko", "pyarrow"
]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# アプリ起動
app.run_server(debug=True, port=8050)
```