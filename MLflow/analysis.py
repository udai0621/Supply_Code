“””
MLflowに記録されたベイズ最適化結果の分析ツール
“””

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional, Tuple
import sqlite3
from config import MLFLOW_CONFIG, VISUALIZATION_CONFIG

class BayesianOptimizationAnalyzer:
“””
ベイズ最適化結果の分析クラス
“””

```
def __init__(self, tracking_uri: Optional[str] = None):
    """
    初期化
    
    Args:
        tracking_uri: MLflowのトラッキングURI
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif MLFLOW_CONFIG.get("tracking_uri"):
        mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    
    self.client = MlflowClient()
    
def get_all_experiments(self) -> List[Dict]:
    """
    すべての実験の基本情報を取得
    
    Returns:
        実験情報のリスト
    """
    experiments = self.client.search_experiments()
    exp_info = []
    
    for exp in experiments:
        if exp.name != "Default":
            runs = self.client.search_runs(exp.experiment_id)
            
            # 基本統計の計算
            scores = [run.data.metrics.get("final_best_score") for run in runs]
            scores = [s for s in scores if s is not None]
            
            info = {
                "experiment_name": exp.name,
                "experiment_id": exp.experiment_id,
                "total_runs": len(runs),
                "completed_runs": len(scores),
                "best_score": max(scores) if scores else None,
                "avg_score": np.mean(scores) if scores else None,
                "std_score": np.std(scores) if scores else None
            }
            exp_info.append(info)
    
    return exp_info

def get_experiment_details(self, experiment_name: str) -> pd.DataFrame:
    """
    特定の実験の詳細データを取得
    
    Args:
        experiment_name: 実験名
        
    Returns:
        実験データのDataFrame
    """
    exp = self.client.get_experiment_by_name(experiment_name)
    if not exp:
        raise ValueError(f"実験 '{experiment_name}' が見つかりません")
    
    runs = self.client.search_runs(exp.experiment_id)
    
    # データの整理
    data = []
    for run in runs:
        row = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "status": run.info.status,
            "start_time": pd.to_datetime(run.info.start_time, unit='ms'),
            "end_time": pd.to_datetime(run.info.end_time, unit='ms') if run.info.end_time else None
        }
        
        # メトリクスの追加
        row.update(run.data.metrics)
        
        # パラメータの追加
        row.update(run.data.params)
        
        # タグの追加
        for key, value in run.data.tags.items():
            if not key.startswith("mlflow."):
                row[f"tag_{key}"] = value
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # データ型の変換
    numeric_columns = ["final_best_score", "total_iterations", "execution_time", 
                      "learning_rate", "batch_size", "dropout_rate"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def analyze_convergence(self, experiment_name: str) -> Dict:
    """
    収束性の分析
    
    Args:
        experiment_name: 実験名
        
    Returns:
        収束分析の結果
    """
    df = self.get_experiment_details(experiment_name)
    
    analysis = {
        "total_experiments": len(df),
        "convergence_stats": {},
        "performance_stats": {}
    }
    
    if "total_iterations" in df.columns and "final_best_score" in df.columns:
        # 収束統計
        analysis["convergence_stats"] = {
            "avg_iterations": df["total_iterations"].mean(),
            "median_iterations": df["total_iterations"].median(),
            "min_iterations": df["total_iterations"].min(),
            "max_iterations": df["total_iterations"].max()
        }
        
        # 性能統計
        analysis["performance_stats"] = {
            "best_score": df["final_best_score"].max(),
            "avg_score": df["final_best_score"].mean(),
            "worst_score": df["final_best_score"].min(),
            "score_std": df["final_best_score"].std()
        }
        
        # 効率性分析
        if len(df) > 1:
            correlation = df["total_iterations"].corr(df["final_best_score"])
            analysis["efficiency"] = {
                "iterations_score_correlation": correlation,
                "efficient_runs": len(df[
                    (df["total_iterations"] < df["total_iterations"].median()) & 
                    (df["final_best_score"] > df["final_best_score"].median())
                ])
            }
    
    return analysis

def compare_acquisition_functions(self, experiment_name: str) -> Dict:
    """
    獲得関数の比較分析
    
    Args:
        experiment_name: 実験名
        
    Returns:
        獲得関数の比較結果
    """
    df = self.get_experiment_details(experiment_name)
    
    if "tag_acquisition_function" not in df.columns:
        return {"error": "獲得関数の情報が見つかりません"}
    
    comparison = {}
    
    for acq_func in df["tag_acquisition_function"].unique():
        if pd.isna(acq_func):
            continue
            
        subset = df[df["tag_acquisition_function"] == acq_func]
        
        comparison[acq_func] = {
            "runs_count": len(subset),
            "best_score": subset["final_best_score"].max() if "final_best_score" in subset.columns else None,
            "avg_score": subset["final_best_score"].mean() if "final_best_score" in subset.columns else None,
            "avg_iterations": subset["total_iterations"].mean() if "total_iterations" in subset.columns else None,
            "success_rate": len(subset[subset["final_best_score"] > df["final_best_score"].median()]) / len(subset)
            if "final_best_score" in subset.columns and len(subset) > 0 else 0
        }
    
    return comparison

def visualize_optimization_progress(self, experiment_name: str, save_plots: bool = True):
    """
    最適化進行状況の可視化
    
    Args:
        experiment_name: 実験名
        save_plots: プロットを保存するかどうか
    """
    df = self.get_experiment_details(experiment_name)
    
    fig, axes = plt.subplots(2, 2, figsize=VISUALIZATION_CONFIG["figure_size"])
    fig.suptitle(f'Optimization Analysis: {experiment_name}', fontsize=16)
    
    # 1. スコア分布
    if "final_best_score" in df.columns:
        axes[0, 0].hist(df["final_best_score"].dropna(), bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].axvline(df["final_best_score"].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["final_best_score"].mean():.4f}')
        axes[0, 0].set_xlabel('Final Best Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 反復回数 vs スコア
    if "total_iterations" in df.columns and "final_best_score" in df.columns:
        axes[0, 1].scatter(df["total_iterations"], df["final_best_score"], alpha=0.6)
        axes[0, 1].set_xlabel('Total Iterations')
        axes[0, 1].set_ylabel('Final Best Score')
        axes[0, 1].set_title('Iterations vs Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 相関係数を表示
        corr = df["total_iterations"].corr(df["final_best_score"])
        axes[0, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[0, 1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 3. 獲得関数別の性能比較
    if "tag_acquisition_function" in df.columns and "final_best_score" in df.columns:
        acq_scores = []
        acq_labels = []
        for acq_func in df["tag_acquisition_function"].unique():
            if pd.notna(acq_func):
                scores = df[df["tag_acquisition_function"] == acq_func]["final_best_score"].dropna()
                if len(scores) > 0:
                    acq_scores.append(scores)
                    acq_labels.append(acq_func)
        
        if acq_scores:
            axes[1, 0].boxplot(acq_scores, labels=acq_labels)
            axes[1, 0].set_ylabel('Final Best Score')
            axes[1, 0].set_title('Performance by Acquisition Function')
            axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 実行時間の分析
    if "execution_time" in df.columns:
        df_clean = df.dropna(subset=["execution_time"])
        if len(df_clean) > 0:
            axes[1, 1].hist(df_clean["execution_time"], bins=20, alpha=0.7, color='lightgreen')
            axes[1, 1].set_xlabel('Execution Time (seconds)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Execution Time Distribution')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"analysis_{experiment_name.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=VISUALIZATION_CONFIG["dpi"])
        print(f"分析グラフを保存しました: {filename}")
    
    plt.show()

def parameter_importance_analysis(self, experiment_name: str) -> Dict:
    """
    パラメータ重要度の分析
    
    Args:
        experiment_name: 実験名
        
    Returns:
        パラメータ重要度の結果
    """
    df = self.get_experiment_details(experiment_name)
    
    if "final_best_score" not in df.columns:
        return {"error": "スコア情報が見つかりません"}
    
    # 数値パラメータを特定
    numeric_params = []
    for col in df.columns:
        if col not in ["run_id", "run_name", "status", "start_time", "end_time", "final_best_score"] and \
           df[col].dtype in ['float64', 'int64']:
            numeric_params.append(col)
    
    importance = {}
    
    for param in numeric_params:
        param_data = df[[param, "final_best_score"]].dropna()
        if len(param_data) > 1:
            correlation = param_data[param].corr(param_data["final_best_score"])
            importance[param] = {
                "correlation": correlation,
                "abs_correlation": abs(correlation),
                "data_points": len(param_data)
            }
    
    # 重要度順にソート
    sorted_importance = dict(sorted(importance.items(), 
                                  key=lambda x: x[1]["abs_correlation"], 
                                  reverse=True))
    
    return sorted_importance

def generate_summary_report(self, experiment_name: str) -> str:
    """
    実験のサマリーレポートを生成
    
    Args:
        experiment_name: 実験名
        
    Returns:
        レポートのテキスト
    """
    df = self.get_experiment_details(experiment_name)
    convergence = self.analyze_convergence(experiment_name)
    acq_comparison = self.compare_acquisition_functions(experiment_name)
    param_importance = self.parameter_importance_analysis(experiment_name)
    
    report = f"""
```

ベイズ最適化実験レポート: {experiment_name}
{’=’*60}

実験概要:

- 総実行回数: {len(df)}
- 完了回数: {convergence.get(‘total_experiments’, 0)}
- 実験期間: {df[‘start_time’].min()} ～ {df[‘start_time’].max()}

性能統計:
“””

```
    if convergence.get("performance_stats"):
        stats = convergence["performance_stats"]
        report += f"""- 最良スコア: {stats.get('best_score', 'N/A'):.4f}
```

- 平均スコア: {stats.get(‘avg_score’, ‘N/A’):.4f}
- 最悪スコア: {stats.get(‘worst_score’, ‘N/A’):.4f}
- スコア標準偏差: {stats.get(‘score_std’, ‘N/A’):.4f}
  “””
  
  ```
    if convergence.get("convergence_stats"):
        conv = convergence["convergence_stats"]
        report += f"""
  ```

収束統計:

- 平均反復回数: {conv.get(‘avg_iterations’, ‘N/A’):.1f}
- 中央値反復回数: {conv.get(‘median_iterations’, ‘N/A’):.1f}
- 最少反復回数: {conv.get(‘min_iterations’, ‘N/A’)}
- 最多反復回数: {conv.get(‘max_iterations’, ‘N/A’)}
  “””
  
  ```
    if acq_comparison and "error" not in acq_comparison:
        report += "\n獲得関数比較:\n"
        for acq_func, stats in acq_comparison.items():
            report += f"- {acq_func}: 最良={stats.get('best_score', 'N/A'):.4f}, "
            report += f"平均={stats.get('avg_score', 'N/A'):.4f}, "
            report += f"成功率={stats.get('success_rate', 'N/A'):.2%}\n"
    
    if param_importance and "error" not in param_importance:
        report += "\nパラメータ重要度（相関係数）:\n"
        for param, stats in list(param_importance.items())[:5]:  # 上位5つ
            report += f"- {param}: {stats.get('correlation', 'N/A'):.3f}\n"
    
    return report
  ```

def main():
“””
分析のメイン実行関数
“””
analyzer = BayesianOptimizationAnalyzer()

```
print("MLflow ベイズ最適化結果分析ツール")
print("="*50)

# 全実験の一覧
experiments = analyzer.get_all_experiments()

if not experiments:
    print("分析可能な実験が見つかりません。")
    return

print("利用可能な実験:")
for i, exp in enumerate(experiments):
    print(f"{i+1}. {exp['experiment_name']} (実行回数: {exp['total_runs']})")

# ユーザーに実験選択を促す（実際の使用では自動化可能）
print("\n最初の実験を分析します...")
selected_exp = experiments[0]["experiment_name"]

print(f"\n=== {selected_exp} の分析 ===")

# 詳細分析
try:
    # サマリーレポート
    report = analyzer.generate_summary_report(selected_exp)
    print(report)
    
    # 可視化
    analyzer.visualize_optimization_progress(selected_exp)
    
    # 詳細データの表示
    df = analyzer.get_experiment_details(selected_exp)
    print(f"\n詳細データ (最初の5行):")
    print(df.head())
    
except Exception as e:
    print(f"分析エラー: {e}")
```

if **name** == “**main**”:
main()