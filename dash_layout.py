import dash
from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# サンプルデータを作成

np.random.seed(42)
dates = pd.date_range(‘2024-01-01’, periods=30)
data1 = np.random.randn(30).cumsum()
data2 = np.random.randn(30) * 10 + 50
data3 = np.random.choice([‘A’, ‘B’, ‘C’, ‘D’], 30)
data4 = np.random.randn(30) * 5 + 20

# グラフを作成する関数

def create_sample_graphs():
# グラフ1: 線グラフ
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=dates, y=data1, mode=‘lines’, name=‘データ1’))
fig1.update_layout(title=“時系列データ”, height=300, margin=dict(l=20, r=20, t=40, b=20))

```
# グラフ2: 散布図
fig2 = px.scatter(x=data2, y=data4, title="散布図")
fig2.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))

# グラフ3: 棒グラフ
fig3 = px.bar(x=['A', 'B', 'C', 'D'], 
              y=[list(data3).count(i) for i in ['A', 'B', 'C', 'D']], 
              title="カテゴリ別集計")
fig3.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))

# グラフ4: ヒストグラム
fig4 = px.histogram(x=data4, title="分布グラフ")
fig4.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))

return fig1, fig2, fig3, fig4
```

# Dashアプリを初期化

app = dash.Dash(**name**)

# グラフを作成

fig1, fig2, fig3, fig4 = create_sample_graphs()

# レイアウトを定義

app.layout = html.Div([
html.H1(“ダッシュボード - (2,2), (3,1)レイアウト”,
style={‘text-align’: ‘center’, ‘margin-bottom’: ‘30px’}),

```
# 最初の行: (2,2)の比率
html.Div([
    html.Div([
        dcc.Graph(figure=fig1)
    ], style={
        'width': '66.67%',  # 2/3の幅
        'display': 'inline-block',
        'vertical-align': 'top'
    }),
    html.Div([
        dcc.Graph(figure=fig2)
    ], style={
        'width': '33.33%',  # 1/3の幅
        'display': 'inline-block',
        'vertical-align': 'top'
    })
], style={'margin-bottom': '20px'}),

# 2番目の行: (3,1)の比率
html.Div([
    html.Div([
        dcc.Graph(figure=fig3)
    ], style={
        'width': '75%',     # 3/4の幅
        'display': 'inline-block',
        'vertical-align': 'top'
    }),
    html.Div([
        dcc.Graph(figure=fig4)
    ], style={
        'width': '25%',     # 1/4の幅
        'display': 'inline-block',
        'vertical-align': 'top'
    })
])
```

], style={
‘margin’: ‘20px’,
‘font-family’: ‘Arial, sans-serif’
})

if **name** == ‘**main**’:
app.run_server(debug=True)