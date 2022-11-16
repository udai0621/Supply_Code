class Prepro():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    def __init__(self, df):
        self.df = df
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
    
    # OneHotエンコーディング
    def one_hot(self,df):
        
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        
        # 文字型のカラムを抽出
        str_col = df.select_dtypes(include = 'O')
        # 抽出したカラムに対して、One-hot
        str_col_dummies = pd.get_dummies(df[str_col.columns], drop_first = True)
        # 特徴量と正解ラベルに分割（DF型）
        # 特徴量
        explain = df.iloc[:, :-1]
        # 正解ラベル
        target = df.iloc[:, -1]
        # One-Hotの対象となる、特徴量の列を削除
        explain = explain.drop(str_col.columns, axis = 1)
        # One-Hot化したデータを結合
        df_One_Hot = pd.concat([explain, str_col_dummies, target], axis = 1)
        return df_One_Hot

    # 必要な特徴量の抽出
    def drop_feature(self, df):
        
        df = df[['RestingBP',
                 'FastingBS',
                 'Age',
                 'Sex',
                 'Oldpeak',
                 'ST_Slope_Flat',
                 'ST_Slope_Up',
                 'ChestPainType_ATA',
                 'ChestPainType_NAP',
                 'HeartDisease']]
        
        return df

    # 特徴量と正解ラベルに分割
    def label_split(self, df):
        
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split

        from sklearn.preprocessing import StandardScaler
        
        # 特徴量
        X = np.array(df.iloc[:, :-1])
        # 標準化
        ss = StandardScaler()
        X = ss.fit_transform(X)
        # 正解ラベル
        y = np.array(df.iloc[:, -1])
        return X, y

    # 特徴量と正解ラベルを8:2の割合で分割する。(X_train, X_test, y_train, y_test)
    # 訓練データを7:3で分割する(X_train, X_valid, y_train, y_valid）
    def split(self, X, y):
        
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.3, random_state = 0, stratify = y_train)

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def output(self, y, pred):

        from sklearn.metrics import roc_curve, auc
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

        print('accuracy : ', accuracy_score(y_true = y, y_pred = pred))
        print('precision : ', precision_score(y_true =y, y_pred = pred))
        print('recall : ', recall_score(y_true = y, y_pred = pred))
        print('f1 score : ', f1_score(y_true = y, y_pred = pred))
    
    