import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from deepface import DeepFace
from datetime import datetime

class Output():
    
    def __init__(self, img_path):
        self.img_path = img_path
    
    # DeepFace.defectFace()を実行
    def _defect(self):
        image = DeepFace.detectFace(self.img_path)
        plt.imshow(image)
        plt.show()
        
    # DeepFace.analyze()を実行
    def _analyze(self):
        object = DeepFace.analyze(img_path = self.img_path)
        return object
    
    # DataFrameを作成
    def _df(self):
        # 呼び出し
        object = self._analyze()
        
        # 要素抽出
        emotion = object.get('dominant_emotion')
        age = object.get('age')
        gender = object.get('gender')
        race = object.get('dominant_race')
        
        # array作成
        arr = np.array([age, gender, emotion, race])
        
        # pathをimage_nameに変換
        path_name = pathlib.Path(self.img_path)
        img_name = str(path_name)[5:]
        
        # DataFrameに変換
        return pd.DataFrame(arr.reshape(1,4), index = [img_name], columns = ['age', 'gender', 'emotion', 'race'])
    
    # 結果出力
    def result(self):
        self._defect()
        return self._df()
    
    # CSV出力
    def to_csv(self):
        df = self._df()
        path = './result/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        df.to_csv(path)