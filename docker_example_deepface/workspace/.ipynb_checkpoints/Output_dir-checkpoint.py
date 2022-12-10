import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

from glob import glob
from deepface import DeepFace
from datetime import datetime


class Output():
    
    def __init__(self, directory):
        self.files = glob(directory + '**.jpg')
        self.directory = directory
    
    # DeepFace.defectFace()を実行
    def _defect(self):
        for i in range(len(self.files)):
            image = DeepFace.detectFace(self.files[i])
            plt.imshow(image)
            plt.show()
        
    # DeepFace.analyze()を実行
    def _analyze(self):
        object = []
        for i in range(len(self.files)):
            object.append(DeepFace.analyze(img_path = self.files[i]))
        return object
    
    # DataFrameを作成
    def _df(self):
        # 呼び出し
        object = self._analyze()
        
        emotion = []
        age = []
        gender = []
        race = []
        
        img_name = []
        
        # 要素抽出
        for i in range(len(object)):
            age.append(object[i].get('age'))
            gender.append(object[i].get('gender'))
            emotion.append(object[i].get('dominant_emotion'))
            race.append(object[i].get('dominant_race'))
            
            # pathをimage_nameに変換
            path_name = pathlib.Path(self.files[i])
            img_name.append(str(path_name)[(len(self.directory)-2):])
        
        # array作成
        arr = np.stack([age, gender, emotion, race], axis = 1)
        
        # DataFrameに変換
        return pd.DataFrame(arr, index = img_name, columns = ['age', 'gender', 'emotion', 'race'])
    
    # 結果出力
    def result(self, csv = False):
        
        df = self._df()
        
        if csv == True:
            path = './result/result_' + datetime.now().strftime("%Y%m%d-%H%M%S")
            df.to_csv(path)
        else:
            pass
        
        self._defect()
        return df
    
    # CSV出力
    def to_csv(self):
        df = self._df()
        path = './result/result_' + datetime.now().strftime("%Y%m%d-%H%M%S")
        df.to_csv(path)