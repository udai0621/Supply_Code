# pythonの3.8.0をベースにする
FROM python:3.8.0

# linuxの環境設定
RUN apt-get update \
    && apt-get upgrade -y \
    # imageのサイズを小さくするためにキャッシュ削除
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # pipのアップデート
    && pip install --upgrade pip

# requirements.txtをコピーする
COPY requirements.txt ${PWD}

# pythonのパッケージをインストール
RUN pip install -r ./requirements.txt
