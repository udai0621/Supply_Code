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

RUN apt-get -y update && \
    apt-get install -y build-essential g++-8 libopenblas-dev \
            libgtk2.0-dev pkg-config python-dev python-numpy \
            libgl1-mesa-dev

# requirements.txtをコピーする
COPY requirements.txt ${PWD}

# pythonのパッケージをインストール
RUN pip install -r ./requirements.txt