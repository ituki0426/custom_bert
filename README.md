# Get Start

```shell
 docker compose up --build 
```

```shell
# Python3.10.15のインストール
pyenv install 3.10.15
# Python3.10.15を規定バージョンに指定
pyenv global 3.10.15
# 仮想環境の作成
python -m venv test_env
# 仮想環境の有効化
source test_env/bin/activate
# pipのアップグレード
python -m pip install -U pip
# PyTorchのインストール
# https://pytorch.org/ を参照
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

```