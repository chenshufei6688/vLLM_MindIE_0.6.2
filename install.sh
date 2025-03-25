set -e

if [ -d "./vllm" ]; then
    echo "./vllm directory has already exist!"
    exit 1
fi

git clone -b v0.6.2 https://github.com/vllm-project/vllm.git vllm

yes | cp -r cover/* vllm/

cd vllm
pip install -r requirements-npu.txt
python setup.py install