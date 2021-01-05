# MLL

## Installation

Create virtual environment:
```
conda create --name mll
activate mll
conda install -c anaconda tensorflow quandl ipykernel
ipython kernel install --user --name=mll
```

```
conda create --name mll_gpu python=3.6.6 jupyter quandl matplotlib scikit-learn plotly nbformat
conda create --name tf_cluster_test python=3.6.6
activate tf_cluster_test
pip install tensorflow==2.2.0 numpy==1.19.3
pip install pytorch==1.7.0
ipython kernel install --user --name=mll_gpu
```
pytorch=1.7.0 - otherwise torch.cuda.is_available() wouldn't be true
To run tests in PyCharm, please, run `conda install -c anaconda tensorflow=2.0` outside of `mll` virtual environment.


To use __yfinance__:

!!!Last installation tensorflow==2.1.0 for C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin:
```
conda create --name tf_cluster_test python=3.7.9 numpy=1.19.2 tensorflow==2.1.0 pandas=1.1.3
activate tf_cluster_test
```
