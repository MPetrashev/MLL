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
conda create --name mll_gpu python jupyter pytorch=1.7.0 tensorflow-gpu==2.1 quandl matplotlib scikit-learn plotly nbformat
activate mll_gpu
ipython kernel install --user --name=mll_gpu
```
pytorch=1.7.0 - otherwise torch.cuda.is_available() wouldn't be true
To run tests in PyCharm, please, run `conda install -c anaconda tensorflow=2.0` outside of `mll` virtual environment.


To use __yfinance__:
```
conda install numpy=1.19.2 python=3.7.9 pandas=1.1.3
```
