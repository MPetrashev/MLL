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
conda create --name mll_gpu python ipykernel tensorflow-gpu quandl matplotlib
activate mll_gpu
ipython kernel install --user --name=mll_gpu
```
To run tests in PyCharm, please, run `conda install -c anaconda tensorflow=2.0` outside of `mll` virtual environment.
