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
conda create --name mll python ipykernel tensorflow quandl matplotlib
ipython kernel install --user --name=mll
```
To run tests in PyCharm, please, run `conda install -c anaconda tensorflow=2.0` outside of `mll` virtual environment.
