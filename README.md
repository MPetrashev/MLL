# MLL

## Installation

Create virtual environment:

```
conda install -c anaconda tensorflow=2.0
pip install ipykernel
ipython kernel install --user --name=mll
conda create --name mll
activate mll
conda install -c anaconda quandl
```
In this code we do install `tensorflow` not in the `mll` virtual environemnt to fix a problem with PyCharm which otherwise can't run tests.
