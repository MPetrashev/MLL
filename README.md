# MLL

## Installation

Create virtual environment:
```
conda create --name mll jupyter ipykernel pytorch==1.7.1 plotly matplotlib pandas scipy -c pytorch
activate mll
ipython kernel install --user --name=mll
```

```
conda create --name mll_gpu python=3.6.6 jupyter quandl matplotlib scikit-learn plotly nbformat keras
conda create --name tf_cluster_test python=3.6.6
activate tf_cluster_test
pip install tensorflow==2.2.0 numpy==1.19.3
pip install pytorch==1.7.0 ipykernel
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
## GPU Memory exception

* Kill `dwm` process (usually it takes most Dedicated GPU memory: see Task Manager | Details tab) 

## AutoML
```
conda create --name automl pip requests tabulate scikit-learn jupyter ipykernel
activate automl
ipython kernel install --user --name=automl
pip uninstall h2o
pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
```

## Best config/params lookup
* Original:
    * Torch: Mean error = 0.0001, StDev = 0.0012
* v * 100. put price, n_epochs = 6000, n_layers = 4, n_hidden = 100
    * Torch: Mean error = 0.0042583657125902195, 0.0905587802924326
    * TFApproximator (5m 30.298s): -0.010638369223945138, 0.08173625354977596
    * TorchApproximator (1m 24.836s): 0.0042583657125902195, 0.0905587802924326