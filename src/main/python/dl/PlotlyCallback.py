from typing import Tuple
from tensorflow.python.keras.callbacks import LambdaCallback
import logging
logger = logging.getLogger(__file__)


import plotly.graph_objects as go


def plotly_callback(n_epochs: int) -> Tuple[go.FigureWidget, LambdaCallback]:
    """
    :return: Figure widget (to be plotted in Jupyter Notebook cell) and callback instance which can be passed to tf.Model.fit method to plot interactively current loss values
    """
    figure = f = go.FigureWidget(layout = {
        'title':'Epoch losses',
        'xaxis':{
            'title': '# Epoch',
            'range':[0,n_epochs-1]
        },
        'yaxis':{'title': 'Loss'}
    })

    def on_epoch_end(epoch, logs):
        loss = logs['loss']
        logger.info('Epoch: %s,\t Loss: %.3f', epoch, loss)
        if not figure.data:
            figure.add_scatter(y=[loss])
        else:
            figure.data[0].y += (loss,)

    return figure, LambdaCallback(on_epoch_end=on_epoch_end)
