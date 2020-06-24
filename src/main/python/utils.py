import plotly.graph_objs as go


class lazy_property():
    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__ if hasattr(fget, '__name__') else fget.__func__.__name__

    def __get__(self, obj, cls):
        if obj is None:
            value = self.fget.__func__()
            setattr(cls, self.func_name, value)
        else:
            value = self.fget(obj)
            setattr(obj, self.func_name, value)

        return value


def plot_df(df, x_title: str = None, y_title: str = None, title: str = None):
    """
    Plots df columns as a set of lines
    :param df:
    :param x_title:
    :param y_title:
    :param title:
    :return:
    """
    x = df.index
    data = [go.Scatter(x=x, y=values, name=column) for column, values in df.iteritems()] # todo is df.itertuples faster?
    layout = go.Layout(showlegend=True, xaxis={'title': x_title} if x_title else None
                       , yaxis={'title': y_title} if y_title else None, title=title)
    return go.FigureWidget(data=data, layout=layout)
