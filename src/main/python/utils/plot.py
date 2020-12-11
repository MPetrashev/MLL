import pandas as pd
import datetime as dt
import plotly.graph_objs as go
import plotly.offline as py

excel_t0_date = dt.datetime(1899, 12, 30)


def from_xl_date(excel_date: int) -> dt.datetime:
    """
    :param excel_date:
    :return: Converts the excel date (int representation) into the python datetime object.
    """
    return dt.datetime.fromordinal(excel_t0_date.toordinal() + excel_date)


def plot_df(df, x_title: str = None, y_title: str = None, title: str = None):
    """
    Plots df columns as a set of lines
    :param df:
    :param title: Title of the graph
    :param x_title: X-axis title
    :param y_title: Y-axis title
    :return:
    """
    x = df.index
    data = [go.Scatter(x=x, y=values, name=column, visible='legendonly') for column, values in df.iteritems()]
    data[0].visible = data[-1].visible = True
    layout = go.Layout(showlegend=True, xaxis={'title': x_title} if x_title else None
                       , yaxis={'title': y_title} if y_title else None, title=title)
    py.iplot(go.Figure(data=data, layout=layout), filename='basic-line')


def plot_df_by_dates(df: pd.DataFrame, x_title: str = None, y_title: str = None, title: str = None
                     , date_column: str = 'Date', anchor_column: str = None) -> None:
    """
    Plots df column value as a set of lines sliced by date.
    :param df: DataFrame with columns: [Date, TimeSeries1, TimeSeries2,...]
    :param title: Title of the graph
    :param x_title: X-axis title
    :param y_title: Y-axis title
    :param date_column: The name of df column that represents dates
    :param anchor_column: The name of df column with the anchor values. The next column after the date column if missed
    :return: Plots time series against each other. To make the visualization of time series chart more representative,
    sorts the DataFrame values by @anchor_column in ascending order. So the table data:
    Date    | X     | Y
    1.1.21  | 2.2   | 2.1
    2.1.21  | 8.3   | 8.4
    ...
    31.5.21 | 3.5   | 3.4

    would be sorted and plotted as:
    Date    | X     | Y
    1.1.21  | 2.2   | 2.1
    31.5.21 | 3.5   | 3.4
    2.1.21  | 8.3   | 8.4
    ...
    """
    if not anchor_column:
        anchor_column = next(c for c in df.columns if c != date_column)
    dates = sorted(df[date_column].unique())
    buttons = []
    data = []
    n = len(df) * 2
    x = list(range(int(len(df) / len(dates))))
    for i, pricing_day in enumerate(dates):
        date = from_xl_date(int(pricing_day)).strftime('%d.%m.%Y')
        args = [False] * n
        args[2 * i] = args[2 * i + 1] = True
        buttons.append({'label': date,
                        'method': 'update',
                        'args': [{'visible': args},
                                 {'title': 'Prices at {}'.format(date)}]})
        subset = df[df[date_column] == pricing_day].sort_values(anchor_column)
        for column in df.columns:
            if column != date_column:
                data.append({'x': x, 'y': subset[column], 'name': '{} {}'.format(column, i + 1), 'visible': i == 0})

    layout = go.Layout(showlegend=True, xaxis={'title': x_title} if x_title else None
                       , yaxis={'title': y_title} if y_title else None, title=title
                       , updatemenus=[{'active': 0, 'buttons': buttons}])
    py.iplot(go.Figure(data=data, layout=layout), filename='basic-line')
