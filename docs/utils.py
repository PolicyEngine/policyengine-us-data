import plotly.io as pio
from IPython.display import HTML


def show(fig):
    html = pio.to_html(fig)
    return HTML(html)
