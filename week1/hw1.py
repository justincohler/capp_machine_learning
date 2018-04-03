import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Image
import pygal
from functools import reduce
import collections
from datetime import datetime

def retrieveData():
    """Return dataframes for alley, graffiti, and building 311 calls from downloaded csvs."""
    dt = datetime(2017, 1, 1, 0, 0, 0, 0)

    alley = pd.read_csv("311_Service_Requests_-_Alley_Lights_Out.csv")
    alley["Creation Date"] = pd.to_datetime(alley["Creation Date"])
    alley["Completion Date"] = pd.to_datetime(alley["Completion Date"])
    alley = alley[alley["Creation Date"] >= dt]

    graffiti = pd.read_csv("311_Service_Requests_-_Graffiti_Removal.csv")
    graffiti["Creation Date"] = pd.to_datetime(graffiti["Creation Date"])
    graffiti["Completion Date"] = pd.to_datetime(graffiti["Completion Date"])
    graffiti = graffiti[graffiti["Creation Date"] >= dt]

    building = pd.read_csv("311_Service_Requests_-_Vacant_and_Abandoned_buildings_Reported.csv")
    building["DATE SERVICE REQUEST WAS RECEIVED"] = pd.to_datetime(building["DATE SERVICE REQUEST WAS RECEIVED"])
    building = building[building["DATE SERVICE REQUEST WAS RECEIVED"] >= dt]

    return (alley, graffiti, building)

def groupByType(df, type):
    """Return a list of key-values for plotting of a dataframe grouped by the given type."""
    df_groups = dfByType(df, type)

    subtypes = []

    for index, row in df_groups.iterrows():
        subtypes.append({"value": row['COUNT'], "label": row[type]})

    return subtypes

def dfByType(df, type):
    """Return a dataframe grouped by type and counted."""
    return df[[type]].groupby([type]).size().reset_index(name='COUNT')

def galplot(chart):
    """
    Credit https://stackoverflow.com/questions/36322683/pygal-charts-not-displaying-tooltips-in-jupyter-ipython-notebook.

    As a workaround for pygal's SVG interactivity, the following HTML snippet renders in Jupyter.
    """
    base_html = """
        <!DOCTYPE html>
        <html>
          <head>
          <script type="text/javascript" src="http://kozea.github.com/pygal.js/javascripts/svg.jquery.js"></script>
          <script type="text/javascript" src="https://kozea.github.io/pygal.js/2.0.x/pygal-tooltips.min.js""></script>
          </head>
          <body>
            <figure>
              {rendered_chart}
            </figure>
          </body>
        </html>
        """

    rendered_chart = chart.render(is_unicode=True)
    plot_html = base_html.format(rendered_chart=rendered_chart)
    display(HTML(plot_html))
