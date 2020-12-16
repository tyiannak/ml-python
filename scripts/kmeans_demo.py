"""

Instructions:

Maintainer: Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

# -*- coding: utf-8 -*-
import argparse
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

colors = {'background': '#111111', 'text': '#7FDBDD'}
def_font = dict(family="Courier New, Monospace", size=10, color='#000000')

# global vars related to data loaded
global data
data = {'x': []}
global clusters
clusters = {'means': [], 'labels': []}

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

def_layout_margin = dict(l=30, r=30, b=30, t=130, pad=0)
def_layout_height = 190

def generate_data(data_val):
    global data
    n_samples = 10
    global is_training
    is_training = False
    global w
    w = np.random.random((3,))
    global clusters
    clusters = {'means': [], 'labels': []}
    if data_val == "data1":
        m1 = [-2, -2]
        m2 = [2, 2]
        a = np.concatenate([np.random.normal(m1[0], 1, n_samples),
                            np.random.normal(m2[0], 1, n_samples)])
        b = np.concatenate([np.random.normal(m1[1], 1, n_samples),
                            np.random.normal(m2[1], 1, n_samples)])

        data['x'] = np.array([a, b]).T
    if data_val == "data2":
        m1 = [-4.5, -0.5]
        m2 = [0.5, 0.5]
        a = np.concatenate([np.random.normal(m1[0], 1, n_samples),
                            np.random.normal(m2[0], 1, n_samples)])
        b = np.concatenate([np.random.normal(m1[1], 1, n_samples),
                            np.random.normal(m2[1], 1, n_samples)])

        data['x'] = np.array([a, b]).T

    if data_val == "data3":
        data['x'] = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])


def draw_data():
    global data
    if len(data['x'])>0:
        figure = {'data': [
                            go.Scatter(x=data['x'][:, 0],
                                       y=data['x'][:, 1],
                                       mode='markers', name='y = -1',
                                       marker_size=10,
                                       marker_color='rgba(22, 182, 255, .9)'
                                       )
                       ],
            'layout': go.Layout(
                xaxis=dict(range=[min(data['x'][:, 0]) - np.mean(np.abs(data['x'][:, 0])),
                                  max(data['x'][:, 0]) + np.mean(np.abs(data['x'][:, 0]))
                        ]),
                yaxis=dict(range=[min(data['x'][:, 1]) - np.mean(np.abs(data['x'][:, 1])),
                                  max(data['x'][:, 1]) + np.mean(np.abs(data['x'][:, 1]))
                                                  ]))}
    else:
        figure = {}
    return dcc.Graph(figure=figure)


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="BPP SIMULATION UI")
    return parser.parse_args()


def get_layout():
    """
    Initialize the UI layout
    """
    global data

    layout = dbc.Container([
        # Title
        dbc.Row(dbc.Col(html.H2("Basic Classification Training Simulations",
                                style={'textAlign': 'center',
                                       'color': colors['text']}))),

        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        figure={'data': [go.Scatter(x=[1], y=[1], name='F')]}),
                    id="main_graph"),

            ], className="h-1"),

        dbc.Row(
            [
                dbc.Col(html.Button('Data 1', id='btn-data-1', n_clicks=0)),
                dbc.Col(html.Button('Data 2', id='btn-data-2', n_clicks=0)),
                dbc.Col(html.Button('Data 3', id='btn-data-3', n_clicks=0)),
                dbc.Col(html.Button('Perceptron Step', id='btn-next',
                                    n_clicks=0)),
                html.Div(id='container-button-timestamp')
            ], className="h-1"),

    ], style={"height": "100vh"})

    return layout


if __name__ == "__main__":
    args = parse_arguments()

    app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = get_layout()


    @app.callback(dash.dependencies.Output('main_graph', 'children'),
                  dash.dependencies.Input('btn-data-1', 'n_clicks'),
                  dash.dependencies.Input('btn-next', 'n_clicks'),
                  dash.dependencies.Input('btn-data-2', 'n_clicks'),
                  dash.dependencies.Input('btn-data-3', 'n_clicks'))
    def displayClick(btn1, btn2, btn3, btn4):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'btn-data-1' in changed_id:
            generate_data('data1')
        if 'btn-data-2' in changed_id:
            generate_data('data2')
        if 'btn-data-3' in changed_id:
            generate_data('data3')
        elif 'btn-next' in changed_id:
            global clusters
            
            if len(clusters['means'])==0:
                clusters['means'] = data['x'][np.random.randint(len(data['x']), size=2)]




        elif 'btn-data-2' in changed_id:
            msg = 'Button 3 was most recently clicked'
        else:
            msg = 'None of the buttons have been clicked yet'

        return draw_data()

    app.run_server(debug=True)
