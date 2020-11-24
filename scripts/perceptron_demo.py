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
import pandas as pd
import plotly.graph_objs as go

colors = {'background': '#111111', 'text': '#7FDBDD'}
def_font = dict(family="Courier New, Monospace", size=10, color='#000000')

from datetime import date

# global vars related to data loaded
global data
global w
global is_training
global data_seen
global is_seen_error
global data_error

data_error = 0
data_seen = 0
is_seen_error = 0
is_training = False

w = np.array([0.1, 0.1, 0.1])

data = {'x': [], 'y': []}

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
    n_samples = 5
    global is_training
    is_training = False
    global w
    w = np.random.random((3,))
    if data_val == "data1":
        m1 = [-1, -1]
        m2 = [0, 1]
        a = np.concatenate([np.random.normal(m1[0], 1, n_samples),
                            np.random.normal(m2[0], 1, n_samples)])
        b = np.concatenate([np.random.normal(m1[1], 1, n_samples),
                            np.random.normal(m2[1], 1, n_samples)])

        data['x'] = np.array([a, b]).T
        data['y'] = \
            np.concatenate([-np.ones((n_samples,)),
                             np.ones((n_samples,))])
    if data_val == "data2":
        m1 = [-0.5, -0.5]
        m2 = [0.5, 0.5]
        a = np.concatenate([np.random.normal(m1[0], 1, n_samples),
                            np.random.normal(m2[0], 1, n_samples)])
        b = np.concatenate([np.random.normal(m1[1], 1, n_samples),
                            np.random.normal(m2[1], 1, n_samples)])

        data['x'] = np.array([a, b]).T
        data['y'] = \
            np.concatenate([-np.ones((n_samples,)),
                             np.ones((n_samples,))])

    if data_val=="data3":
        data['x'] = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
        data['y'] = np.array([-1, -1, 1, 1])
        
    global data_seen
    data_seen = 0



def draw_data():
    global data
    if len(data['x'])>0:
        figure = {'data': [
                            go.Scatter(x=data['x'][data['y'] == -1, 0],
                                       y=data['x'][data['y'] == -1, 1],
                                       mode='markers', name='y = -1',
                                       marker_size=10,
                                       marker_color='rgba(22, 182, 255, .9)'
                                       ),
                            go.Scatter(x=data['x'][data['y'] == 1, 0],
                                       y=data['x'][data['y'] == 1, 1],
                                       mode='markers', name='y = 1',
                                       marker_size=10,
                                       marker_color='rgba(182, 22, 255, .9)')
                       ],
            'layout': go.Layout(
                xaxis=dict(range=[min(data['x'][:,0]) - np.mean(np.abs(data['x'][:, 0])),
                                  max(data['x'][:,0]) + np.mean(np.abs(data['x'][:, 0]))
                        ]),
                yaxis=dict(range=[min(data['x'][:,1]) - np.mean(np.abs(data['x'][:,1])),
                                  max(data['x'][:,1]) + np.mean(np.abs(data['x'][:, 1]))
                                                  ]))}

        # draw line:
        x_space = np.linspace(np.min(data['x'][:, 0]), np.max(data['x'][:, 0]))
        y_space = -w[1] / (w[2] + 0.0000001) * x_space - w[0] / (w[2]+ 0.0000001)


        figure['data'].append(go.Scatter(x=x_space, y=y_space, name='f(x)=0'))
#        figure['data'].append(go.Scatter(x=x_space, y=y_space_prev, name='prev f(x)=0'))

        if is_seen_error == 0:
            figure['data'].append(
                go.Scatter(x=[data['x'][data_seen-1, 0]],
                           y=[data['x'][data_seen-1, 1]], name='cur example',
                           mode='markers',
                           marker_line_color='rgba(0, 255, 0, .9)',
                           marker_line_width=2, marker_size=15,
                           marker_color='rgba(0, 255, 0, .2)'))
        else:
            figure['data'].append(
                go.Scatter(x=[data['x'][data_seen - 1, 0]],
                           y=[data['x'][data_seen - 1, 1]], name='cur example',
                           mode='markers',
                           marker_line_color='rgba(255, 0, 0, .9)',
                           marker_line_width=2, marker_size=15,
                           marker_color='rgba(255, 0, 0, .2)'))
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
            global w
            global data_error
            global is_training
            global data_seen
            global is_seen_error

            is_training = True
            x_cur = data['x'][data_seen]
            y_cur = data['y'][data_seen]
            y_pred = w[0] + w[1] * x_cur[0] + w[2] * x_cur[1] # w^T * x
            y_has_error = y_cur * y_pred                      # y * w^T * x
            if y_has_error <= 0:
                data_error += 1
                w += (y_cur * np.array([1.0, x_cur[0], x_cur[1]]))
                is_seen_error = 1
            else:
                is_seen_error = 0
            data_seen += 1
            if data_seen >= data['x'].shape[0]:
                data_seen = 0
                data_error = 0


        elif 'btn-data-2' in changed_id:
            msg = 'Button 3 was most recently clicked'
        else:
            msg = 'None of the buttons have been clicked yet'

        return draw_data()

    app.run_server(debug=True)
