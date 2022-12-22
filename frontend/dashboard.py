from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import numpy as np
import ModelClient as model

app = Dash(__name__)


@app.callback(
    Output('img-content', 'figure'),
    Input('select-content', 'value'))
def showContent(content):
    fig=px.imshow(model.content_images[content].numpy()[0])
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@app.callback(
    Output('img-style', 'figure'),
    Input('select-style', 'value'))
def showContent(style):
    fig=px.imshow(model.style_images[style].numpy()[0])
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@app.callback(
    Output('img-generated', 'figure'),
    Input('select-content', 'value'),
    Input('select-style', 'value'))
def showGeneration(content,style):
    response=model.getGeneration(content,style)
    fig=px.imshow(np.array(response.json()['predictions'][0]))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

app.layout = html.Div(children=[
    html.H1(children='TensorServe Model Client Frontend'),
    html.Div(style={'width': '32%', 'display': 'inline-block'}),
    html.Div(children=[
        html.Label('Content:'),
        dcc.Dropdown(list(model.content_images.keys()),'sea_turtle', id='select-content'),
        html.Label('Style:'),
        dcc.Dropdown(list(model.style_images.keys()),'munch_scream', id='select-style'),
    ], style={'width': '32%', 'padding': 10, 'flex': 1, 'display': 'inline-block'}),
    html.Div(style={'width': '32%', 'display': 'inline-block'}),

    html.Div(children=[dcc.Graph(id='img-content')], style={'width': '32%', 'display': 'inline-block'}),
    html.Div(children=[dcc.Graph(id='img-style')], style={'width': '32%', 'display': 'inline-block'}),
    html.Div(children=[dcc.Graph(id='img-generated')], style={'width': '32%', 'display': 'inline-block'}),


])


def start():
    app.run_server(host='0.0.0.0')


if __name__ == '__main__':
    start()
