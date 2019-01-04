# -*- coding: utf-8 -*-
import base64

import numpy as np

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly.colors import DEFAULT_PLOTLY_COLORS as colors

from ..config import YAMLConfig
from ..types.state import StateMutableSequence
from ..types.detection import Clutter
from ..webui import app, cache

dash_app = dash.Dash(server=app)

dash_app.layout = html.Div(children=[
    html.Div(id='config', style={'display': "none"}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ])),
    dcc.Graph(
        id='graph',
        style={
            'width': '90vw',
            'height': '70vh',
        },
    ),
    dcc.RangeSlider(
        id="time-slider",
        min=0,
        max=100,
        value=[0, 100],
        allowCross=False,
    ),
    html.Table(id='track-table'),
])


@cache.memoize(timeout=86400)
def run_tracker(contents):
    _, content_string = contents.split(',')
    yaml_string = base64.b64decode(content_string)

    yaml = YAMLConfig()
    tracker = yaml.load(yaml_string)
    groundtruth_source = yaml.groundtruth_readers.pop()
    detections_source = yaml.detection_readers.pop()
    measurement_model = yaml.measurement_models.pop()

    groundtruth_paths = set()
    detections = set()
    tracks = set()
    for n, (time, current_tracks) in enumerate(tracker.tracks_gen()):
        if n == 0:
            stime = time
        groundtruth_paths |= groundtruth_source.groundtruth_paths
        detections |= detections_source.detections
        tracks |= current_tracks
        if n % 10 == 0:
            print("Loop {}".format(n))

    etime = time

    return stime, etime, list(groundtruth_paths), StateMutableSequence(list(detections)), list(tracks), measurement_model


@dash_app.callback(Output('config', 'children'),
                   [Input('upload-data', 'contents')])
def load_config(contents):
    if contents is None:
        return None
    run_tracker(contents)
    return contents


@dash_app.callback(Output('time-slider', 'marks'),
                   [Input('config', 'children')])
def set_slider_marks(contents):
    stime, etime, groundtruth_paths, detections, tracks, measurement_model = run_tracker(contents)
    return {0: str(stime), 100: str(etime)}


@dash_app.callback(Output('graph', 'figure'),
                   [Input('config', 'children'),
                    Input('time-slider', 'value')],
                   [State('graph', 'relayoutData')])
def plot_data(contents, slider_value, relayout_data):
    if contents is None:
        return {'data': []}

    stime, etime, groundtruth_paths, detections, tracks, measurement_model = run_tracker(contents)
    time_steps = (etime - stime) / 100
    start_index = stime + time_steps * slider_value[0]
    end_index = stime + time_steps * slider_value[1]

    traces = []
    first = True
    for n, groundtruth_path in enumerate(groundtruth_paths):
        states = groundtruth_path[start_index:end_index]
        if states:
            data = np.hstack([
                measurement_model.function(state.state_vector)
                for state in states
            ])
            traces.append(go.Scatter(
                x=data[0, :].ravel(),
                y=data[1, :].ravel(),
                mode="lines",
                legendgroup='groundtruth',
                name="Ground Truth",
                showlegend=first,
                text=[state.timestamp for state in states],
                line={
                    'dash': 'dot',
                    'color': colors[n % len(colors)],
                },
            ))
            first = False

    first = True
    for n, track in enumerate(tracks):
        states = track[start_index:end_index]
        if states:
            data = np.hstack([
                measurement_model.function(state.state_vector)
                for state in states
            ])
            traces.append(go.Scatter(
                x=data[0, :].ravel(),
                y=data[1, :].ravel(),
                mode="markers+lines",
                legendgroup='tracks',
                name="Track",
                showlegend=first,
                text=[state.timestamp for state in states],
                line={'color': colors[n % len(colors)]},
                customdata=[("Track", n, state_n) for state_n in range(len(states))],
            ))
            first = False

    detections_data = [
        detection.state_vector
        for detection in detections[start_index:end_index]
        if not isinstance(detection, Clutter)]
    if detections_data:
        data = np.hstack(detections_data)
        traces.append(go.Scatter(
            x=data[0, :].ravel(),
            y=data[1, :].ravel(),
            mode='markers',
            name="Detections",
            line={'color': colors[0]},
            text=[detection.timestamp
                  for detection in detections[start_index:end_index]
                  if not isinstance(detection, Clutter)],
        ))
    clutter_data = [
        clutter.state_vector
        for clutter in detections[start_index:end_index]
        if isinstance(clutter, Clutter)]
    if clutter_data:
        data = np.hstack(clutter_data)
        traces.append(go.Scatter(
            x=data[0, :].ravel(),
            y=data[1, :].ravel(),
            mode='markers',
            name="Clutter",
            line={'color': colors[1]},
            text=[detection.timestamp
                  for detection in detections[start_index:end_index]
                  if isinstance(detection, Clutter)],
        ))

    figure = {
        'data': traces,
        'layout': {
            'hovermode': "closest",
            'xaxis': {},
            'yaxis': {},
        },
    }

    if relayout_data and 'xaxis.range[0]' in relayout_data:
        figure['layout']['xaxis']['range'] = [
            relayout_data['xaxis.range[0]'],
            relayout_data['xaxis.range[1]']]
    if relayout_data and 'yaxis.range[0]' in relayout_data:
        figure['layout']['yaxis']['range'] = [
            relayout_data['yaxis.range[0]'],
            relayout_data['yaxis.range[1]']]

    return figure


@dash_app.callback(Output('track-table', 'children'),
                   [Input('config', 'children'),
                    Input('time-slider', 'value'),
                    Input('graph', 'selectedData')],
                   )
def track_table(contents, slider_value, selected_data):
    if contents is None or selected_data is None:
        return None
    stime, etime, groundtruth_paths, detections, tracks, measurement_model = run_tracker(contents)
    time_steps = (etime - stime) / 100
    start_index = stime + time_steps * slider_value[0]
    end_index = stime + time_steps * slider_value[1]

    selected_track_ids = {
        point['customdata'][1]
        for point in selected_data['points']
        if 'customdata' in point and point['customdata'][0] == "Track"}
    selected_tracks = [tracks[track_id] for track_id in sorted(selected_track_ids)]

    fields = ["__class__", "state_vector", "covar", "timestamp"]
    header = [html.Tr([html.Th(header.replace("_", " ").title()) for header in fields])]
    rows = [
        html.Tr([html.Td(str(getattr(state, field, None))) for field in fields])
        for track in selected_tracks for state in track[start_index:end_index]]

    return header + rows
