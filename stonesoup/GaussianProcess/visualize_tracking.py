# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:44:54 2023

@author: 007
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize(state, truth, measurements, x_data, y_data, num_steps, Size_win):
    truth_x = [state.state_vector[0] for state in truth]
    truth_y = [state.state_vector[2] for state in truth]

    Xm = [m.state_vector[0] for m in measurements]
    Ym = [m.state_vector[1] for m in measurements]
    Xm = np.array(Xm).reshape(-1, 1)
    Ym = np.array(Ym).reshape(-1, 1)

    fig = make_subplots(rows=1, cols=1, subplot_titles=['Tracking process'])

    trace1 = go.Scatter(
        x=[], y=[], mode='lines+markers', name='Ground Truth',
        marker=dict(size=10, color='orange')
    )
    fig.add_trace(trace1, row=1, col=1)

    trace2 = go.Scatter(
        x=[], y=[], mode='markers', name='Measurement',
        marker=dict(size=10, color='blue')
    )
    fig.add_trace(trace2, row=1, col=1)

    trace3 = go.Scatter(
        x=[], y=[], mode='lines+markers', name='GP result',
        marker=dict(size=10, color='darkmagenta')
    )
    fig.add_trace(trace3, row=1, col=1)

    fig.update_xaxes(range=[-5, 40], row=1, col=1)
    fig.update_yaxes(range=[-5, 80], row=1, col=1)

    fig.update_layout(width=800, height=600, font=dict(size=16))

    frames = [
        go.Frame(
            data=[
                go.Scatter(
                    x=truth_x[max(0, k-Size_win):k],
                    y=truth_y[max(0, k-Size_win):k],
                    mode='lines+markers', name='Ground Truth',
                    marker=dict(size=5, color='orange'),
                    line=dict(color='orange')
                ),
                go.Scatter(
                    x=Xm[max(0, k-Size_win):k].flatten(),
                    y=Ym[max(0, k-Size_win):k].flatten(),
                    mode='markers', name='Measurement',
                    marker=dict(size=5, color='blue')
                ),
                go.Scatter(
                    x=x_data[max(0, k-Size_win):k].flatten(),
                    y=y_data[max(0, k-Size_win):k].flatten(),
                    mode='lines+markers', name='GP result',
                    marker=dict(size=5, color='darkmagenta')
                )
            ],
            name=str(k)
        ) for k in range(num_steps-1)
    ]

    fig.update(frames=frames)

    animation_settings = dict(
        frame=dict(duration=300, redraw=True), fromcurrent=True
    )
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons', showactive=False,
                buttons=[
                    dict(
                        label='play', method='animate',
                        args=[None, animation_settings]
                    ),
                    dict(
                        label='stop', method='animate',
                        args=[[], animation_settings]
                    )
                ]
            )
        ]
    )

    return fig


def visualize_DGP(state, truth, meas, x_data, y_data, num_steps, sensor_data):
    # Create the figure object
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Tracking process'])

    trace1 = go.Scatter(
        x=[], y=[], mode='lines+markers',
        name='Ground Truth', marker=dict(size=10, color='orange')
    )
    trace2 = go.Scatter(
        x=[], y=[], mode='markers',
        name='measurement', marker=dict(size=10, color='blue')
    )
    trace3 = go.Scatter(
        x=[], y=[], mode='lines+markers',
        name='GP result', marker=dict(size=10, color='purple')
    )

    # Add traces
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)
    fig.add_trace(trace3, row=1, col=1)

    fig.update_xaxes(range=[-5, 40], row=1, col=1)
    fig.update_yaxes(range=[-5, 80], row=1, col=1)

    fig.update_layout(
        width=800, height=600, font=dict(size=16)
    )

    # Process measurements
    Xm = [m.state_vector[0] for m in meas]
    Ym = [m.state_vector[1] for m in meas]
    Xm = np.array(Xm).reshape(-1, 1)
    Ym = np.array(Ym).reshape(-1, 1)

    # Extract truth_x and truth_y
    truth_x = [state.state_vector[0] for state in truth]
    truth_y = [state.state_vector[2] for state in truth]

    # Create animation frames
    frames = [
        go.Frame(
            data=[
                go.Scatter(
                    x=Xm[0:k].flatten(), y=Ym[0:k].flatten(), mode='markers',
                    name='Measurement', marker=dict(size=5, color='blue')
                ),
                go.Scatter(
                    x=truth_x[0:k], y=truth_y[0:k], mode='lines+markers',
                    name='Ground Truth', marker=dict(size=5, color='orange'),
                    line=dict(color='orange')
                ),
                go.Scatter(
                    x=x_data[0:k].flatten(), y=y_data[0:k].flatten(),
                    mode='lines+markers', name='GP result',
                    marker=dict(size=5, color='darkmagenta')
                ),
                go.Scatter(
                    x=[data['position'][0] for data in sensor_data],
                    y=[data['position'][1] for data in sensor_data],
                    mode='markers', name='Sensor',
                    marker=dict(size=5, color='rgba(0, 0, 0, 0.3)',
                                symbol='circle')
                )
            ],
            name=str(k)
        ) for k in range(num_steps-1)
    ]

    # Add sensor range circles
    for data in sensor_data:
        circle = go.layout.Shape(
            type="circle",
            x0=data['position'][0] - data['range'],
            y0=data['position'][1] - data['range'],
            x1=data['position'][0] + data['range'],
            y1=data['position'][1] + data['range'],
            line=dict(color="gray"),
            fillcolor="rgba(255, 255, 0, 0.05)"
        )
        fig.add_shape(circle)

    # Add animation frames and set up animation
    fig.update(frames=frames)
    animation_settings = dict(frame=dict(duration=300, redraw=True),
                              fromcurrent=True)
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons', showactive=False,
                buttons=[
                    dict(
                        label='play', method='animate',
                        args=[None, animation_settings]
                    ),
                    dict(
                        label='stop', method='animate',
                        args=[[None], animation_settings]
                    )
                ]
            )
        ]
    )

    fig.show()
