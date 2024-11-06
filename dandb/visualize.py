import typing as tp
import torchvision
import itertools
from IPython.display import display as ipython_display, Audio
import math
import numpy as np
import panel as pn
import torch 


import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torchbend as tb



def make_audio_grid(
    audios_rec: torch.Tensor, 
    audios_bended: torch.Tensor,
    sr: int = 44100, 
    ncols: int = 4, 
):
    """
    Makes a pretty little grid of audios, displaying both 
    the standard and bended reconstructions
    """
    audios = []
    for au_rec, au_bended in zip(audios_rec, audios_bended):
        audios.append(pn.Column(
            pn.pane.Str('Rec.'), 
            pn.pane.Audio(
                object=au_rec.squeeze().numpy(), 
                sample_rate=sr), 
            pn.pane.Str('Bended'), 
            pn.pane.Audio(
                object=au_bended.squeeze().numpy(), 
                sample_rate=sr), 
            pn.pane.Str('\n'*2), 

        )
                      )
    return pn.GridBox(*audios, ncols=ncols)


def make_widgets(
    ops: tp.List[str]
):
    """
    Assuming each activation will endure affine bending, 
    builds the appropriate widgets to dynamically bend the model
    """
    widgets = {}
    for op in ops:
        widgets[op+'/name'] = pn.widgets.StaticText(name=op, value=op)
        widgets[op+'/toggle'] = pn.widgets.Checkbox(name='Toggle')
        widgets[op+'/cluster'] = pn.widgets.IntSlider(value=0, start=0, end=4, name='Cluster')
        widgets[op+'/scale'] = pn.widgets.FloatSlider(value=1, start=-1, end=3, step=.1, name='Scale')
        widgets[op+'/bias'] = pn.widgets.FloatSlider(value=0, start=-2, end=2, step=.1, name='Bias')
    return widgets


def update_affine_params(
    clustered_affine_cb: tp.Dict[str, tb.BendingCallback], 
    **params, 
    ) -> tp.Dict[str, tb.BendingCallback]:
    """
    Updates the scale, bias, and cluster index of affine callbacks
    """
    for op_param_name, op_val in params.items():
        op, _action = op_param_name.split('/')
        if _action=='bias':
            clustered_affine_cb[op]._callback.bias = op_val    
        if _action=='scale':
            clustered_affine_cb[op]._callback.scale = op_val
        if _action=='cluster':
            clustered_affine_cb[op].cluster_idx = op_val 
        if _action=='toggle':
            clustered_affine_cb[op].toggled = op_val 

    return clustered_affine_cb    


def make_widget_box(op_names, **widgets):
    widget_boxes = {op_name: [] for op_name in op_names}
    for widget_name, widget in widgets.items():
        op_name = widget_name.split('/')[0]
        widget_boxes[op_name].append(widget)
    return pn.Row(*[pn.WidgetBox(*op_widget) for op_widget in widget_boxes.values()])


def get_grid_shape(n):
    return math.floor(math.sqrt(n)), math.ceil(math.sqrt(n))
    

def plot_1d_activation(name, activations, plot_picker = None, height=250, max_samples=16384, display=False, channel_idx=None):
    def _downgrade(x, max_samples=max_samples):
        t = np.arange(x.shape[-1])
        downscale = math.ceil(x.shape[-1] / max_samples)
        return t[..., ::downscale], x[..., ::downscale]

    plot_picker = _downgrade
    if activations.ndim == 1: 
        activations = activations[None, None]
    elif activations.ndim == 2:
        activations = activations[None]
    elif activations.ndim > 3:
        activations = activations.reshape(activations.shape[:2], -1)

    n_plots = activations.shape[0]
    # if activations.ndim == 2 : 
    t, data = plot_picker(activations)
    # if activations.ndim == 3:
    #     data = list(map(lambda x: plot_picker(x[0,0]), activations.split(1, dim=1)))

    n_rows, n_columns = get_grid_shape(n_plots)
    fig = make_subplots(rows=n_rows, cols=n_columns)
    fig.update_layout(
        title=name,
        height=height,
        margin=dict(l=0,r=0,b=0,t=0,pad=40)
    )

    for i in range(n_rows): 
        for j in range(n_columns):
            idx = i * n_columns + j 
            if idx >= n_plots : break
            d = data[idx]
            for k, dd in enumerate(d):
                idx_c = k if channel_idx is None else channel_idx[k]
                p = go.Scatter(x = t, y = dd, name="channel %d"%(idx_c), showlegend=True)
                fig.add_trace(p, row=i+1, col=j+1)
        if idx >= n_plots: break
    if display:
        ipython_display(fig)
    else: 
        return fig



def plot_audio(audio, plot_waveform=True, plot_spectrogram=True, height=200, sr=44100, n_fft=2048, name=None, display = False):
    audio_output = widgets.Output()
    with audio_output:
        ipython_display(Audio(audio.numpy(), rate=sr))
    audio_obj = [audio_output]
    if (plot_waveform and plot_spectrogram):
        if plot_waveform:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = torch.arange(audio.shape[-1]) / sr, y = audio[0].numpy()))
            fig.update_layout(title=name, height=height - 30, margin=dict(l=0,r=0,b=0,t=0,pad=4))
            audio_obj.insert(0, go.FigureWidget(fig))
        if plot_spectrogram:
            input_stft = torch.stft(audio, n_fft, window=torch.hann_window(n_fft), return_complex=True).abs()
            fig = go.Figure()
            fig.update_layout(title=name, margin=dict(l=0,r=0,b=0,t=0,pad=4), height=height)
            fig.add_trace(go.Heatmap(x = torch.linspace(0, audio.shape[-1] / sr, input_stft.shape[-1]), 
                                     y = torch.arange(input_stft.shape[-2]) / input_stft.shape[-2] * (sr / 2),
                                     z =input_stft[0].numpy(), 
                                     showlegend=False, showscale=False,
                                     colorbar=None))
            spec = go.FigureWidget(fig)
    if plot_spectrogram: 
        out = widgets.HBox([widgets.VBox(audio_obj), spec])
    else:
        out = widgets.VBox(audio_obj)
    if display: 
        ipython_display(out)
    else:
        return out


def plot_reconstructions(original, reconstruction, display=True, **kwargs):
    outs = widgets.HBox([plot_audio(original, **kwargs), plot_audio(reconstruction, **kwargs)])
    if display: 
        ipython_display(outs)
    else:
        return outs


def plot_latent_trajs(latent_trajs, name="latent trajectories"):
    return plot_1d_activation(name, latent_trajs)


def plot_kernel_grid(kernels, height=300, width=250, margin = 3, display=False):
    n_column, n_row = math.ceil(math.sqrt(kernels.shape[0])), math.floor(math.sqrt(kernels.shape[0]))
    fig= make_subplots(rows=n_row, cols=n_column)
    fig.update_layout(
        height=height,
        margin=dict(l=margin,r=margin,b=margin,t=margin, pad=2)
    )
    for i, j in itertools.product(range(n_row), range(n_column)):
        plot = go.Scatter(y=kernels[i*n_column+j].numpy(), showlegend=False)
        fig.add_trace(plot, row=i+1, col=j+1)       
    if display: 
        ipython_display(fig)
    else:
        return fig

def preprocess_image(x):
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.shape[-1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return torch.round(x * 255)

def plot_image_grid(images, height=300, width=250, margin = 0, display=False):
    n_column, n_row = math.ceil(math.sqrt(images.shape[0])), math.floor(math.sqrt(images.shape[0]))
    fig= make_subplots(rows=n_row, cols=n_column)
    fig.update_layout(
        height=height,
        
        margin=dict(l=margin,r=margin,b=margin,t=margin, pad=0)
    )
    for i, j in itertools.product(range(n_row), range(n_column)):
        img = (images[i*n_column+j].permute(1, 2, 0).repeat(1, 1, 3) * 255).round()
        plot = go.Image(z=img)
        fig.add_trace(plot, row=i+1, col=j+1)
        fig.update_layout(**{f"xaxis{i*n_column+j+1}": {"showticklabels": False}, f"yaxis{i*n_column+j+1}": {"showticklabels": False}})
    if display: 
        ipython_display(fig)
    else:
        return fig


def plot_image_reconstructions(x, y, **kwargs):
    images = torch.cat([x, y], -1)
    return plot_image_grid(images, **kwargs)

def plot_image_activations_list(activations, height=300, margin = 0, display=False, n_rows=None, name=None):
    for i, act in enumerate(activations):
        nrow = math.floor(math.sqrt(act.shape[0]))
        act = torchvision.utils.make_grid(preprocess_image(act), nrow=nrow, padding=1, pad_value=255)
        act = act.permute(1, 2, 0)
        activations[i] = act
    if n_rows is None:
        n_rows, n_columns = get_grid_shape(len(activations))
    else:
        n_columns = math.ceil(len(activations) / n_rows)


    fig = make_subplots(rows=n_rows, cols=n_columns)
    for k, act in enumerate(activations):
        i = k // n_columns
        j = k % n_columns
        fig.add_trace(go.Image(z=activations[k]), row=i+1, col=j+1)
        fig.update_layout(**{f"xaxis{i*n_columns+j+1}": {"showticklabels": False}, f"yaxis{i*n_columns+j+1}": {"showticklabels": False}})

    fig.update_layout(
            height=height,
            margin=dict(l=margin,r=margin,b=margin,t=margin, pad=0),
            xaxis= {"showticklabels": False},
            yaxis = {"showticklabels": False},
                title=name,
        )
    if display: 
        ipython_display(fig)
    else:
        return fig


def plot_image_activations(activations, height=300, margin = 0, display=False, n_rows=None, name=None):
    if isinstance(activations, list):
        return plot_image_activations_list(activations, height=height, margin=margin, display=display, name=name, n_rows=n_rows)
    nrow = math.floor(math.sqrt(activations.shape[0]))
    activations = torchvision.utils.make_grid(preprocess_image(activations), nrow=nrow, padding=1, pad_value=255)
    activations = activations.permute(1, 2, 0)
    fig=go.Figure()
    fig.update_layout(
        height=height,
        # margin=dict(l=margin,r=margin,b=margin,t=margin, pad=0),
        xaxis= {"showticklabels": False},
        yaxis = {"showticklabels": False},
        title=name,
    )
    fig.add_trace(go.Image(z=activations))
    if display: 
        ipython_display(fig)
    else:
        return fig