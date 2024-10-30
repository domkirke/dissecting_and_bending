import typing as tp

import numpy as np
import panel as pn
import torch 

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
        )
                      )
    return pn.GridBox(*audios, nclos=ncols)


def make_widgets(
    ops: tp.List[str]
):
    """
    Assuming each activation will endure affine bending, 
    builds the appropriate widgets to dynamically bend the model
    """
    widgets = {}
    for op in ops:
        widgets[op] = {
            'toggle': pn.widgets.Checkbox(name='Toggle'), 
            'cluster': pn.widgets.IntSlider(value=0, start=0, end=4, name='Cluster'),
            'scale': pn.widgets.FloatSlider(value=1, start=-1, end=3, step=.1, name='Scale'), 
            'bias': pn.widgets.FloatSlider(value=0, start=-2, end=2, step=.1, name='Bias')
        }
    return widgets