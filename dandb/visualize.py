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

