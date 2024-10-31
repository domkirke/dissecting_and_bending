import typing as tp

import numpy as np
import torch 

import torchbend as tb


class ClusterCallback(tb.BendingCallback):
    weight_compatible = False 
    activation_compatible = True
    jit_compatible = True 
    nntilde_compatible = True 
    controllable_params = ['cluster_idx', 'toggled']

    def __init__(
        self, 
        callback: tb.BendingCallback, 
        clusters: tp.List[tp.List[int]], 
        cluster_idx: int = 0, 
        toggled: bool = True
        ) -> None:
        """
        Wrapper for any activation-based callback to apply it on a cluster of channels 
        """
        
        super().__init__()
        self._callback = callback
        self.clusters = [torch.from_numpy(np.sort(cluster)) for cluster in clusters]
        self.cluster_idx = cluster_idx
        self.toggled = toggled

    @property
    def cluster_size(self) -> int:
        return len(self.cluster)

    @property
    def cluster(self) -> tp.List[int]:
        return self.clusters[self.cluster_idx]

    def forward(self, x: torch.Tensor, 
                name: tp.Optional[str] = None) -> torch.Tensor:
        
        if not self.toggled: 
            return x
        
        x_mod = self._callback(x, name)
        x_mod = torch.index_select(x_mod, dim=1, index=self.cluster)
        for idx, feat_idx in enumerate(self.cluster):
            x[:, feat_idx, :] = x_mod[:, idx, :]
        return x 


def make_clustered_bending_callbacks(
    tb_callbacks: tp.Dict[str, tb.BendingCallback], 
    clusters: tp.Dict[str, tp.List[tp.List[str]]]
) -> tp.Dict[str, tb.BendingCallback]:
    """
    Given a dictionnary of torchbend Callbacks, and a dictionnary of activation, 
    will create clustered callbacks (i.e. applied to specific cluster of channels)
    """
    clustered_callbacks = {}
    for op, cb in tb_callbacks.items():
        clustered_callbacks[op] = ClusterCallback(cb, clusters[op])
    return clustered_callbacks


def make_affine_bending_modules(
    ops: tp.List[str]
) -> tp.Dict[str, tb.Affine]:
    return {op: tb.Affine(bias=0, scale=1) for op in ops}