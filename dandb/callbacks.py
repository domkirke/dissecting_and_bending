import typing as tp

import numpy as np
import torch 

import torchbend as tb


class ClusterCallback(tb.BendingCallback):
    weight_compatible = False 
    activation_compatible = True
    jit_compatible = True 
    nntilde_compatible = True 
    controllable_params = ['cluster_idx']

    def __init__(
        self, 
        callback: tb.BendingCallback, 
        clusters: tp.List[tp.List[int]], 
        cluster_idx: int = 0
        ) -> None:
        """
        Wrapper for any activation-based callback to apply it on a cluster of channels 
        """
        
        super().__init__()
        self._callback = callback
        self.clusters = [torch.from_numpy(np.sort(cluster)) for cluster in clusters]
        self.cluster_idx = cluster_idx

    @property
    def cluster_size(self) -> int:
        return len(self.cluster)

    @property
    def cluster(self) -> tp.List[int]:
        return self.clusters[self.cluster_idx]

    def forward(self, x: torch.Tensor, 
                name: tp.Optional[str] = None) -> torch.Tensor:
        x_mod = self._callback(x, name)
        x_mod = torch.index_select(x_mod, dim=1, index=self.cluster)
        for idx, feat_idx in enumerate(self.cluster):
            x[:, feat_idx, :] = x_mod[:, idx, :]
        return x 