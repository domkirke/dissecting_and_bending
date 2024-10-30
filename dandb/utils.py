import os
from pathlib import Path
import typing as tp

import numpy as np
import torch 
import torch.nn as nn 

from einops import rearrange
import gin
import rave
import torchbend as tb

torch.set_grad_enabled(False)


def _inifinite_iterator():
    i = 0
    while True:
        yield i
        i+=1


def extract_activations(
    bended_model: tb.BendedModule, 
    op_names: tp.List[str], 
    loader: torch.utils.data.DataLoader, 
    device: str, 
    max_batches: int = 200, 
    avg_batch: bool = False
    ) -> tp.Dict[str, torch.Tensor]:
    """
    Given a neural network (wrapped inside a BendedMModule), a data loader, 
    and a list of layers, will extract the output of each of these layers across 
    the dataloader
    """
    all_acts = {op:[] for op in op_names}
    _iterator = range(max_batches) if max_batches != -1 else _inifinite_iterator()
    for x, batch_idx in zip(loader, _iterator):
        x = x.to(device)
        acts = bended_model.get_activations(*op_names, x=x, fn="forward")
        for op in op_names:
            if not avg_batch:
                flat_act = rearrange(acts[op], 'b c t -> c (b t)')
            else:
                flat_act = acts[op].mean(0)
            flat_act = flat_act.cpu()
            all_acts[op].append(flat_act) 
    return {k: torch.cat(v, dim=-1) for k, v in all_acts.items()}


def compute_activations_similarity(
    activations: tp.Dict[str, torch.Tensor], 
    device: str = 'cpu'
) -> tp.Dict[str, torch.Tensor]:
    """
    Given a dictionnary of activations, computes channel-wise and op-wise L1 similarity
    across these activations 
    """
    sim_matrices = {op: torch.zeros(acts.shape[0], acts.shape[0], device=device) 
                    for op, acts in activations.items()}
    similarity_fn = nn.CosineSimilarity(dim=-1).to(device)
    for op, acts in activations.items():
        acts = acts.to(device)
        for channel_idx, channel_acts in enumerate(acts):
            channel_acts = channel_acts.unsqueeze(0).repeat(acts.shape[0], 1)
            sim_matrices[op][channel_idx] = similarity_fn(channel_acts, acts)
    return sim_matrices


def save_activations_similarity(
    sim_matrices: tp.Dict[str, torch.Tensor], 
    path: str
    ) -> None:
    torch.save(sim_matrices, path)


def load_activations_similarity(
    path: str
    ) -> tp.Dict[str, torch.Tensor]:
    return torch.load(path)


def compute_clusters(
    sim_matrices: tp.Dict[str, torch.Tensor], 
    threshold: float =.8
    ) -> tp.Dict[str, tp.List[tp.List[int]]]:
    """
    Given a dictionnary of similarity scores, computes clusters of activations having 
    a similarity score above a given threshold
    """
    assert threshold>0 and threshold<=1, ValueError(f'Threshold should be between 0 and 1, got {threshold}')
    clusters = {}
    for op, sim_scores in sim_matrices.items():
        op_clusters = []
        handled = []
        for idx, row in enumerate(sim_scores):
            if idx in handled:
                continue
            similar_features = torch.argwhere(row>=threshold)
            similar_features = [i.item() for i in similar_features]
            op_clusters.append(similar_features)
            handled+=similar_features
        clusters[op] = op_clusters
        return clusters


def compute_non_singleton_clusters(
    clusters: tp.Dict[str, tp.List[tp.List[int]]]
    ) -> tp.Dict[str, tp.List[tp.List[int]]] :
    """
    Keeps clusters that have more than 1 member
    """
    return {op_name: [c for c in op_clusters if len(c)>1]
            for op_name, op_clusters in clusters.items()}

    
def sort_clusters(    
    clusters: tp.Dict[str, tp.List[tp.List[int]]]
    ) -> tp.Dict[str, tp.List[tp.List[int]]] :
    """
    Sort clusters by descending cardinality
    """
    sorted_clusters = {}
    for op, op_clusters in clusters.items():
        sorted_idx = np.argsort([len(c) for c in op_clusters])[::-1]
        sorted_clusters[op] = [op_clusters[idx] for idx in sorted_idx]
    return sorted_clusters


def _search_for_config(folder):
    if os.path.isfile(folder):
        folder = os.path.dirname(folder)
    configs = list(map(str, Path(folder).rglob("config.gin")))
    if configs != []:
        return os.path.abspath(os.path.join(folder, "config.gin"))
    configs = list(map(str, Path(folder).rglob("../config.gin")))
    if configs != []:
        return os.path.abspath(os.path.join(folder, "../config.gin"))
    configs = list(map(str, Path(folder).rglob("../../config.gin")))
    if configs != []:
        return os.path.abspath(os.path.join(folder, "../../config.gin"))
    else:
        return None
    

def load_rave(
    run_path: str, 
    ckpt_path: str
) -> nn.Module:
    """
    Given a path to a RAVE run, as well a specific checkpoint, instantiates
    RAVE and loads the weights of the model
    """
    ckpt_path = os.path.join(run_path, ckpt_path)
    gin.parse_config_file(_search_for_config(run_path))
    rave_model = rave.RAVE()
    weights = torch.load(ckpt_path)['state_dict']
    rave_model.load_state_dict(weights, strict=False)
    rave_model.eval()
    for m in rave_model.modules():
        if hasattr(m, 'weight_g'):
            nn.utils.remove_weight_norm(m)
    return rave_model
