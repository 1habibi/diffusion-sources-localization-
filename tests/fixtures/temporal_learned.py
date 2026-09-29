import hashlib

import networkx as nx
import numpy as np

from diffusion_sources.temporal_learned_data import CandidateTable


def tiny_graph():
    return nx.path_graph(6)


def tiny_table():
    k = np.tile(np.arange(1, 4), 2)
    ids = np.tile(np.arange(6), 6)
    p = np.tile(np.array([.9, .8, .7, .3, .2, .1]), 6)
    early = np.tile([1, 0, 1, 0, 0, 0], 6)
    features = np.column_stack([p, early, early * .5, np.ones(36) * .5,
                                np.ones(36) * .6, p * early])
    labels = np.concatenate([np.arange(6) < c for c in k])
    return CandidateTable(features, ids, p, early, labels, np.arange(0, 37, 6),
                          np.arange(6), k.copy(), k, np.zeros(6, dtype=bool),
                          tuple(frozenset(range(c)) for c in k),
                          hashlib.sha256(b'tiny full masks').hexdigest())


def repeated_table(indices, *, perfect_v3=False):
    """Whole-cascade synthetic fixture; not a metric from the real model."""
    from dataclasses import replace
    t=tiny_table()
    n=len(indices)
    row=np.tile(np.arange(36), (n+5)//6)[:n*6]
    case=np.arange(n)%6
    result=replace(t,features=t.features[row].copy(),candidate_ids=t.candidate_ids[row],
        probabilities=t.probabilities[row],early_flags=t.early_flags[row].copy(),labels=t.labels[row],
        offsets=np.arange(0,n*6+1,6),indices=np.array(indices,dtype=np.int64),
        predicted_counts=t.predicted_counts[case],true_counts=t.true_counts[case],
        early_empty=t.early_empty[case].copy(),snapshot_sources=tuple(t.snapshot_sources[i] for i in case))
    if perfect_v3:
        result.early_flags[:]=0
        result.features[:,[1,2,5]]=0
        result.early_empty[:]=True
    return result
