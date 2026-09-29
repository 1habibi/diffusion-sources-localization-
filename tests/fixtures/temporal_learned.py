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
