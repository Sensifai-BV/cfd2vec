"""Nearest-neighbour retrieval in the frozen embedding space."""
from __future__ import annotations

import numpy as np


class EmbeddingIndex:
    def __init__(self, ids, embeddings: np.ndarray, payload=None):
        E = np.asarray(embeddings, np.float32)
        self.mu = E.mean(0); self.sd = E.std(0) + 1e-6
        Z = (E - self.mu) / self.sd
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        self.ids = list(ids); self.payload = payload

    def query(self, e: np.ndarray, k: int = 5):
        z = (np.asarray(e, np.float32) - self.mu) / self.sd
        z /= np.linalg.norm(z) + 1e-12
        s = self.Z @ z
        o = np.argsort(-s)[:k]
        return [(self.ids[i], float(s[i])) for i in o]
