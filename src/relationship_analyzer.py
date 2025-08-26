import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import networkx as nx

class RelationshipAnalyzer:
    def __init__(self, faiss_index: faiss.Index, id_map: dict):
        """
        Parameters
        ----------
        faiss_index : faiss.Index
            The FAISS index storing your vectors.
        id_map : dict
            Maps internal FAISS IDs to external series IDs (e.g., filenames).
        """
        self.index = faiss_index
        self.id_map = id_map

    def get_all_vectors(self):
        """Extract all vectors stored in FAISS as a numpy array."""
        ntotal = self.index.ntotal
        if ntotal == 0:
            return np.empty((0, self.index.d), dtype=np.float32)
        
        # For newer FAISS versions, we need to reconstruct vectors one by one
        vectors = []
        for i in range(ntotal):
            vectors.append(self.index.reconstruct(i))
        
        return np.array(vectors, dtype=np.float32)

    def compute_similarity(self, metric="cosine"):
        """
        Compute pairwise similarity/distance matrix.

        Parameters
        ----------
        metric : str
            "cosine" for cosine similarity,
            "euclidean" for euclidean distance.

        Returns
        -------
        np.ndarray : similarity/distance matrix
        """
        X = self.get_all_vectors()
        if X.shape[0] == 0:
            return np.array([[]])
        if metric == "cosine":
            return cosine_similarity(X)
        elif metric == "euclidean":
            return euclidean_distances(X)
        else:
            raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")

    def build_graph(self, metric="cosine", threshold=0.7):
        """
        Build a graph where nodes are time series and edges connect related ones.

        Parameters
        ----------
        metric : str
            "cosine" (similarity) or "euclidean" (distance).
        threshold : float
            For cosine: keep edges with similarity > threshold.
            For euclidean: keep edges with distance < threshold.

        Returns
        -------
        networkx.Graph
        """
        mat = self.compute_similarity(metric=metric)
        n = mat.shape[0]

        G = nx.Graph()
        for i in range(n):
            G.add_node(self.id_map[i])

        for i in range(n):
            for j in range(i + 1, n):
                value = mat[i, j]
                if (metric == "cosine" and value > threshold) or \
                   (metric == "euclidean" and value < threshold):
                    G.add_edge(self.id_map[i], self.id_map[j], weight=value)

        return G

    def top_k_relationships(self, k=5, metric="cosine"):
        """
        For each series, find top-k most related series.

        Returns
        -------
        dict : {series_id: [(neighbor_id, score), ...]}
        """
        mat = self.compute_similarity(metric=metric)
        n = mat.shape[0]

        results = {}
        if n == 0:
            return results

        for i in range(n):
            series_id = self.id_map[i]
            scores = [(self.id_map[j], mat[i, j]) for j in range(n) if i != j]

            # Sort: high score = similar (cosine), low score = close (euclidean)
            if metric == "cosine":
                scores = sorted(scores, key=lambda x: -x[1])
            else:
                scores = sorted(scores, key=lambda x: x[1])

            results[series_id] = scores[:k]

        return results