import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import networkx as nx
from typing import Dict, List, Tuple

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
        self.dimension = self.index.d if hasattr(self.index, 'd') else None

    def get_all_vectors(self):
        """Extract all vectors stored in FAISS as a numpy array."""
        ntotal = self.index.ntotal
        if ntotal == 0:
            return np.empty((0, self.index.d), dtype=np.float32)
        
        # Try batch reconstruction if available
        if hasattr(self.index, 'reconstruct_n'):
            return self.index.reconstruct_n(0, ntotal)
        else:
            # Fallback to individual reconstruction
            vectors = []
            for i in range(ntotal):
                vectors.append(self.index.reconstruct(i))
            return np.array(vectors, dtype=np.float32)

    def compute_similarity(self, metric: str = "cosine") -> np.ndarray:
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

    def build_graph(self, metric: str = "cosine", threshold: float = 0.7) -> nx.Graph:
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
        
        # Add all nodes first
        for i in range(n):
            G.add_node(self.id_map[i])
        
        # Use vectorized operations for better performance
        if metric == "cosine":
            # Create mask for similarities above threshold
            mask = np.triu(mat > threshold, k=1)
        else:  # euclidean
            # Create mask for distances below threshold
            mask = np.triu(mat < threshold, k=1)
        
        # Get indices of edges to add
        edges_i, edges_j = np.where(mask)
        edges_data = [(self.id_map[i], self.id_map[j], {'weight': mat[i, j]}) 
                      for i, j in zip(edges_i, edges_j)]
        
        G.add_edges_from(edges_data)
        return G

    def top_k_relationships(self, k: int = 5, metric: str = "cosine") -> Dict[str, List[Tuple[str, float]]]:
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

        # Set diagonal to values that will be excluded in sorting
        if metric == "cosine":
            np.fill_diagonal(mat, -np.inf)  # So self-similarity is not selected
        else:  # euclidean
            np.fill_diagonal(mat, np.inf)   # So self-distance is not selected

        for i in range(n):
            series_id = self.id_map[i]
            
            if metric == "cosine":
                # For cosine, we want highest values
                top_k_indices = np.argsort(mat[i])[-k:][::-1]
            else:  # euclidean
                # For euclidean, we want lowest values
                top_k_indices = np.argsort(mat[i])[:k]
            
            results[series_id] = [(self.id_map[j], mat[i, j]) for j in top_k_indices]

        return results

    def get_relationships_for_id(self, series_id: str, k: int = 5, metric: str = "cosine") -> List[Tuple[str, float]]:
        """
        Get top-k relationships for a specific series ID.
        
        Parameters
        ----------
        series_id : str
            The external ID of the series to analyze
        k : int
            Number of relationships to return
        metric : str
            "cosine" or "euclidean"
            
        Returns
        -------
        list : [(neighbor_id, score), ...]
        """
        # Find the internal index for this series_id
        internal_id = None
        for i, ext_id in self.id_map.items():
            if ext_id == series_id:
                internal_id = i
                break
                
        if internal_id is None:
            raise ValueError(f"Series ID {series_id} not found in id_map")
            
        mat = self.compute_similarity(metric=metric)
        n = mat.shape[0]
        
        if metric == "cosine":
            mat[internal_id, internal_id] = -np.inf
            top_k_indices = np.argsort(mat[internal_id])[-k:][::-1]
        else:  # euclidean
            mat[internal_id, internal_id] = np.inf
            top_k_indices = np.argsort(mat[internal_id])[:k]
            
        return [(self.id_map[j], mat[internal_id, j]) for j in top_k_indices]