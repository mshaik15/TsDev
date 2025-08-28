import numpy as np
import faiss
import pickle
import networkx as nx
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
from TS_to_vector import build_feature_matrix

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

class TimeSeriesVectorStore:
    def __init__(self, dimension: int, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.metadata = []
        self.is_trained = False
        self.sources = set()  # Track all data sources
        
        self._create_index()
    
    def _create_index(self):
        if self.index_type == "flat":
            # Exact search
            self.index = faiss.IndexFlatL2(self.dimension)
            self.is_trained = True
            
        elif self.index_type == "ivf":
            # Inverted file index
            nlist = 100  # number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
        elif self.index_type == "hnsw":
            # Hierarchical NSW
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.is_trained = True
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add_time_series(self, time_series_list: List[np.ndarray], window_size: int, stride: int, 
                       fft_components: int, metadata_list: Optional[List[Dict]] = None, source: str = "unknown"):
        """
        Add time series to the vector store with a source identifier
        
        Parameters:
        -----------
        source : str
            Identifier for the data source (e.g., "avocado", "donut_searches")
        """
        all_embeddings = []
        all_metadata = []
        
        for i, ts in enumerate(time_series_list):
            feature_matrix = build_feature_matrix(ts, window_size, stride, fft_components)
            ts_embedding = np.mean(feature_matrix, axis=1).reshape(1, -1)
            all_embeddings.append(ts_embedding)
            
            meta = {"ts_index": i, "length": len(ts), "source": source}
            if metadata_list and i < len(metadata_list):
                meta.update(metadata_list[i])
            all_metadata.append(meta)
        
        self.sources.add(source)

        embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)

        if not self.is_trained:
            print(f"Training FAISS index with {len(embeddings_matrix)} embeddings...")
            self.index.train(embeddings_matrix)
            self.is_trained = True
        
        # Add to index
        self.index.add(embeddings_matrix)
        self.metadata.extend(all_metadata)
        
        print(f"Added {len(embeddings_matrix)} embeddings from {source} to index. Total: {self.index.ntotal}")
    
    def add_multiple_datasets(self, datasets: Dict[str, List[np.ndarray]], window_size: int, stride: int, fft_components: int):
        """
        Add multiple datasets to the vector store.
        
        Parameters:
        -----------
        datasets : Dict[str, List[np.ndarray]]
            Dictionary where keys are source names and values are lists of time series.
        """
        for source, time_series_list in datasets.items():
            self.add_time_series(time_series_list, window_size, stride, fft_components, source=source)
    
    def get_vectors_by_source(self, source: str):
        """Get all vectors and metadata for a specific source"""
        indices = [i for i, meta in enumerate(self.metadata) if meta.get("source") == source]
        vectors = [self.index.reconstruct(i) for i in indices]
        metadata = [self.metadata[i] for i in indices]
        return vectors, metadata, indices
    
    def search_similar(self, query_ts: np.ndarray, k: int = 5, 
                      window_size: int = 50, stride: int = 1, 
                      fft_components: int = 20, source_filter: Optional[Union[str, List[str]]] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search for similar time series, optionally filtered by source
        
        Parameters:
        -----------
        source_filter : str or list of str, optional
            If provided, only return results from this source(s)
        """
        query_features = build_feature_matrix(query_ts, window_size, stride, fft_components)
        query_embedding = np.mean(query_features, axis=1).reshape(1, -1).astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get metadata for results
        result_metadata = []
        result_distances = []
        result_indices = []
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                # Apply source filter if provided
                if source_filter is not None:
                    if isinstance(source_filter, str):
                        if meta.get("source") != source_filter:
                            continue
                    else:  # list of sources
                        if meta.get("source") not in source_filter:
                            continue
                result_metadata.append(meta)
                result_distances.append(dist)
                result_indices.append(idx)
        
        return np.array(result_distances), np.array(result_indices), result_metadata
    
    def build_similarity_graph(self, k_neighbors: int = 5, distance_threshold: Optional[float] = None, 
                              source_filter: Optional[Union[str, List[str]]] = None) -> nx.Graph:
        """
        Build a similarity graph, optionally filtered by source
        """
        if self.index.ntotal == 0:
            raise ValueError("No embeddings in index. Add embeddings first.")
        
        # Get indices for the specified source (or all if no filter)
        if source_filter:
            if isinstance(source_filter, str):
                indices = [i for i, meta in enumerate(self.metadata) if meta.get("source") == source_filter]
            else:
                indices = [i for i, meta in enumerate(self.metadata) if meta.get("source") in source_filter]
            if not indices:
                raise ValueError(f"No embeddings found for source: {source_filter}")
            print(f"Building similarity graph for {source_filter} with {len(indices)} nodes...")
        else:
            indices = list(range(self.index.ntotal))
            print(f"Building similarity graph with {self.index.ntotal} nodes...")
        
        # Get embeddings for the selected indices
        all_embeddings = np.array([self.index.reconstruct(i) for i in indices], dtype=np.float32)
        
        # Search for neighbors of each embedding
        distances, neighbor_indices = self.index.search(all_embeddings, k_neighbors + 1)  # +1 because first result is self
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes with metadata
        for idx in indices:
            node_attrs = {"embedding_id": idx}
            if idx < len(self.metadata):
                node_attrs.update(self.metadata[idx])
            G.add_node(idx, **node_attrs)
        
        # Add edges
        edge_count = 0
        for i, idx in enumerate(indices):
            for j in range(1, len(neighbor_indices[i])):  # Skip first result (self)
                neighbor_idx = neighbor_indices[i][j]
                distance = distances[i][j]
                
                # Filter by distance threshold if provided
                if distance_threshold is None or distance <= distance_threshold:
                    # Apply source filter to edges if provided
                    if source_filter is not None:
                        if neighbor_idx >= len(self.metadata):
                            continue
                        neighbor_source = self.metadata[neighbor_idx].get("source")
                        if isinstance(source_filter, str):
                            if neighbor_source != source_filter:
                                continue
                        else:
                            if neighbor_source not in source_filter:
                                continue
                    if not G.has_edge(idx, neighbor_idx):  # Avoid duplicate edges
                        G.add_edge(idx, neighbor_idx, weight=distance, similarity=1.0/(1.0 + distance))
                        edge_count += 1
        
        print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def cross_dataset_similarity(self, source1: str, source2: str, metric: str = "cosine", top_k: int = 5):
        """
        Find the most similar vectors between two datasets
        
        Parameters:
        -----------
        source1, source2 : str
            Names of the datasets to compare
        metric : str
            Similarity metric ("cosine" or "euclidean")
        top_k : int
            Number of top matches to return for each vector
        
        Returns:
        --------
        Dict[int, List[Tuple[int, float]]]
            Mapping from source1 vector indices to list of (source2 vector index, similarity) tuples
        """
        # Get vectors from both sources
        vecs1, meta1, indices1 = self.get_vectors_by_source(source1)
        vecs2, meta2, indices2 = self.get_vectors_by_source(source2)
        
        if not vecs1 or not vecs2:
            raise ValueError(f"One or both sources not found: {source1}, {source2}")
        
        vecs1 = np.array(vecs1, dtype=np.float32)
        vecs2 = np.array(vecs2, dtype=np.float32)
        
        # Calculate similarity matrix
        if metric == "cosine":
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(vecs1, vecs2)
        elif metric == "euclidean":
            from sklearn.metrics.pairwise import euclidean_distances
            dist_matrix = euclidean_distances(vecs1, vecs2)
            # Convert distance to similarity
            sim_matrix = 1 / (1 + dist_matrix)
        else:
            raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")
        
        # Find top-k matches for each vector in source1
        results = {}
        for i, (vec_idx, similarities) in enumerate(zip(indices1, sim_matrix)):
            # Get indices of top-k matches
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            # Map back to original indices
            matches = [(indices2[j], similarities[j]) for j in top_indices]
            results[vec_idx] = matches
        
        return results
    
    def save_index(self, filepath: str):
        """Save FAISS index and metadata to disk."""
        filepath = Path(filepath)
        
        # Save FAISS index
        faiss.write_index(self.index, str(filepath.with_suffix('.faiss')))
        
        # Save metadata and config
        config = {
            'metadata': self.metadata,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': self.is_trained,
            'sources': list(self.sources)
        }
        
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Saved index to {filepath.with_suffix('.faiss')} and {filepath.with_suffix('.pkl')}")
    
    def load_index(self, filepath: str):
        """Load FAISS index and metadata from disk."""
        filepath = Path(filepath)
        
        # Load FAISS index
        self.index = faiss.read_index(str(filepath.with_suffix('.faiss')))
        
        # Load metadata and config
        with open(filepath.with_suffix('.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        self.metadata = config['metadata']
        self.dimension = config['dimension']
        self.index_type = config['index_type']
        self.is_trained = config['is_trained']
        self.sources = set(config.get('sources', []))
        
        print(f"Loaded index with {self.index.ntotal} embeddings from {filepath}")
        print(f"Sources in index: {', '.join(self.sources)}")

    @staticmethod
    def calculate_embedding_dimension(window_size: int, fft_components: int = 20) -> int:
        """Calculate the dimension of time series embeddings"""
        # The number of FFT features is limited by window size
        # We can't have more FFT features than window_size - 1 (since we exclude DC component)
        actual_fft_components = min(fft_components, window_size - 1)
        return 7 + actual_fft_components  # 7 statistical features + FFT components