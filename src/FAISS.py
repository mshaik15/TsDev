import numpy as np
import faiss
import pickle
import networkx as nx
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# Import your existing time series to vector functions
from TS_to_vector import build_feature_matrix

class TimeSeriesVectorStore:
    def __init__(self, dimension: int, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.metadata = []  # Store metadata for each embedding (optional)
        self.is_trained = False
        
        self._create_index()
    
    def _create_index(self):
        if self.index_type == "flat":
            # Exact search (good for smaller datasets)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.is_trained = True
            
        elif self.index_type == "ivf":
            # Inverted file index (faster for large datasets)
            nlist = 100  # number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
        elif self.index_type == "hnsw":
            # Hierarchical NSW (good balance of speed/accuracy)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.is_trained = True
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add_time_series(self, time_series_list: List[np.ndarray], 
                       window_size: int = 50, stride: int = 1, 
                       fft_components: int = 20, 
                       metadata_list: Optional[List[Dict]] = None):

        all_embeddings = []
        all_metadata = []
        
        for i, ts in enumerate(time_series_list):
            # Use your existing pipeline to create feature matrix
            feature_matrix = build_feature_matrix(ts, window_size, stride, fft_components)
            
            # Each column is an embedding for one window
            # We'll take the mean across windows to get one embedding per time series
            # Alternative: you could store each window as separate embedding
            ts_embedding = np.mean(feature_matrix, axis=1).reshape(1, -1)
            all_embeddings.append(ts_embedding)
            
            # Store metadata
            meta = {"ts_index": i, "length": len(ts)}
            if metadata_list and i < len(metadata_list):
                meta.update(metadata_list[i])
            all_metadata.append(meta)
        
        # Concatenate all embeddings
        embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
        
        # Train index if needed
        if not self.is_trained:
            print(f"Training FAISS index with {len(embeddings_matrix)} embeddings...")
            self.index.train(embeddings_matrix)
            self.is_trained = True
        
        # Add to index
        self.index.add(embeddings_matrix)
        self.metadata.extend(all_metadata)
        
        print(f"Added {len(embeddings_matrix)} embeddings to index. Total: {self.index.ntotal}")
    
    def search_similar(self, query_ts: np.ndarray, k: int = 5, 
                      window_size: int = 50, stride: int = 1, 
                      fft_components: int = 20) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:

        # Convert query to embedding using your pipeline
        query_features = build_feature_matrix(query_ts, window_size, stride, fft_components)
        query_embedding = np.mean(query_features, axis=1).reshape(1, -1).astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get metadata for results
        result_metadata = [self.metadata[idx] for idx in indices[0] if idx < len(self.metadata)]
        
        return distances[0], indices[0], result_metadata
    
    def build_similarity_graph(self, k_neighbors: int = 5, distance_threshold: Optional[float] = None) -> nx.Graph:
        if self.index.ntotal == 0:
            raise ValueError("No embeddings in index. Add embeddings first.")
        
        print(f"Building similarity graph with {self.index.ntotal} nodes...")
        
        # Get all embeddings from index
        all_embeddings = self.index.reconstruct_n(0, self.index.ntotal)
        
        # Search for neighbors of each embedding
        distances, indices = self.index.search(all_embeddings, k_neighbors + 1)  # +1 because first result is self
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes with metadata
        for i in range(self.index.ntotal):
            node_attrs = {"embedding_id": i}
            if i < len(self.metadata):
                node_attrs.update(self.metadata[i])
            G.add_node(i, **node_attrs)
        
        # Add edges
        edge_count = 0
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # Skip first result (self)
                neighbor_idx = indices[i][j]
                distance = distances[i][j]
                
                # Filter by distance threshold if provided
                if distance_threshold is None or distance <= distance_threshold:
                    if not G.has_edge(i, neighbor_idx):  # Avoid duplicate edges
                        G.add_edge(i, neighbor_idx, weight=distance, similarity=1.0/(1.0 + distance))
                        edge_count += 1
        
        print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
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
            'is_trained': self.is_trained
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
        
        print(f"Loaded index with {self.index.ntotal} embeddings from {filepath}")


# Example usage and integration with your pipeline
def example_pipeline():
    """Example showing how to integrate with your time series pipeline."""
    
    # Generate some example time series data
    np.random.seed(42)
    time_series_data = []
    
    # Create some synthetic time series with different patterns
    for i in range(20):
        # Mix of trends, seasonality, and noise
        t = np.arange(200)
        trend = 0.01 * i * t
        seasonal = 5 * np.sin(2 * np.pi * t / 50) + 2 * np.cos(2 * np.pi * t / 20)
        noise = np.random.normal(0, 1, len(t))
        ts = trend + seasonal + noise
        time_series_data.append(ts)
    
    # Calculate dimension based on your pipeline
    # 7 statistical features + K FFT components
    K = 20  # FFT components
    dimension = 7 + K
    
    # Initialize vector store
    vector_store = TimeSeriesVectorStore(dimension=dimension, index_type="flat")
    
    # Add time series to the store
    metadata = [{"series_id": f"ts_{i}", "pattern": "mixed"} for i in range(len(time_series_data))]
    vector_store.add_time_series(
        time_series_data, 
        window_size=50, 
        stride=1, 
        fft_components=K,
        metadata_list=metadata
    )
    
    # Save the index
    vector_store.save_index("my_timeseries_index")
    
    # Search for similar time series
    query_ts = time_series_data[0]  # Use first time series as query
    distances, indices, result_metadata = vector_store.search_similar(
        query_ts, k=5, window_size=50, stride=1, fft_components=K
    )
    
    print("\nSimilarity search results:")
    for i, (dist, idx, meta) in enumerate(zip(distances, indices, result_metadata)):
        print(f"  Rank {i+1}: Index {idx}, Distance: {dist:.4f}, Metadata: {meta}")
    
    # Build similarity graph
    similarity_graph = vector_store.build_similarity_graph(k_neighbors=3)
    
    print(f"\nGraph statistics:")
    print(f"  Nodes: {similarity_graph.number_of_nodes()}")
    print(f"  Edges: {similarity_graph.number_of_edges()}")
    print(f"  Average clustering: {nx.average_clustering(similarity_graph):.4f}")
    
    # Optional: Save graph
    nx.write_gexf(similarity_graph, "timeseries_similarity_graph.gexf")
    
    return vector_store, similarity_graph


if __name__ == "__main__":
    # Run the example
    vector_store, graph = example_pipeline()
    
    # Load the index back (demonstration)
    new_store = TimeSeriesVectorStore(dimension=27)  # 7 + 20
    new_store.load_index("my_timeseries_index")
    print(f"\nLoaded index has {new_store.index.ntotal} embeddings")