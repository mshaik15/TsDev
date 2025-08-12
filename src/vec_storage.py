import numpy as np
import faiss
import pickle
import networkx as nx
import pathlib
from typing import List, Tuple, Dict, Optional
import pinecone

class VectorStore:
    # Constructor
    def __init__(self, dim: int, index_type:str):
        self.dim = dim
        self.index_type = index_type
        self.index = None
        self.metadata = []
        self.is_trained = False

    # Create FAISS index based on type
    def create_index(self):
        if self.index_type == 'Flat':
            self.index = faiss.IndexFlatL2(self.dim)
        elif self.index_type == 'IVF':
            nlist = 100
            quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_L2)
            self.is_trained = False
        elif self.index_type == 'HNSW':
            m = 32
            self.index = faiss.IndexHNSWFlat(self.dim, m)
        else:
            raise ValueError("Unsupported index type")


