# `TimeSeriesVectorStore` Pipeline Documentation

## Overview

This documentation explains the **workflow** of the `TimeSeriesVectorStore` class and its related Pinecone utility functions.
It shows how raw time series data can be **transformed into embeddings**, stored in a **vector index** (FAISS or Pinecone),
and then searched or analyzed for **similar patterns**.

The system supports:

* **Embedding generation** (from raw time series)
* **FAISS vector search** (exact, IVF, or HNSW)
* **Graph-based similarity analysis**
* **Cloud-based storage with Pinecone** (optional)

---

## Main Class: `TimeSeriesVectorStore`

### Purpose

A local in-memory **vector store** for time series data,
powered by **FAISS** (Facebook AI Similarity Search).
It lets you add, search, and analyze time series in **high-dimensional vector space**.

---

### **Initialization**

```python
store = TimeSeriesVectorStore(dimension=27, index_type="flat")
```

| Parameter    | Type  | Description                                                             |
| ------------ | ----- | ----------------------------------------------------------------------- |
| `dimension`  | `int` | Size of the vector embedding (depends on FFT components)                |
| `index_type` | `str` | `"flat"` (exact search), `"ivf"` (clustered), or `"hnsw"` (graph-based) |

**Example in Jupyter Notebook:**

```python
from ts_to_vector import build_feature_matrix

store = TimeSeriesVectorStore(
    dimension=calculate_embedding_dimension(fft_components=20),
    index_type="flat"
)
```

---

## Workflow Summary

```text
Step 1: Prepare raw time series (list of NumPy arrays)
Step 2: Convert each time series into embeddings with build_feature_matrix(...)
Step 3: Add embeddings to FAISS index (store.add_time_series)
Step 4: Search for similar patterns (store.search_similar)
Step 5: (Optional) Build similarity graph (store.build_similarity_graph)
Step 6: (Optional) Save/Load index for later use
```

---

## Function Reference

### 1. `add_time_series(...)`

* **Purpose**: Convert raw time series into embeddings and store in FAISS.
* **Parameters**:

| Parameter          | Type                   | Description                                |
| ------------------ | ---------------------- | ------------------------------------------ |
| `time_series_list` | `List[np.ndarray]`     | List of time series arrays                 |
| `window_size`      | `int`                  | Sliding window size for feature extraction |
| `stride`           | `int`                  | Step size between windows                  |
| `fft_components`   | `int`                  | Number of FFT frequency components         |
| `metadata_list`    | `Optional[List[dict]]` | Extra information for each series          |

**Example:**

```python
import numpy as np

# Two simple time series
ts1 = np.sin(np.linspace(0, 10, 100))
ts2 = np.cos(np.linspace(0, 10, 100))

store.add_time_series(
    [ts1, ts2],
    window_size=50,
    stride=1,
    fft_components=20,
    metadata_list=[{"label": "sine"}, {"label": "cosine"}]
)
```

---

### 2. `search_similar(...)`

* **Purpose**: Find the most similar stored time series to a query.
* **Returns**: Distances, indices, and metadata.

**Example:**

```python
query = np.sin(np.linspace(0, 10, 100))
distances, indices, metadata = store.search_similar(query, k=3)

print("Distances:", distances)
print("Metadata:", metadata)
```

---

### 3. `build_similarity_graph(...)`

* **Purpose**: Create a **NetworkX graph** where:

  * Nodes = stored time series
  * Edges = similarity connections
* **Useful for**: Clustering, community detection, and visualization.

**Example:**

```python
import matplotlib.pyplot as plt
import networkx as nx

G = store.build_similarity_graph(k_neighbors=3)
nx.draw(G, with_labels=True)
plt.show()
```

---

### 4. Save & Load

```python
store.save_index("my_timeseries_index")
store.load_index("my_timeseries_index")
```

---

## Pinecone Cloud Utilities

If you want to store and search time series in the **cloud**,
you can use the helper functions:

### 1. `create_pinecone_index(...)`

Creates a new Pinecone index in the cloud.

```python
index = create_pinecone_index(
    api_key="YOUR_PINECONE_KEY",
    index_name="timeseries-demo",
    dimension=calculate_embedding_dimension(fft_components=20)
)
```

---

### 2. `add_timeseries_to_pinecone(...)`

Uploads embeddings to Pinecone.

```python
add_timeseries_to_pinecone(
    index,
    [ts1, ts2],
    window_size=50,
    stride=1,
    fft_components=20,
    metadata_list=[{"label": "sine"}, {"label": "cosine"}]
)
```

---

### 3. `search_pinecone(...)`

Finds the most similar embeddings stored in Pinecone.

```python
results = search_pinecone(index, query, k=3)
print(results)
```

---

### 4. `delete_pinecone_index(...)`

Deletes an index from Pinecone.

```python
delete_pinecone_index("YOUR_PINECONE_KEY", "timeseries-demo")
```

---

## Helper: `calculate_embedding_dimension(...)`

```python
dim = calculate_embedding_dimension(fft_components=20)
print(dim)  # 27
```

This ensures you set the **correct dimension** when initializing your store.

---

## Example: Full Jupyter Workflow

```python
import numpy as np

# Step 1: Create a store
store = TimeSeriesVectorStore(
    dimension=calculate_embedding_dimension(fft_components=20)
)

# Step 2: Add data
ts_data = [
    np.sin(np.linspace(0, 10, 100)),
    np.cos(np.linspace(0, 10, 100))
]
store.add_time_series(ts_data, window_size=50, stride=1, fft_components=20)

# Step 3: Search
query_ts = np.sin(np.linspace(0, 10, 100))
distances, indices, metadata = store.search_similar(query_ts, k=2)
print("Matches:", metadata)

# Step 4: Build and plot graph
G = store.build_similarity_graph(k_neighbors=2)
import matplotlib.pyplot as plt
import networkx as nx
nx.draw(G, with_labels=True)
plt.show()
```

---

## Notes

* FAISS must be installed (`pip install faiss-cpu`).
* Pinecone features require `pip install pinecone-client`.
* Embeddings are **mean-pooled** over sliding windows for compactness.
* For larger datasets, consider `ivf` or `hnsw` for faster searches.
