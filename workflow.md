# Hidden Structures in Time Series

This project extracts and visualizes hidden relationships in time series data using machine learning and vector similarity search.

```mermaid
flowchart LR
    A[User Upload] --> B[Preprocess Data]
    B --> C[Detect Timestamps]
    C --> D[Aggregate to Time Series]
    D --> E[Extract Features]
    E --> F[Convert to Vectors (TS2Vec/TST)]
    F --> G[Store in Vector DB (FAISS/Pinecone)]
    G --> H[Run Similarity Search]
    H --> I[Analyze Alignments]
    I --> J[Visualize (UMAP/PCA)]
    J --> K{Make Decisions}
    K --> L[Alerting]
    K --> M[Forecasting]
```
