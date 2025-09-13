# credit-fraud Unsupervised using custom GNN based auto-encoders

Testing both **conventional supervised learning techniques** and **unsupervised custom GNN-based autoencoders** for credit-card fraud detection.

The workflow is:

1. **Supervised models** → Train and evaluate classifiers like **Decision Tree** and **LightGBM** to see how they handle the **extreme class imbalance**.  
2. **Unsupervised embeddings** → Build **KNN graphs**, train **GAT/GCN autoencoders**, extract **embeddings**, and then apply **clustering** (HDBSCAN, KMeans) to test whether fraud vs non-fraud **naturally separates**.

> Note: The **supervised** and **unsupervised** tracks are **independent experiments**. I first explored the supervised models; then, separately, I explored GNN autoencoders + clustering.

---

## Table of Contents

- [Motivation](#motivation)  
- [Key Contributions](#key-contributions)  
- [Dataset](#dataset)  
- [Approach Overview](#approach-overview)  
  - [Supervised Track](#supervised-track)  
  - [Unsupervised Track (GNN Autoencoders)](#unsupervised-track-gnn-autoencoders)  
- [Graph Construction (KNN)](#graph-construction-knn)  
- [Training & Evaluation Protocol](#training--evaluation-protocol)  
- [Results (Representative)](#results-representative)  
- [Interpretation & Theory](#interpretation--theory)  
  - [Why GNN Autoencoders Help](#why-gnn-autoencoders-help)  
  - [GAT vs GCN (ClusterGCN): Differences & Observations](#gat-vs-gcn-clustergcn-differences--observations)  
- [Reproducibility](#reproducibility)  
- [How to Run](#how-to-run)  
- [Design Decisions & Pitfalls](#design-decisions--pitfalls)  
- [Limitations](#limitations)  
- [Next Steps / Roadmap](#next-steps--roadmap)  
- [Ethical Use](#ethical-use)  
- [Repository Structure (suggested)](#repository-structure-suggested)  
- [License](#license)

---

## Motivation

- Fraud datasets are **highly imbalanced** (positives are extremely rare) and **noisy**.  
- Supervised models like **tree ensembles** are very strong, but I wanted to explore **graph-based autoencoders** because there’s relatively **limited end-to-end work** on applying **GNN AEs** to **tabular** fraud data.  
- By converting rows into a **graph** (via **KNN**), a **GNN** can learn **geometry-aware embeddings** that may reveal **fraud patterns** more clearly in **unsupervised** settings.

---

## Key Contributions

- **Two independent tracks**: (1) **Supervised** (Decision Tree, LightGBM) and (2) **Unsupervised** (GAT/GCN autoencoders + clustering).
- **Graphify tabular data** using **KNN** to inject local structure.
- Show that **ClusterGCN (GCN) AE + HDBSCAN** can yield a **cleaner near-binary separation** (normal vs fraud-heavy cluster) on embeddings, while **GAT AE** highlights **fraud-heavy micro-clusters**.

---

## Dataset

- Standard **credit-card transactions** dataset with label **`Class`** (`1 = fraud`, `0 = non-fraud`).  
- Fraud rate ≈ **0.1727%** (extremely imbalanced).  
- Features: anonymized numeric components (PCA-like) plus the label.

---

## Approach Overview

### Supervised Track

- **Decision Tree**  
  - Trained directly; also tried **oversampling** on the **training set** (SMOTE and simple replication).  
  - Useful to get a feel for feature separability and error modes.

- **LightGBM**  
  - Trained with **SMOTE (train-only)** and **imbalance handling** (`scale_pos_weight`, early stopping).  
  - Later, **threshold tuning** recommended (PR curve / cost-sensitive Fβ) to align with business costs.

**Metrics (supervised)**: **Precision**, **Recall**, **F1** for the positive class (fraud), **AUC**, and **Accuracy** (reported but less informative under heavy imbalance).

---

### Unsupervised Track (GNN Autoencoders)

1. **Graphify** the data via **KNN** on scaled features (`k = 5, 10`).  
2. Train two **autoencoder** families:  
   - **GAT-based AE**: attention-weighted neighborhood aggregation; **MLP decoder** reconstructs features.  
   - **ClusterGCN (GCN)-based AE**: Laplacian smoothing; **ClusterGCN** partitions the graph for scalable mini-batches; MLP decoder.  
3. **Cluster** the learned **embeddings** with **HDBSCAN** (density-based) and **KMeans (k=2)**.  
4. Evaluate **cluster compositions/purity** and qualitative separation of fraud vs non-fraud.

---

## Graph Construction (KNN)

- Build an **undirected** KNN graph over standardized features.  
- Typical values tried: **`k ∈ {5, 10}`**.  
- Optional: use **cosine similarity** as edge weights; prune weak edges.



---

## Training & Evaluation Protocol

- **Split**: 80/20 train/test.  
- **Imbalance handling (supervised)**:  
  - **SMOTE** or **replication** on **train only**.  
  - For LightGBM: tune `scale_pos_weight`, **early stopping**, and later the **classification threshold**.  
- **Autoencoders**: optimize **feature reconstruction loss** (e.g., MSE/MAE).  
- **Clustering**: run **HDBSCAN** and **KMeans (k=2)** on embeddings; summarize cluster counts by label.  
- **Reproducibility**: set seeds for `numpy`, `torch`, `random`.

---

## Results (Representative)

### Supervised

| Model | Accuracy | Precision (fraud) | Recall (fraud) | F1 | AUC |
|------|---------:|-------------------:|---------------:|---:|----:|
| Decision Tree | 99.90% | 66.67% | 71.11% | 68.82% | — |
| Decision Tree + oversampling | 99.92% | 81.25% | 68.42% | 74.29% | — |
| LightGBM (SMOTE + pos_weight) | **99.986%** | **0.9888** | **0.9263** | **0.9565** | **0.99745** |

> **Observation**: LightGBM is **highly effective**. In practice, choose an **operating threshold** via **PR curves** or **cost-sensitive Fβ (β>1)** to trade off **missed fraud** vs **false positives**.

### Unsupervised (Embeddings → Clustering)

- **GAT AE → HDBSCAN**  
  - Multiple **micro-clusters**; at least one cluster had **51 fraud / 9 non-fraud** → **strong fraud concentration**.  
  - Good separability but **fragmented** across several clusters.

- **ClusterGCN (GCN) AE → HDBSCAN**  
  - Emerged **two large clusters** with **high purity**:  
    - Cluster A: **56,610 non-fraud / 25 fraud** (~**99.956%** non-fraud)  
    - Cluster B: **49 fraud / 8 non-fraud** (~**85.965%** fraud)  
  - **KMeans (k=2)** on the same embeddings was less aligned (e.g., ~5.65% fraud in one cluster), but **HDBSCAN** matched the density structure **very well**.

> **Observation**: On KNN graphs from tabular features, **GCN/ClusterGCN embeddings + HDBSCAN** yielded a **cleaner two-cluster story** (one mostly normal, one predominantly fraud). **GAT** provided **strong but more granular** separation.

---

## Interpretation & Theory

### Why GNN Autoencoders Help

1. **Local Manifold Smoothing**  
   The KNN graph encodes neighborhood structure. **GCN/GAT** propagate information along edges, learning **smooth embeddings** that respect **local manifolds**.

2. **Label Efficiency**  
   Autoencoders minimize **reconstruction loss**—they can learn useful structure **without labels**, which is valuable when **fraud is rare**.

3. **Imbalance-Friendly**  
   Fraud points that are **locally similar** form **dense pockets**. Graph smoothing can **pull them together** and **separate** them from the larger normal manifold, improving **clusterability**.

4. **Downstream Flexibility**  
   Embeddings can feed **unsupervised** (HDBSCAN), **semi-supervised**, or **supervised** detectors—or be concatenated with tabular features for a **hybrid** model.

### GAT vs GCN (ClusterGCN): Differences & Observations

- **GCN / ClusterGCN**  
  - Operates as learned **normalized neighbor averaging** (Laplacian smoothing).  
  - Works especially well when **homophily** holds—neighbors are truly similar (as in a **KNN feature graph**).  
  - **ClusterGCN** uses **graph partitions** for scalable, regularized training.

- **GAT**  
  - Learns **attention weights** over neighbors → **neighbor-specific** aggregation.  
  - More expressive for **heterogeneous** neighborhoods.  
  - Tends to highlight nuanced local patterns → **multiple fraud-heavy micro-clusters**.

**In this project**  
- With **KNN graphs on tabular features**, **GCN/ClusterGCN** produced **globally cleaner, near-binary separation** under **HDBSCAN**.  
- **GAT** gave **strong separation**, but **fragmented** across several clusters—useful if you want **fine-grained segments** of suspicious behavior.

---

## Reproducibility

**Environment**

- Python ≥ **3.10**

Core packages:
- `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `lightgbm`, `matplotlib`
- `torch`, `torch-geometric` (match Torch/CUDA versions)
- `hdbscan`


