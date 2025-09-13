# credit-fraud

Testing both **conventional supervised learning techniques** and **unsupervised custom GNN-based autoencoders** for credit card fraud detection.  
The workflow is:

1. **Supervised models** → Train and evaluate classifiers like Decision Tree and LightGBM to see how well they handle the extreme class imbalance.  
2. **Unsupervised embeddings** → Build KNN graphs, train GAT/GCN autoencoders, extract embeddings, and then apply clustering (HDBSCAN, KMeans) to test whether fraud vs non-fraud naturally separates.

---

## Motivation

- Fraud datasets are **heavily imbalanced** and challenging.  
- Supervised models (trees, boosting) perform very well, but I wanted to explore **graph-based autoencoders** because there has been limited work in this direction.  
- The idea: by converting tabular rows into a **graph** (via KNN), a GNN autoencoder can learn **geometry-aware embeddings** that make fraud stand out more clearly in an unsupervised setting.

---

## Dataset

- Standard **credit card fraud dataset** with label `Class` (`1 = fraud`, `0 = non-fraud`).  
- Fraud rate: **~0.1727%** (extremely imbalanced).  
- Features: anonymized numeric components.

---

## Methods

### Supervised Track
- **Decision Tree:** trained directly and also with oversampling (SMOTE and replication).  
- **LightGBM:** trained with SMOTE on the training set only, plus imbalance-handling (`scale_pos_weight=2.0`, early stopping).  
- Metrics: Precision, Recall, F1, AUC, Accuracy.

### Unsupervised Track
- **Graph Construction:** KNN graphs (`k = 5` and `10`) built over scaled features.  
- **Autoencoders:**
  - **GAT-based AE:** attention-driven neighborhood aggregation, reconstructs features.  
  - **ClusterGCN-based AE:** Laplacian smoothing with partitioned training for scalability.  
- **Clustering on Embeddings:** HDBSCAN (density-based) and KMeans (k=2).  
- Evaluation: cluster composition, fraud purity, and separation quality.

---

## Results (Representative)

### Supervised

| Model | Accuracy | Precision (fraud) | Recall (fraud) | F1 | AUC |
|-------|----------|-------------------|----------------|----|-----|
| Decision Tree | 99.90% | 66.67% | 71.11% | 68.82% | — |
| Decision Tree + oversampling | 99.92% | 81.25% | 68.42% | 74.29% | — |
| LightGBM (SMOTE + pos_weight) | **99.986%** | **0.9888** | **0.9263** | **0.9565** | **0.99745** |

➡️ **Observation:** LightGBM is highly effective. Threshold tuning (PR curve / Fβ) is critical for aligning with business costs.

### Unsupervised

- **GAT Autoencoder → HDBSCAN**:  
  - Multiple micro-clusters emerged, with fraud concentrated (e.g., one cluster: 51 fraud / 9 non-fraud).  
  - Strong separation but fragmented across clusters.  

- **ClusterGCN Autoencoder → HDBSCAN**:  
  - Two large clusters with high purity:  
    - Cluster A: 56,610 non-fraud / 25 fraud (~99.956% non-fraud)  
    - Cluster B: 49 fraud / 8 non-fraud (~85.965% fraud)  
  - Clearer binary separation compared to GAT.

➡️ **Observation:**  
GCN/ClusterGCN embeddings yielded **cleaner two-cluster separation**, while GAT embeddings highlighted **fraud-heavy micro-clusters**.

---

## Interpretation

### Why GNN Autoencoders?
- Encode **local manifolds**: smooth embeddings across KNN neighborhoods.  
- Work **without labels**, good for rare fraud.  
- **Imbalance-friendly**: fraud points gather into dense pockets separated from normal data.  
- Provide embeddings useful for **clustering, semi-supervised learning, or hybrid pipelines**.

### GAT vs GCN
- **GCN/ClusterGCN:** great when neighbors are already homophilous (like KNN graphs). Produced a nearly binary separation.  
- **GAT:** more expressive with heterogeneous neighborhoods. Here, it fragmented fraud into multiple micro-clusters instead of one fraud cluster.  
- **Conclusion:** GCN worked better for this dataset’s KNN structure; GAT may be stronger in settings with noisier or mixed neighborhoods.

---

## Reproducibility

**Environment:**  
- Python ≥ 3.10  
- Key libs: `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `lightgbm`, `torch`, `torch-geometric`, `hdbscan`

**Data:**  
- Place `creditcard.csv` in `./data/`.

**Run:**  
- Execute the notebook `notebooks/credit-fraud (1).ipynb` step by step:
  1. Supervised models  
  2. Graph building  
  3. GAT & ClusterGCN autoencoders  
  4. Clustering on embeddings  

---

## Limitations
- Graph built from the same features (risk of leakage if not split carefully).  
- HDBSCAN/KMeans results vary with hyperparameters.  
- Pure reconstruction loss favors majority class; contrastive or edge losses could help.  
- GNNs are compute-intensive compared to tree ensembles.

---

## Next Steps
- Tune LightGBM thresholds with PR curves.  
- Add embedding-quality metrics (purity, NMI/ARI, silhouette).  
- Try weighted edges, different k values.  
- Extend autoencoders with link-prediction or contrastive losses.  
- Hybrid pipeline: combine GNN embeddings with LightGBM.  

---

