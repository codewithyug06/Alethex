# 📊 ALETHEX Training & Evaluation Report

**Generated on:** 2026-08-16 00:20:47
**Results Directory:** `D:\Text Analytics Project\alethex\results\training_50_epochs`

---

## 🏆 Key Performance Indicators (Best Checkpoint)

| Metric | Peak Score | Epoch Achieved | Status |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | **94.88%** | Epoch 44.0 | 🟢 Optimal |
| **Weighted F1 Score** | **94.43%** | Epoch 44.0 | 🟢 Optimal |
| **Minimum Validation Loss** | **0.2275** | Epoch 44.0 | 🟢 Converged |

---

## 📈 Generated Visual Artifacts

The following publication-grade charts were generated and saved in `results\training_50_epochs\plots`:

1. **`01_loss_convergence.png`** — Training vs. Validation loss convergence curve.
2. **`02_accuracy_f1_progression.png`** — Accuracy and F1 score trajectory across epochs.
3. **`03_per_class_metrics.png`** — Precision, Recall, and F1 comparisons for Entailment, Neutral, and Contradiction.
4. **`04_confusion_matrices.png`** — Dual absolute count and row-normalized confusion matrix heatmaps.
5. **`05_learning_rate_schedule.png`** — Learning rate warmup and linear decay schedule.
6. **`07_comprehensive_dashboard_summary.png`** — Unified 4-panel executive summary dashboard.

---

## 🔬 Model Behavioral Insights

1. **Contradiction Sensitivity**:
   * The model shows highest precision on **Contradiction** pairs (>90%), which is critical for preventing hallucinations and outdated facts in downstream RAG memory filtering.
2. **Temporal Supersession**:
   * Timestamps and temporal intervals provide clear separation between historical facts and updated states.
3. **Convergence Dynamics**:
   * The DeBERTa-v3 architecture achieves deep convergence with minimal overfitting when paired with warm-up and weight decay.

