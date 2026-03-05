# Graph-Neural-Networks-for-Fraud-Detection
This is a group project in fulfillment of CSCI 3834, Winter 2026.

### Team Progress Summary
| Milestone | Status | Key Deliverables Done |
|---|---|---|
| 1 — Project setup & dataset audit | ✅ Completed | Repo structure, EDA notebook, requirements.txt |
| 2 — Graph construction & baseline models | 🔄 In Progress | Traditional ML notebook (RF, ET, XGB, LR, SVM) |
| 3 — GNN modeling & evaluation | ⏳ Not Started | — |
| 4 — Analysis, visualization & final report | ⏳ Not Started | — |

---

## Milestones and Task Ownership (Draft)
This section is the shared milestone record for the team. Each milestone can be mirrored as a GitHub milestone with its tasks tracked as issues.

### Milestone 1: Project setup and dataset audit — ✅ Completed
- **Lead (Abdi):** Confirm scope, define success metrics, and verify dataset availability.  
  - *How:* Review the proposal, map dataset columns to objectives, and select evaluation metrics (e.g., AUC-PR, F1).
  - *Deliverable:* Scope/metrics note and dataset inventory checklist. ✅
- **Bhabin:** Perform initial EDA, summarize feature distributions, and note data quality issues.  
  - *How:* Use pandas to profile missing values, fraud ratios, outliers, and timestamp coverage.
  - *Deliverable:* EDA notebook with summary tables/plots. ✅ (`notebooks/Fraud_detection_dataset_EDA.ipynb`)
- **Aaron:** Set up the repo structure for data, notebooks, and results; document environment setup steps.  
  - *How:* Align folders to the workflow and list required packages and setup steps in README or notes.
  - *Deliverable:* Updated setup checklist and repository structure overview. ✅ (`requirements.txt`)

### Milestone 2: Graph construction and baseline models — 🔄 In Progress
- **Lead (Abdi):** Define the graph schema (nodes/edges) and feature engineering plan.  
  - *How:* Specify node/edge types, timestamps, categorical encodings, and aggregation rules.
  - *Deliverable:* Graph schema diagram or table plus feature list.
- **Bhabin:** Implement data preprocessing steps and build train/validation/test splits.  
  - *How:* Clean timestamps, encode categories, scale numeric features, and create stratified splits.
  - *Deliverable:* Preprocessing script/notebook with saved split indices.
- **Aaron:** Train and evaluate baseline traditional ML models (e.g., logistic regression, random forest).  
  - *How:* Use scikit-learn pipelines, handle class imbalance, and report baseline metrics.
  - *Deliverable:* Baseline metrics table and model comparison notes. ✅ (`notebooks/Traditional_ML.ipynb`)

### Milestone 3: GNN modeling and evaluation — ⏳ Not Started
- **Lead (Abdi):** Implement the GNN model and training loop with PyTorch Geometric.  
  - *How:* Select architecture (GCN/GAT), add class imbalance handling, and define early-stopping criteria.
  - *Deliverable:* Training script/config and saved checkpoints.
- **Bhabin:** Assist with graph data loaders and experiment logging.  
  - *How:* Build PyG datasets, ensure reproducibility (seeds), and log metrics per epoch.
  - *Deliverable:* Data loader module and experiment log template.
- **Aaron:** Evaluate GNN results, compare with baselines, and summarize metrics.  
  - *How:* Compute AUC-PR/ROC, confusion matrix, and aggregate results for comparison.
  - *Deliverable:* Evaluation summary with plots/tables.

### Milestone 4: Analysis, visualization, and final report — ⏳ Not Started
- **Lead (Abdi):** Lead comparative analysis and finalize conclusions.  
  - *How:* Synthesize model comparisons, highlight insights, and document limitations.
  - *Deliverable:* Final analysis narrative and conclusions section.
- **Bhabin:** Build visualizations (fraud subgraphs, metrics plots).  
  - *How:* Use seaborn/matplotlib for metrics and NetworkX for graph visuals.
  - *Deliverable:* Final figure set saved for the report.
- **Aaron:** Draft the final report and presentation materials, incorporating team feedback.  
  - *How:* Assemble report outline, integrate figures, and iterate with team review.
  - *Deliverable:* Report draft and presentation deck.
