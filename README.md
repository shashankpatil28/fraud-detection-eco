# AI-Based Early Fraud Detection for Sustainable Growth Metrics (MRR, CAC, LTV)

**Course:** Economics (Term Paper)
**Group:** GROUP 27

### Team Members

* **Shashank Patil (IIT2022226)**
* **Milan Bhatiya (IIT2022176)**
* **Rajat (IIT2022227)**

---

## 📌 Project Overview

This project builds an **AI-powered early fraud detection framework** focused on identifying fraudulent users within the *first 30 days of acquisition*. Early-stage fraud distorts key business metrics used for growth planning and valuation.

By detecting fraud early, the system protects and stabilizes:

* **MRR** — avoids inflated recurring revenue from fake accounts
* **CAC** — reduces wasted marketing spend on fraudulent leads
* **LTV** — prevents artificial inflation by removing synthetic churners

This repository contains a **from-scratch ML & DL pipeline** (no scikit‑learn) using:

* PyTorch
* NumPy
* Pandas
* NetworkX
* Matplotlib

---

## 🎯 Aims & Objectives

### **Primary Aim**

Develop a system to detect fraud in the first 30 days and protect financial metrics.

### **Research Objective**

Evaluate a **multi‑model AI ensemble**:

* Graph Neural Network (GNN)
* Random Forest (custom NumPy implementation)
* Isolation Forest (custom NumPy implementation)
* WGAN‑GP for synthetic fraud generation

### **Economic Objective**

Simulate impact on business KPIs:

* Target **15–25% improvement** in CAC‑to‑LTV ratio
* Projected **>680% ROI** from reduced fraud losses

---

## 🧠 AI Methodology

This project is **AI‑based**, not rule‑based. It learns patterns from data using a layered modeling approach.

| Component                  | Type              | Purpose                                                  |
| -------------------------- | ----------------- | -------------------------------------------------------- |
| **GNN**                    | Deep Learning     | Learns graph‑based fraud patterns (transaction networks) |
| **Random Forest**          | ML Ensemble       | Tabular pattern learning via scratch‑built tree ensemble |
| **Isolation Forest**       | Anomaly Detection | Finds unusual points without labels                      |
| **WGAN‑GP**                | Generative AI     | Creates synthetic fraud samples to fix data imbalance    |
| **Ensemble (2‑of‑3 vote)** | Meta‑model        | Robust final fraud decision                              |

---

## 📂 Dataset

**Dataset:** ULB Credit Card Fraud Dataset
**Rows:** 284,807 transactions
**Fraud cases:** 492 (0.172%) — extremely imbalanced

Why this dataset?

* Industry‑standard benchmark
* Realistic anonymized PCA features
* Extreme imbalance → ideal for GAN and anomaly‑based models

---

## 🚀 How to Run

```bash
git clone <repo-url>
cd eco

python3 -m venv .venv
source .venv/bin/activate

pip install torch numpy pandas networkx matplotlib scipy
```

Download `creditcard.csv` into `data/raw/`

```bash
mkdir -p figures tables data/processed data/synthetic
python3 main.py
```

Outputs:

* **figures/** → plots
* **tables/** → performance tables

---

## 📊 Current Results (as of Nov 05, 2025)

### Table 1 — Model Performance

| Model            | Accuracy | Precision | Recall | F1    |
| ---------------- | -------- | --------- | ------ | ----- |
| GNN              | 0.99827  | 0.0       | 0.0    | 0.0   |
| Random Forest    | 0.99827  | 0.0       | 0.0    | 0.0   |
| Isolation Forest | 0.99737  | 0.239     | 0.239  | 0.239 |
| **Ensemble**     | 0.99827  | 0.0       | 0.0    | 0.0   |

**Insight:** High accuracy but **0 recall → models predict "not fraud" always**

> Accuracy is meaningless in imbalanced problems

### Table 2 — Economic Simulation

| Scenario         | CAC | LTV | CAC/LTV |
| ---------------- | --- | --- | ------- |
| Before Detection | 120 | 300 | 0.4     |
| After Detection  | 120 | 300 | 0.4     |

**Why no change?**
Ensemble caught **0 fraud → correct output** until recall improves

---

## 🔍 Key Insights

* Accuracy ≠ quality in imbalanced datasets
* Isolation Forest shows promise → detects anomalies
* GNN & RF need balancing + class‑weighted loss

---

## 🛠️ Next Steps (Fix Plan)

✅ Use **WGAN‑GP synthetic samples** in training
✅ **Class‑weighted loss** for GNN

```python
criterion = nn.NLLLoss(weight=torch.FloatTensor([1.0, 578.0]))
```

✅ Improve anomaly threshold tuning for IF
✅ Retrain ensemble & evaluate economic impact

Expected outcome:

* Increase **recall > 0.5**
* Positive CAC/LTV improvement
* Realistic economic simulation

---

## 🧾 Conclusion

This project demonstrates:

* A real‑world financial fraud challenge
* Deep learning + anomaly detection pipeline
* From‑scratch ML implementation

Upcoming improvements will unlock:

* Higher fraud recall
* Proven positive business impact
* Strong economic insights for fintech & growth metrics

---

### 📎 End of README
