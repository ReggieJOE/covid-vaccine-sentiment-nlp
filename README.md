# 🧬 COVID-19 Vaccine Sentiment Analysis

**Competition:** [Zindi – To Vaccinate or Not to Vaccinate](https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate)  
**Final Rank:** Top 60 out of 800+ participants  
**Metric:** Root Mean Squared Error (RMSE)  
**Best Score:** ~0.58 (Public Leaderboard)

---

## 🎯 Objective
Build a machine learning model to classify Twitter posts related to vaccinations
as positive, neutral, or negative — helping governments monitor public sentiment
toward COVID-19 vaccines.

---

## 🗂️ Dataset
| File | Rows | Description |
|------|------|-------------|
| Train.csv | 10,000 | Labelled tweets (-1, 0, 1) |
| Test.csv | 5,177 | Unlabelled tweets for prediction |

**Target:** Continuous value between -1 (negative) and 1 (positive)

---

## 🧠 Approach

### Model
- **Base Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - Pre-trained on 124M tweets — ideal for this domain
- **Architecture:** RoBERTa backbone + CLS/Mean pooling head + regression layer
- **Loss:** SmoothL1Loss (robust to noisy labels)

### Training Strategy
- 5-Fold Stratified Cross-Validation
- Cosine LR schedule with warmup
- Gradient clipping
- Agreement-weighted samples (higher annotator agreement = higher weight)

### Post-processing
- Threshold-based label snapping toward hard labels {-1, 0, 1}
- Matched prediction distribution to train label distribution

---

## 📁 Repository Structure
