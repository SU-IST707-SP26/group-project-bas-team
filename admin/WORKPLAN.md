# WORKPLAN.md

## Active Plan

### Milestone 1: Project Planning
- [✅] M1.T1 — Determine project idea (Team)
- [✅] M1.T2 — Identify dataset to use (Team)
- [✅] M1.T3 — Create appropriate files in project repository (Alexa)

### Milestone 2: Data Acquisition
- [✅] M2.T1 — Download Open E-commerce dataset from Harvard Dataverse (Alexa)
- [✅] M2.T2 — Initial exploratory data analysis

### Milestone 3: Data Preparation
- [✅] M3.T1 — Join purchase data with survey data on SurveyResponseID field
- [✅] M3.T2 — Clean full dataset (remove or impute missing values, standardize data, encode categorical data, convert data types)
- [✅] M3.T3 Identify and remove irrelevant features
- [✅] M3.T4 Conduct temporal feature engineering
- [✅] M3.T5 Consolidate categories into broader parent categories

### Milestone 4: Visualize Data
- [✅] M4.T1 Perform dimensionality reduction on cleaned dataset
- [✅] M4.T2 Visualize reduced data

### Milestone 5: Modeling
- [✅] M5.T1 — Split data into training and test sets by date
- [✅] M5.T2 — Association rule mining
- [✅] M5.T4 — Random forest model - Shahaan
- [✅] M5.T5 — XGBoost model - Shahaan
- [✅] M5.T6 — Long Short-Term Memory (LSTM) model
- [✅] M5.T7 — Wide and deep neural networks - Alexa
- [✅] M5.T9 — Bayesian rule mining
- [✅] M5.T10 — Predict demographics from purchases

### Milestone 6: Evaluation
- [✅] M6.T1 — Calculate precision, recall, and F1 scores
- [⏳] M6.T4 — Final documentation (Team)

---

## Changelog

### 2026-03-01
- (Alexa) 🆕 M4.T1, M4.T2 — Added visualization milestone

### 202-05-02
- (Alexa) ❌ M5.T3 — Abandoned "Sequential pattern mining" - Unable to implement, decided to just incorpate sequential features in other modeling attempts.
- (Alexa) ❌ M5.T8 — Abandoned "Clustering methods" - Unable to derive meaningful results from clusters.
- (Alexa) ❌ M6.T3 — Abandoned "Perform Cross Validation" - Cross validation to be performed when calculation accuracy, F1-scores, etc. No need to implement separately.
- (Alexa) 🆕 M5.T10 — Added demographic prediction milestone
- (Alexa) 🆕 M3.T5 — Added category consolidation milestone.

### 2026-05-05
- (Alexa) ❌ M6.T2 — Abandoned "Analyze model performance by demographic segment" - Unable to complete in time to submit project.
