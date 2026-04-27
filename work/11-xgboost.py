import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, make_scorer,
    confusion_matrix
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold
import time
import warnings
warnings.filterwarnings('ignore')

scorer = make_scorer(f1_score, average='weighted', zero_division=0)

# ── 1. Data loading ────────────────────────────────────────────────────────────
print('Loading data...')
data = pd.read_csv('../data/cleaned_data.csv')
print(f'Dataset shape: {data.shape}')
print(f'Unique categories: {data["Category"].nunique()}')

train = data[data['order_year'] <= 2021].copy()
test  = data[data['order_year'] > 2021].copy()
print(f'Train: {len(train):,} | Test: {len(test):,}')

train_cats = set(train['Category'].unique())
unseen = set(test['Category'].unique()) - train_cats
if unseen:
    test = test[~test['Category'].isin(unseen)]
    print(f'Removed {len(unseen)} unseen categories. Adjusted test: {len(test):,}')

drop_cols = ['Title', 'ASIN/ISBN (Product Code)']
X_train = train.drop(['Category'] + drop_cols, axis=1)
X_test  = test.drop(['Category'] + drop_cols, axis=1)
feature_names = X_train.columns

# XGBoost requires contiguous integer labels [0, num_class)
le = LabelEncoder()
le.fit(train['Category'])
y_train = le.transform(train['Category'])
y_test  = le.transform(test['Category'])
num_classes = len(le.classes_)
print(f'Features: {X_train.shape[1]} | Classes: {num_classes}')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print('Features scaled.')

# ── 2. Baseline XGBoost ────────────────────────────────────────────────────────
print('\n--- Baseline XGBoost (full 113K, CPU) ---')
t0 = time.time()
xgb_baseline = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    objective='multi:softprob',
    num_class=num_classes,
    tree_method='hist',
    device='cpu',
    n_jobs=-1,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_baseline.fit(X_train_scaled, y_train)
print(f'Baseline trained in {time.time() - t0:.1f}s')

y_pred_baseline  = xgb_baseline.predict(X_test_scaled)
baseline_acc     = accuracy_score(y_test, y_pred_baseline)
baseline_f1_weighted = f1_score(y_test, y_pred_baseline, average='weighted', zero_division=0)
print(f'Baseline Accuracy:      {baseline_acc:.4f}')
print(f'Baseline F1 (weighted): {baseline_f1_weighted:.4f}')

# ── 3. Feature importance ──────────────────────────────────────────────────────
importances = xgb_baseline.feature_importances_
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=False).reset_index(drop=True)
print('\nTop 20 features:')
print(feat_imp.head(20).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
top_feats = feat_imp.head(30)
ax.barh(range(30), top_feats['Importance'].values, color='darkorange')
ax.set_yticks(range(30))
ax.set_yticklabels(top_feats['Feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (gain)')
ax.set_title('Top 30 Feature Importances — XGBoost')
plt.tight_layout()
plt.savefig('xgb_feature_importance.png', dpi=150)
plt.close()
print('Saved: xgb_feature_importance.png')

feat_imp['Cumulative'] = feat_imp['Importance'].cumsum()
n_90 = (feat_imp['Cumulative'] <= 0.90).sum() + 1
n_95 = (feat_imp['Cumulative'] <= 0.95).sum() + 1
print(f'Features for 90% importance: {n_90} / {len(feature_names)}')
print(f'Features for 95% importance: {n_95} / {len(feature_names)}')

# ── 4. Hyperparameter tuning (30K subsample) ───────────────────────────────────
print('\n--- Hyperparameter Search (30K subsample) ---')
SEARCH_SIZE = 30_000
np.random.seed(42)
search_idx = np.random.choice(len(X_train_scaled), SEARCH_SIZE, replace=False)
X_search = X_train_scaled[search_idx]
y_search_raw = y_train[search_idx]

# Re-encode to contiguous [0, n) for the subsample's classes
le_search = LabelEncoder()
y_search = le_search.fit_transform(y_search_raw)
num_classes_search = len(le_search.classes_)

param_distributions = {
    'n_estimators':      [30, 50, 100],
    'max_depth':         [3, 4, 5],
    'learning_rate':     [0.05, 0.1, 0.2],
    'subsample':         [0.8, 1.0],
    'colsample_bytree':  [0.8, 1.0],
    'min_child_weight':  [1, 3]
}

t0 = time.time()
random_search = RandomizedSearchCV(
    xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes_search,
        tree_method='hist',
        device='cpu',
        n_jobs=-1,
        random_state=42,
        eval_metric='mlogloss'
    ),
    param_distributions,
    n_iter=10, cv=3, scoring=scorer,
    random_state=42, n_jobs=1, verbose=1
)
random_search.fit(X_search, y_search)
print(f'Search done in {time.time() - t0:.1f}s')
print(f'Best score (F1 weighted): {random_search.best_score_:.4f}')
print(f'Best params: {random_search.best_params_}')
best_params = random_search.best_params_

# ── 5. Tuned model — full training ─────────────────────────────────────────────
print(f'\n--- Tuned XGBoost (full 113K) with params: {best_params} ---')
t0 = time.time()
xgb_tuned = xgb.XGBClassifier(
    **best_params,
    objective='multi:softprob',
    num_class=num_classes,
    tree_method='hist',
    device='cpu',
    n_jobs=-1,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_tuned.fit(X_train_scaled, y_train)
print(f'Tuned model trained in {time.time() - t0:.1f}s')

y_pred_tuned     = xgb_tuned.predict(X_test_scaled)
tuned_acc        = accuracy_score(y_test, y_pred_tuned)
tuned_prec_macro = precision_score(y_test, y_pred_tuned, average='macro', zero_division=0)
tuned_rec_macro  = recall_score(y_test, y_pred_tuned, average='macro', zero_division=0)
tuned_f1_macro   = f1_score(y_test, y_pred_tuned, average='macro', zero_division=0)
tuned_f1_weighted= f1_score(y_test, y_pred_tuned, average='weighted', zero_division=0)

print(f'Tuned Accuracy:          {tuned_acc:.4f}')
print(f'Tuned Precision (macro): {tuned_prec_macro:.4f}')
print(f'Tuned Recall (macro):    {tuned_rec_macro:.4f}')
print(f'Tuned F1 (macro):        {tuned_f1_macro:.4f}')
print(f'Tuned F1 (weighted):     {tuned_f1_weighted:.4f}')
print(f'Improvement over baseline: {tuned_acc - baseline_acc:+.4f}')

top_15_cats = pd.Series(y_test).value_counts().head(15).index.tolist()
mask_15 = pd.Series(y_test).isin(top_15_cats).values
original_top_15 = le.inverse_transform(top_15_cats)
cm = confusion_matrix(y_test[mask_15], y_pred_tuned[mask_15], labels=top_15_cats)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=original_top_15, yticklabels=original_top_15, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — Top 15 Categories (XGBoost)')
plt.tight_layout()
plt.savefig('xgb_confusion_matrix.png', dpi=150)
plt.close()
print('Saved: xgb_confusion_matrix.png')

# ── 6. Cross-validation (30K subsample) ───────────────────────────────────────
print('\n--- 5-Fold Cross-Validation (30K subsample) ---')
CV_SIZE = 30_000
np.random.seed(0)
cv_idx = np.random.choice(len(X_train_scaled), CV_SIZE, replace=False)
X_cv_sub = X_train_scaled[cv_idx]
y_cv_raw = y_train[cv_idx]

le_cv = LabelEncoder()
y_cv = le_cv.fit_transform(y_cv_raw)
num_classes_cv = len(le_cv.classes_)

# Filter to categories with >= 5 samples for StratifiedKFold
cv_series = pd.Series(y_cv)
valid_mask = cv_series.isin(cv_series.value_counts()[cv_series.value_counts() >= 5].index).values
X_cv = X_cv_sub[valid_mask]
y_cv = y_cv[valid_mask]

# Re-encode after filtering so classes are contiguous again
le_cv2 = LabelEncoder()
y_cv = le_cv2.fit_transform(y_cv)
num_classes_cv2 = len(le_cv2.classes_)
print(f'CV subsample: {len(X_cv):,} samples, {num_classes_cv2} categories')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_cv = xgb.XGBClassifier(
    **best_params,
    objective='multi:softprob',
    num_class=num_classes_cv2,
    tree_method='hist',
    device='cpu',
    n_jobs=-1,
    random_state=42,
    eval_metric='mlogloss'
)

t0 = time.time()
cv_scores     = cross_val_score(xgb_cv, X_cv, y_cv, cv=cv, scoring=scorer, n_jobs=1)
cv_acc_scores = cross_val_score(xgb_cv, X_cv, y_cv, cv=cv, scoring='accuracy', n_jobs=1)
print(f'CV done in {time.time() - t0:.1f}s')
print(f'CV F1 (weighted): {cv_scores.mean():.4f} +/- {cv_scores.std() * 2:.4f}')
print(f'CV Accuracy:      {cv_acc_scores.mean():.4f} +/- {cv_acc_scores.std() * 2:.4f}')

# ── 7. Summary ─────────────────────────────────────────────────────────────────
print()
print('=' * 55)
print('          XGBOOST — RESULTS SUMMARY')
print('=' * 55)
print(f'  Dataset:              {data.shape[0]:,} samples, {X_train.shape[1]} features')
print(f'  Categories:           {num_classes}')
print(f'  Train size:           {len(y_train):,}')
print(f'  Test size:            {len(y_test):,}')
print()
print(f'  --- Baseline XGBoost ---')
print(f'  Accuracy:             {baseline_acc:.4f}')
print(f'  F1 (weighted):        {baseline_f1_weighted:.4f}')
print()
print(f'  --- Tuned XGBoost (full 113K train) ---')
print(f'  Best params:          {best_params}')
print(f'  Accuracy:             {tuned_acc:.4f}')
print(f'  Precision (macro):    {tuned_prec_macro:.4f}')
print(f'  Recall (macro):       {tuned_rec_macro:.4f}')
print(f'  F1 (macro):           {tuned_f1_macro:.4f}')
print(f'  F1 (weighted):        {tuned_f1_weighted:.4f}')
print()
print(f'  --- Cross-Validation (5-fold, 30K subsample) ---')
print(f'  F1 (weighted):        {cv_scores.mean():.4f} +/- {cv_scores.std() * 2:.4f}')
print(f'  Accuracy:             {cv_acc_scores.mean():.4f} +/- {cv_acc_scores.std() * 2:.4f}')
print()
print(f'  --- Model Comparison ---')
print(f'  Random Forest (tuned, full data): 7.28% accuracy')
print(f'  Neural Network (best):            7.10% accuracy')
print(f'  XGBoost (tuned, full data):       {tuned_acc*100:.2f}% accuracy')
print('=' * 55)
