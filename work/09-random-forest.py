import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless — no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, make_scorer
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

# Temporal train/test split
train = data[data['order_year'] <= 2021].copy()
test  = data[data['order_year'] > 2021].copy()
print(f'Train: {len(train):,} | Test: {len(test):,}')

# Remove categories unseen during training
train_cats = set(train['Category'].unique())
test_cats  = set(test['Category'].unique())
unseen = test_cats - train_cats
if unseen:
    test = test[~test['Category'].isin(unseen)]
    print(f'Removed {len(unseen)} unseen categories from test. Adjusted test: {len(test):,}')

# Features
drop_cols = ['Title', 'ASIN/ISBN (Product Code)']
X_train = train.drop(['Category'] + drop_cols, axis=1)
y_train = train['Category']
X_test  = test.drop(['Category'] + drop_cols, axis=1)
y_test  = test['Category']
feature_names = X_train.columns
print(f'Features: {X_train.shape[1]}')

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print('Features scaled.')

# ── 2. Baseline Random Forest ──────────────────────────────────────────────────
print('\n--- Baseline RF (full 113K) ---')
t0 = time.time()
rf_baseline = RandomForestClassifier(
    n_estimators=100,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)
rf_baseline.fit(X_train_scaled, y_train)
print(f'Baseline trained in {time.time() - t0:.1f}s')

y_pred_baseline = rf_baseline.predict(X_test_scaled)
baseline_acc        = accuracy_score(y_test, y_pred_baseline)
baseline_f1_weighted = f1_score(y_test, y_pred_baseline, average='weighted', zero_division=0)
print(f'Baseline Accuracy:      {baseline_acc:.4f}')
print(f'Baseline F1 (weighted): {baseline_f1_weighted:.4f}')

# ── 3. Feature importance ──────────────────────────────────────────────────────
importances = rf_baseline.feature_importances_
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=False).reset_index(drop=True)
print('\nTop 20 features:')
print(feat_imp.head(20).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
top_feats = feat_imp.head(30)
ax.barh(range(30), top_feats['Importance'].values, color='steelblue')
ax.set_yticks(range(30))
ax.set_yticklabels(top_feats['Feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (Gini)')
ax.set_title('Top 30 Feature Importances — Random Forest')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=150)
plt.close()
print('Saved: rf_feature_importance.png')

feat_imp['Cumulative'] = feat_imp['Importance'].cumsum()
n_90 = (feat_imp['Cumulative'] <= 0.90).sum() + 1
n_95 = (feat_imp['Cumulative'] <= 0.95).sum() + 1
print(f'Features for 90% importance: {n_90} / {len(feature_names)}')
print(f'Features for 95% importance: {n_95} / {len(feature_names)}')

# ── 4. Hyperparameter tuning (30K subsample) ───────────────────────────────────
print('\n--- Hyperparameter Search (30K subsample) ---')
SEARCH_SIZE = 30_000
np.random.seed(42)
search_idx = np.random.choice(len(X_train_scaled), size=SEARCH_SIZE, replace=False)
X_search = X_train_scaled[search_idx]
y_search = y_train.iloc[search_idx]

param_distributions = {
    'n_estimators':     [50, 100, 200],
    'max_depth':        [10, 20, 30],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features':     ['sqrt', 'log2'],
    'class_weight':     ['balanced', 'balanced_subsample', None]
}

t0 = time.time()
random_search = RandomizedSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=42),
    param_distributions,
    n_iter=15, cv=3, scoring=scorer,
    random_state=42, n_jobs=1, verbose=1
)
random_search.fit(X_search, y_search)
print(f'Search done in {time.time() - t0:.1f}s')
print(f'Best score (F1 weighted): {random_search.best_score_:.4f}')
print(f'Best params: {random_search.best_params_}')

# ── 5. Tuned model — full training ─────────────────────────────────────────────
safe_params = dict(random_search.best_params_)
if safe_params.get('max_features') is None:
    safe_params['max_features'] = 'sqrt'
if safe_params.get('max_depth') is None or safe_params.get('max_depth', 0) > 30:
    safe_params['max_depth'] = 30
best_params = safe_params

print(f'\n--- Tuned RF (full 113K) with params: {best_params} ---')
t0 = time.time()
rf_tuned = RandomForestClassifier(**best_params, n_jobs=-1, random_state=42)
rf_tuned.fit(X_train_scaled, y_train)
print(f'Tuned model trained in {time.time() - t0:.1f}s')

y_pred_tuned     = rf_tuned.predict(X_test_scaled)
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

# Confusion matrix — top 15
top_15_cats = y_test.value_counts().head(15).index.tolist()
mask_15 = y_test.isin(top_15_cats)
cm = confusion_matrix(y_test[mask_15], y_pred_tuned[mask_15], labels=top_15_cats)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=top_15_cats, yticklabels=top_15_cats, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — Top 15 Categories')
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=150)
plt.close()
print('Saved: rf_confusion_matrix.png')

# ── 6. Cross-validation ────────────────────────────────────────────────────────
print('\n--- 5-Fold Cross-Validation ---')
train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
train_df['Category'] = y_train.values
cat_counts = train_df['Category'].value_counts()
valid_cats = cat_counts[cat_counts >= 5].index
train_df_filtered = train_df[train_df['Category'].isin(valid_cats)]
X_cv = train_df_filtered.drop('Category', axis=1).values
y_cv = train_df_filtered['Category'].values
print(f'CV data: {len(X_cv):,} samples, {len(valid_cats)} categories')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv = RandomForestClassifier(**best_params, n_jobs=-1, random_state=42)

t0 = time.time()
cv_scores = cross_val_score(rf_cv, X_cv, y_cv, cv=cv, scoring=scorer, n_jobs=1)
cv_acc_scores = cross_val_score(rf_cv, X_cv, y_cv, cv=cv, scoring='accuracy', n_jobs=1)
print(f'CV done in {time.time() - t0:.1f}s')
print(f'CV F1 (weighted): {cv_scores.mean():.4f} +/- {cv_scores.std() * 2:.4f}')
print(f'CV Accuracy:      {cv_acc_scores.mean():.4f} +/- {cv_acc_scores.std() * 2:.4f}')

# ── 7. Summary ─────────────────────────────────────────────────────────────────
print()
print('=' * 55)
print('        RANDOM FOREST — RESULTS SUMMARY')
print('=' * 55)
print(f'  Dataset:              {data.shape[0]:,} samples, {X_train.shape[1]} features')
print(f'  Categories:           {data["Category"].nunique()}')
print(f'  Train size:           {len(y_train):,}')
print(f'  Test size:            {len(y_test):,}')
print()
print(f'  --- Baseline (100 trees, depth=30) ---')
print(f'  Accuracy:             {baseline_acc:.4f}')
print(f'  F1 (weighted):        {baseline_f1_weighted:.4f}')
print()
print(f'  --- Tuned RF ---')
print(f'  Best params:          {best_params}')
print(f'  Accuracy:             {tuned_acc:.4f}')
print(f'  Precision (macro):    {tuned_prec_macro:.4f}')
print(f'  Recall (macro):       {tuned_rec_macro:.4f}')
print(f'  F1 (macro):           {tuned_f1_macro:.4f}')
print(f'  F1 (weighted):        {tuned_f1_weighted:.4f}')
print()
print(f'  --- Cross-Validation (5-fold) ---')
print(f'  F1 (weighted):        {cv_scores.mean():.4f} +/- {cv_scores.std() * 2:.4f}')
print(f'  Accuracy:             {cv_acc_scores.mean():.4f} +/- {cv_acc_scores.std() * 2:.4f}')
print('=' * 55)
