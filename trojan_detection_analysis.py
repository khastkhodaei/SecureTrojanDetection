#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from imblearn.over_sampling import RandomOverSampler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import seaborn as sns


# Read the data
df_train = pd.read_csv("dataset/HEROdata2.csv")

# Data preprocessing
df_train['Sequential Internal Power'].fillna(df_train['Sequential Internal Power'].median(), inplace=True)
df_train['Sequential Total Power'].fillna(df_train['Sequential Total Power'].median(), inplace=True)
df_train.drop(['Circuit'], axis=1, inplace=True)
df_train = df_train.sample(frac=1).reset_index(drop=True)
df_train['Label'].replace({"'Trojan Free'": 0, "'Trojan Infected'": 1}, inplace=True)

# Define features and target variable
x = df_train.drop(['Label'], axis=1)
y = df_train['Label']

# Exploratory Data Analysis
corr_matrix = x.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(round(corr_matrix, 2), annot=True, cmap="coolwarm", fmt='.2f', linewidths=.05)
plt.title('Features Correlation')
plt.show()

# Feature Scaling
scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)

# Handling multicollinearity
threshold = 0.96
corr_matrix = x.corr()
iters = range(len(corr_matrix.columns) - 1)
drop_cols = []
for i in iters:
    for j in range(i + 1):
        item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
        col = item.columns
        row = item.index
        val = abs(item.values)
        if val >= threshold:
            drop_cols.append(col.values[0])

drops = set(drop_cols)
x = x.drop(columns=drops)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Modeling with RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
rf_score = rf_model.score(x_test, y_test)
rf_y_pred = rf_model.predict(x_test)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)
rf_classification_report = classification_report(y_test, rf_y_pred)
rf_recall_score = recall_score(y_test, rf_y_pred)

# Hyperparameter Tuning for RandomForestClassifier
param_grid = {
    'n_estimators': [100],
    'max_features': ['sqrt'],
    'max_depth': [20],
    'max_leaf_nodes': [25]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
grid_search.fit(x_train, y_train)
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(x_train, y_train)
best_rf_score = best_rf_model.score(x_test, y_test)
best_rf_y_pred = best_rf_model.predict(x_test)
best_rf_conf_matrix = confusion_matrix(y_test, best_rf_y_pred)
best_rf_classification_report = classification_report(y_test, best_rf_y_pred)

# Cross-Validation
rf_cross_val_results = cross_validate(rf_model, scaled_x, y, cv=5,
                                      scoring=('accuracy', 'precision', 'recall', 'f1'), return_train_score=True)

# Modeling with XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)
xgb_score = xgb_model.score(x_test, y_test)
xgb_y_pred = xgb_model.predict(x_test)
xgb_conf_matrix = confusion_matrix(y_test, xgb_y_pred)
xgb_classification_report = classification_report(y_test, xgb_y_pred)

# Ensemble Model with XGBRFClassifier
ensemble_model = XGBClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.8)
ensemble_model.fit(x_train, y_train)
ensemble_score = ensemble_model.score(x_test, y_test)
ensemble_y_pred = ensemble_model.predict(x_test)
ensemble_conf_matrix = confusion_matrix(y_test, ensemble_y_pred)

# Print results
print("Random Forest Classifier:")
print("Accuracy Score:", rf_score)
print("Confusion Matrix:\n", rf_conf_matrix)
print("Classification Report:\n", rf_classification_report)
print("Recall Score:", rf_recall_score)

print("\nBest Random Forest Classifier after Hyperparameter Tuning:")
print("Accuracy Score:", best_rf_score)
print("Confusion Matrix:\n", best_rf_conf_matrix)
print("Classification Report:\n", best_rf_classification_report)

print("\nCross-Validation Results for Random Forest Classifier:")
print(rf_cross_val_results)

print("\nXGBoost Classifier:")
print("Accuracy Score:", xgb_score)
print("Confusion Matrix:\n", xgb_conf_matrix)
print("Classification Report:\n", xgb_classification_report)

print("\nEnsemble Model with XGBRFClassifier:")
print("Accuracy Score:", ensemble_score)
print("Confusion Matrix:\n", ensemble_conf_matrix)
