import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# Load the dataset
df_train = pd.read_csv("dataset/HEROdata2.csv")

# Preprocessing
df_train.drop(['Circuit'], axis=1, inplace=True)
df_train['Label'].replace({"'Trojan Free'": 0, "'Trojan Infected'": 1}, inplace=True)
df_train.fillna(df_train.median(), inplace=True)

# Split features and target
X = df_train.drop(['Label'], axis=1)
y = df_train['Label']

# Balance the dataset using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# Neural Network model
nn_model = Sequential([
    Dense(16, input_shape=(X.shape[1],), activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=80, batch_size=10, validation_split=0.2, verbose=1)
nn_score = nn_model.evaluate(X_test, y_test, verbose=0)
print("Neural Network Model Accuracy:", nn_score[1])
