# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# %%
# Load dataset
df = pd.read_csv(r"C:\Users\Sameer\OneDrive\Documents\Desktop\BIA_PROJECT\water_potability.csv")

df.head()


# %%
# Check missing values
print(df.isnull().sum())

# Fill missing values with mean
df = df.fillna(df.mean())

# Distribution plots
df.hist(figsize=(12,10))
plt.show()

# Outlier check
sns.boxplot(data=df)
plt.show()


# %%
# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()


# %%
X = df.drop("Potability", axis=1)
y = df["Potability"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE()
X_res, y_res = smote.fit_resample(X_scaled, y)


# %%
models = {
    "SVC": SVC(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

results = []
for name, model in models.items():
    scores = cross_val_score(model, X_res, y_res, cv=5, scoring="accuracy")
    results.append([name, scores.mean()])

results_df = pd.DataFrame(results, columns=["Model", "CV Mean Accuracy"])
print(results_df)


# %%
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_res, y_res)

print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)


# %%
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

best_model = RandomForestClassifier(**grid.best_params_)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.show()

fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.legend()
plt.show()


# %%
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=importances.index, palette='viridis')
plt.title("Feature Importances - Best Model")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# %%
# Missing values ko median se fill karna
df['ph'] = df['ph'].fillna(df['ph'].median())
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].median())
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median())

# Check karna ki koi missing value bachi toh nahi
print("Missing values check:")
print(df.isnull().sum())

# %%
# Features (X) aur Target (y) ko alag karna
X = df.drop('Potability', axis=1)
y = df['Potability']

print("X shape:", X.shape)
print("y shape:", y.shape)

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data ko Train (80%) aur Test (20%) mein baantna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Features ko scale (standardize) karna
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train scaled shape:", X_train_scaled.shape)
print("X_test scaled shape:", X_test_scaled.shape)

# %%
from imblearn.over_sampling import SMOTE

# SMOTE se training data ko balance karna
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print("SMOTE ke baad Class Distribution:")
print(pd.Series(y_train_res).value_counts())

# %%
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# Sabhi algorithms ki dictionary
models = {
    'SVC': SVC(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

results = []

# Loop chalakar har model ko train aur evaluate karna
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy (%)': round(acc * 100, 2),
        'F1 Score (%)': round(f1 * 100, 2)
    })

# Results ko table format mein display karna
results_df = pd.DataFrame(results).sort_values(by='Accuracy (%)', ascending=False)
results_df

# %%
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Best performing model (Random Forest) par evaluation
best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train_res, y_train_res)
y_pred_best = best_model.predict(X_test_scaled)

print("--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred_best))

# Confusion Matrix Visualisation
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix - Random Forest')
plt.show()

# %%
from sklearn.model_selection import GridSearchCV

# Random Forest ke hyperparameters fine-tune karna
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_res, y_train_res)

print("Best Parameters:", grid.best_params_)
print("Best Tuning Score:", round(grid.best_score_ * 100, 2), "%")

# %%



