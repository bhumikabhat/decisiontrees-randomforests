import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Load dataset ===
df = pd.read_csv("heart.csv")  # Make sure heart.csv is in same folder
print("Dataset Shape:", df.shape)
print(df.head())

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 2. Decision Tree Classifier ===
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nClassification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt))

# === 3. Visualize Decision Tree using Matplotlib ===
plt.figure(figsize=(20, 10))
tree.plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=['No Disease', 'Disease'],
    filled=True
)
plt.savefig("decision_tree_heart.png")
plt.close()
print("✅ Decision tree saved as decision_tree_heart.png (Matplotlib)")

# === 4. Decision Tree with limited depth to avoid overfitting ===
dt_model_limited = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model_limited.fit(X_train, y_train)
y_pred_dt_limited = dt_model_limited.predict(X_test)
print("\nDecision Tree (max_depth=4) Accuracy:",
      accuracy_score(y_test, y_pred_dt_limited))

# === 5. Random Forest Classifier ===
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# === 6. Feature Importance Plot (Random Forest) ===
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
plt.title('Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.close()
print("✅ Feature importance plot saved as feature_importances.png")

# === 7. Cross-validation Scores ===
dt_cv = cross_val_score(dt_model, X, y, cv=5)
rf_cv = cross_val_score(rf_model, X, y, cv=5)
print("\nCross-validation Accuracy (Decision Tree):", dt_cv.mean())
print("Cross-validation Accuracy (Random Forest):", rf_cv.mean())
