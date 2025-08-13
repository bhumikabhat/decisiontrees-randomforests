# Decision Trees & Random Forests on Heart Disease Dataset

## 📌 Overview
This project demonstrates **tree-based machine learning models** — **Decision Tree** and **Random Forest** — for **classification** using a heart disease dataset (`heart.csv`).  
We train, evaluate, visualize, and compare the models, and also analyze **feature importance**.

## 🛠 Tools & Libraries
- **Python 3.x**
- [pandas](https://pandas.pydata.org/) — data handling
- [scikit-learn](https://scikit-learn.org/) — machine learning models
- [matplotlib](https://matplotlib.org/) — visualization
- [seaborn](https://seaborn.pydata.org/) — styled plots

## 📂 Dataset
The dataset `heart.csv` contains the following features:
- `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`  
- `target` → **0 = No Disease**, **1 = Disease**

Shape: **1025 rows × 14 columns**

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install pandas scikit-learn matplotlib seaborn
2️⃣ Place dataset
Put heart.csv in the same directory as the Python script.

3️⃣ Run the script
bash
Copy
Edit
python file.py
The script will:

Train a Decision Tree and Random Forest classifier.

Save:

decision_tree_heart.png — visualization of the Decision Tree.

feature_importances.png — feature importance chart from Random Forest.

Print accuracy, classification report, and cross-validation scores.

📊 Outputs & Results
Example metrics (may vary slightly):

Decision Tree Accuracy: ~98.5%

Decision Tree (max depth = 4) Accuracy: ~83.9%
→ Shows reduced overfitting.

Random Forest Accuracy: ~100%

Cross-validation Accuracy:

Decision Tree: ~100%

Random Forest: ~99.7%

📈 Visualizations
Decision Tree

Displays splits, features, thresholds, and class predictions.

Feature Importances

Shows which features contribute most to predictions.

💡 Notes
This version does not require Graphviz — the Decision Tree is plotted using Matplotlib.

You can adjust hyperparameters like max_depth in DecisionTreeClassifier or n_estimators in RandomForestClassifier to see their effect.

📜 License
This project is for educational purposes only. Dataset source may vary.
