# Bank Marketing Decision Tree Classifier (Readable & Professional)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("bank_marketing.csv", sep=";")

print("\nDataset Preview:")
print(df.head())
print("\nDataset Shape:", df.shape)

# ===============================
# 2. Encode Categorical Variables
# ===============================
df_encoded = pd.get_dummies(df, drop_first=True)

# ===============================
# 3. Features & Target
# ===============================
X = df_encoded.drop("y_yes", axis=1)
y = df_encoded["y_yes"]

# ===============================
# 4. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5. Train Decision Tree (PRUNED FOR READABILITY)
# ===============================
model = DecisionTreeClassifier(
    max_depth=4,              # controls tree size
    min_samples_leaf=100,     # avoids tiny nodes
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 6. Evaluation
# ===============================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# 7. Plot Readable Decision Tree
# ===============================
plt.figure(figsize=(36, 18))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("Readable Decision Tree – Bank Marketing Dataset")
plt.savefig("decision_tree_readable.png", dpi=300, bbox_inches="tight")
plt.show()

# ===============================
# 8. Feature Importance Plot
# ===============================
importances = model.feature_importances_
features = X.columns

feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"]
)
plt.gca().invert_yaxis()
plt.title("Top 10 Important Features")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()

# ===============================
# 9. Insights (Printed for Screenshot)
# ===============================
print("\nKey Insights:")
print("1. Call duration is the strongest predictor of subscription.")
print("2. Contact method significantly impacts customer response.")
print("3. Previous campaign outcomes influence future conversions.")
print("4. Decision Trees provide clear, interpretable decision rules.")
