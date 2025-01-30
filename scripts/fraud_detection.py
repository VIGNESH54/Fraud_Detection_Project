import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
dataset_path = os.path.expanduser("~/Desktop/Fraud_Detection_Project/data/transactions.csv")

try:
    df = pd.read_csv(dataset_path)
    print("\n✅ Dataset Loaded Successfully!\n")
except FileNotFoundError:
    print("\n❌ Error: transactions.csv not found. Check your data folder.\n")
    exit()

# Display basic dataset info
print(df.info())
print(df.head())

# Create output folder for images if not exists
images_folder = os.path.expanduser("~/Desktop/Fraud_Detection_Project/images")
os.makedirs(images_folder, exist_ok=True)

# ---------------------------
# Step 1: Fraud Distribution Analysis
# ---------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Fraud", data=df, palette=["green", "red"])
plt.title("Fraud vs. Non-Fraud Transactions")
plt.xlabel("Fraud (0 = Legit, 1 = Fraudulent)")
plt.ylabel("Transaction Count")
plt.savefig(os.path.join(images_folder, "fraud_distribution.png"))
plt.show()

# ---------------------------
# Step 2: Data Preprocessing
# ---------------------------
# Convert categorical features to numerical values
categorical_cols = ["Card_Type", "Merchant", "Location"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for reference

# Remove non-relevant features
X = df.drop(["TransactionID", "Fraud"], axis=1)
y = df["Fraud"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Step 3: Train Fraud Detection Model
# ---------------------------
print("\n🚀 Training RandomForest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# ---------------------------
# Step 4: Model Evaluation
# ---------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")

print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------
# Step 5: Feature Importance Visualization
# ---------------------------
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=feature_importance.index, palette="Blues_r")
plt.title("Feature Importance in Fraud Detection")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.savefig(os.path.join(images_folder, "feature_importance.png"))
plt.show()

print("\n✅ Fraud Detection Analysis Completed! All charts saved in images/ folder.")
