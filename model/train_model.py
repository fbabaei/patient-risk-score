import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Ensure output directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ----------------------------
# 1️⃣ Generate synthetic dataset
# ----------------------------
np.random.seed(42)
num_samples = 1000

data = pd.DataFrame({
    "age": np.random.randint(18, 90, size=num_samples),
    "bmi": np.random.uniform(18, 40, size=num_samples),
    "blood_pressure": np.random.randint(90, 180, size=num_samples),
    "cholesterol": np.random.randint(150, 300, size=num_samples),
    "days_in_hospital": np.random.randint(1, 15, size=num_samples),
    "num_prev_visits": np.random.randint(0, 10, size=num_samples),
    "glucose_level": np.random.randint(70, 200, size=num_samples),
})

# Target: readmitted (1=yes, 0=no)
# Use a rule-based synthetic logic for realism
prob = (
    0.02 * (data["age"] - 40)
    + 0.03 * (data["bmi"] - 25)
    + 0.04 * (data["days_in_hospital"])
    + 0.05 * (data["num_prev_visits"])
    + 0.02 * (data["glucose_level"] - 100)
)
prob = 1 / (1 + np.exp(-prob / 100))  # sigmoid scaling
data["readmitted"] = np.random.binomial(1, prob)

data.to_csv("data/patients.csv", index=False)
print("✅ Synthetic patient data created at data/patients.csv")

# ----------------------------
# 2️⃣ Train-test split
# ----------------------------
X = data.drop("readmitted", axis=1)
y = data["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 3️⃣ Preprocess and train model
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)
print(f"📊 Train Accuracy: {train_acc:.3f}")
print(f"📊 Test Accuracy: {test_acc:.3f}")

# ----------------------------
# 4️⃣ Save model + preprocessor
# ----------------------------
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/preprocessor.pkl", "wb"))
print("💾 Model and scaler saved in /model")

# ----------------------------
# 5️⃣ Generate a few test samples
# ----------------------------
sample = X_test.sample(5, random_state=42)
sample["predicted_prob"] = model.predict_proba(scaler.transform(sample))[:, 1]
sample["predicted_class"] = (sample["predicted_prob"] > 0.5).astype(int)
sample.to_csv("data/test_samples.csv", index=False)
print("🧪 Test samples saved at data/test_samples.csv")

print("\n✅ Training pipeline completed successfully!")
