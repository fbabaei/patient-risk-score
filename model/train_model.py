import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load sample dataset (replace with real hospital data)
data = pd.read_csv("data/patients.csv")

X = data.drop("readmitted", axis=1)
y = data["readmitted"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model & preprocessor
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/preprocessor.pkl", "wb"))

print("✅ Model training complete. Files saved in /model")
