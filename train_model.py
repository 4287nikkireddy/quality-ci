
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample student data
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 6, 9, 3, 7, 8, 4, 10, 1, 9, 6],
    "attendance": [40, 45, 50, 60, 65, 70, 75, 85, 50, 68, 72, 95, 55, 80, 90, 62, 98, 35, 92, 78],
    "result": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", accuracy)

metrics = {
    "accuracy": float(accuracy),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved to metrics.json")
