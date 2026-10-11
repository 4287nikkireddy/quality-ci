
import json

# Use the accuracy variable already calculated by your model.
metrics = {
    "accuracy": float(accuracy)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved to metrics.json")
