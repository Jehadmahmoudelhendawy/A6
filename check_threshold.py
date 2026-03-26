import os
import mlflow

THRESHOLD = 0.85

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")

if accuracy is None:
    raise SystemExit("Accuracy metric not found.")

if accuracy < THRESHOLD:
    raise SystemExit(f"Deployment blocked: accuracy {accuracy:.4f} is below {THRESHOLD}")

print("Threshold passed.")