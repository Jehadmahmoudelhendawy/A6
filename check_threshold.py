import mlflow

THRESHOLD = 0.80

mlflow.set_tracking_uri("file:./mlruns")

with open("model_info.txt") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
acc = run.data.metrics.get("accuracy")

print("Accuracy:", acc)

if acc is None:
    raise SystemExit("FAILED: accuracy not found")

if acc < THRESHOLD:
    raise SystemExit("FAILED: accuracy below threshold")

print("PASSED")