THRESHOLD = 0.98

with open("model_info.txt") as f:
    run_id = f.read().strip()

import os

metrics_path = None

for root, dirs, files in os.walk("mlruns"):
    if run_id in root and "metrics" in root:
        metrics_path = os.path.join(root, "accuracy")
        break

if not metrics_path or not os.path.exists(metrics_path):
    raise SystemExit("FAILED: accuracy file not found")

with open(metrics_path) as f:
    lines = f.readlines()

if not lines:
    raise SystemExit("FAILED: accuracy file empty")

last_line = lines[-1].strip().split()
acc = float(last_line[1])

print("Accuracy:", acc)

if acc < THRESHOLD:
    raise SystemExit("FAILED: accuracy below threshold")

print("PASSED")