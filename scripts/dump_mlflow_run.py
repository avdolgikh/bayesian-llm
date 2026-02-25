"""Dump MLflow run details to stdout."""

import sys

import mlflow


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/dump_mlflow_run.py <run_id>")
        sys.exit(1)

    run_id = sys.argv[1]
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    run = mlflow.get_run(run_id)

    print("=== PARAMS ===")
    for k, v in sorted(run.data.params.items()):
        print(f"  {k}: {v}")

    print("\n=== METRICS ===")
    for k, v in sorted(run.data.metrics.items()):
        print(f"  {k}: {v}")

    print("\n=== TAGS ===")
    for k, v in sorted(run.data.tags.items()):
        if not k.startswith("mlflow."):
            print(f"  {k}: {v}")

    print("\n=== MLFLOW TAGS ===")
    for k, v in sorted(run.data.tags.items()):
        if k.startswith("mlflow."):
            print(f"  {k}: {v}")

    print("\n=== RUN INFO ===")
    print(f"  status: {run.info.status}")
    print(f"  start: {run.info.start_time}")
    print(f"  end: {run.info.end_time}")
    print(f"  artifact_uri: {run.info.artifact_uri}")


if __name__ == "__main__":
    main()
