"""Dump MLflow run details to stdout."""

import sys

import mlflow


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    if len(sys.argv) < 2 or sys.argv[1] == "latest":
        # No arg or "latest" → fetch the most recent run
        client = mlflow.tracking.MlflowClient()
        all_experiments = client.search_experiments()
        exp_ids = [e.experiment_id for e in all_experiments]
        runs = client.search_runs(
            experiment_ids=exp_ids,
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            print("No runs found.")
            sys.exit(1)
        run_id = runs[0].info.run_id
        print(f"(latest run: {run_id})\n")
    else:
        run_id = sys.argv[1]

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
