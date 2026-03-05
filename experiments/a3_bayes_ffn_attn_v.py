"""A3 - Bayesian FFN + attention value projection on AG News (BPE)."""

from runner import run_experiment


def main() -> None:
    run_experiment(
        milestone="a3",
        description="A3 Bayesian FFN + attention value projection",
    )


if __name__ == "__main__":
    main()
