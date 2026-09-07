from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("This public repository ships checked-in synthetic samples and anonymized metrics.")
    print("Run the notebooks to regenerate analysis tables and figures from the public sample data.")
    for path in [PROJECT_ROOT / "data_sample", PROJECT_ROOT / "results_public", PROJECT_ROOT / "notebooks"]:
        files = sorted(p.name for p in path.glob("*"))
        print({"path": str(path.relative_to(PROJECT_ROOT)), "files": files})
    metrics = pd.read_csv(PROJECT_ROOT / "results_public" / "public_metrics_summary.csv")
    print(metrics)


if __name__ == "__main__":
    main()
