import pandas as pd
from pathlib import Path


def save_dataframe(dataframe: pd.DataFrame, output_dir: str | Path, filename: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(f"{output_dir / Path(filename)}.csv", index=False)