"""Build typed runtime Census surname tables from the published CSV files."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "model_training" / "data" / "census"
OUTPUT_DIR = ROOT / "src" / "ethnicolr2" / "data" / "census"
PERCENT_COLUMNS = [
    "pctwhite",
    "pctblack",
    "pctapi",
    "pctaian",
    "pct2prace",
    "pcthispanic",
]


def build_table(year: int) -> pd.DataFrame:
    """Return one normalized Census surname table.

    Census represents percentages based on one to four people as ``(S)``.
    The documented Census preparation divides each row's remaining percentage
    equally among its suppressed cells.
    """
    source = SOURCE_DIR / f"census_{year}.csv"
    table = pd.read_csv(source, usecols=["name", *PERCENT_COLUMNS])
    table = table.dropna(subset=["name"]).reset_index(drop=True)
    table["name"] = table["name"].astype("string")

    percentages = table[PERCENT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    suppressed = percentages.isna()
    suppressed_count = suppressed.sum(axis=1)
    remaining = 100.0 - percentages.sum(axis=1)
    imputed = remaining.div(suppressed_count.where(suppressed_count.gt(0)))
    table[PERCENT_COLUMNS] = percentages.mask(suppressed, imputed, axis=0).astype(
        "float64"
    )

    if table[PERCENT_COLUMNS].isna().any(axis=None):
        raise ValueError(f"Census {year} still contains missing percentages")
    if not table[PERCENT_COLUMNS].ge(0).all(axis=None):
        raise ValueError(f"Census {year} contains a negative percentage")

    return table


def main() -> None:
    """Write the 2000 and 2010 runtime tables as compressed Parquet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in (2000, 2010):
        table = build_table(year)
        table.to_parquet(
            OUTPUT_DIR / f"census_{year}.parquet",
            compression="zstd",
            index=False,
        )


if __name__ == "__main__":
    main()
