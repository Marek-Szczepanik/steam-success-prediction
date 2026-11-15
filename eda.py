"""Exploratory data analysis of the Steam games metadata.

Only analyzes the raw data (prints statistics, later saves plots);
it does not modify anything. Findings drive the cleaning in prepare_data.py.
"""
import pandas as pd

from prepare_data import download_metadata, load_metadata


def check_data_quality(df: pd.DataFrame) -> None:
    """
        Check zeros, extreme values, dates and duplicates in the raw data.
    """
    n_reviews = df["positive"] + df["negative"]

    # Zero reviews - real, or missing data?
    zero = n_reviews == 0
    print("Games with 0 reviews:", zero.sum())
    print("  of which recommendations > 0 (label missing):", (zero & (df["recommendations"] > 0)).sum())

    # Extreme values
    cols = ["name", "price", "positive", "negative", "required_age", "dlc_count"]
    print("\nMost expensive:\n", df.nlargest(5, "price")[cols])
    print("\nMost reviews:\n", df.loc[n_reviews.nlargest(5).index, cols])
    print("\nrequired_age values:", sorted(df["required_age"].unique()))
    print("dlc_count max:", df["dlc_count"].max())

    # Dates - unparsable and future ones
    release = pd.to_datetime(df["release_date"], format="mixed", errors="coerce")
    print("\nUnparsable dates:", release.isna().sum())
    print("Release date range:", release.min(), "-", release.max())
    print("Released after the download date:", (release > pd.Timestamp("2026-11-15")).sum())

    # Duplicates
    print("\nDuplicate app_id:", df.index.duplicated().sum())
    name_dev = df["name"].str.lower() + " | " + df["developers"].map(", ".join)
    print("Same name + developers under several app_ids:", name_dev.duplicated(keep=False).sum())


if __name__ == "__main__":
    df = load_metadata(download_metadata())

    check_data_quality(df)
