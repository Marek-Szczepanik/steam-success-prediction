"""Exploratory data analysis of the Steam games metadata.

Only analyzes the raw data (prints statistics, later saves plots);
it does not modify anything. Findings drive the cleaning in prepare_data.py.
"""
import pandas as pd

from prepare_data import download_metadata, load_metadata


def check_data_quality(df: pd.DataFrame) -> None:
    """Check data quality and print findings."""
    # Basic stats
    pos = df["positive"].sum()
    neg = df["negative"].sum()
    total = pos + neg
    print("Total reviews:", total)
    print(f"Positive reviews: {pos} ({pos / total:.1%})")
    print(f"Negative reviews: {neg} ({neg / total:.1%})")

    # Zero reviews - real, or missing data?
    n_reviews = df["positive"] + df["negative"]
    zero = n_reviews == 0
    print("Games with 0 reviews:", zero.sum())
    print("  of which recommendations > 0 (label missing):", (zero & (df["recommendations"] > 0)).sum())

    # Extreme values
    cols = ["name", "price", "positive", "negative", "required_age", "dlc_count"]
    top = df.loc[n_reviews.nlargest(5).index, cols].copy()
    for c in ["positive", "negative"]:
        top[c] = [f"{v} ({v / n:.1%})" for v, n in zip(top[c], n_reviews[top.index])]
    print("\nMost expensive:\n", df.nlargest(5, "price")[cols])
    print("\nMost reviews:\n", top)
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


def inspect_duplicates(df: pd.DataFrame, n: int = 20) -> None:
    """Show games sharing the same name + developers under several app_ids."""
    key = df["name"].str.lower() + " | " + df["developers"].map(", ".join)
    dup = df[key.duplicated(keep=False)].assign(
        key=key, n_reviews=df["positive"] + df["negative"]
    )
    print(f"\n{dup.shape[0]} games share the same name + developers under several app_ids:")
    print(dup.sort_values("key")[["name", "release_date", "price", "n_reviews"]].head(n))


def zero_reviews_by_year(df: pd.DataFrame) -> None:
    """Share of games with 0 reviews per release year.

    missing_share: among games with recommendations > 0 (they surely have
    reviews), how many still have positive + negative == 0 -> label missing.
    """
    year = pd.to_datetime(df["release_date"], format="mixed").dt.year
    zero = (df["positive"] + df["negative"]) == 0
    has_rec = df["recommendations"] > 0
    table = pd.DataFrame({
        "games": zero.groupby(year).size(),
        "zero_share": zero.groupby(year).mean(),
        "missing_share": zero[has_rec].groupby(year[has_rec]).mean(),
    }).round(3)

    print("Share of games with 0 reviews by release year:")
    print(table.loc[2010:])

if __name__ == "__main__":
    df = load_metadata(download_metadata())

    check_data_quality(df)
    inspect_duplicates(df)
    zero_reviews_by_year(df)
