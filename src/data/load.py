

import pandas as pd


COLUMN_NAMES = ["target", "id", "date", "flag", "user", "text"]


def load_raw_tweets(path: str, has_header: bool = False) -> pd.DataFrame:
    """
    Read raw tweet CSV into a DataFrame with at least 'text' and 'target' columns.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.
    has_header : bool
        Set True if the CSV already has a header row.

    Returns
    -------
    pd.DataFrame with columns ['target', 'text'] (extra columns kept if present)
    """
    if has_header:
        df = pd.read_csv(path, encoding="latin-1")
    else:
        df = pd.read_csv(
            path,
            encoding="latin-1",
            names=COLUMN_NAMES,
            header=None,
        )

    if "target" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"Expected 'target' and 'text' columns, got: {list(df.columns)}"
        )

    return df


def save_processed(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = load_raw_tweets("data/raw/tweets.csv")
    print(df.shape)
    print(df["target"].value_counts())
    print(df.head())
