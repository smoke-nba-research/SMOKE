"""
Loads raw shot data, performs cleaning and standardization,
and writes a cleaned shot table to data/interim/shots_clean.parquet.
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/shots")
OUTPUT_FILE = Path("data/interim/shots_clean.parquet")


def load_shots():
    frames = []

    for file in sorted(RAW_DIR.glob("shots_*.parquet")):
        print(f"Loading {file.name}")

        df = pd.read_parquet(file)

        df["SEASON"] = file.stem.replace("shots_", "")

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def clean_shots(df):
    df = df.drop_duplicates()

    df = df.dropna(subset=["SHOT_DISTANCE", "LOC_X", "LOC_Y"])

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    return df


def main():
    shots = load_shots()

    print("Raw shape:", shots.shape)

    shots = clean_shots(shots)

    print("Clean shape:", shots.shape)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    shots.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
