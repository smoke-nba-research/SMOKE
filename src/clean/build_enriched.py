"""
Builds an enriched shot-level dataset by joining shots,
players, teams, and games data and writes the result
to data/processed/shots_enriched.parquet.
"""

from pathlib import Path
import pandas as pd

SHOTS_FILE = Path("data/interim/shots_clean.parquet")
PLAYERS_FILE = Path("data/raw/players.parquet")
TEAMS_FILE = Path("data/raw/teams.parquet")
GAMES_DIR = Path("data/raw/games")
OUTPUT_FILE = Path("data/processed/shots_enriched.parquet")


def load_games():
    frames = []

    for file in sorted(GAMES_DIR.glob("games_*.parquet")):
        print(f"Loading {file.name}")
        df = pd.read_parquet(file)
        df["SEASON"] = file.stem.replace("games_", "")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def main():
    shots = pd.read_parquet(SHOTS_FILE)
    players = pd.read_parquet(PLAYERS_FILE)
    teams = pd.read_parquet(TEAMS_FILE)
    games = load_games()

    players_small = players[
        ["PERSON_ID", "DISPLAY_FIRST_LAST", "FROM_YEAR", "TO_YEAR"]
    ].drop_duplicates()

    teams_small = teams[
        ["id", "full_name", "abbreviation", "city", "state", "year_founded"]
    ].drop_duplicates()

    games_small = games[
        ["GAME_ID", "TEAM_ID", "SEASON", "MATCHUP", "WL", "PTS", "PLUS_MINUS"]
    ].drop_duplicates()

    enriched = shots.merge(
        players_small, left_on="PLAYER_ID", right_on="PERSON_ID", how="left"
    ).drop(columns=["PERSON_ID"])

    enriched = enriched.merge(
        teams_small, left_on="TEAM_ID", right_on="id", how="left"
    ).drop(columns=["id"])

    enriched = enriched.merge(
        games_small, on=["GAME_ID", "TEAM_ID", "SEASON"], how="left"
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    enriched.to_parquet(OUTPUT_FILE, index=False)

    print("Shots shape:", shots.shape)
    print("Enriched shape:", enriched.shape)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
