# Data Cleaning Pipeline

This folder contains scripts used to clean and enrich the NBA shot data used throughout the project.

## Cleaning Scripts

| File                   | Purpose                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------- |
| `shots.py`             | Builds the cleaned shot dataset.                                                      |
| `build_enriched.py`    | Builds the enriched modeling dataset.                                                 |

---

## shots.py

`shots.py` processes raw shot data and prepares a cleaned shot dataset for later stages.

The script currently:

* Loads all shot parquet files from `data/raw/shots/`
* Combines seasons into a single dataset
* Adds a `SEASON` column
* Removes duplicate rows
* Removes rows missing shot location information (`SHOT_DISTANCE`, `LOC_X`, `LOC_Y`)
* Converts `GAME_DATE` to a datetime field

### Output

```text
data/interim/shots_clean.parquet
```

---

## build_enriched.py

`build_enriched.py` creates an enriched shot-level dataset by combining cleaned shot data with player, team, and game information.

The script currently:

* Loads the cleaned shot dataset
* Loads player data
* Loads team data
* Loads game data
* Joins related datasets into a single enriched table
* Writes the final enriched dataset for EDA and modeling

### Join Keys

| Dataset         | Join Key                       |
| --------------- | ------------------------------ |
| Shots ↔ Players | `PLAYER_ID` ↔ `PERSON_ID`      |
| Shots ↔ Teams   | `TEAM_ID` ↔ `id`               |
| Shots ↔ Games   | `GAME_ID`, `TEAM_ID`, `SEASON` |

### Output

```text
data/processed/shots_enriched.parquet
```

---

## Usage

### Run the cleaning pipeline

From the project root:

```powershell
python -m src.clean.shots
```

Expected output:

```text
data/interim/shots_clean.parquet
```

### Run the enrichment pipeline

From the project root:

```powershell
python -m src.clean.build_enriched
```

Expected output:

```text
data/processed/shots_enriched.parquet
```

---
