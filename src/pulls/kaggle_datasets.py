"""Download our two supporting Kaggle datasets.

Before this works:
1. `pip install kaggle` (already in requirements.txt)
2. A Kaggle API token at ~/.kaggle/kaggle.json (Windows: %USERPROFILE%\\.kaggle\\kaggle.json)
   Grab yours from kaggle.com -> account settings -> "Create New API Token"

Each dataset lands in its own subfolder under data/raw/kaggle/.
Idempotent — already-downloaded datasets get skipped.

If one of the slugs below stops working (Kaggle URLs can change),
look up the current one at kaggle.com and update the constant.
"""

from __future__ import annotations
from pathlib import Path

from src.pulls._paths import KAGGLE_DIR, ensure_dirs

# Dataset slugs (the "owner/dataset" part of the Kaggle URL).
# Both are current as of project setup. If one 404s, look up the
# current slug at kaggle.com and swap it in.
SHOT_LOGS_SLUG = (
    "dansbecker/nba-shot-logs"  # 128k shots, 2014-15 only (for prototyping)
)
HISTORICAL_STATS_SLUG = (
    "sumitrodatta/nba-aba-baa-stats"  # 1947-present, actively maintained
)

DATASETS = {
    "shot_logs_2014_15": SHOT_LOGS_SLUG,
    "historical_stats": HISTORICAL_STATS_SLUG,
}


def _has_extracted_files(dest: Path) -> bool:
    """True if the folder exists and has at least one non-hidden file in it."""
    if not dest.exists():
        return False
    return any(p.is_file() and not p.name.startswith(".") for p in dest.iterdir())


def download_dataset(slug: str, dest: Path) -> None:
    """Download and unzip one Kaggle dataset into `dest`."""
    # Import here so the module can be loaded without Kaggle creds being set up
    # (handy for inspection); creds only matter at actual download time.
    from kaggle import api as kaggle_api  # type: ignore

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {slug} -> {dest}")
    kaggle_api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)


def main() -> None:
    ensure_dirs()
    for name, slug in DATASETS.items():
        dest = KAGGLE_DIR / name
        if _has_extracted_files(dest):
            print(f"Skip {name} (already extracted at {dest})")
            continue
        try:
            download_dataset(slug, dest)
            print(f"  Done: {name}")
        except Exception as exc:
            print(f"  FAILED: {name} ({slug}) -> {exc}")
            print(
                "    If that's a 404, check the slug at kaggle.com and update kaggle_datasets.py"
            )


if __name__ == "__main__":
    main()
