"""One-shot helper to set up Kaggle credentials.

Run this once per machine. It'll prompt you for your Kaggle username and
API key, write them to the right place, set the right permissions, and
test that the kaggle CLI can authenticate.

Usage:
    python -m src.pulls.setup_kaggle

Where to get your username + key:
    1. Go to https://www.kaggle.com/settings
    2. Scroll to the "API" section
    3. Click "Create New API Token"
    4. A box pops up showing your username and key (or downloads a
       kaggle.json file with both values inside).

If Kaggle downloaded a kaggle.json file and you don't want to retype the
values, you can just open that file in Notepad to see them, then paste
them here when prompted.
"""

from __future__ import annotations
import json
import stat
from pathlib import Path


def main() -> None:
    print("=" * 60)
    print("Kaggle credentials setup")
    print("=" * 60)
    print()
    print("This will create ~/.kaggle/kaggle.json so the project's")
    print("Kaggle pulls can authenticate as you.")
    print()
    print("If you've never created an API token before, go to:")
    print("    https://www.kaggle.com/settings")
    print("Scroll to the 'API' section -> click 'Create New API Token'.")
    print()

    username = input("Kaggle username: ").strip()
    if not username:
        print("ERROR: username cannot be empty. Aborting.")
        return

    key = input("Kaggle API key: ").strip()
    if not key:
        print("ERROR: key cannot be empty. Aborting.")
        return

    # Find the right folder. ~/.kaggle/ on every OS.
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    creds_path = kaggle_dir / "kaggle.json"

    if creds_path.exists():
        ans = (
            input(f"\n{creds_path} already exists. Overwrite? [y/N]: ").strip().lower()
        )
        if ans != "y":
            print("Aborted. Existing file untouched.")
            return

    # Write the credentials. Kaggle expects a single-line JSON object.
    with creds_path.open("w", encoding="utf-8") as f:
        json.dump({"username": username, "key": key}, f)

    # On Linux/Mac, the kaggle CLI complains if the file is world-readable.
    # On Windows this is a no-op but doesn't hurt.
    try:
        creds_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass

    print(f"\nWrote credentials to: {creds_path}")
    print()

    # Verify by asking the kaggle library to authenticate.
    print("Testing authentication...")
    try:
        from kaggle import api as kaggle_api  # type: ignore

        kaggle_api.authenticate()
        print("  SUCCESS - kaggle CLI can authenticate as you.")
        print()
        print("You're set. To run the Kaggle dataset downloads:")
        print("    python -m src.pulls.kaggle_datasets")
        print()
        print("Or to run every data pull in order:")
        print("    python -m src.pulls.fetch_all")
    except Exception as exc:
        print(f"  FAILED: {exc}")
        print()
        print("Double-check that the username and key you entered match what")
        print("Kaggle shows under https://www.kaggle.com/settings -> API.")
        print(f"If you need to retry, delete {creds_path} and re-run this script.")


if __name__ == "__main__":
    main()
